import sys
from RBPNet import *

from metrics import *
from module import Module
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seqpro as sp
import seqdata as sd
from tqdm import tqdm
import numpy as np
import logomaker
import torch.nn as nn
from modiscolite.tfmodisco import TFMoDISco
from scipy.stats import entropy
from seqexplainer import attribute

def plot_patterns(patterns, entropy_threshold = 1, out_prefix=''):
    f, axes = plt.subplots(len(patterns), 1, figsize = (3,3), sharex = True)

    try:
        len(axes)
    except:
        axes = [axes]
    for i, (p, ax) in enumerate(zip(patterns, axes)):
        pwm_df = pd.DataFrame(p.sequence,
                     columns = ["A", "C", "G", "U"]
                             )
        pwm_df.to_csv(f'{out_prefix}.{i}.pwm.csv')
        e = entropy(pwm_df,axis = 1)
        try:
            start = np.where(e<entropy_threshold)[0][0]
            end = np.where(e<entropy_threshold)[0][-1]
            sub = pwm_df.loc[start:end]
            sub.index = np.arange(sub.shape[0])
            logomaker.Logo(sub, ax = ax)
        except Exception as e:
            print(e)
    plt.show()
    plt.savefig(f'{out_prefix}.pdf')

def find_patterns(seq, model):
    ism_attrs = attribute(
    model,
    inputs=seq,
    method="DeepLift",
    target=0,
    batch_size=128
    )

    pos_patterns, neg_patterns = TFMoDISco(
        hypothetical_contribs=ism_attrs.transpose(0, 2, 1),
        one_hot=seq.cpu().numpy().transpose(0, 2, 1),
        sliding_window_size=12,
    )

    return ism_attrs, pos_patterns, neg_patterns

class RBPNet_wrapper(nn.Module):
    def __init__(self, module):
        super(RBPNet_wrapper, self).__init__()
        self.rbpnet = module.arch
    def forward(self, x):
        x_total, x_signal, x_ctl, x_mix, d_log_odds = self.rbpnet(x)
        return d_log_odds.unsqueeze(dim = 1)
    
def predict(model, x, batch_size=128, verbose=True):
    with torch.no_grad():
        device = model.device
        model.eval()
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x.astype(np.float32))
        totals = []
        clips = []
        inputs = []
        pis = []
        dlogodds = []
        for _, i in tqdm(
            enumerate(range(0, len(x), batch_size)),
            desc="Predicting on batches",
            total=len(x) // batch_size,
            disable=not verbose,
        ):
            batch = x[i : i + batch_size].to(device)
            # x_total, x_signal, x_ctl, x_mix, d_log_odds
            outputs = model.arch(batch)

            # eCLIP csln
            out = outputs[0].detach().cpu()
            totals.append(out)

            clips.append(outputs[1].detach().cpu())

            inputs.append(outputs[2].detach().cpu())

            # mixing coef
            pi = outputs[3].detach().cpu()
            pis.append(pi)

            # dlog
            d = outputs[4].detach().cpu()
            dlogodds.append(d)

        totals = torch.cat(totals)
        inputs = torch.cat(inputs)
        clips = torch.cat(clips)
        pis = torch.cat(pis)
        dlogodds = torch.cat([i if i.dim()==1 else i.unsqueeze(-1) for i in dlogodds])

        ip_probs = torch.nn.functional.softmax(clips, dim=1)
        in_probs = torch.nn.functional.softmax(inputs, dim=1)
        
        return totals, ip_probs, in_probs, pis, dlogodds
def find_latest_ckp(log_directory):
    checkpoints = list((Path(log_directory)/'checkpoints').glob("*.ckpt"))
    checkpoint_epochs = [int(i.name.split('=')[1].split('-')[0]) for i in checkpoints]
    latest_checkpoint = [x for _, x in sorted(zip(checkpoint_epochs, checkpoints))][-1]

    print(latest_checkpoint)
    return latest_checkpoint

if __name__ == '__main__':


    data_dir = Path(sys.argv[1])
    log_dir = Path(sys.argv[2])

    # load model
    module = Module.load_from_checkpoint(find_latest_ckp(log_dir),
                                    arch = RBPNet())
    m=RBPNet_wrapper(module)
    
    # load test data
    test_sdata = sd.open_zarr(data_dir/ 'test.zarr').load()

    # score
    # find the strongest tests
    test_seq = torch.tensor(sp.ohe(test_sdata['seq'].values,alphabet=sp.alphabets.DNA).transpose(0, 2, 1),
                        dtype = torch.float32)
    _, test_ip_prob, test_in_prob, _, test_dlogodds = predict(module, test_seq)

    from scipy.stats import zscore
    test_dlogodds_z = zscore(test_dlogodds, axis=0, ddof=0, nan_policy='propagate').numpy()

    size = 256
    # 99 precentile
    index_99=np.where(test_dlogodds_z>3)[0][:size]
    index_97=np.where((test_dlogodds_z>2) & (test_dlogodds_z<3))[0][:size]
    index_85=np.where((test_dlogodds_z>1) & (test_dlogodds_z<2))[0][:size]
    index_50=np.where((test_dlogodds_z>0) & (test_dlogodds_z<1))[0][:size]

    for percentile, index in zip([99,97,85,50],
        [index_99, index_97, index_85, index_50]
        ):
        try:
            ism_attrs, pos_patterns, neg_patterns = find_patterns(
                test_seq[index,:,:], m)
            
            if pos_patterns is not None and len(pos_patterns) > 0:
                plot_patterns(pos_patterns, out_prefix = log_dir / f'{percentile}')
        except Exception as e:
            print(e)
            print(len(index))
            print(index)
        
    
    with open(log_dir / f'motif_done', 'w') as f:
        f.write('motif done')
    