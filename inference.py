
from RBPNet import *
import seqdata as sd
from pathlib import Path
from tqdm import tqdm
import seqpro as sp
import pandas as pd
import numpy as np
import torch

def find_latest_ckp(log_directory):
    checkpoints = list((Path(log_directory)/'checkpoints').glob("*.ckpt"))
    checkpoint_epochs = [int(i.name.split('=')[1].split('-')[0]) for i in checkpoints]
    latest_checkpoint = [x for _, x in sorted(zip(checkpoint_epochs, checkpoints))][-1]

    print(latest_checkpoint)
    return latest_checkpoint

def predict(model, x, batch_size=128, verbose=True):
    with torch.no_grad():
        device = model.device
        model.eval()
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x.astype(np.float32))
        outs = []
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
            outs.append(out)

            # mixing coef
            pi = outputs[3].detach().cpu()
            pis.append(pi)

            # dlog
            d = outputs[4].detach().cpu()
            dlogodds.append(d)

        outs = torch.cat(outs)
        pis = torch.cat(pis)
        dlogodds = torch.cat([i if i.dim()==1 else i.unsqueeze(-1) for i in dlogodds])
        return outs, pis, dlogodds
def score_fa(fa, module):
    name = fa.name
    infer_sdata = sd.read_flat_fasta(
                name = 'seq',
                fasta = fa,
                out = f'/tscc/nfs/home/hsher/scratch/{name}.zarr',
                batch_size = 512,
                fixed_length = False,
            overwrite = True)
    
    
    X_infer = sp.ohe(infer_sdata["seq"].values, 
                alphabet=sp.alphabets.DNA).transpose(0, 2, 1)
    y_pred_infer, pis_infer, dlogodds_infer = predict(module, X_infer)
    
    output = pd.DataFrame({'dlogodds_pred': dlogodds_infer,
                           'ID': infer_sdata['_sequence']
                          })
    return output

