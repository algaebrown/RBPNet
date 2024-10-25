
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
            x = torch.from_numpy(x.astype(np.float32)) # explodes in memory when x is large
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

def seq_trans(x):
        x = sp.ohe(x, sp.alphabets.DNA)
        x = x.swapaxes(1, 2)
        x = x[..., 32:-32]
        return x

def score_fa(fa, module, batch_size = 256, zarr_outdir=''):
    name = str(fa).replace('/', '-')
    if Path(f'{zarr_outdir}/{name}.zarr').exists():
        print('using processed data')
        infer_sdata = sd.open_zarr(f'{zarr_outdir}/{name}.zarr').load()
    else:
        print('making .zarr from scratch')
        infer_sdata = sd.read_flat_fasta(
                    name = 'seq',
                    fasta = fa,
                    out = f'{zarr_outdir}/{name}.zarr',
                    batch_size = 512,
                    fixed_length = False,
                overwrite = True) ## used to be true

    # make dataloader
    infer_dl = sd.get_torch_dataloader(
        infer_sdata,
        sample_dims=['_sequence'],
        variables=['seq'],
        num_workers=0,
        prefetch_factor=None,
        batch_size=batch_size,
        transforms={
            'seq': seq_trans,
        },
        return_tuples=False,
        shuffle=False,
    )

    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    module.eval()
    module.to(device)

    d_log_odds = []
    
    with torch.no_grad():
        for i, batch in tqdm(enumerate(infer_dl)):
            
            X = batch['seq'].to(device)
            
            x_total, x_signal, x_ctl, x_mix, d= module.arch(X)
            del X
            d_log_odds.append(d)
    
    dlogodds = torch.cat([i if i.dim()==1 else i.unsqueeze(-1) for i in d_log_odds])
    output = pd.DataFrame({'dlogodds_pred': dlogodds.cpu(),
                           'ID': infer_sdata['_sequence']
                          })

    return output



