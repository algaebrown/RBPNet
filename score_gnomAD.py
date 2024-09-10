from RBPNet import *
import seqdata as sd
import sys
from module import Module
from pathlib import Path
from tqdm import tqdm
import seqpro as sp
import pandas as pd
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
        return torch.cat(outs), torch.cat(pis), torch.cat(dlogodds)
    
if __name__ == '__main__':
    log_dir = Path(sys.argv[1]) # where model is saved
    fa = Path(sys.argv[2])
    mask = 100
    
    module = Module.load_from_checkpoint(find_latest_ckp(log_dir),
                                    arch = RBPNet(mask = mask)) ###
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

    alt_output = pd.DataFrame({'dlogodds_pred': dlogodds_infer,
                           'ID': infer_sdata['_sequence']
                          })

