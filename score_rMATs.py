import sys
sys.path.append('/tscc/nfs/home/hsher/projects/splice_snake/rules')
import pybedtools
pybedtools.helpers.set_bedtools_path('/tscc/nfs/home/hsher/miniconda3/envs/my_metadensity/bin/')
from constants import *
from pathlib import Path
import seqdata as sd
import seqpro as sp
import xarray as xr
import pandas as pd
import torch
from RBPNet import *
from inference import find_latest_ckp
from pybedtools import BedTool
from module import Module
import numpy as np
from tqdm import tqdm


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

if __name__ == '__main__':
    grp = ['control', 'included', 'excluded']
    event = sys.argv[1]
    rbp = sys.argv[2] # rMATs ID
    splice_dir = Path(sys.argv[3]) # '/tscc/nfs/home/hsher/ps-yeolab5/encode_kd/output/rMATs/'
    log_dir = Path(sys.argv[4])
    model_name = log_dir.name
    fasta = sys.argv[5]
    chromsize = sys.argv[6]
    out_dir = Path(sys.argv[7])
    mask = 100

   # load model
    module = Module.load_from_checkpoint(find_latest_ckp(log_dir),
                                    arch = RBPNet(mask=mask))

    # inference
    score_matrix = []
    for region in event_type2_region_type[event]:
        sdatas = []
        regions = []
        for group in grp:
            orig = BedTool(splice_dir/f'{rbp}/regions/{event}/{group}/{region}.bed')
            bed = orig.slop(l = mask, r = mask, g=chromsize).saveas()
            region_df = bed.to_dataframe(names = ['chrom','chromStart', 'chromEnd', 'ID',
                                                    'InclusionLevelDifference', 'strand']).copy()
            region_df['identity']=region_df.apply(
                lambda row: row['chrom']+':'+str(row['chromStart'])+'-'+str(row['chromEnd'])+'('+row['strand']+')',
                                        axis = 1)
            dedup = region_df.drop_duplicates(subset = ['identity'])

            # get original coords
            
            infer_sdata = sd.from_region_files(
                sd.GenomeFASTA('seq',
                fasta,
                batch_size=2048,
                n_threads=4,
            ),
                bed=dedup[['chrom','chromStart', 'chromEnd', 'identity',
                                                    'InclusionLevelDifference', 'strand']],
                path = out_dir /f'{rbp}.{event}.{group}.{region}.zarr',
                fixed_length = False,
                max_jitter = 0,
                overwrite = True)
            sdatas.append(infer_sdata)
            regions.append(region_df)
        
        sdatas = xr.concat(sdatas, dim="_sequence")
        regions = pd.concat(regions, axis = 0)
        assert not set(regions['identity'])-set(sdatas['identity'].values)
        X_infer = sp.ohe(sdatas["seq"].values, 
                    alphabet=sp.alphabets.DNA).transpose(0, 2, 1)
        _, ip_probs, in_probs, _, dlogodds_infer = predict(module, X_infer)
        output = pd.DataFrame({'coord':sdatas['identity'],
                            'score': dlogodds_infer.numpy()
                        }).drop_duplicates(subset = 'coord')
        #output.to_csv(out_dir / f'{rbp}.{event}.{region}.nn_score_bed.csv')
        
        # ['chrom','chromStart', 'chromEnd', 'ID','InclusionLevelDifference', 'strand']
        
        regions[f'{region}.RBPNet_score'] = regions['identity'].map(output.set_index('coord')['score'])
        score_matrix.append(regions.set_index('ID')[f'{region}.RBPNet_score'])
        
        

    score_matrix = pd.concat(score_matrix, axis = 1)
    print(score_matrix.describe())
    score_matrix.to_csv(out_dir / f'{rbp}.{event}.{model_name}.csv')