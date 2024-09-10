import os
import torch
import numpy as np
from scipy.stats import pearsonr
from RBPNet import *

from metrics import rbpnet_metrics, pearson_corr, dlog_odds_from_data
from module import Module
from dataload import DataloaderWrapper

import seqpro as sp
import seqdata as sd
import sys
from pathlib import Path

from yaml import load, dump
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

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

if __name__ == '__main__':
    skipper_config = sys.argv[1]
    exp = sys.argv[2]
    data_dir = Path(sys.argv[3]) # with test.zarr
    log_dir = Path(sys.argv[4]) # where model is saved
    mask = 100

    try:
        os.mkdir(log_dir/'valid')
    except Exception as e:
        print(e)
    
    config = load(open(skipper_config), Loader=Loader)

    skipper_dir = Path(config['WORKDIR'])

    checkpoint = find_latest_ckp(log_dir)

    # load model
    module = Module.load_from_checkpoint(find_latest_ckp(log_dir),
                                    arch = RBPNet(mask = mask))
    
    # plot training curve
    f, axes = plt.subplots(1,4,figsize = (12,3))
    metric_df = pd.read_csv(log_dir / 'metrics.csv')
    metric_df.plot.scatter(x = 'epoch', y = 'train_binom_loss_epoch',
                        ax = axes[0], color = 'grey')
    metric_df.plot.scatter(x = 'epoch', y = 'val_binom_loss_epoch',
                        ax = axes[0], color = 'tomato')

    metric_df.plot.scatter(x = 'epoch', 
                        y = 'train_d_log_odds_profile_pearson_epoch', 
                        ax = axes[1], color = 'grey')
    metric_df.plot.scatter(x = 'epoch', 
                        y = 'val_d_log_odds_profile_pearson_epoch', 
                        ax = axes[1], color = 'tomato')

    metric_df.plot.scatter(x = 'epoch', y = 'train_eCLIP_loss_epoch', 
                        ax = axes[2], color = 'grey')
    metric_df.plot.scatter(x = 'epoch', y = 'val_eCLIP_loss_epoch', 
                        ax = axes[2], color = 'tomato')

    metric_df.plot.scatter(x = 'epoch', y = 'train_eCLIP_profile_pearson_epoch', 
                        ax = axes[3], color = 'grey')
    metric_df.plot.scatter(x = 'epoch', y = 'val_eCLIP_profile_pearson_epoch', 
                        ax = axes[3], color = 'tomato')
    sns.despine()
    plt.tight_layout()
    plt.savefig(log_dir / 'valid' / f'training_curve.pdf')

    # forward on test
    test_sdata = sd.open_zarr(os.path.join(data_dir, 'test.zarr')).load()
    X = sp.ohe(test_sdata["seq"].values[..., 32:-32], alphabet=sp.alphabets.DNA).transpose(0, 2, 1)
    y_pred, pis, dlogodds = predict(module, X)

    y_clip = torch.tensor(test_sdata["signal"].values[..., 32:-32].astype("float32"), dtype=torch.float32)
    y_clip.shape

    # sum along sequence axis (to get total counts)
    y_total = torch.sum(y_clip, axis=1)
    y_total = y_total.unsqueeze(1)
    y_total.shape

    # compute softmax of logits along sequence axis
    probs = torch.nn.functional.softmax(y_pred, dim=1)

    # return expected counts, i.e. probs * total_counts
    expected_counts = probs * y_total

    pearsons = pearson_corr(expected_counts, y_clip).detach().numpy()
    pearsons.mean()

    y_logodd, y_dlogodd = dlog_odds_from_data({'n_IP': torch.from_numpy(test_sdata['n_IP'].values),
                     'n_IN': torch.from_numpy(test_sdata['n_IN'].values),
                     'gc_fraction': torch.from_numpy(test_sdata['gc_fraction'].values),
                    }
                     )
    
    # calculate metrics
    test_metrics = pd.DataFrame({
        'profile_pearsons': pearsons,
        'pi': pis.squeeze(),
        'dlogodds_pred': dlogodds,
        'dlogodds': y_dlogodd,
        'logodds': y_logodd,
        'n_IP': torch.from_numpy(test_sdata['n_IP'].values),
        'n_IN': torch.from_numpy(test_sdata['n_IN'].values),
        'gc_fraction': torch.from_numpy(test_sdata['gc_fraction'].values),
        'name': test_sdata['name'].values,
        'strand': test_sdata['strand'].values,
    }
                            )
    test_metrics['total'] = test_metrics['n_IP']+test_metrics['n_IN']

    data_output = {}
    for nread in [10,50,100,200]:
        pearson_means = test_metrics.loc[test_metrics['total']>nread, 'profile_pearsons'].mean()
        pearson_std = test_metrics.loc[test_metrics['total']>nread, 'profile_pearsons'].std()
        data_output[f'mean profile_pearson(total>{nread})']=pearson_means
        data_output[f'std profile_pearson(total>{nread})']=pearson_std

    data_output['dlogodds pearson'], _=pearsonr(test_metrics['dlogodds'],test_metrics['dlogodds_pred'])
    for nread in [10,50,100,200]:
        sub = test_metrics[test_metrics['total']>nread]
        r,p = pearsonr(sub['dlogodds'],sub['dlogodds_pred'])
        data_output[f'dlogodds pearson(total>{nread})']=r

        r,p = pearsonr(sub['pi'],sub['dlogodds_pred'])
        data_output[f'pi pearson(total>{nread})']=r
    
    pd.Series(data_output).to_csv(log_dir / 'valid' / f'test_data_metric.csv')

    f, axes = plt.subplots(1,4, figsize = (12,3))
    sns.histplot(data = test_metrics,x='total', y='profile_pearsons', ax = axes[0])
    sns.regplot(data = test_metrics,x='total', y='profile_pearsons', scatter = False, ax = axes[0],lowess=True)
    axes[0].set_xlim(0,1000)

    sns.histplot(data = test_metrics, x='dlogodds', y='dlogodds_pred', ax = axes[1])

    sns.histplot(data = test_metrics, x='dlogodds', y='pi', ax = axes[2])

    sns.histplot(data =test_metrics, x='total', y='dlogodds', ax = axes[3])
    sns.regplot(data =test_metrics, x='total', y='dlogodds', scatter = False, ax = axes[3],lowess=True)
    axes[3].set_xscale('log')
    sns.despine()
    plt.savefig(log_dir / 'valid' / f'correlation_plot.pdf')

    # correlate with reproducible enriched windows
    reproducible_enriched_windows = pd.read_csv(
        skipper_dir / f'output/reproducible_enriched_windows/{exp}.reproducible_enriched_windows.tsv.gz',
    sep = '\t')

    merged = pd.merge(test_metrics, reproducible_enriched_windows,
         left_on = 'name', right_on = 'name')
    try:
        f, axes = plt.subplots(1,2, figsize = (6,3))
        sns.histplot(data =merged, y='dlogodds_pred', x='enrichment_l2or_mean', 
                        ax = axes[0])
        sns.regplot(data =merged, y='dlogodds_pred', x='enrichment_l2or_mean', 
                    scatter = False, ax = axes[0],lowess=True)
        r,p= pearsonr(merged['enrichment_l2or_mean'], merged['dlogodds_pred'])
        axes[0].set_title(f'{p:.2e} pearonsr={r:.2f}')

        sns.histplot(data =merged, y='pi', x='enrichment_l2or_mean', 
                        ax = axes[1])
        sns.regplot(data =merged, y='pi', x='enrichment_l2or_mean', 
                    scatter = False, ax = axes[1],lowess=True)
        r,p= pearsonr(merged['enrichment_l2or_mean'], merged['pi'])
        axes[1].set_title(f'{p:.2e} pearonsr={r:.2f}')
        sns.despine()
        plt.savefig(log_dir / 'valid' / f'correlation_with_re.pdf')
    except Exception as e:
        print(e)

            