# Imports
import os
import pandas as pd
import xarray as xr
import seqdata as sd
import sys
from eugene import preprocess as pp
import pandas as pd
import pybedtools
pybedtools.helpers.set_bedtools_path('/tscc/nfs/home/hsher/miniconda3/envs/my_metadensity/bin/')
from pybedtools import BedTool
from pathlib import Path
from yaml import load, dump
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

from pathlib import Path

def find_bigwigs_for_eclip(exp, ip_rep, in_rep, skipper_dir):
    signals = [skipper_dir / f'output/bigwigs/unscaled/{strand}/{exp}_IP_{ip_rep}.unscaled.{strand}.bw' 
           for strand in ['plus', 'minus']]
    controls = [skipper_dir / f'output/bigwigs/unscaled/{strand}/{exp}_IN_{in_rep}.unscaled.{strand}.bw' 
                for strand in ['plus', 'minus']]
    bigwigs = signals + controls
    sample_names = ['signal+', 'signal-', 'control+', 'control-']
    return bigwigs, sample_names

def get_gc_odds_ratio(gc_bin_sum, exp, ip_rep, in_rep):
    gc_odds = (gc_bin_sum[f'{exp}_IP_{ip_rep}']/gc_bin_sum[f'{exp}_IN_{in_rep}']
          )/(total[f'{exp}_IP_{ip_rep}']/total[f'{exp}_IN_{in_rep}'])

    return gc_odds
def gc_fraction(gc_bin_sum, exp, ip_rep, in_rep):
    gc_frac = gc_bin_sum[f'{exp}_IP_{ip_rep}']/(gc_bin_sum[f'{exp}_IN_{in_rep}']+gc_bin_sum[f'{exp}_IP_{ip_rep}'])
    
    return gc_frac
def make_training_regions(counts, gc_frac, read_thres, exp, ip_rep, in_rep):
    counts['gc_bin_fraction']=counts['gc_bin'].map(gc_frac)
    counts['total']=counts[[f'{exp}_IN_{in_rep}',f'{exp}_IP_{ip_rep}']].sum(axis = 1)
    tested_df = counts.loc[counts['total']>read_thres]
    peaks = Path(out_dir) / f"{exp}.rep{ip_rep}.training.bed"
    bed_tested_window=BedTool.from_dataframe(
                           tested_df[['chr', 'start', 'end', 'name', 'gc_bin_fraction', 'strand',
                                     f'{exp}_IP_{ip_rep}', f'{exp}_IN_{in_rep}']])
    bed_tested_window.slop(l = length, r = length, g=chromsize).saveas(peaks)

    # make verbose name
    peaks_df = BedTool(peaks).to_dataframe().rename({'score':'gc_fraction',
                          'thickStart':'n_IP',
                          'thickEnd':'n_IN',
                         'start':'chromStart',
                        'end':'chromEnd'},
                           axis = 1)

    return peaks_df
def prepare_sdata(bigwigs, sample_names, out, peaks_df, fasta):
    # load everything
    sdata = sd.from_region_files(
        sd.GenomeFASTA('seq',
            fasta,
            batch_size=2048,
            n_threads=4,
        ),
        sd.BigWig(
            'cov',
            bigwigs,
            sample_names,
            batch_size=2048,
            n_jobs=4,
            threads_per_job=2,
        ),
        path=out,
        fixed_length=300,
        bed=peaks_df,
        overwrite=True,
        max_jitter=32
    )
    sdata.load()

    # sel
    # Split cov and control
    sdata['signal+'] = (
        sdata.cov.sel(cov_sample=['signal+'])
        .drop_vars("cov_sample").squeeze()
    )
    sdata['signal-'] = (
        sdata.cov.sel(cov_sample=['signal-'])
        .drop_vars("cov_sample").squeeze()
    )
    sdata['control+'] = (
        sdata.cov.sel(cov_sample=['control+'])
        .drop_vars("cov_sample").squeeze()
    )
    sdata['control-'] = (
        sdata.cov.sel(cov_sample=['control-'])
        .drop_vars("cov_sample").squeeze()
    )
    # Get rid of aggregated cov
    sdata = sdata.drop_vars("cov")
    sdata = sdata.drop_vars("cov_sample")

    # make this strand specific
    # Split into two SeqDatas, one for positive strand and one for negative strand
    pos_sdata = sdata.sel(_sequence=sdata["strand"] == "+")
    pos_sdata = pos_sdata.drop_vars(["signal-", "control-"])
    pos_sdata = pos_sdata.rename_vars({"signal+": "signal", "control+": "control"})
    neg_sdata = sdata.sel(_sequence=sdata["strand"] == "-")
    neg_sdata = neg_sdata.drop_vars(["signal+", "control+"])
    neg_sdata = neg_sdata.rename_vars({"signal-": "signal", "control-": "control"})

    # concat
    # Combine
    sdata = xr.concat([pos_sdata, neg_sdata], dim="_sequence")

    return sdata

if __name__ == '__main__':
    skipper_config = sys.argv[1]
    exp = sys.argv[2]
    out_dir = Path(sys.argv[3])

    config = load(open(skipper_config), Loader=Loader)

    fasta = config['GENOME']
    chromsize=config['CHROM_SIZES']
    skipper_dir = Path(config['WORKDIR'])

    print(config['protocol'])
    print(config['protocol']=='ENCODE3')
    length = 100
    read_thres = 5

    try:
        os.mkdir(out_dir)
    except Exception as e:
        print(e)

    # load counts
    counts = pd.read_csv(skipper_dir /f'output/counts/genome/tables/{exp}.tsv.gz', sep = '\t')
    counts = counts.loc[counts['chr']!='chrM']
    counts['gc_bin']=pd.qcut(counts['gc'], q = 10)

    # calculate gcbias
    gc_bin_sum = counts.groupby(by = 'gc_bin')[counts.columns[7:-1]].sum()
    total = gc_bin_sum.sum(axis = 0)

    # do the thing
    ip_reps = [1,2]
    in_reps = [1,1] if config['protocol']=='ENCODE3' else [1,2]
    gc_odds_across_rep = []
    sdata_across_rep = []
    for ip_rep, in_rep in zip(ip_reps, in_reps):
        print( f'processing IP rep={ip_rep}, IN rep={in_rep}')
        bigwigs, sample_names = find_bigwigs_for_eclip(exp, ip_rep, in_rep, skipper_dir)
        out = out_dir/f"{exp}.rep{ip_rep}.zarr"

        gc_odds = get_gc_odds_ratio(gc_bin_sum, exp, ip_rep, in_rep)
        gc_odds_across_rep.append(gc_odds)

        gc_frac = gc_fraction(gc_bin_sum, exp, ip_rep, in_rep)
        peaks_df = make_training_regions(counts, gc_frac, read_thres, exp, ip_rep, in_rep)
        peaks_df['rep']=ip_rep

        sdata = prepare_sdata(bigwigs, sample_names, out, peaks_df, fasta)
        sdata_across_rep.append(sdata)

    sdata_across_rep = xr.concat(sdata_across_rep, dim="_sequence")

    # GC bias
    gc_bias = pd.concat(gc_odds_across_rep, axis = 1)
    gc_bias.columns = [f'Odds Ratio in IP_1', f'Odds Ratio in IP_2']
    gc_bias.to_csv(out_dir / 'gc_odds.csv')

    # train test split by chromosome
    # train test split
    training_chroms = ['chr{}'.format(i) for i in [3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 17, 18, 19, 20, 21, 22]]
    valid_chroms = ["chr1", "chr8", "chr15"]
    test_chroms = ["chr2", "chr9", "chr16"]

    sdata_across_rep = sdata_across_rep.sel(_sequence=(
        (sdata_across_rep["chrom"].isin(training_chroms + valid_chroms + test_chroms))))
    sdata_across_rep["chrom"].to_series().value_counts()

    # Create split columns
    pp.train_test_chrom_split(sdata_across_rep, test_chroms=test_chroms, train_var="train_test")
    pp.train_test_chrom_split(sdata_across_rep, test_chroms=valid_chroms, train_var="train_val")

    # Split SeqDatas
    train_sdata = sdata_across_rep.sel(_sequence=(sdata_across_rep["train_val"] & sdata_across_rep["train_test"]))
    valid_sdata = sdata_across_rep.sel(_sequence=~sdata_across_rep["train_val"])
    test_sdata = sdata_across_rep.sel(_sequence=~sdata_across_rep["train_test"])

    # Check how many of each
    size = {'train':train_sdata.dims["_sequence"],
            'valid':valid_sdata.dims["_sequence"], 
            'test': test_sdata.dims["_sequence"]
    }
    print(size)
    pd.Series(size).to_csv(out_dir/ 'size.csv')

    # Save them
    sd.to_zarr(train_sdata, out_dir/'train.zarr', mode='w')
    sd.to_zarr(valid_sdata, out_dir/'valid.zarr', mode='w')
    sd.to_zarr(test_sdata, out_dir/'test.zarr', mode='w')

    with open(out_dir/'prep_done', 'w') as f:
        f.write('prep done')