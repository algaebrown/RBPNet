import sys
from RBPNet import *
from inference import *
from module import Module
from pathlib import Path


    
if __name__ == '__main__':
    log_dir = Path(sys.argv[1]) # where model is saved
    fa = Path(sys.argv[2])
    mask = 0
    outf = sys.argv[3]
    try:
        zarr_outdir = Path(sys.argv[4])
    except:
        zarr_outdir = Path(outf).parent
    
    module = Module.load_from_checkpoint(find_latest_ckp(log_dir),
                                    arch = RBPNet(mask = mask),
                                    zarr_outdir = zarr_outdir) ###
    
    output = score_fa(fa, module)
    output.to_csv(outf)
