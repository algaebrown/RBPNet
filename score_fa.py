import sys
from RBPNet import *
from inference import *
from module import Module
from pathlib import Path


    
if __name__ == '__main__':
    log_dir = Path(sys.argv[1]) # where model is saved
    fa = Path(sys.argv[2])
    mask = 100
    outf = sys.argv[3]
    
    module = Module.load_from_checkpoint(find_latest_ckp(log_dir),
                                    arch = RBPNet(mask = mask)) ###
    
    output = score_fa(fa, module)
    output.to_csv(outf)
