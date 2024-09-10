import os
import torch
import numpy as np
from RBPNet import RBPNet
from metrics import rbpnet_metrics, dlog_odds_from_data
from module import Module
from dataload import DataloaderWrapper
from losses import rbpnet_loss
import seqpro as sp
import seqdata as sd
import sys
from pathlib import Path
from tqdm import tqdm
import warnings
#from eugene import preprocess as pp

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
    if not torch.cuda.is_available():
        warnings.warn('CUDA NOT AVAILABLE!!')

    data_dir = Path(sys.argv[1])
    log_dir = Path(sys.argv[2])
    mask = 100
    #d_log_odds_loss_weight = 30

    try:
        os.mkdir(log_dir)
    except Exception as e:
        print(e)
    
    # Load Data
    train_sdata = sd.open_zarr(os.path.join(data_dir, 'train.zarr')).load()
    valid_sdata = sd.open_zarr(os.path.join(data_dir, 'valid.zarr')).load()

    # architecture
    arch = RBPNet(mask = mask)

    ### load training data ###
    def seq_trans(x):
        x = np.char.upper(x)
        x = sp.ohe(x, sp.alphabets.DNA)
        x = x.swapaxes(1, 2)
        return x

    def cov_dtype(x):
        return tuple(arr.astype('f4') for arr in x) # float32

    def jitter(x):
        return sp.jitter(*x, max_jitter=32, length_axis=-1, jitter_axes=0)

    def to_tensor(x):
        return tuple(torch.tensor(arr, dtype=torch.float32) for arr in x)
    
    # Get the train dataloader
    train_dl = sd.get_torch_dataloader(
        train_sdata,
        sample_dims=['_sequence'],
        variables=['seq', 'control', 'signal', 'gc_fraction', 'n_IN', 'n_IP'], # gc_baseline,ip_count, in_count
        num_workers=0,
        prefetch_factor=None,
        batch_size=128,
        transforms={
            ('seq', 'control', 'signal'): jitter,
            'seq': seq_trans,
            ('control', 'signal', 'gc_fraction', 'n_IN', 'n_IP'): cov_dtype,
            ('control', 'seq', 'signal','gc_fraction', 'n_IN', 'n_IP'): to_tensor,
        },
        return_tuples=False,
        shuffle=True,
        drop_last=True # when batch size = 1, batch norm fails
    )
    #train_dl = DataloaderWrapper(train_dl, batch_per_epoch=1000)

    ### load validation data ###
    def seq_trans(x):
        x = np.char.upper(x)
        x = sp.ohe(x, sp.alphabets.DNA)
        x = x.swapaxes(1, 2)
        x = x[..., 32:-32]
        return x

    def cov_trans(x):
        x = x[..., 32:-32]
        return torch.as_tensor(x.astype('f4'))

    def to_f4(x):
        return torch.as_tensor(x.astype('f4'))

    valid_dl = sd.get_torch_dataloader(
        valid_sdata,
        sample_dims=['_sequence'],
        variables=['seq', 'control', 'signal', 'gc_fraction', 'n_IN', 'n_IP'],
        num_workers=0,
        prefetch_factor=None,
        batch_size=256,
        transforms={
            'seq': seq_trans,
            'control': cov_trans,
            'signal': cov_trans,
            'gc_fraction': to_f4,
            'n_IN': to_f4,
            'n_IP': to_f4
        },
        return_tuples=False,
        shuffle=False,
    )
    print(len(valid_sdata['n_IP']), 'Validation data size', len(valid_dl), '# batch')
    ### Module ###
    # LightningModule
    module = Module(
        arch=arch,
        input_variables=["seq"],
        output_variables=["eCLIP_profile", "signal_profile", "control_profile", "mixing_coefficient",
                        "d_log_odds"],
        target_variables=["signal", "control", "gc_fraction", "n_IN", "n_IP"],
        loss_fxn=rbpnet_loss,
        metrics_fxn=rbpnet_metrics
    )

    # Logger
    from pytorch_lightning.loggers import CSVLogger
    logger = CSVLogger(save_dir=log_dir, name="", version="")

    # Set-up callbacks
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.callbacks.early_stopping import EarlyStopping
    from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
    callbacks = []
    model_checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(
            logger.save_dir, 
            logger.name, 
            logger.version, 
            "checkpoints"
        ),
        save_top_k=5,
        monitor="val_loss_epoch",
    )
    callbacks.append(model_checkpoint_callback)
    early_stopping_callback = EarlyStopping(
        monitor="val_loss_epoch",
        patience=10,
        mode="min",
        verbose=True,
        #check_on_train_epoch_end=False ###https://github.com/Lightning-AI/pytorch-lightning/issues/9151
    )
    callbacks.append(early_stopping_callback)
    callbacks.append(LearningRateMonitor())
    callbacks

    # Trainer
    from pytorch_lightning import Trainer
    trainer = Trainer(
        max_epochs=100,
        logger=logger,
        devices="auto",
        accelerator="auto",
        callbacks=callbacks,
        num_sanity_val_steps=2
    )

    # Fit
    trainer.fit(
        module, 
        train_dataloaders=train_dl, 
        val_dataloaders=valid_dl,
    )

    with open(log_dir/'training_done', 'w') as f:
        f.write('training done')