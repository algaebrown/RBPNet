import argparse
import os
import json
from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from scipy.stats import pearsonr

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning import Trainer

import sys
# Make sure we can import from RBPNet module
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "RBPNet"))
from RBPNet import RBPNet
from module import Module
from losses import rbpnet_loss
from metrics import rbpnet_metrics, dlog_odds_from_data, pearson_corr

# --- Helper Functions ---
def custom_ohe(seq, alphabet="ACGT"):
    """One-hot encodes a single sequence string or array of characters."""
    char_to_idx = {char: idx for idx, char in enumerate(alphabet)}
    if isinstance(seq, np.ndarray) and seq.ndim == 0:
        seq = str(seq.item())
    chars = np.array(list(seq) if isinstance(seq, str) else [str(c) for c in seq])
    seq_encoded = np.zeros((len(chars), len(alphabet)), dtype=np.float32)
    for i, char in enumerate(chars):
        if char in char_to_idx:
            seq_encoded[i, char_to_idx[char]] = 1.0
    return seq_encoded

def apply_jitter_single(seq, control, signal, max_jitter, seed=None):
    """Applies jitter to a single example's arrays."""
    if max_jitter <= 0:
        return seq, control, signal

    rng = np.random.default_rng(seed)
    original_length = seq.shape[-1]
    jittered_length = original_length - 2 * max_jitter

    if jittered_length <= 0:
        raise ValueError(f"Array length {original_length} is too small for max_jitter {max_jitter}")

    start_idx = rng.integers(0, 2 * max_jitter + 1)
    end_idx = start_idx + jittered_length

    jittered_seq = seq[..., start_idx:end_idx]
    jittered_control = control[..., start_idx:end_idx]
    jittered_signal = signal[..., start_idx:end_idx]

    return jittered_seq, jittered_control, jittered_signal

# --- Dataset ---
class ECLIPDataset(Dataset):
    def __init__(self, hf_dataset, max_jitter=0, seed=None):
        self.ds = hf_dataset
        self.max_jitter = max_jitter
        self.seed = seed

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        ex = self.ds[idx]
        seq = np.char.upper(ex["seq"])
        seq = custom_ohe(seq, "ACGT").swapaxes(0, 1)

        control = np.asarray(ex["control"], dtype=np.float32)
        signal = np.asarray(ex["signal"], dtype=np.float32)

        if self.max_jitter > 0:
            item_seed = self.seed + idx if self.seed is not None else None
            seq, control, signal = apply_jitter_single(seq, control, signal, self.max_jitter, item_seed)
        else:
            # Statically trim by 32 if no jitter so validation/test match jittered train length
            seq = seq[..., 32:-32]
            control = control[..., 32:-32]
            signal = signal[..., 32:-32]

        return {
            "seq": torch.from_numpy(seq).float(),
            "control": torch.from_numpy(control).float(),
            "signal": torch.from_numpy(signal).float(),
            "gc_fraction": torch.tensor(ex["gc_fraction"], dtype=torch.float32),
            "n_IN": torch.tensor(ex["n_IN"], dtype=torch.float32),
            "n_IP": torch.tensor(ex["n_IP"], dtype=torch.float32),
        }

# --- Evaluation Function ---
def evaluate_and_save(module, test_dl, output_dir):
    module.eval()
    device = next(module.parameters()).device

    all_pred, all_pi, all_dlogodds = [], [], []
    all_signal, all_gc, all_nIP, all_nIN = [], [], [], []

    print("Running evaluation on test set...")
    with torch.no_grad():
        for batch in tqdm(test_dl, desc="Predicting"):
            x = batch["seq"].to(device)
            outputs = module.arch(x)

            all_pred.append(outputs[0].cpu())
            all_pi.append(outputs[3].cpu())
            all_dlogodds.append(outputs[4].cpu())

            all_signal.append(batch["signal"].cpu())
            all_gc.append(batch["gc_fraction"].cpu())
            all_nIP.append(batch["n_IP"].cpu())
            all_nIN.append(batch["n_IN"].cpu())

    y_pred = torch.cat(all_pred)
    pis = torch.cat(all_pi).squeeze()
    dlogodds = torch.cat([x if x.dim() == 1 else x.unsqueeze(-1) for x in all_dlogodds])

    y_clip = torch.cat(all_signal)
    gc_fraction = torch.cat(all_gc)
    n_IP = torch.cat(all_nIP)
    n_IN = torch.cat(all_nIN)

    y_total = y_clip.sum(dim=1, keepdim=True)
    probs = torch.softmax(y_pred, dim=1)
    expected_counts = probs * y_total

    pearsons = pearson_corr(expected_counts, y_clip).cpu().numpy()
    y_logodd, y_dlogodd = dlog_odds_from_data({
        "n_IP": n_IP,
        "n_IN": n_IN,
        "gc_fraction": gc_fraction,
    })

    test_metrics = pd.DataFrame({
        'profile_pearsons': pearsons,
        'pi': pis.squeeze(),
        'dlogodds_pred': dlogodds.squeeze().numpy(),
        'dlogodds': y_dlogodd.numpy(),
        'logodds': y_logodd.numpy(),
        'n_IP': n_IP.numpy(),
        'n_IN': n_IN.numpy(),
        'gc_fraction': gc_fraction.numpy(),
    })
    test_metrics['total'] = test_metrics['n_IP'] + test_metrics['n_IN']

    # Calculate aggregate summary stats
    data_output = {}
    for nread in [10, 50, 100, 200]:
        pearson_means = test_metrics.loc[test_metrics['total'] > nread, 'profile_pearsons'].mean()
        pearson_std = test_metrics.loc[test_metrics['total'] > nread, 'profile_pearsons'].std()
        data_output[f'mean_profile_pearson(total>{nread})'] = float(pearson_means)
        data_output[f'std_profile_pearson(total>{nread})'] = float(pearson_std)

    data_output['dlogodds_pearson'], _ = pearsonr(test_metrics['dlogodds'], test_metrics['dlogodds_pred'])
    data_output['dlogodds_pearson'] = float(data_output['dlogodds_pearson'])

    for nread in [10, 50, 100, 200]:
        sub = test_metrics[test_metrics['total'] > nread]
        if len(sub) > 1:
            r, p = pearsonr(sub['dlogodds'], sub['dlogodds_pred'])
            data_output[f'dlogodds_pearson(total>{nread})'] = float(r)

            r, p = pearsonr(sub['pi'], sub['dlogodds_pred'])
            data_output[f'pi_pearson(total>{nread})'] = float(r)

    # Save to files
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    test_metrics_path = out_path / "test_metrics.csv.gz"
    test_metrics.to_csv(test_metrics_path, index=False, compression='gzip')

    summary_path = out_path / "test_metrics_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(data_output, f, indent=4)

    print(f"Test metrics saved to: {test_metrics_path}")
    print(f"Test summary saved to: {summary_path}")
    print("Test Summary:")
    print(json.dumps(data_output, indent=4))

# --- Main Script ---
def main():
    parser = argparse.ArgumentParser(description="Train RBPNet with configurable hyperparameters.")
    parser.add_argument("--experiment_id", type=str, required=True, help="Unique identifier for this experiment run.")
    parser.add_argument("--max_jitter", type=int, default=32, help="Max jitter for training data.")
    parser.add_argument("--loss_w", type=float, default=30.0, help="Weight parameter (w) for rbpnet_loss.")
    parser.add_argument("--batch_size", type=int, default=128, help="Training batch size.")
    parser.add_argument("--output_path", type=str, default="./hyperparameter_search", help="Base directory for experiment outputs.")
    parser.add_argument("--max_epochs", type=int, default=100, help="Maximum number of training epochs.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for DataLoaders.")
    parser.add_argument("--dataset_id", type=str, default="RBFOX2_HepG2_ENCSR987FTF", help="The name/ID of the dataset from HuggingFace.")
    args = parser.parse_args()

    # Create output directory for this specific experiment
    experiment_dir = os.path.join(args.output_path, args.experiment_id)
    os.makedirs(experiment_dir, exist_ok=True)

    # Save the arguments configuration
    with open(os.path.join(experiment_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    print(f"Starting experiment: {args.experiment_id}")
    print(f"Configuration saved to {experiment_dir}/config.json")

    # Load dataset
    print(f"Loading dataset from HuggingFace (ID: {args.dataset_id})...")
    ds = load_dataset("yeolab/eCLIP", name=args.dataset_id)

    # Create dataloaders
    train_dataset = ECLIPDataset(ds["train"], max_jitter=args.max_jitter)
    train_dl = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers)

    valid_dataset = ECLIPDataset(ds["validation"], max_jitter=0)
    # Use 2x batch size for validation/test to speed up evaluation
    valid_dl = DataLoader(valid_dataset, batch_size=args.batch_size * 2, shuffle=False, drop_last=False, num_workers=args.num_workers)

    test_dataset = ECLIPDataset(ds["test"], max_jitter=0)
    test_dl = DataLoader(test_dataset, batch_size=args.batch_size * 4, shuffle=False, drop_last=False, num_workers=args.num_workers)

    # Initialize model
    print("Initializing model...")
    arch = RBPNet(mask=100)
    module = Module(
        arch=arch,
        input_variables=["seq"],
        output_variables=["eCLIP_profile", "signal_profile", "control_profile", "mixing_coefficient", "d_log_odds"],
        target_variables=["signal", "control", "gc_fraction", "n_IN", "n_IP"],
        loss_fxn=rbpnet_loss,
        loss_kwargs={"w": args.loss_w},  # Pass hyperparameter here
        metrics_fxn=rbpnet_metrics
    )

    # Setup Logging and Checkpoints
    logger = CSVLogger(save_dir=args.output_path, name=args.experiment_id)

    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    model_checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        save_top_k=5,
        monitor="val_loss_epoch",
        filename="best-{epoch}-{val_loss_epoch:.2f}"
    )

    early_stopping_callback = EarlyStopping(
        monitor="val_loss_epoch",
        patience=10,
        mode="min",
        verbose=True,
    )

    callbacks = [model_checkpoint_callback, early_stopping_callback, LearningRateMonitor()]

    # Trainer
    trainer = Trainer(
        max_epochs=args.max_epochs,
        logger=logger,
        devices="auto",
        accelerator="auto",
        callbacks=callbacks,
        num_sanity_val_steps=2
    )

    # Train
    print("Starting training...")
    trainer.fit(module, train_dataloaders=train_dl, val_dataloaders=valid_dl)
    print("Training complete!")

    # Evaluate best model on test set
    best_model_path = model_checkpoint_callback.best_model_path
    if best_model_path:
        print(f"Loading best model from {best_model_path} for testing...")
        best_module = Module.load_from_checkpoint(
            best_model_path,
            arch=RBPNet(mask=100),
            input_variables=["seq"],
            output_variables=["eCLIP_profile", "signal_profile", "control_profile", "mixing_coefficient", "d_log_odds"],
            target_variables=["signal", "control", "gc_fraction", "n_IN", "n_IP"],
            loss_fxn=rbpnet_loss,
            loss_kwargs={"w": args.loss_w},
            metrics_fxn=rbpnet_metrics
        )
        evaluate_and_save(best_module, test_dl, output_dir=experiment_dir)
    else:
        print("No checkpoint found. Evaluating current model state.")
        evaluate_and_save(module, test_dl, output_dir=experiment_dir)

if __name__ == "__main__":
    main()