from pytorch_lightning.core import LightningModule
from pytorch_lightning.utilities.model_summary import ModelSummary
from typing import Callable, Optional, Tuple, Union, Literal, List, Mapping, Any
import torch.optim as opt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch


class Module(LightningModule):

    def __init__(
        self,
        arch: torch.nn.Module,
        input_variables: List[str] = ["seq"],
        output_variables: List[str] = ["output"],
        target_variables: List[str] = ["target"],
        loss_fxn: Union[str, Callable] = "mse",
        loss_kwargs: Mapping[str, Any] = {},
        metrics_fxn: Union[str, Callable] = "pearson",
        metrics_kwargs: Mapping[str, Any] = {},
        optimizer: Literal["adam", "sgd"] = "adam",
        optimizer_lr: Optional[float] = 1e-3,
        optimizer_kwargs: Optional[dict] = {},
        scheduler: Optional[str] = None,
        scheduler_monitor: str = "val_loss_epoch",
        scheduler_kwargs: dict = {},
    ):
        super().__init__()
        
        # Set the base architecture 
        self.arch = arch
        
        # Set variables
        self.input_variables = input_variables
        self.output_variables = output_variables
        self.target_variables = target_variables
        
        # Loss function
        self.loss_fxn = loss_fxn
        self.loss_kwargs = loss_kwargs
        
        # Metrics function
        self.metrics_fxn = metrics_fxn
        self.metrics_kwargs = metrics_kwargs
        
        # Optimizer
        self.optimizer = opt.Adam
        self.optimizer_lr = optimizer_lr if optimizer_lr is not None else 1e-3
        self.optimizer_kwargs = optimizer_kwargs
        
        # Scheduler
        self.scheduler = ReduceLROnPlateau
        self.scheduler_monitor = scheduler_monitor
        self.scheduler_kwargs = scheduler_kwargs

        
    def forward(self, inputs_dict) -> torch.Tensor:
        """
        Forward pass of the arch.

        Parameters
        ----------
        x : torch.Tensor
            inputs
        """
        return self.arch(*inputs_dict.values())
    
    def _common_step(self, batch, batch_idx, stage: str):
        
        # Initialize dictionaries for inputs, outputs, and targets
        inputs_dict = {var: batch[var] for var in self.input_variables}
        targets_dict = {var: batch[var] for var in self.target_variables}

        # Forward pass through the model
        outputs = self(inputs_dict)
        outputs_dict = {var: outputs[ind] for ind, var in enumerate(self.output_variables)}

        # Ensure outputs is a dictionary matching output_variables
        if not isinstance(outputs_dict, dict):
            raise ValueError("Model output must be a dictionary when using _common_step.")
            
        # Calculate losses using the rbpnet_loss function
        losses_dict = self.loss_fxn(outputs_dict, targets_dict, **self.loss_kwargs)

        # Calculate metrics using the rbpnet_metrics function
        metrics_dict = self.metrics_fxn(outputs_dict, targets_dict, **self.metrics_kwargs)

        # Merge losses and metrics into a single dictionary
        step_dict = {**losses_dict, **metrics_dict}

        return step_dict
    
    
    def training_step(self, batch, batch_idx):
        """Training step"""
        step_dict = self._common_step(batch, batch_idx, "train")

        # Log loss on step
        self.log(
            "train_loss", 
            step_dict["loss"].mean(), 
            on_step=True, 
            on_epoch=False, 
            prog_bar=True
        )
        
        # Log everything else on epoch
        self.log_dict(
            {f"train_{k}_epoch": v.mean() for k, v in step_dict.items()}, 
            on_step=False, 
            on_epoch=True
        )
        return step_dict["loss"].mean()
    
    def validation_step(self, batch, batch_idx):
        """Validation step"""
        step_dict = self._common_step(batch, batch_idx, "val")
        print({f"val_{k}_epoch": v.mean() for k, v in step_dict.items()})
        # Log validation loss and metrics
        self.log_dict(
            {f"val_{k}_epoch": v.mean() for k, v in step_dict.items()}, 
            on_step=False, 
            on_epoch=True,
            prog_bar=True
        )

    def configure_optimizers(self):
        """Configure optimizers

        Returns:
        ----------
        torch.optim.Optimizer:
            optimizer
        torch.optim.lr_scheduler._LRScheduler:
            learning rate scheduler
        """
        optimizer = self.optimizer(
            self.parameters(), lr=self.optimizer_lr, **self.optimizer_kwargs
        )
        if self.scheduler is None:
            return optimizer
        else:
            scheduler = self.scheduler(optimizer, **self.scheduler_kwargs)
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": self.scheduler_monitor,
            }