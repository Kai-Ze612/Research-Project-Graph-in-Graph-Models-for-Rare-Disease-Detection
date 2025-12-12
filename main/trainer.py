import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models import GiG

class GiGTrainer(pl.LightningModule):
    """
    PyTorch Lightning Module for training the GiG architecture.
    Handles optimization, logging, and graph warm-up strategy.
    """
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.save_hyperparameters(config)
        self.automatic_optimization = False
        
        self.model = GiG(config)
        
        # Loss definitions
        self.ce_loss = nn.CrossEntropyLoss()
        self.alpha = config["alpha"]

    def forward(self, data):
        return self.model(data)

    def _shared_step(self, data, stage: str):
        # Forward pass
        logits, adj_matrix, kl_loss = self.model(data)
        
        # Warmup Logic: Disable graph influence in early epochs to stabilize features
        is_warmup = self.current_epoch < self.config.get('warmup_epochs', 0)
        
        # Label processing
        labels = data.y.view(-1).long().to(self.device)
        classification_loss = self.ce_loss(logits, labels)
        
        # Robustness against NaN loss
        if torch.isnan(classification_loss): 
            classification_loss = torch.tensor(10.0, device=self.device, requires_grad=True)
        
        # Combined Loss
        total_loss = classification_loss + (0.0 if is_warmup else self.alpha * kl_loss)
        
        # Metrics
        acc = torchmetrics.functional.accuracy(logits.argmax(-1), labels, task="multiclass", num_classes=logits.shape[1])
        
        return {
            f"{stage}_loss": total_loss,
            f"{stage}_acc": acc,
            f"{stage}_kl": kl_loss
        }

    def training_step(self, batch, batch_idx):
        opt_main, opt_lgl = self.optimizers()
        opt_main.zero_grad(); opt_lgl.zero_grad()
        
        metrics = self._shared_step(batch, "train")
        loss = metrics["train_loss"]
        
        if not torch.isfinite(loss): return None
        
        self.manual_backward(loss)
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        
        opt_main.step()
        # Only step the graph optimizer after warmup
        if self.current_epoch >= self.config.get('warmup_epochs', 0):
            opt_lgl.step()
        
        self.log_dict(metrics, prog_bar=True, batch_size=len(batch.y))
        return loss

    def validation_step(self, batch, batch_idx):
        metrics = self._shared_step(batch, "val")
        self.log_dict(metrics, prog_bar=True, batch_size=len(batch.y))
        return metrics
    
    def test_step(self, batch, batch_idx):
        metrics = self._shared_step(batch, "test")
        self.log_dict(metrics, prog_bar=True, batch_size=len(batch.y))
        return metrics

    def on_validation_epoch_end(self):
        sch = self.lr_schedulers()
        if isinstance(sch, list): sch = sch[0]
        
        val_loss = self.trainer.callback_metrics.get("val_loss")
        if val_loss is not None and isinstance(sch, ReduceLROnPlateau):
            sch.step(val_loss)

    def configure_optimizers(self):
        # 1. Main Parameters (F1 Feature Extractor + F3 Classifier)
        main_params = [p for n, p in self.model.named_parameters() if "population_level_module" not in n]
        
        # 2. Graph Learning Parameters (F2 Population Module)
        lgl_params = list(self.model.population_level_module.parameters())
        
        opt_main = torch.optim.Adam(main_params, 
                                    lr=self.config["lr"], 
                                    weight_decay=self.config.get("weight_decay", 1e-5))
                                    
        opt_lgl = torch.optim.Adam(lgl_params, 
                                   lr=self.config["lr_theta_temp"])
        
        sched = ReduceLROnPlateau(opt_main, mode="min", patience=5, factor=0.5)
        return [opt_main, opt_lgl], [sched]