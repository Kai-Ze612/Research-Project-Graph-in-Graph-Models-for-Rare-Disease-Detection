import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from models import GiG, GlobalNodeEmbedding

class GiGTrainer(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters(config)
        self.automatic_optimization = False
        self.model = GiG(config)
        
        # Initialize Global Embeddings
        # We use a larger standard deviation for initialization to help gradients flow
        self.global_node_embedding = GlobalNodeEmbedding(num_global_nodes=105220, embedding_dim=config["input_dim"])
        
        loss_dict = {'CrossEntropyLoss': nn.CrossEntropyLoss(), 'BCEWithLogitsLoss': nn.BCEWithLogitsLoss()}
        self.initial_loss = loss_dict[config["loss"]]
        self.alpha = config["alpha"]

    def forward(self, data):
        return self.model(data)

    def _shared_step(self, data, addition):
        # 1. Prepare Inputs
        data.x = self.global_node_embedding(data.original_ids.long().to(self.device))
        
        # 2. Forward Pass with Warmup Logic
        # Get embeddings from F1
        node_embeddings, graph_embeddings = self.model.node_level_module(data)
        
        # Get graph structure from F2
        adjacency_matrix, edge_index, edge_weight, kl_loss = self.model.population_level_module(graph_embeddings)
        
        # --- WARMUP LOGIC ---
        # If we are in warmup phase, IGNORE the learned graph.
        # Force the classifier to treat patients independently (Like the Baseline).
        is_warmup = self.current_epoch < self.config.get('warmup_epochs', 0)
        
        if is_warmup:
            # Create empty edge index (No connections between patients)
            # GCNConv with empty edges behaves like a Linear Layer (MLP)
            # This replicates the "Baseline" architecture exactly
            batch_size = graph_embeddings.size(0)
            edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
            edge_weight = None
            kl_loss = torch.tensor(0.0, device=self.device) # No KL loss during warmup
        
        # 3. Classifier (F3)
        # Uses either the empty graph (Warmup) or the learned graph (GiG)
        logits = self.model.classifier(h=graph_embeddings, edge_index=edge_index, edge_weight=edge_weight)
        
        # 4. Loss Calculation
        labels = data.y.view(-1).long().to(self.device)
        class_loss = self.initial_loss(logits, labels)
        
        # Nan Guard
        if torch.isnan(class_loss): 
            print(f"NaN Loss in {addition}")
            class_loss = torch.tensor(5.0, device=self.device, requires_grad=True)
        
        # Only add KL loss if we are NOT in warmup
        total_loss = class_loss + (0.0 if is_warmup else self.alpha * kl_loss)
        
        # 5. Metrics
        acc = torchmetrics.functional.accuracy(logits.argmax(-1), labels, task="multiclass", num_classes=logits.shape[1])
        f1 = torchmetrics.functional.f1_score(logits.argmax(-1), labels, task="multiclass", num_classes=logits.shape[1])
        
        metrics = {
            f"{addition}_loss": total_loss, 
            f"{addition}_acc": acc, 
            f"{addition}_f1": f1,
            f"{addition}_kl": kl_loss
        }
        return metrics, total_loss

    def training_step(self, batch, batch_idx):
        opt1, opt2 = self.optimizers()
        opt1.zero_grad(); opt2.zero_grad()
        
        metrics, loss = self._shared_step(batch, "train")
        
        # Skip update if loss is broken
        if not torch.isfinite(loss): return None
        
        self.manual_backward(loss)
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        
        opt1.step()
        # Only step the Structure Optimizer (opt2) if we are past warmup
        if self.current_epoch >= self.config.get('warmup_epochs', 0):
            opt2.step()
        
        self.log_dict(metrics, prog_bar=True, batch_size=len(batch.y))
        return loss

    def validation_step(self, batch, batch_idx):
        metrics, _ = self._shared_step(batch, "val")
        self.log_dict(metrics, prog_bar=True, batch_size=len(batch.y))
        return metrics
    
    def test_step(self, batch, batch_idx):
        metrics, _ = self._shared_step(batch, "test")
        self.log_dict(metrics, prog_bar=True, batch_size=len(batch.y))
        return metrics

    def on_train_epoch_end(self):
        # Step schedulers
        sch = self.lr_schedulers()[0] if isinstance(self.lr_schedulers(), list) else self.lr_schedulers()
        if isinstance(sch, CosineAnnealingLR): sch.step()

    def on_validation_epoch_end(self):
        # Step schedulers
        sch = self.lr_schedulers()[0] if isinstance(self.lr_schedulers(), list) else self.lr_schedulers()
        if isinstance(sch, ReduceLROnPlateau):
            val_loss = self.trainer.callback_metrics.get("val_loss")
            if val_loss is not None: sch.step(val_loss)

    def configure_optimizers(self):
        # 1. Collect Main Parameters (F1, F3, Classifier, AND Embeddings)
        # Filter out F2 (population module) parameters to optimize them separately
        main_params = [
            p for n, p in self.model.named_parameters() 
            if "population_level_module" not in n or n.split('.')[-1] not in ["temp", "theta", "mu", "sigma"]
        ]
        
        # --- CRITICAL FIX: Add the Global Embeddings to the optimizer ---
        main_params.extend(self.global_node_embedding.parameters())
        # ----------------------------------------------------------------
        
        # 2. Collect Structure Learning Parameters (F2)
        lgl_params = [
            self.model.population_level_module.temp, 
            self.model.population_level_module.theta, 
            self.model.population_level_module.mu, 
            self.model.population_level_module.sigma
        ]
        
        # 3. Define Optimizers
        # Use the learning rate from config
        opt_main = torch.optim.Adam(
            main_params, 
            lr=self.config["lr"], 
            weight_decay=self.config.get("weight_decay", 1e-5)
        )
        
        opt_lgl = torch.optim.Adam(
            lgl_params, 
            lr=self.config["lr_theta_temp"]
        )
        
        # 4. Define Scheduler
        if self.config["scheduler"] == "ReduceLROnPlateau":
            sched = ReduceLROnPlateau(opt_main, mode="min", patience=5, factor=0.5)
        else:
            sched = CosineAnnealingLR(opt_main, T_max=10)
            
        return [opt_main, opt_lgl], [sched]