import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import os

from data_loader import get_dataloaders
from trainer import GiGTrainer

# --- CONFIGURATION ---
# Optimized for High-Cardinality Classification (2405 classes)
config = {
    # Architecture
    "input_dim": 7,           # Raw biological features
    "hidden_dim": 512,        # High capacity for feature extraction
    "embedding_dim": 512,     # High capacity for patient representation
    "latent_dim": 512,        # Bottleneck for graph learning
    "gnn_hidden_dim": 512,
    
    # Model Type
    "conv_type": "GCN",       # GCN for stability
    "gnn_layers": 3,
    "dropout": 0.3,           
    "heads": 4, 
    
    # Training Dynamics
    "lr": 0.001,
    "optimizer_lr": 0.001,
    "lr_theta_temp": 0.001,
    
    # Graph Learning Strategy
    "alpha": 0.001,           # Gentle graph regularization
    "warmup_epochs": 10,      # Stabilize features before graph construction
    
    # Logistics
    "loss": "CrossEntropyLoss",
    "weight_decay": 1e-5,
    "scheduler": "ReduceLROnPlateau",
    "batch_size": 1024,       
    "debug_mode": True,       # Enable diagnostics for first batch
}

# --- SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.normpath(os.path.join(current_dir, '..', 'saved_data_pkl'))

if not os.path.exists(file_path):
    raise FileNotFoundError(f"Data directory not found at: {file_path}")

# Data Loading
print(f"Initializing data loaders from: {file_path}")
train_loader, val_loader, test_loader, num_classes = get_dataloaders(
    file_path, 
    batch_size_train=config["batch_size"], 
    batch_size_val=config["batch_size"]
)
config["num_classes"] = num_classes
print(f"Configuration: {num_classes} classes detected.")

# Model & Logging
model = GiGTrainer(config)
wandb_logger = WandbLogger(project="gig-model-academic", name="high-capacity-gcn")

ckpt_path = os.path.join(current_dir, '..', 'checkpoints')
os.makedirs(ckpt_path, exist_ok=True)

callbacks = [
    ModelCheckpoint(
        monitor='val_loss', 
        dirpath=ckpt_path, 
        filename='gig-{epoch:02d}-{val_loss:.4f}', 
        save_top_k=2, 
        mode='min'
    ),
    EarlyStopping(
        monitor='val_loss', 
        min_delta=0.001, 
        patience=30, 
        mode='min', 
        verbose=True
    ),
    LearningRateMonitor(logging_interval='epoch')
]

# Trainer
trainer = Trainer(
    max_epochs=1000,
    accelerator='gpu',
    devices='auto',
    callbacks=callbacks,
    logger=wandb_logger,
    benchmark=True,
    check_val_every_n_epoch=1,
    precision='16-mixed' # Optimization for RTX 5090
)

if __name__ == "__main__":
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)