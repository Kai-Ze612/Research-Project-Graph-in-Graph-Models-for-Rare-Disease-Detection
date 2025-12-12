import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import wandb
import os

from data_loader import get_dataloaders
from trainer import GiGTrainer

# 1. Config with Fixes
config = {
    # Architecture
    "input_dim": 64,
    "hidden_dim": 256,     
    "embedding_dim": 256,   
    "latent_dim": 256,      
    "gnn_hidden_dim": 512,  
    "conv_type": "GCN",
    "gnn_layers": 4,
    "dropout": 0.3,
    
    # Training - UPDATED
    "lr": 0.001,             # Critical for learning embeddings
    "optimizer_lr": 0.001,
    "lr_theta_temp": 0.001,  
    "alpha": 0.01,
    "weight_decay": 1e-4,    
    
    # Warmup Parameter
    "warmup_epochs": 20,     # First 20 epochs = Baseline Mode (No F2)
    
    # Misc
    "loss": "CrossEntropyLoss",
    "scheduler": "ReduceLROnPlateau",
}

# 2. Setup Data - FIXED PATH
# We use the absolute path to guarantee the file is found
file_path = '../saved_data_pkl'

print(f"Loading data from: {file_path}")
train_loader, val_loader, test_loader, num_classes = get_dataloaders(file_path)
config["num_classes"] = num_classes
print(f"Detected {num_classes} unique classes.")

# 3. Setup Model
model = GiGTrainer(config)

# 4. Setup Logger & Callbacks
wandb_logger = WandbLogger(project="gig-model")
callbacks = [
    ModelCheckpoint(
        monitor='val_loss', 
        dirpath='../checkpoints',  # Save outside the 'main' folder
        filename='gig-{epoch:02d}-{val_loss:.2f}', 
        save_top_k=3, 
        mode='min'
    ),
    EarlyStopping(monitor='val_loss', min_delta=0.01, patience=30, mode='min', verbose=True),
    LearningRateMonitor(logging_interval='epoch')
]

# 5. Train
trainer = Trainer(
    max_epochs=1000,
    accelerator='gpu',
    devices='auto',
    callbacks=callbacks,
    logger=wandb_logger,
    benchmark=True,
    check_val_every_n_epoch=1 
)

trainer.fit(model, train_loader, val_loader)
trainer.test(model, test_loader)