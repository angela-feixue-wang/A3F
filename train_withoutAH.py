import os
import random
import numpy as np
import pandas as pd
import datetime
import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision import transforms
from torchvision.models import ResNet50_Weights, vit_b_16, ViT_B_16_Weights
from PIL import Image
from pathlib import Path
from thop import profile, clever_format

from utils import set_seed, focal_loss_binary, focal_loss_multiclass
from dataset import ImageDatasetFourL, ImageDatasetFourRGB
from models import SonoNet, FourTaskWrapperSonoNet, FourTaskWrapperResNet
from models import OneTaskVisionTransformer, FourTaskWrapperViT


### Set up logging train loss and val loss during training
os.makedirs("./ckpt", exist_ok=True)
start_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    filename=f"./ckpt/Run_{start_time}.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)
print(f"\nStart Time: {start_time}\n")

### --------------------------------------------------- ###
### Define variables
### --------------------------------------------------- ###
img_dir = "/path/to/directory/containing/all/training/and/validation/images"
csv_train = "/path/to/csv/file/listing/training/image/names/and/ground/truth/labels/fold1_train_withaug.csv"
csv_val = "/path/to/csv/file/listing/validation/image/names/and/ground/truth/labels/fold1_val_withaug.csv"
ckpt_path = None

### Valid model names:
# "resnet50"
# "sononet"
# "vitb16"

# ===================================================================
# Hyperparameter Optimisation
# ===================================================================
model_name = "vitb16"
backbone_lr = 5e-05
head_lr = 1e-05
weight_decay = 0.0001
# Loss function weights
w_mag = 0.561
w_gain = 0.561
w_centering = 0.714
w_shadow = 1.289
loss_func = "focal" # "ce" for CrossEntropyLoss, "focal" for Focal Loss
# Focal loss parameters
gamma_mag = 2.0
gamma_gain = 3.0
gamma_centering = 1.5
gamma_shadow = 3.0
# ===================================================================
# ===================================================================
logging.info(f"Hyperparameters: ")
logging.info(f"img_dir: {img_dir}")
logging.info(f"csv_train: {csv_train}")
logging.info(f"csv_val: {csv_val}")
logging.info(f"model_name: {model_name}")
logging.info(f"backbone_lr: {backbone_lr}, head_lr: {head_lr}, weight_decay: {weight_decay}")
logging.info(f"w_mag: {w_mag}, w_gain: {w_gain}, w_centering: {w_centering}, w_shadow: {w_shadow}")
logging.info(f"loss_func: {loss_func}")
logging.info(f"gamma_mag: {gamma_mag}, gamma_gain: {gamma_gain}, gamma_centering: {gamma_centering}, gamma_shadow: {gamma_shadow}\n")
# ===================================================================
# ===================================================================
# ===================================================================

# Number of classes for anatomy, magnification, gain, centering, shadow
num_ana = 9
num_mag = 3
num_gain = 3
num_centering = 5
num_shadow = 1 # binary classification

# Training parameters
batch_size = 16
num_workers = 4

# Set random seed
set_seed(42)

### --------------------------------------------------- ###
### Step 1: Define how to read labels from a csv file
### --------------------------------------------------- ###
# ImageDataset class moved to dataset.py (2026-Jun-08)
    
### --------------------------------------------------- ###
### Step 2: Define image transform to model input format
### --------------------------------------------------- ###
if model_name in ["resnet50", "sononet"]:
    channel = 1
elif model_name == "vitb16":
    channel = 3
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=channel),  # since conv1 expects 1 channel
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

### --------------------------------------------------- ###
### Step 3: Define train and val datasets
### --------------------------------------------------- ###
train_dataset = ImageDatasetFourRGB(csv_file=csv_train, img_dir=img_dir, transform=transform)
val_dataset = ImageDatasetFourRGB(csv_file=csv_val, img_dir=img_dir, transform=transform)

### --------------------------------------------------- ###
### Step 4: Build dataloaders
### --------------------------------------------------- ###
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

### --------------------------------------------------- ###
### Step 5: Build Model
### --------------------------------------------------- ###
# Define gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_model(model_name, device, ckpt_path=None,
                num_mag=3, num_gain=3, num_centering=5, num_shadow=1):

    # ==========================================
    # ResNet50
    # ==========================================
    if model_name == "resnet50":
        backbone = models.resnet50(weights=None) # use "weights=ResNet50_Weights.IMAGENET1K_V2" for ImageNet weights
        backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()

        model = FourTaskWrapperResNet(
            backbone=backbone, 
            in_features=in_features,
            num_mag=num_mag, 
            num_gain=num_gain, 
            num_centering=num_centering, 
            num_shadow=num_shadow
        )
    
    # ==========================================
    # SonoNet
    # ==========================================
    elif model_name == "sononet":
        base_model = SonoNet(config="SN64", num_labels=9, weights=False)

        # if ckpt_path is not None:
        #     # Define weights location
        #     ckpt = torch.load(ckpt_path, map_location=device)

        #     # Drop final layer weights to fit 13 classes to 9 classes
        #     # Use strict=True if original model has the same number of classes as the new model
        #     # Use strict=True if number of classes is different so it loads all weights except the final layer weights
        #     # Try the 3 lines below first. If they don't work, comment them out and use the 4th line
        #     keys_to_remove = [k for k in ckpt.keys() if k.startswith("adaption.3") or k.startswith("adaption.4")]
        #     for k in keys_to_remove:
        #         del ckpt[k]

        #     # Load weights
        #     base_model.load_state_dict(ckpt, strict=False)

        model = FourTaskWrapperSonoNet(
            backbone=base_model,
            num_mag=num_mag,
            num_gain=num_gain,
            num_centering=num_centering,
            num_shadow=num_shadow # binary classification
        )

    # ==========================================
    # ViT-B/16
    # ==========================================
    elif model_name == "vitb16":
        # Call the backbone
        backbone = vit_b_16(weights=None) # use "weights=ViT_B_16_Weights.IMAGENET1K_V1" for ImageNet weights

        # Remove the ImageNet classifier
        backbone.heads = nn.Identity()

        # Call the multi-task wrapper
        model = FourTaskWrapperViT(
            backbone=backbone,
            num_mag=num_mag,
            num_gain=num_gain,
            num_centering=num_centering,
            num_shadow=num_shadow
        )

    return model

def finetuning(model, model_name):
    if model_name == "resnet50":
        for param in model.backbone.parameters():
            param.requires_grad = True # False = freeze all blocks, True = unfreeze all blocks

        # for param in model.backbone.layer4[2].parameters():
        #     param.requires_grad = True # False = freeze layer4, True = unfreeze layer4

        for param in model.head_mag.parameters():
            param.requires_grad = True

        for param in model.head_gain.parameters():
            param.requires_grad = True

        for param in model.head_centering.parameters():
            param.requires_grad = True

        for param in model.head_shadow.parameters():
            param.requires_grad = True
    
    elif model_name == "sononet":
        for param in model.parameters():
            param.requires_grad = True # False = freeze all blocks, True = unfreeze all blocks
    
    elif model_name == "vitb16":
        # Freeze everything
        for param in model.backbone.parameters():
            param.requires_grad = True

        # # ViT has 12 transformer blocks in total.
        # # Unfreeze last 2 transformer blocks
        # for block in model.backbone.encoder.layers[-2:]:
        #     for param in block.parameters():
        #         param.requires_grad = True

        # # Unfreeze final LayerNorm
        # for param in model.backbone.encoder.ln.parameters():
        #     param.requires_grad = True

    else:
        raise ValueError(f"Unknown model: {model_name}")

model = build_model(
    model_name=model_name,
    device=device,
    ckpt_path=ckpt_path,
    num_mag=num_mag,
    num_gain=num_gain,
    num_centering=num_centering,
    num_shadow=1  # binary classification
)

finetuning(model,model_name)
### --------------------------------------------------- ###
### Step 6: Move model to device
### --------------------------------------------------- ###
model = model.to(device)

### --------------------------------------------------- ###
### Step 6.1: Use THOP to calculate model params and flops
### --------------------------------------------------- ###
dummy_input = torch.randn(1, channel, 224, 224).to(device) # dummy input shape must match input shape expected by your model
flops, params = profile(model, inputs=(dummy_input,)) # uses thop to calculate model's computational cost and parameter count
flops, params = clever_format([flops, params], "%.3f") # formats thop outputs into human-readable format
print(f"# Model Params: {params} || FLOPS: {flops}")

### --------------------------------------------------- ###
### Step 6.5: Create folder for saving checkpoints
### --------------------------------------------------- ###
os.makedirs("ckpt", exist_ok=True)

### --------------------------------------------------- ###
### Step 7: Define Loss function
### --------------------------------------------------- ###
if loss_func == "ce":
    criterion_mag = nn.CrossEntropyLoss()
    criterion_gain = nn.CrossEntropyLoss()
    criterion_centering = nn.CrossEntropyLoss()
    criterion_shadow = nn.BCEWithLogitsLoss() # Shadow is binary classification
elif loss_func == "focal":
    criterion_mag = lambda x, y: focal_loss_multiclass(x, y, gamma=gamma_mag)
    criterion_gain = lambda x, y: focal_loss_multiclass(x, y, gamma=gamma_gain)
    criterion_centering = lambda x, y: focal_loss_multiclass(x, y, gamma=gamma_centering)
    criterion_shadow = lambda x, y: focal_loss_binary(x, y, gamma=gamma_shadow)

### --------------------------------------------------- ###
### Step 8: Define optimiser
### --------------------------------------------------- ###
# optimizer = torch.optim.Adam(
#     filter(lambda p: p.requires_grad, model.parameters()), # Only pass unfrozen parameters
#     lr=1e-4
# )
optimizer = torch.optim.Adam([
    {"params": model.backbone.parameters(), "lr": backbone_lr},
    {"params": model.head_mag.parameters(), "lr": head_lr},
    {"params": model.head_gain.parameters(), "lr": head_lr},
    {"params": model.head_centering.parameters(), "lr": head_lr},
    {"params": model.head_shadow.parameters(), "lr": head_lr},
], weight_decay=weight_decay)

# Scheduler for reducing learning rate if model does not improve after a number of epochs
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5, # 0.5 means each time learning rate is reduced, it is reduced to {current_learning_rate * 0.5}
    patience=3, # epochs with no val_loss improvement
    min_lr=1e-7,
)

### --------------------------------------------------- ###
### Step 9: Training and validation loop
### --------------------------------------------------- ###
total_epochs = 100

# Initiate early stopping
best_val_loss = float("inf")
patience = 5
counter = 0

for epoch in range(total_epochs):

    model.train() # Set model to train mode
    train_loss = 0

    for images, label_mag, label_gain, label_centering, label_shadow in train_loader:
        images = images.to(device) # Pass images to gpu
        label_mag = label_mag.to(device)
        label_gain = label_gain.to(device)
        label_centering = label_centering.to(device)
        label_shadow = label_shadow.to(device)

        optimizer.zero_grad() # Resets gadients before computing each loop
        logits_mag, logits_gain, logits_centering, logits_shadow = model(images) # Forward pass through the model to produce logits for each image

        # Loss function implementation
        loss_mag = criterion_mag(logits_mag, label_mag)
        loss_gain = criterion_gain(logits_gain, label_gain)
        loss_centering = criterion_centering(logits_centering, label_centering)
        loss_shadow = criterion_shadow(logits_shadow.squeeze(1), label_shadow.float())

        loss = w_mag*loss_mag + w_gain*loss_gain + w_centering*loss_centering + w_shadow*loss_shadow

        loss.backward() # Performs backpropagation
        optimizer.step() # Updates model parameters using the computed gradients

        train_loss += loss.item() # Accumulate train_loss through batches

    train_loss_avg = train_loss / len(train_loader) # Calculate average loss per batch

    # Validation starts
    model.eval() # Set model to evaluation mode
    val_loss = 0
    correct_mag = 0
    correct_gain = 0
    correct_centering = 0
    correct_shadow = 0
    total = 0
    total = 0
    total_mag = 0
    total_gain = 0
    total_centering = 0
    total_shadow = 0

    with torch.no_grad():
        for images, label_mag, label_gain, label_centering, label_shadow in val_loader:
            images = images.to(device) # Pass images to gpu
            label_mag = label_mag.to(device)
            label_gain = label_gain.to(device)
            label_centering = label_centering.to(device)
            label_shadow = label_shadow.to(device)

            logits_mag, logits_gain, logits_centering, logits_shadow = model(images) # Get predictions

            _, pred_mag = torch.max(logits_mag, 1)
            _, pred_gain = torch.max(logits_gain, 1)
            _, pred_centering = torch.max(logits_centering, 1)
            probs_shadow = torch.sigmoid(logits_shadow.squeeze(1)) # binary classification
            pred_shadow = (probs_shadow > 0.5).long() # binary classification

            loss_mag = criterion_mag(logits_mag, label_mag)
            loss_gain = criterion_gain(logits_gain, label_gain)
            loss_centering = criterion_centering(logits_centering, label_centering)
            loss_shadow = criterion_shadow(logits_shadow.squeeze(1), label_shadow.float())

            loss = w_mag*loss_mag + w_gain*loss_gain + w_centering*loss_centering + w_shadow*loss_shadow

            val_loss += loss.item() # Accumulate val_loss through batches

            total += label_mag.size(0) # Counts total number of data samples

            # Counts how many predictions match the true labels
            correct_mag += (pred_mag == label_mag).sum().item()
            correct_gain += (pred_gain == label_gain).sum().item()
            correct_centering += (pred_centering == label_centering).sum().item()
            correct_shadow += (pred_shadow == label_shadow).sum().item()

    # Calculate val accuracy
    acc_mag = 100 * correct_mag / total
    acc_gain = 100 * correct_gain / total
    acc_centering = 100 * correct_centering / total
    acc_shadow = 100 * correct_shadow / total

    acc_avg = (acc_mag + acc_gain + acc_centering + acc_shadow)/4

    val_loss_avg = val_loss / len(val_loader) # Calculate average val loss

    # Save best model
    if val_loss_avg < best_val_loss:
        best_val_loss = val_loss_avg
        filename = f"ckpt/Run_{start_time}_{model_name}_{acc_mag:.0f}_{acc_gain:.0f}_{acc_centering:.0f}_{acc_shadow:.0f}.pth"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss_avg
        }, filename)
        print(f"Epoch {epoch+1} is saved.")

        counter = 0 # counter is used for early stopping

    else:
        counter += 1 # counter is used for early stopping

    # Log training progress to .log file, and print in termial.
    log_line = (
        f"Epoch {epoch+1} | "
        f"Train Loss {train_loss_avg:.4f} | "
        f"Val Loss {val_loss_avg:.4f} | "
        f"Val Acc: mag({acc_mag:.2f}%) gain({acc_gain:.2f}%) centering({acc_centering:.2f}%) shadow({acc_shadow:.2f}%)"
    )
    print(log_line)
    logging.info(log_line)

    # Engage early stopping
    if counter >= patience:
        print("Early stopping triggered.")
        break

# Save final model
filename = f"ckpt/Run_{start_time}_{model_name}_{acc_mag:.0f}_{acc_gain:.0f}_{acc_centering:.0f}_{acc_shadow:.0f}_final.pth"
torch.save({
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "val_loss": val_loss_avg
}, filename)




