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
import optuna
from optuna.exceptions import TrialPruned

from utils import set_seed, focal_loss_binary, focal_loss_multiclass
from dataset import ImageDatasetFiveL, ImageDatasetFiveLSN, ImageDatasetFiveRGB
from models import FiveTaskWrapperResNet
from models import SonoNet, FiveTaskWrapperSonoNet
from models import OneTaskVisionTransformer, FiveTaskVisionTransformer, FiveTaskWrapperViT

###
# Before changing to new model, check if the model requires 1 channel or 3 channel input.
# The following must be changed to match 1 VS 3 channel input:
# (1) ImageDataset class
# (2) transform function

### Set up logging train loss and val loss during training
os.makedirs("./ckpt", exist_ok=True)
start_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
logging.basicConfig(
    filename=f"./ckpt/Optuna_{start_time}.log",
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

model_name = "sononet"

logging.info(f"img_dir: {img_dir}")
logging.info(f"csv_train: {csv_train}")
logging.info(f"csv_val: {csv_val}")
logging.info(f"model_name: {model_name}")

### Valid model names:
# "resnet50"
# "sononet"
# "vitb16"

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
    dataset_class = ImageDatasetFiveLSN
elif model_name == "vitb16":
    channel = 3
    dataset_class = ImageDatasetFiveRGB
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=channel),  # since conv1 expects 1 channel
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

### --------------------------------------------------- ###
### Step 3: Define train and val datasets
### --------------------------------------------------- ###
train_dataset = dataset_class(csv_file=csv_train, img_dir=img_dir, transform=transform)
val_dataset = dataset_class(csv_file=csv_val, img_dir=img_dir, transform=transform)
logging.info(f"Dataset: {train_dataset.__class__.__name__}")

### --------------------------------------------------- ###
### Step 4: Build dataloaders
### --------------------------------------------------- ###
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

### --------------------------------------------------- ###
### Step 5: Functions to Build Model
### --------------------------------------------------- ###
# Define gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_model(model_name, device, ckpt_path=None,
                num_ana=9, num_mag=3, num_gain=3, num_centering=5, num_shadow=1):

    # ==========================================
    # ResNet50
    # ==========================================
    if model_name == "resnet50":
        backbone = models.resnet50(weights=None) # ResNet50_Weights.IMAGENET1K_V2
        backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()

        model = FiveTaskWrapperResNet(
            backbone=backbone, 
            in_features=in_features, 
            num_ana=num_ana, 
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

        if ckpt_path is not None:
            # Define weights location
            ckpt = torch.load(ckpt_path, map_location=device)

            # Drop final layer weights to fit 13 classes to 9 classes
            # Use strict=True if original model has the same number of classes as the new model
            # Use strict=True if number of classes is different so it loads all weights except the final layer weights
            # Try the 3 lines below first. If they don't work, comment them out and use the 4th line
            keys_to_remove = [k for k in ckpt.keys() if k.startswith("adaption.3") or k.startswith("adaption.4")]
            for k in keys_to_remove:
                del ckpt[k]

            # Load weights
            base_model.load_state_dict(ckpt, strict=False)

        model = FiveTaskWrapperSonoNet(
            backbone=base_model,
            num_ana=num_ana,
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
        backbone = vit_b_16(weights=None) # ViT_B_16_Weights.IMAGENET1K_V1

        # Remove the ImageNet classifier
        backbone.heads = nn.Identity()

        # Call the multi-task wrapper
        model = FiveTaskWrapperViT(
            backbone=backbone,
            num_ana=num_ana,
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

        for param in model.head_ana.parameters():
            param.requires_grad = True

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
            param.requires_grad = True # False = freeze all blocks, True = unfreeze all blocks

        # ViT has 12 transformer blocks in total.
        # Unfreeze last 2 transformer blocks
        for block in model.backbone.encoder.layers[-2:]:
            for param in block.parameters():
                param.requires_grad = True

        # Unfreeze final LayerNorm
        for param in model.backbone.encoder.ln.parameters():
            param.requires_grad = True

    else:
        raise ValueError(f"Unknown model: {model_name}")

def train_val_loop(model, optimizer, train_loader, val_loader,
                   criterion_ana, criterion_mag, criterion_gain,
                   criterion_centering, criterion_shadow,
                   w_ana, w_mag, w_gain, w_centering, w_shadow,
                   device, total_epochs=100, patience=5, trial=None):
    
    # Initiate early stopping
    best_val_loss = float("inf")
    counter = 0

    for epoch in range(total_epochs):

        model.train() # Set model to train mode
        train_loss = 0.0

        for images, label_ana, label_mag, label_gain, label_centering, label_shadow in train_loader:
            images = images.to(device) # Pass images to gpu
            label_ana = label_ana.to(device)
            label_mag = label_mag.to(device)
            label_gain = label_gain.to(device)
            label_centering = label_centering.to(device)
            label_shadow = label_shadow.to(device)

            optimizer.zero_grad() # Resets gadients before computing each loop
            logits_ana, logits_mag, logits_gain, logits_centering, logits_shadow = model(images) # Forward pass through the model to produce logits for each image

            # Loss function implementation
            loss_ana = criterion_ana(logits_ana, label_ana)
            loss_mag = criterion_mag(logits_mag, label_mag)
            loss_gain = criterion_gain(logits_gain, label_gain)
            loss_centering = criterion_centering(logits_centering, label_centering)
            loss_shadow = criterion_shadow(logits_shadow.squeeze(1), label_shadow.float())

            loss = w_ana*loss_ana + w_mag*loss_mag + w_gain*loss_gain + w_centering*loss_centering + w_shadow*loss_shadow

            loss.backward() # Performs backpropagation
            optimizer.step() # Updates model parameters using the computed gradients

            train_loss += loss.item() # Accumulate train_loss through batches

        train_loss_avg = train_loss / len(train_loader) # Calculate average loss per batch

        # Validation starts
        model.eval() # Set model to evaluation mode
        val_loss = 0.0
        correct_ana = 0
        correct_mag = 0
        correct_gain = 0
        correct_centering = 0
        correct_shadow = 0
        total = 0

        with torch.no_grad():
            for images, label_ana, label_mag, label_gain, label_centering, label_shadow in val_loader:
                images = images.to(device) # Pass images to gpu
                label_ana = label_ana.to(device)
                label_mag = label_mag.to(device)
                label_gain = label_gain.to(device)
                label_centering = label_centering.to(device)
                label_shadow = label_shadow.to(device)

                logits_ana, logits_mag, logits_gain, logits_centering, logits_shadow = model(images) # Get predictions

                _, pred_ana = torch.max(logits_ana, 1) # Get the class output
                _, pred_mag = torch.max(logits_mag, 1)
                _, pred_gain = torch.max(logits_gain, 1)
                _, pred_centering = torch.max(logits_centering, 1)
                probs_shadow = torch.sigmoid(logits_shadow.squeeze(1)) # binary classification
                pred_shadow = (probs_shadow > 0.5).long() # binary classification

                loss_ana = criterion_ana(logits_ana, label_ana)
                loss_mag = criterion_mag(logits_mag, label_mag)
                loss_gain = criterion_gain(logits_gain, label_gain)
                loss_centering = criterion_centering(logits_centering, label_centering)
                loss_shadow = criterion_shadow(logits_shadow.squeeze(1), label_shadow.float())

                loss = w_ana*loss_ana + w_mag*loss_mag + w_gain*loss_gain + w_centering*loss_centering + w_shadow*loss_shadow

                val_loss += loss.item() # Accumulate val_loss through batches

                total += label_ana.size(0) # Counts total number of data samples

                correct_ana += (pred_ana == label_ana).sum().item() # Counts how many predictions match the true labels
                correct_mag += (pred_mag == label_mag).sum().item()
                correct_gain += (pred_gain == label_gain).sum().item()
                correct_centering += (pred_centering == label_centering).sum().item()
                correct_shadow += (pred_shadow == label_shadow).sum().item()

        acc_ana = 100 * correct_ana / total # Calculate val accuracy
        acc_mag = 100 * correct_mag / total
        acc_gain = 100 * correct_gain / total
        acc_centering = 100 * correct_centering / total
        acc_shadow = 100 * correct_shadow / total

        val_loss_avg = val_loss / len(val_loader) # Calculate average val loss

        # Check for early stopping
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            counter = 0 # counter is used for early stopping
        else:
            counter += 1 # counter is used for early stopping

        # Log training progress to .log file, and print in termial.
        log_line = (
            f"Epoch {epoch+1} | "
            f"Train Loss {train_loss_avg:.4f} | "
            f"Val Loss {val_loss_avg:.4f} | "
            f"Val Acc: ana({acc_ana:.2f}%) mag({acc_mag:.2f}%) gain({acc_gain:.2f}%) centering({acc_centering:.2f}%) shadow({acc_shadow:.2f}%)"
        )
        logging.info(log_line)

        # Pruning terminates clearly underperforming trials to save time
        if trial is not None:
            trial.report(val_loss_avg, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        # Engage early stopping
        if counter >= patience:
            break

    return best_val_loss

# =========================================================
# Optuna objective
# =========================================================
def objective(trial):
    '''
    The optuna objective should contain
    everything that must be rebuilt fresh for each trial.
    '''
    # =========================================================
    # Hyperparameter optimisation
    # =========================================================
    # Learning rates
    backbone_lr = trial.suggest_categorical("backbone_lr", [1e-6, 5e-6, 1e-5, 5e-5, 1e-4]) # log=True samples in log space, ie. 1e-6, 2e-6, 3e-6
    head_lr = trial.suggest_categorical("head_lr", [1e-6, 5e-6, 1e-5, 5e-5, 1e-4])
    weight_decay = trial.suggest_categorical("weight_decay", [1e-6, 5e-6, 1e-5, 5e-5, 1e-4])

    # Loss weights
    w_ana = 1.0
    w_mag = trial.suggest_float("w_mag", 0.5, 2.0)
    w_gain = trial.suggest_float("w_gain", 0.5, 2.0)
    w_centering = trial.suggest_float("w_centering", 0.5, 2.0)
    w_shadow = trial.suggest_float("w_shadow", 0.5, 2.0)

    # Loss function choice
    loss_func = trial.suggest_categorical("loss_func", ["ce", "focal"]) # "ce" for CrossEntropyLoss, "focal" for Focal Loss

    if loss_func == "focal":
        gamma_ana = trial.suggest_float("gamma_ana", 1.0, 3.0, step=0.5)
        gamma_mag = trial.suggest_float("gamma_mag", 1.0, 3.0, step=0.5)
        gamma_gain = trial.suggest_float("gamma_gain", 1.0, 3.0, step=0.5)
        gamma_centering = trial.suggest_float("gamma_centering", 1.0, 3.0, step=0.5)
        gamma_shadow = trial.suggest_float("gamma_shadow", 1.0, 3.0, step=0.5)
    else:
        gamma_ana = gamma_mag = gamma_gain = gamma_centering = gamma_shadow = 2.0

    # =========================================================
    # Build Model
    # =========================================================
    model = build_model(
        model_name=model_name,
        device=device,
        ckpt_path=ckpt_path,
        num_ana=num_ana,
        num_mag=num_mag,
        num_gain=num_gain,
        num_centering=num_centering,
        num_shadow=1  # binary classification
    )

    finetuning(model,model_name)

    model = model.to(device)

    # Define Loss function
    if loss_func == "ce":
        criterion_ana = nn.CrossEntropyLoss()
        criterion_mag = nn.CrossEntropyLoss()
        criterion_gain = nn.CrossEntropyLoss()
        criterion_centering = nn.CrossEntropyLoss()
        criterion_shadow = nn.BCEWithLogitsLoss() # Shadow is binary classification
    elif loss_func == "focal":
        criterion_ana = lambda x, y: focal_loss_multiclass(x, y, gamma=gamma_ana)
        criterion_mag = lambda x, y: focal_loss_multiclass(x, y, gamma=gamma_mag)
        criterion_gain = lambda x, y: focal_loss_multiclass(x, y, gamma=gamma_gain)
        criterion_centering = lambda x, y: focal_loss_multiclass(x, y, gamma=gamma_centering)
        criterion_shadow = lambda x, y: focal_loss_binary(x, y, gamma=gamma_shadow)

    # Define optimiser
    optimizer = torch.optim.Adam([
        {"params": model.backbone.parameters(), "lr": backbone_lr},
        {"params": model.head_ana.parameters(), "lr": head_lr},
        {"params": model.head_mag.parameters(), "lr": head_lr},
        {"params": model.head_gain.parameters(), "lr": head_lr},
        {"params": model.head_centering.parameters(), "lr": head_lr},
        {"params": model.head_shadow.parameters(), "lr": head_lr},
    ], weight_decay=weight_decay)

    # Training
    best_val_loss = train_val_loop(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion_ana=criterion_ana,
        criterion_mag=criterion_mag,
        criterion_gain=criterion_gain,
        criterion_centering=criterion_centering,
        criterion_shadow=criterion_shadow,
        w_ana=w_ana,
        w_mag=w_mag,
        w_gain=w_gain,
        w_centering=w_centering,
        w_shadow=w_shadow,
        device=device,
        total_epochs=100,
        patience=5,
        trial=trial # enables pruning
    )

    return best_val_loss

# =========================================================
# Run the study
# =========================================================
study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner()
)

def log_trial_callback(study, trial):
    logging.info(
        f"Trial {trial.number} finished with value: {trial.value:.6f} and parameters: {trial.params}\n")

study.optimize(objective, n_trials=70, callbacks=[log_trial_callback])

logging.info("\n===== OPTUNA FINISHED =====")
logging.info("Best trial:")
logging.info(f"Final val loss: {study.best_trial.value}")
logging.info(f"Params: {study.best_trial.params}")

print("\nBest trial:")
print(f"Final val loss: {study.best_trial.value}")
print(f"Params: {study.best_trial.params}")


