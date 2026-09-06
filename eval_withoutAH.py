import os
import random
import numpy as np
import pandas as pd
import datetime
import csv
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision import transforms
from torchvision.models import vit_b_16
from PIL import Image
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from utils import SentenceGenerator
from dataset import ImageDatasetEvalL, ImageDatasetEvalRGB
from models import SonoNet, FourTaskWrapperSonoNet, FourTaskWrapperResNet
from models import OneTaskVisionTransformer, FourTaskWrapperViT

model_name = "vitb16"
ckpt = "./ckpt/checkpoint_name.pth"

img_dir_path = "/path/to/directory/containing/all/test/set/images"
gnd_csv_file = "/path/to/csv/file/listing/test/set/image/names/and/ground/truth/labels/test_set.csv"
ckpt_stem = Path(ckpt).stem
print(f"ckpt_stem: {ckpt_stem}")

# Automatically store all results to ./ckpt/checkpoint_name-[idx]
os.makedirs("./ckpt", exist_ok=True)
output_folder_base = "./ckpt"
i = 0
while True:
    output_folder = os.path.join(output_folder_base, f"{ckpt_stem}-{i:02d}")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        break
    i += 1
print(f"Results will store to: {output_folder}")

# Number of classes for anatomy, magnification, gain, centering, shadow
num_ana = 9
num_mag = 3
num_gain = 3
num_centering = 5
num_shadow = 2

# Training parameters
batch_size = 1
num_workers = 4

# ================================================
# Make predictions on test set
# ================================================
if model_name in ["resnet50", "sononet"]:
    channel = 1
    dataset_class = ImageDatasetEvalL
elif model_name == "vitb16":
    channel = 3
    dataset_class = ImageDatasetEvalRGB

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=channel),  # since conv1 expects 1 channel
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

val_dataset = dataset_class(img_dir=img_dir_path,transform=transform)
val_loader = DataLoader(val_dataset,batch_size=batch_size,shuffle=False) #num_workers=num_workers

# Define gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_model_eval(model_name, num_mag=3, num_gain=3, num_centering=5, num_shadow=1):
    
    if model_name == "resnet50":
        backbone = models.resnet50(weights=None)
        backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()

        model = FourTaskWrapperResNet(
            backbone=backbone,
            in_features=in_features,
            num_mag=num_mag,
            num_gain=num_gain,
            num_centering=num_centering,
            num_shadow=num_shadow # binary classification
        )

    elif model_name == "sononet":
        backbone = SonoNet(config="SN64", num_labels=9, weights=False)

        model = FourTaskWrapperSonoNet(
            backbone=backbone,
            num_mag=num_mag,
            num_gain=num_gain,
            num_centering=num_centering,
            num_shadow=num_shadow # binary classification
        )

    elif model_name == "vitb16":
        backbone = vit_b_16(weights=None)
        backbone.heads = nn.Identity()

        model = FourTaskWrapperViT(
            backbone=backbone,
            num_mag=num_mag,
            num_gain=num_gain,
            num_centering=num_centering,
            num_shadow=num_shadow # binary classification
        )

    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    return model

model = build_model_eval(
    model_name=model_name, 
    num_mag=num_mag,
    num_gain=num_gain,
    num_centering=num_centering,
    num_shadow=1
)

# Load checkpoint
checkpoint = torch.load(ckpt, map_location=device)

# # Load weights
# model.load_state_dict(checkpoint["model_state_dict"], strict=True)

# Load weights & remove profiler/bookkeeping entries
if model_name == "resnet50":
    strictness = False
elif model_name == "sononet":
    strictness = True
elif model_name == "vitb16":
    strictness = False
state_dict = checkpoint["model_state_dict"]
state_dict = {k: v for k, v in state_dict.items()
              if k not in ["total_ops", "total_params"]}
model.load_state_dict(state_dict, strict=strictness)

# Move model to device
model = model.to(device)
print("model_name:", model_name)
print("model type:", type(model))
# print("head_ana.weight shape:", model.head_ana.weight.shape)
# print("head_ana.weight in checkpoint:", state_dict["head_ana.weight"].shape)
# print("head_mag.weight in checkpoint:", state_dict["head_mag.weight"].shape)
# print("head_gain.weight in checkpoint:", state_dict["head_gain.weight"].shape)
# print("head_centering.weight in checkpoint:", state_dict["head_centering.weight"].shape)
# print("head_shadow.weight in checkpoint:", state_dict["head_shadow.weight"].shape)
# print("sample keys:", list(state_dict.keys())[:20])

# Set to evaluation mode
model.eval()

# Predict and store results to CSV file: 
results = []
sentence_generator = SentenceGenerator()
with torch.no_grad():
    for images, names in val_loader:
        images = images.to(device)

        # Calculate inference time: Start
        if device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.time()

        # Forward pass
        logits_mag, logits_gain, logits_centering, logits_shadow = model(images)

        # Calculate inference time: End
        if device.type == "cuda":
            torch.cuda.synchronize()
        inference_time_ms = (time.time() - start_time) * 1000

        # Predictions
        pred_mag = logits_mag.argmax(dim=1)
        pred_gain = logits_gain.argmax(dim=1)
        pred_centering = logits_centering.argmax(dim=1)
        prob_shadow = torch.sigmoid(logits_shadow.squeeze(1)) # binary classification
        pred_shadow = (prob_shadow > 0.5).long() # binary classification

        # Confidences
        conf_mag = torch.softmax(logits_mag, dim=1).max(dim=1)[0]
        conf_gain = torch.softmax(logits_gain, dim=1).max(dim=1)[0]
        conf_centering = torch.softmax(logits_centering, dim=1).max(dim=1)[0]
        conf_shadow = torch.maximum(prob_shadow, 1 - prob_shadow) # binary classification

        # Build the input format that Sentence Generator requires and use Sentence Generator
        preds_dict = {
            "mag": pred_mag[0].item(),
            "gain": pred_gain[0].item(),
            "centering": pred_centering[0].item(),
            "shadow": pred_shadow[0].item(),
        }
        
        results.append({
            "frame": names[0], 
            "pred_mag": pred_mag[0].item(),
            "pred_gain": pred_gain[0].item(),
            "pred_centering": pred_centering[0].item(),
            "pred_shadow": pred_shadow[0].item(),
            "conf_mag": round(conf_mag[0].item(), 2),
            "conf_gain": round(conf_gain[0].item(), 2),
            "conf_centering": round(conf_centering[0].item(), 2),
            "conf_shadow": round(conf_shadow[0].item(), 2),
            "inf_time_ms": round(inference_time_ms, 2),
        })

pred_df = pd.DataFrame(results)

gnd_df = pd.read_csv(gnd_csv_file)

# Match rows by filename
concat_df = pred_df.merge(
    gnd_df,
    on="frame",
    how="left"
)

pred_csv = os.path.join(output_folder, "pred.csv")
concat_df.to_csv(pred_csv, index=False)
print(f"Predictions concatenated with gnd and saved to {Path(pred_csv).stem}.")


# ================================================
# Accuracy, Precision, Recall, F1 calculation
# ================================================
full_results_df = pd.read_csv(pred_csv)

# Define column pairs
categories = {
    "Magnification": ("pred_mag", "gnd_mag"),
    "Gain": ("pred_gain", "gnd_gain"),
    "Centering": ("pred_centering", "gnd_centering"),
    "Shadow": ("pred_shadow", "gnd_shadow"),
}

# Accuracy, Precision, Recall, F1 Calculation
metrics_lines = []
for name, (pred_col, gnd_col) in categories.items():
    acc = (full_results_df[pred_col] == full_results_df[gnd_col]).mean()

    if name == "Anatomy":
        y_true = full_results_df[gnd_col]
        y_pred = full_results_df[pred_col]
    else:
        df_filtered = full_results_df # df_filtered = full_results_df[full_results_df["gnd_ana"] != 8]
        y_true = df_filtered[gnd_col]
        y_pred = df_filtered[pred_col]

    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print(f"{name}:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-score:  {f1:.4f}")

    metrics_lines.append(f"{name}:")
    metrics_lines.append(f"  Accuracy: {acc:.4f}")
    metrics_lines.append(f"  Precision: {precision:.4f}")
    metrics_lines.append(f"  Recall:    {recall:.4f}")
    metrics_lines.append(f"  F1-score:  {f1:.4f}")
    metrics_lines.append("")

metrics_path = os.path.join(output_folder, "metrics.txt")
with open(metrics_path, "w") as f:
    f.write(f"ckpt: {ckpt}\n")
    f.write(f"img_dir_path: {img_dir_path}\n")
    f.write(f"model_name: {model_name}\n\n")
    f.write("\n".join(metrics_lines))
print(f"Stored metrics to {Path(metrics_path).stem}.")

# ================================================
# Confusion Matrices
# ================================================
# Define label mapping
map_ana = {
    0: "HC",
    1: "Cerebellum",
    2: "Nose & Lips",
    3: "AC", 
    4: "Femur", 
    5: "Spine (Coronal)", 
    6: "Spine (Sagittal)", 
    7: "Cardiac", 
    8: "Other",
}

map_mag = {
    0: "good_mag",
    1: "increase_mag",
    2: "reduce_mag",
}

map_gain = {
    0: "good_gain",
    1: "increase_gain",
    2: "reduce_gain", 
}

map_centering = {
    0: "good_centering",
    1: "move_left",
    2: "move_right",
    3: "move_up", 
    4: "move_down", 
}

map_shadow = {
    0: "no_shadow",
    1: "shadow",
}

full_results_df = full_results_df
df_filtered = full_results_df

# Extract columns
pred_mag = df_filtered["pred_mag"]
gnd_mag = df_filtered["gnd_mag"]

pred_gain = df_filtered["pred_gain"]
gnd_gain = df_filtered["gnd_gain"]

pred_centering = df_filtered["pred_centering"]
gnd_centering = df_filtered["gnd_centering"]

pred_shadow = df_filtered["pred_shadow"]
gnd_shadow = df_filtered["gnd_shadow"]

# Ensure consistent label ordering
labels_mag = sorted(map_mag.keys())
display_labels_mag = [map_mag[i] for i in labels_mag]

labels_gain = sorted(map_gain.keys())
display_labels_gain = [map_gain[i] for i in labels_gain]

labels_centering = sorted(map_centering.keys())
display_labels_centering = [map_centering[i] for i in labels_centering]

labels_shadow = sorted(map_shadow.keys())
display_labels_shadow = [map_shadow[i] for i in labels_shadow]

# Compute confusion matrix (not normalised)
cm_mag_og = confusion_matrix(gnd_mag, pred_mag, labels=labels_mag)
cm_gain_og = confusion_matrix(gnd_gain, pred_gain, labels=labels_gain)
cm_centering_og = confusion_matrix(gnd_centering, pred_centering, labels=labels_centering)
cm_shadow_og = confusion_matrix(gnd_shadow, pred_shadow, labels=labels_shadow)

# Compute confusion matrix (normalised)
cm_mag_n = confusion_matrix(gnd_mag, pred_mag, labels=labels_mag, normalize="true")
cm_gain_n = confusion_matrix(gnd_gain, pred_gain, labels=labels_gain, normalize="true")
cm_centering_n = confusion_matrix(gnd_centering, pred_centering, labels=labels_centering, normalize="true")
cm_shadow_n = confusion_matrix(gnd_shadow, pred_shadow, labels=labels_shadow, normalize="true")

# Display
def plot_and_save_confusion_matrices(
    cm_mag, cm_gain, cm_centering, cm_shadow,
    display_labels_mag, display_labels_gain,
    display_labels_centering, display_labels_shadow,
    output_svg_path
):
    cms = [cm_mag, cm_gain, cm_centering, cm_shadow]
    titles = ["Magnification", "Gain", "Centering", "Shadow"]
    display_labels = [
        display_labels_mag,
        display_labels_gain,
        display_labels_centering,
        display_labels_shadow
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i in range(4):
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cms[i],
            display_labels=display_labels[i]
        )
        disp.plot(ax=axes[i], cmap="Blues", colorbar=False)
        axes[i].set_title(titles[i])
        axes[i].tick_params(axis="x", rotation=60)

    # Remove unused subplot
    fig.delaxes(axes[4])
    fig.delaxes(axes[5])

    plt.tight_layout()
    fig.savefig(output_svg_path, bbox_inches="tight")
    # plt.show()
    # plt.close(fig)

svg_path_norm = os.path.join(output_folder, "confusion_matrices_norm.svg")
plot_and_save_confusion_matrices(
    cm_mag_n, cm_gain_n, cm_centering_n, cm_shadow_n,
    display_labels_mag, display_labels_gain,
    display_labels_centering, display_labels_shadow,
    svg_path_norm
)
print(f"Saved confusion matrices (normalised) to {svg_path_norm}.")

svg_path_og = os.path.join(output_folder, "confusion_matrices_og.svg")
plot_and_save_confusion_matrices(
    cm_mag_og, cm_gain_og, cm_centering_og, cm_shadow_og,
    display_labels_mag, display_labels_gain,
    display_labels_centering, display_labels_shadow,
    svg_path_og
)
print(f"Saved confusion matrices to {svg_path_og}.")

