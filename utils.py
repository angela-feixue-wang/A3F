import math
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# Set random seed
def set_seed(seed=42):
    random.seed(seed) # Python random
    np.random.seed(seed) # Numpy random
    torch.manual_seed(seed) # PyTorch CPU
    torch.cuda.manual_seed(seed) # PyTorch GPU
    torch.cuda.manual_seed_all(seed) # all GPUs if using multi-GPU

    # For deterministic behavior (may reduce speed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # For Python hash-based operations
    os.environ['PYTHONHASHSEED'] = str(seed)

def focal_loss_multiclass(logits, targets, alpha=1.0, gamma=2.0, reduction="mean"):
    ce_loss = F.cross_entropy(logits, targets, reduction="none")
    p_t = torch.exp(-ce_loss)
    loss = alpha * (1.0 - p_t) ** gamma * ce_loss

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss

def focal_loss_binary(logits, targets, alpha=1.0, gamma=2.0, reduction="mean"):
    targets = targets.float()
    bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = torch.exp(-bce_loss)
    loss = alpha * (1.0 - p_t) ** gamma * bce_loss

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss

class SentenceGenerator:
    """
    This is the final version of SentenceGenerator used in the A3F paper to MICCAI 2026 HAIC workshop.
    """
    def __init__(self):
        # Define label mapping
        self.phrase_maps = {
            # "ana" map is used for "centering" phrasing only.
            # Hence HC (0) and cerebellum (1) both named "head"; 
            # and coronal (5) and sagittal (6) spine both named "spine". 
            "ana": {  
                0: "head", # "HC"
                1: "head", # "Cerebellum"
                2: "lips and nose", # "Lips and nose"
                3: "abdomen", # "Abdomen"
                4: "femur", # "Femur"
                5: "spine", # "Spine (coronal)"
                6: "spine", # "Spine (sagittal)"
                7: "heart", # "Cardiac"
                8: "Not an NHS FASP view", # "Other"
            },
            "mag": {
                0: "good magnification",
                1: "increase magnification",
                2: "reduce magnification",
            },
            "gain": {
                0: "good gain",
                1: "increase gain",
                2: "reduce gain", 
            },
            "centering": {
                0: "good centering",
                1: "to the left",
                2: "to the right",
                3: "up", 
                4: "down", 
            },
            "shadow": {
                0: None, # suppress no_shadow outputs so it doesn't output shadow label when predicts no_shadow
                1: "shadow",
            }
        }

    def join_and(self, items):
        if not items:
            return ""
        if len(items) == 1:
            return items[0]
        if len(items) == 2:
            return " and ".join(items)
        return ", ".join(items[:-1]) + ", and " + items[-1]

    def build_sentence(self, preds):

        # If not a NHS FASP view -> return immediately (silent output)
        if preds["ana"] == 8:
            return ""

        phrases = []

        # Do not output anatomy name as this is redundant information for the sonographer. 

        # Shadow
        shadow_phrase = self.phrase_maps["shadow"].get(preds["shadow"])
        if shadow_phrase:
            phrases.append(shadow_phrase)

        # Only outputs instructions, ignores "this is good" statements
        adjust_phrases = []
        for head in ["mag", "gain"]:
            pred = preds[head]
            phrase = self.phrase_maps[head][pred]
            if pred != 0:
                adjust_phrases.append(phrase)

        centering_pred = preds["centering"]
        centering_phrase = self.phrase_maps["centering"][centering_pred]
        if centering_phrase:
            ana_phrase = self.phrase_maps["ana"][preds["ana"]]
            adjust_phrases.append(f"move {ana_phrase} {centering_phrase}")

        if adjust_phrases:
            phrases.append(self.join_and(adjust_phrases))

        sentence = ", ".join(phrases) + "."
        return sentence[:1].upper() + sentence[1:] # First letter of the sentence should be capital letter.
