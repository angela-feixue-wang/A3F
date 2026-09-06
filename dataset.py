import math
import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

'''Note on Dataset class naming convention: 
1. Image = still images; Video = video clips
2. Dataset
3. Five = w/ anatomy head; Four = w/o anatomy head; Eval = inference mode (labels aren't processed)
4. L = images converted to greyscale; RGB = images converted to RGB
5. SN = sample-based normalisation; GN = global normalisation; [BLANK] = no normalisation
'''

class ImageDatasetEvalLSN(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.image_files = sorted([
            f for f in os.listdir(img_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

    def __len__(self): # Calculate dataset size
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.img_dir, img_name) # Define path to jpg image
        image = Image.open(img_path).convert("L") # Convert image to greyscale

        if self.transform:
            image = self.transform(image)

        # Sample-based normalisation (ie.e normalise each image by its own average and standard deviation)
        mean = image.mean()
        std = image.std()
        if std == 0:
            std = 1.0
        image = (image - mean) / std

        return image, img_name

class ImageDatasetEvalL(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.image_files = sorted([
            f for f in os.listdir(img_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

    def __len__(self): # Calculate dataset size
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.img_dir, img_name) # Define path to jpg image
        image = Image.open(img_path).convert("L") # Convert image to greyscale

        if self.transform:
            image = self.transform(image)

        return image, img_name

class ImageDatasetEvalRGB(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.image_files = sorted([
            f for f in os.listdir(img_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

    def __len__(self): # Calculate dataset size
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.img_dir, img_name) # Define path to jpg image
        image = Image.open(img_path).convert("RGB") # Convert image to greyscale

        if self.transform:
            image = self.transform(image)

        return image, img_name
    
class ImageDatasetFiveLSN(Dataset):
    '''
    csv_file must be .csv and contain columns with headers named exactly as the following:
    "frame"
    "gnd_ana"
    "gnd_mag"
    "gnd_gain"
    "gnd_centering"
    "gnd_shadow"
    '''
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file) # use "header=None" if csv file doesn't have headers
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self): # Calculate dataset size
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        img_name = str(row["frame"])
        label_ana = int(row["gnd_ana"])
        label_mag = int(row["gnd_mag"])
        label_gain = int(row["gnd_gain"])
        label_centering = int(row["gnd_centering"])
        label_shadow = int(row["gnd_shadow"])

        img_path = os.path.join(self.img_dir, img_name) # Define path to jpg image
        image = Image.open(img_path).convert("L") # Convert image to greyscale

        if self.transform:
            image = self.transform(image)

        # Sample-based normalisation (ie.e normalise each image by its own average and standard deviation)
        mean = image.mean()
        std = image.std()
        if std == 0:
            std = 1.0
        image = (image - mean) / std

        return image, label_ana, label_mag, label_gain, label_centering, label_shadow

class ImageDatasetFiveL(Dataset):
    '''
    csv_file must be .csv and contain columns with headers named exactly as the following:
    "frame"
    "gnd_ana"
    "gnd_mag"
    "gnd_gain"
    "gnd_centering"
    "gnd_shadow"
    '''
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file) # use "header=None" if csv file doesn't have headers
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self): # Calculate dataset size
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        img_name = str(row["frame"])
        label_ana = int(row["gnd_ana"])
        label_mag = int(row["gnd_mag"])
        label_gain = int(row["gnd_gain"])
        label_centering = int(row["gnd_centering"])
        label_shadow = int(row["gnd_shadow"])

        img_path = os.path.join(self.img_dir, img_name) # Define path to jpg image
        image = Image.open(img_path).convert("L") # Convert image to greyscale

        if self.transform:
            image = self.transform(image)

        return image, label_ana, label_mag, label_gain, label_centering, label_shadow

class ImageDatasetFiveRGBSN(Dataset):
    '''
    csv_file must be .csv and contain columns with headers named exactly as the following:
    "frame"
    "gnd_ana"
    "gnd_mag"
    "gnd_gain"
    "gnd_centering"
    "gnd_shadow"
    '''
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file) # use "header=None" if csv file doesn't have headers
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self): # Calculate dataset size
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        img_name = str(row["frame"])
        label_ana = int(row["gnd_ana"])
        label_mag = int(row["gnd_mag"])
        label_gain = int(row["gnd_gain"])
        label_centering = int(row["gnd_centering"])
        label_shadow = int(row["gnd_shadow"])

        img_path = os.path.join(self.img_dir, img_name) # Define path to jpg image
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Sample-based normalisation (ie.e normalise each image by its own average and standard deviation)
        mean = image.mean()
        std = image.std()
        if std == 0:
            std = 1.0
        image = (image - mean) / std

        return image, label_ana, label_mag, label_gain, label_centering, label_shadow

class ImageDatasetFiveRGB(Dataset):
    '''
    csv_file must be .csv and contain columns with headers named exactly as the following:
    "frame"
    "gnd_ana"
    "gnd_mag"
    "gnd_gain"
    "gnd_centering"
    "gnd_shadow"
    '''
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file) # use "header=None" if csv file doesn't have headers
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self): # Calculate dataset size
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        img_name = str(row["frame"])
        label_ana = int(row["gnd_ana"])
        label_mag = int(row["gnd_mag"])
        label_gain = int(row["gnd_gain"])
        label_centering = int(row["gnd_centering"])
        label_shadow = int(row["gnd_shadow"])

        img_path = os.path.join(self.img_dir, img_name) # Define path to jpg image
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label_ana, label_mag, label_gain, label_centering, label_shadow

class ImageDatasetFourLSN(Dataset):
    '''
    csv_file must be .csv and contain columns with headers named exactly as the following:
    "frame"
    "gnd_mag"
    "gnd_gain"
    "gnd_centering"
    "gnd_shadow"
    '''
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file) # use "header=None" if csv file doesn't have headers
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self): # Calculate dataset size
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        img_name = str(row["frame"])
        label_mag = int(row["gnd_mag"])
        label_gain = int(row["gnd_gain"])
        label_centering = int(row["gnd_centering"])
        label_shadow = int(row["gnd_shadow"])

        img_path = os.path.join(self.img_dir, img_name) # Define path to jpg image
        image = Image.open(img_path).convert("L") # Convert image to greyscale

        if self.transform:
            image = self.transform(image)

        # Sample-based normalisation (ie.e normalise each image by its own average and standard deviation)
        mean = image.mean()
        std = image.std()
        if std == 0:
            std = 1.0
        image = (image - mean) / std

        return image, label_mag, label_gain, label_centering, label_shadow
    
class ImageDatasetFourL(Dataset):
    '''
    csv_file must be .csv and contain columns with headers named exactly as the following:
    "frame"
    "gnd_mag"
    "gnd_gain"
    "gnd_centering"
    "gnd_shadow"
    '''
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file) # use "header=None" if csv file doesn't have headers
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self): # Calculate dataset size
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        img_name = str(row["frame"])
        label_mag = int(row["gnd_mag"])
        label_gain = int(row["gnd_gain"])
        label_centering = int(row["gnd_centering"])
        label_shadow = int(row["gnd_shadow"])

        img_path = os.path.join(self.img_dir, img_name) # Define path to jpg image
        image = Image.open(img_path).convert("L") # Convert image to greyscale

        if self.transform:
            image = self.transform(image)

        return image, label_mag, label_gain, label_centering, label_shadow
    
class ImageDatasetFourRGBSN(Dataset):
    '''
    csv_file must be .csv and contain columns with headers named exactly as the following:
    "frame"
    "gnd_mag"
    "gnd_gain"
    "gnd_centering"
    "gnd_shadow"
    '''
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file) # use "header=None" if csv file doesn't have headers
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self): # Calculate dataset size
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        img_name = str(row["frame"])
        label_mag = int(row["gnd_mag"])
        label_gain = int(row["gnd_gain"])
        label_centering = int(row["gnd_centering"])
        label_shadow = int(row["gnd_shadow"])

        img_path = os.path.join(self.img_dir, img_name) # Define path to jpg image
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Sample-based normalisation (ie.e normalise each image by its own average and standard deviation)
        mean = image.mean()
        std = image.std()
        if std == 0:
            std = 1.0
        image = (image - mean) / std

        return image, label_mag, label_gain, label_centering, label_shadow

class ImageDatasetFourRGB(Dataset):
    '''
    csv_file must be .csv and contain columns with headers named exactly as the following:
    "frame"
    "gnd_mag"
    "gnd_gain"
    "gnd_centering"
    "gnd_shadow"
    '''
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file) # use "header=None" if csv file doesn't have headers
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self): # Calculate dataset size
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        img_name = str(row["frame"])
        label_mag = int(row["gnd_mag"])
        label_gain = int(row["gnd_gain"])
        label_centering = int(row["gnd_centering"])
        label_shadow = int(row["gnd_shadow"])

        img_path = os.path.join(self.img_dir, img_name) # Define path to jpg image
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label_mag, label_gain, label_centering, label_shadow
    
