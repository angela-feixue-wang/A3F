# A3F: Anatomy-Aware Actionable Feedback for Fetal Ultrasound

Official repository for the 2026 2nd MICCAI Workshop on Human-AI Collaboration paper:

A3F: Anatomy-Aware Actionable Feedback for Fetal Ultrasound

## Dependencies

To install the dependencies into a new conda environment, simpy run:

```bash
conda env create -f environment.yml
conda activate a3f
```

Alternatively, you can install the packages manually:
```bash
conda create -n a3f python=3.12
conda activate a3f
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
conda install pandas=2.3.3
conda install matplotlib=3.10.8
pip install decord
pip install opencv-python==4.10.0.84
pip install grad-cam==1.5.5
pip install ultralytics-thop
pip install optuna
```

Run any Python script from the terminal using:
```bash
python3 file_name.py
```
Replace `file_name.py` with the name of the script you wish to run. 

## Computing environment

The experiments in this research were run in the following software and hardware environment:
- Operating system: Linux Ubuntu 24.04
- Environment management: Miniconda
- Python: 3.12.12
- PyTorch: 2.7.1 + CUDA 12.8
- torchvision: 0.22.1 + CUDA 12.8
- GPU: NVIDIA RTX Pro 4500, compute capability 12.0 (`sm_120`)

## Usage

### Data
The private clinical dataset **PULSE** is not allowed to be released to the public, so it is not included here. Instead, we describe below the expected dataset organisation and file formats required by the data-loading pipeline used in this research.

The training and validation images (raw images and their augmented variants if data augmentation is being used) should be stored together in a single directory, while the test images (raw images only) should be stored in a separate directory. 

The training, validation, and test splits are specified in separate CSV files, which contain the image filenames and their corresponding labels. These files are essential for maintaining consistent data partitioning and preventing data leakage. [`Dataset_Splits`](./Dataset_Splits) provides example CSV files to demonstrate the required labelling format compatible with the data-loading pipeline. 

The expected directory structure is:

```text
A3F/
├── Dataset/
│   ├── train_val_images/
│   │   ├── image0001.jpg
│   │   ├── image0002.jpg
│   │   ├── image0003.jpg
│   │   └── ...
│   └── test_images/
│       ├── image9001.jpg
│       ├── image9002.jpg
│       ├── image9003.jpg
│       └── ...
│
└── Dataset_Splits/
    ├── fold1_train_withaug.csv
    ├── fold1_val_withaug.csv
    ├── fold2_train_withaug.csv
    ├── fold2_val_withaug.csv
    ├── ...
    └── test_set.csv
```

### Scripts
The main Python scripts in this repository are listed below. Click a script name to view its source code.
| Script | Description |
|--------|-------------|
| [`annotator.py`](./annotator.py) | The custom annotation tool used in the research. It was developed and tested on Linux Ubuntu 24.04. Running the tool on Windows may require minor code modifications to accommodate platform-specific GUI bindings. |
| [`augmentation.ipynb`](./augmentation.ipynb) | Performs data augmentation for training and validation images. |
| [`dataset.py`](./dataset.py) | Contains dataset classes used in the research. |
| [`eval_visualiser.py`](./eval_visualiser.py) | A visualiser used to inspect test images with their predictions and ground truth labels. |
| [`eval_withAH.py`](./eval_withAH.py) | Evaluates model performance on test set. Generates a `pred.csv` listing all test images with predicted results and ground truth labels, a `metrics.txt` containing computed evaluation metrics, and confusion matrices stored as SVG files. Used for models that have an anatomy head. |
| [`eval_withoutAH.py`](./eval_withoutAH.py) | Evaluates model performance on test set. Generates a `pred.csv` listing all test images with predicted results and ground truth labels, a `metrics.txt` containing computed evaluation metrics, and confusion matrices stored as SVG files. Used for models that don't have an anatomy head. |
| [`models.py`](./models.py) | Contains models used in the research. |
| [`train_withAH.py`](./train_withAH.py) | Trains the model with anatomy head. |
| [`train_withAH_optuna.py`](./train_withAH_optuna.py) | Trains the model with anatomy head, using Optuna for hyperparameter optimisation. |
| [`train_withoutAH.py`](./train_withoutAH.py) | Trains the model without anatomy head. |
| [`utils.py`](./utils.py) | Contains utility functions (`set_seed`, loss functions, `SentenceGenerator`). |

### Typical workflow

The scripts are generally run in the following order:

1. Annotate the data: [`annotator.py`](./annotator.py)
2. Data augmentation: [`augmentation.ipynb`](./augmentation.ipynb)
3. Train the model: [`train_withAH.py`](./train_withAH.py)
4. Evaluate model: [`eval_withAH.py`](./eval_withAH.py)
5. Inspect results: [`eval_visualiser.py`](./eval_visualiser.py)

## Citation
If you find A3F useful, please cite the following BibTeX entry:

```bibtex
@inproceedings{Wang2026,
  title={A3F: Anatomy-Aware Actionable Feedback for Fetal Ultrasound},
  author={Wang, Angela Feixue and Lamdouar, Hala and Lander, Jayne and Papageorghiou, Aris T. and Noble, J. Alison},
  booktitle={2nd MICCAI Workshop on Human-AI Collaboration},
  year={2026}
}
```

## License
MIT License
