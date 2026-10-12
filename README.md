# ENGE707 Assessment 3 – Cat & Dog Facial Feature Classification

Deep learning pipeline for the ENGE707-2026-S2 Kaggle competition: classifying images of
cat and dog facial features into eight classes (`CatEye`, `CatFace`, `CatMouth`, `CatNose`,
`DogEye`, `DogFace`, `DogMouth`, `DogNose`).

- **Author:** James Walters
- **Kaggle username:** jameswalters1

## Repository contents

| File / folder | Purpose |
|---|---|
| `train.py` | Full pipeline: data split, preprocessing/augmentation, model training, validation metrics, Kaggle submission file |
| `Assignment_2.ipynb` | Notebook used to run `train.py` and produce figures (data exploration, learning curves, confusion matrix, misclassified examples) |
| `requirements.txt` | Exact package versions used |
| `runs/<run-name>/` | Outputs from each training run (see below) |

## Setup

Tested on Windows 11, Python 3.14, PyTorch 2.14.1 with an Intel Arc 140V GPU (PyTorch XPU backend).
The script also runs on NVIDIA GPUs (CUDA) or CPU; the device is selected automatically.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# PyTorch for Intel GPUs (XPU). For NVIDIA or CPU, use the matching command from https://pytorch.org
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/xpu

python -m pip install -r requirements.txt
```

Check the GPU is detected:

```powershell
python -c "import torch; print(torch.xpu.is_available())"
```

## Data

The dataset is not included in this repository. Download it from the competition's
Data tab on Kaggle and extract it so it sits one level above this repository folder:

The pretrained model weights are downloaded automatically on the first run.

### Main options

| Option | Default | Description |
|---|---|---|
| `--model` | `efficientnet_b0` | Any model name from the `timm` library |
| `--img-size` | `160` | Input image size (pixels, square) |
| `--epochs` | `12` | Maximum training epochs |
| `--batch-size` | `64` | Images per batch |
| `--lr` | `3e-4` | Peak learning rate (OneCycle schedule) |
| `--weight-decay` | `0.05` | AdamW weight decay |
| `--label-smoothing` | `0.1` | Cross-entropy label smoothing |
| `--val-frac` | `0.15` | Fraction of training images held out for validation |
| `--patience` | `3` | Early stopping patience (epochs without validation-loss improvement) |
| `--seed` | `42` | Random seed for the split and training |
| `--no-pretrained` | off | Train from random weights instead of ImageNet weights |
| `--no-amp` | off | Disable bfloat16 mixed precision |

Run `python train.py --help` for the full list.

### Outputs

Each run writes to `runs/<run-name>/`:

| File | Contents |
|---|---|
| `config.json` | All settings used for the run |
| `history.csv` | Per-epoch training/validation loss and accuracy |
| `best.pt` | Model weights from the best epoch (not committed – too large) |
| `val_report.txt` | Per-class precision, recall and F1 on the validation set |
| `confusion_matrix.csv` | Validation confusion matrix (rows = true, columns = predicted) |
| `val_predictions.csv` | Prediction and confidence for every validation image |
| `submission.csv` | Kaggle submission file (`image_id,class`) |

## Results

| Run | Model | Image size | Best epoch | Validation accuracy | Kaggle public score |
|---|---|---|---|---|---|
| `effb0_160` | EfficientNet-B0 (ImageNet pretrained) | 160 | 6 of 9 | 0.9883 | 0.9705 |

## Acknowledgements

- PyTorch – https://pytorch.org
- timm (PyTorch Image Models), R. Wightman – https://github.com/huggingface/pytorch-image-models
- EfficientNet: Tan, M., & Le, Q. (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.* ICML.
- Pretrained weights: `efficientnet_b0.ra_in1k` (ImageNet-1k), via timm / Hugging Face Hub
- scikit-learn, pandas, NumPy, Pillow, Matplotlib
- AI-assisted coding tools were used for coding support, as permitted by the assessment guidelines.
