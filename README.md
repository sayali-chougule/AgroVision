# 🌿 AgroVision - Crop Pest & Disease Classification

AgroVision is a data-driven image classification system that leverages machine learning/ deep learning to diagnose crop pests and diseases from images. The project focuses on data understanding, model development, and performance evaluation. The final output includes a trained model, evaluation metrics, and documented insights that can be used to diagnose crop health for crops local to Ghana, as well as serve as a foundation for future models specific to other regions of the globe.

### Dataset:
[Link](https://data.mendeley.com/datasets/bwh3zbpkpv/1)


### Development Environment:

This project was primarily developed and executed using Google Colab with Google Drive mounted for dataset storage and pipeline execution.

### Pipeline stages:

```
Raw Images → Ingest → Validate → Preprocess → Split → Train → Evaluate → Deploy
```

### Project Setup:
```
agrovision/
├── config.yaml                  ← All settings (paths, hyperparameters)
├── pipeline.py                  ← Master runner — runs all stages
├── run.py                       ← Launch deployment server
├── start_agrovision.bat         ← Double-click to start app (Windows)
├── start_agrovision.sh          ← Shell script to start app (Mac/Linux)
├── requirements.txt             ← Training dependencies
├── requirements_deploy.txt      ← Deployment dependencies
│
├── src/
│   ├── stage_01_ingest.py       ← Scan dataset, build manifest CSV
│   ├── stage_02_validate.py     ← Remove corrupt/duplicate/low-res images
│   ├── stage_03_preprocess.py   ← Resize to 224×224, save processed images
│   ├── stage_04_split.py        ← Encode labels, stratified train/val/test split
│   ├── stage_05_train.py        ← Train EfficientNet-B3 with checkpointing
│   ├── stage_06_evaluate.py     ← Classification report on test set
│   └── serve/
│       ├── __init__.py
│       └── app.py               ← FastAPI backend (loads best.pth, serves predictions)
│
├── templates/
│   └── index.html               ← Web UI (drag-and-drop disease classifier)
│
├── static/                      ← Static assets (icons, images)
│
├── data/
│   └── raw/                     ← Put your dataset here (or sync from Drive)
│       ├── Tomato__Early_Blight/
│       ├── Tomato__Healthy/
│       └── ...
│
├── logs/
│   ├── pipeline.log             ← Full pipeline log
│   ├── .done_ingest             ← Checkpoint flags (auto-created)
│   ├── .done_validate
│   └── ...
│
└── output/
    ├── manifest.csv
    ├── manifest_clean.csv
    ├── manifest_processed.csv
    ├── class_map.json
    ├── evaluation_report.txt
    ├── processed/               ← Resized images
    ├── splits/
    │   ├── train.csv
    │   ├── val.csv
    │   └── test.csv
    └── checkpoints/
        └── best.pth             ← Trained model weights (saved here)              ←
```
### Part 1 — Training on Google Colab

The entire training pipeline runs on Google Colab using the dataset stored on Google Drive.

#### 1. Step 1 - Open Google Colab

Create New notebook

#### Step 2 — Enable GPU

```
Runtime → Change runtime type → Hardware accelerator → GPU (T4) → Save
```

#### Step 3 — Mount Google Drive
``` bash
from google.colab import drive
drive.mount('/content/drive')
```

#### Step 4 — Install dependencies
``` bash
!pip install timm albumentations imagehash tqdm pandas opencv-python-headless -q
```

#### Step 5 — Upload project to Colab
Either upload manually via the Files panel (left sidebar in Colab), or clone from GitHub:

``` bash 
!git clone https://github.com/yourname/agrovision.git
%cd agrovision
```

#### Step 6 — Run the full pipeline 

```bash
!python pipeline.py
```
This runs all 6 stages in order automatically. All outputs are saved back to Google Drive.

#### Step 7 — If Colab disconnects mid-training
Colab free tier disconnects after ~90 min of inactivity. Just re-run — completed stages are automatically skipped thanks to checkpoint flags in logs/:

```bash
# Resume from a specific stage (skips everything before it)
!python pipeline.py train
!python pipeline.py split
```

#### Step 8 — Pipeline stages reference

| Stage | File | What it does | Output |
|-------|------|-------------|--------|
| 1 | `stage_01_ingest.py` | Scans Drive dataset folder | `manifest.csv` |
| 2 | `stage_02_validate.py` | Removes corrupt / duplicate / low-res images | `manifest_clean.csv` |
| 3 | `stage_03_preprocess.py` | Resizes all images to 224×224 RGB | `output/processed/` |
| 4 | `stage_04_split.py` | Encodes labels, 70/15/15 stratified split | `splits/train/val/test.csv` |
| 5 | `stage_05_train.py` | Trains EfficientNet-B3, saves best model | `checkpoints/best.pth` |
| 6 | `stage_06_evaluate.py` | Classification report on unseen test set | `evaluation_report.txt` |


#### Step 9 — Download best.pth after training
Once training is complete, download the trained model to your local machine:

```bash
from google.colab import files
files.download('/content/drive/MyDrive/AgroVision/output/checkpoints/best.pth')
```

It saves to your browser's Downloads folder. Need this for local deployment.
