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