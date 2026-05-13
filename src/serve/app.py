"""
AgroVision FastAPI Backend
Loads output/checkpoints/best.pth produced by stage_05_train.py
"""

import io, json, time, logging, sys
from pathlib import Path

import cv2
import numpy as np
import torch
import timm
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from torch.cuda.amp import autocast
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ── make src/ importable so stage_05 transforms are reusable ──────────────────
ROOT = Path(__file__).resolve().parents[2]   # agrovision/
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AgroVision API",
    description="Crop Pest & Disease Classification",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files and UI
TEMPLATES = ROOT / "templates"
app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")

# ── Globals ───────────────────────────────────────────────────────────────────
CKPT_PATH = ROOT / "output" / "checkpoints" / "best.pth"
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

model       = None
class_map   = None
num_classes = None
device      = None

val_tfm = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2()
])

# ── Mock classes for demo mode (when best.pth not found) ─────────────────────
MOCK_CLASSES = {
    "0": "Apple__Apple_Scab",
    "1": "Apple__Healthy",
    "2": "Corn__Common_Rust",
    "3": "Corn__Healthy",
    "4": "Potato__Early_Blight",
    "5": "Potato__Healthy",
    "6": "Tomato__Bacterial_Spot",
    "7": "Tomato__Early_Blight",
    "8": "Tomato__Healthy",
    "9": "Tomato__Late_Blight",
}

import gdown

MODEL_URL = "https://drive.google.com/uc?id=1GH6htk09rs1N5zIcpEwLiPoSMx6DnkuC"

CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)

try:
    if not CKPT_PATH.exists():
        print("Downloading model...")
        gdown.download(MODEL_URL, str(CKPT_PATH), quiet=False)
        print("Download completed")
    else:
        print("Checkpoint already exists")
except Exception as e:
    print("Model download failed:", e)

# ── Model loader ──────────────────────────────────────────────────────────────
def load_model():
    global model, class_map, num_classes, device

    if not CKPT_PATH.exists():
        log.warning(f"No checkpoint found at {CKPT_PATH} — running in demo mode")
        return False

    device = (
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    log.info(f"Loading model on {device} ...")

    bundle      = torch.load(CKPT_PATH, map_location=device)
    class_map   = bundle["class_map"]
    num_classes = bundle["num_classes"]

    model = timm.create_model(
        bundle["model_name"],
        pretrained=False,
        num_classes=num_classes
    )
    model.load_state_dict(bundle["model_state"])
    model = model.to(device).eval()

    log.info(f"Model ready — {num_classes} classes on {device}")
    return True

# ── Predict helper ────────────────────────────────────────────────────────────
def run_inference(img_array: np.ndarray, top_k: int = 5) -> dict:
    start = time.time()

    if model is None:
        # Demo mode — return plausible random scores
        np.random.seed(int(img_array.mean()) % 100)
        scores = np.random.dirichlet(np.ones(len(MOCK_CLASSES)) * 0.4)
        preds  = sorted(
            [{"class_id": int(k), "class_name": v,
              "confidence": round(float(scores[int(k)]), 4)}
             for k, v in MOCK_CLASSES.items()],
            key=lambda x: x["confidence"], reverse=True
        )
        return {
            "predictions":  preds[:top_k],
            "inference_ms": round((time.time() - start) * 1000, 1),
            "mode":         "demo"
        }

    tensor = val_tfm(image=img_array)["image"].unsqueeze(0).to(device)

    with torch.no_grad(), autocast():
        probs = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()

    top_idx = probs.argsort()[::-1][:top_k]
    preds   = [
        {"class_id":   int(i),
         "class_name": class_map[str(i)],
         "confidence": round(float(probs[i]), 4)}
        for i in top_idx
    ]

    return {
        "predictions":  preds,
        "inference_ms": round((time.time() - start) * 1000, 1),
        "mode":         "model"
    }

# ── Routes ────────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    load_model()

@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse(content=(TEMPLATES / "index.html").read_text())

@app.get("/health")
async def health():
    return {
        "status":       "ok",
        "model_loaded": model is not None,
        "device":       str(device) if device else "demo",
        "num_classes":  num_classes or len(MOCK_CLASSES),
        "checkpoint":   str(CKPT_PATH)
    }

@app.get("/classes")
async def get_classes():
    cm = class_map or MOCK_CLASSES
    return {"classes": cm, "total": len(cm)}

@app.post("/predict")
async def predict(file: UploadFile = File(...), top_k: int = 5):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image (jpg/png)")

    contents = await file.read()
    try:
        img = np.array(Image.open(io.BytesIO(contents)).convert("RGB"))
    except Exception:
        raise HTTPException(400, "Could not read image file")

    try:
        result           = run_inference(img, top_k=min(top_k, 10))
        result["filename"] = file.filename
        return result
    except Exception as e:
        log.error(f"Inference error: {e}")
        raise HTTPException(500, f"Prediction failed: {e}")

@app.post("/predict/batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    if len(files) > 10:
        raise HTTPException(400, "Max 10 images per batch")

    results = []
    for f in files:
        contents = await f.read()
        img      = np.array(Image.open(io.BytesIO(contents)).convert("RGB"))
        r        = run_inference(img, top_k=3)
        r["filename"] = f.filename
        results.append(r)

    return {"batch_results": results, "count": len(results)}