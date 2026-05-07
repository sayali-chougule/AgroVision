import torch, timm, json, yaml, logging, pandas as pd, cv2
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

MEAN, STD = [0.485,0.456,0.406], [0.229,0.224,0.225]

class AgroDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.transform = transform
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.cvtColor(cv2.imread(row.path), cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, torch.tensor(row.label_id, dtype=torch.long)

def get_transforms():
    train = A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.7),
        A.GaussNoise(var_limit=(5.0, 25.0), p=0.3),
        A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.3),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2()
    ])
    val = A.Compose([A.Normalize(mean=MEAN, std=STD), ToTensorV2()])
    return train, val

def run(cfg):
    tc   = cfg["training"]
    sp   = Path(cfg["data"]["splits_dir"])
    ckpt = Path(cfg["data"]["checkpoints_dir"])
    ckpt.mkdir(parents=True, exist_ok=True)

    with open(cfg["data"]["class_map"]) as f:
        class_map = json.load(f)
    num_classes = len(class_map)

    # device
    if tc["device"] == "auto":
        device = "cuda" if torch.cuda.is_available() else \
                 "mps"  if torch.backends.mps.is_available() else "cpu"
    else:
        device = tc["device"]
    log.info(f"Device: {device} | Classes: {num_classes}")

    train_tfm, val_tfm = get_transforms()
    train_loader = DataLoader(AgroDataset(sp/"train.csv", train_tfm),
        batch_size=tc["batch_size"], shuffle=True,
        num_workers=tc["num_workers"], pin_memory=True)
    val_loader = DataLoader(AgroDataset(sp/"val.csv", val_tfm),
        batch_size=tc["batch_size"]*2, shuffle=False,
        num_workers=tc["num_workers"], pin_memory=True)

    model = timm.create_model(tc["model_name"], pretrained=True,
                              num_classes=num_classes).to(device)
    optimizer = AdamW(model.parameters(), lr=tc["learning_rate"],
                      weight_decay=tc["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=tc["epochs"], eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(label_smoothing=tc["label_smoothing"])
    scaler    = GradScaler()
    best_acc  = 0.0

    for epoch in range(tc["epochs"]):
        # --- train ---
        model.train()
        correct, total, loss_sum = 0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast():
                out  = model(imgs)
                loss = criterion(out, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
            loss_sum += loss.item()
            correct  += (out.argmax(1) == labels).sum().item()
            total    += labels.size(0)
        scheduler.step()

        # --- validate ---
        model.eval(); vc, vt = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                with autocast(): out = model(imgs)
                vc += (out.argmax(1) == labels).sum().item()
                vt += labels.size(0)

        val_acc = vc / vt
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model_state": model.state_dict(),
                        "class_map": class_map,
                        "num_classes": num_classes,
                        "model_name": tc["model_name"]},
                       ckpt / "best.pth")

        log.info(f"Epoch {epoch+1:02d}/{tc['epochs']} | "
                 f"loss {loss_sum/len(train_loader):.4f} | "
                 f"train {correct/total:.3f} | "
                 f"val {val_acc:.3f} | best {best_acc:.3f}")

    log.info(f"Training complete. Best val accuracy: {best_acc:.3f}")

if __name__ == "__main__":
    cfg = yaml.safe_load(open("config.yaml"))
    run(cfg)