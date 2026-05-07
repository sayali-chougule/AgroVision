import torch, timm, json, yaml, logging, pandas as pd, cv2
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from pathlib import Path
from stage_05_train import AgroDataset, get_transforms

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

def run(cfg):
    ckpt_path = Path(cfg["data"]["checkpoints_dir"]) / "best.pth"
    bundle    = torch.load(ckpt_path, map_location="cpu")
    class_map = bundle["class_map"]
    num_cls   = bundle["num_classes"]
    names     = [class_map[str(i)] for i in range(num_cls)]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = timm.create_model(bundle["model_name"], pretrained=False,
                               num_classes=num_cls)
    model.load_state_dict(bundle["model_state"])
    model  = model.to(device).eval()

    _, val_tfm = get_transforms()
    test_ds    = AgroDataset(Path(cfg["data"]["splits_dir"])/"test.csv", val_tfm)
    loader     = DataLoader(test_ds, batch_size=64, num_workers=4)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            with autocast(): out = model(imgs.to(device))
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())

    report = classification_report(all_labels, all_preds,
                                   target_names=names, digits=3)
    log.info(f"\n{report}")

    out_path = Path(cfg["data"]["output_dir"]) / "evaluation_report.txt"
    out_path.write_text(report)
    log.info(f"Report saved → {out_path}")

if __name__ == "__main__":
    cfg = yaml.safe_load(open("config.yaml"))
    run(cfg)