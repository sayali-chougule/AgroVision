import os, pandas as pd, yaml, logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

def run(cfg):
    raw_dir = Path(cfg["data"]["raw_dir"])
    exts    = set([e.lower() for e in cfg["validation"]["valid_extensions"]])
    records = []

    log.info(f"Scanning dataset at: {raw_dir}")

    for crop in sorted(os.listdir(raw_dir)):
        crop_path = raw_dir / crop
        if not crop_path.is_dir():
            continue

        for disease in os.listdir(crop_path):
            disease_path = crop_path / disease
            if not disease_path.is_dir():
                continue

            for f in os.listdir(disease_path):
                if Path(f).suffix.lower() in exts:
                    records.append({
                        "path": str(disease_path / f),
                        "label": disease,   # disease class
                        "crop": crop        # optional (extra info)
                    })

    df = pd.DataFrame(records)

    if df.empty:
        raise ValueError("No images found — check dataset path or structure")

    out = Path(cfg["data"]["manifest"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    log.info(f"Total images : {len(df)}")
    log.info(f"Total classes: {df['label'].nunique()}")
    log.info(f"Manifest saved → {out}")

    return df


if __name__ == "__main__":
    cfg = yaml.safe_load(open("config.yaml"))
    run(cfg)