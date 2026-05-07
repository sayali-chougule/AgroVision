import pandas as pd, yaml, logging, imagehash
from PIL import Image
from tqdm import tqdm
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

def run(cfg):
    df      = pd.read_csv(cfg["data"]["manifest"])
    min_sz  = cfg["validation"]["min_image_size"]
    seen    = {}

    def check(row):
        try:
            img = Image.open(row.path).convert("RGB")
            if min(img.size) < min_sz:
                return "low_res"
            h = str(imagehash.phash(img, hash_size=8))
            if h in seen:
                return "duplicate"
            seen[h] = row.path
            return "ok"
        except:
            return "corrupt"

    tqdm.pandas(desc="Validating")
    df["status"] = df.progress_apply(check, axis=1)

    log.info(f"Validation results:\n{df.status.value_counts().to_string()}")
    clean = df[df.status == "ok"].drop(columns="status")
    out   = Path(cfg["data"]["manifest_clean"])
    clean.to_csv(out, index=False)
    log.info(f"Clean images: {len(clean)} / {len(df)} → {out}")
    return clean

if __name__ == "__main__":
    cfg = yaml.safe_load(open("config.yaml"))
    run(cfg)