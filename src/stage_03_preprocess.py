import cv2, pandas as pd, yaml, logging
from tqdm import tqdm
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

def run(cfg):
    df      = pd.read_csv(cfg["data"]["manifest_clean"])
    size    = cfg["preprocessing"]["image_size"]
    pro_dir = Path(cfg["data"]["processed_dir"])
    skipped, failed = 0, 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing"):
        dst = pro_dir / row.label / Path(row.path).name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            skipped += 1
            continue
        img = cv2.imread(row.path)
        if img is None:
            failed += 1
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(dst), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    log.info(f"Done | Skipped: {skipped} | Failed: {failed}")

    # rebuild manifest with new paths
    records = []
    for ld in pro_dir.iterdir():
        if ld.is_dir():
            for ip in ld.iterdir():
                records.append({"path": str(ip), "label": ld.name})

    out = Path(cfg["data"]["manifest_processed"])
    pd.DataFrame(records).to_csv(out, index=False)
    log.info(f"Processed manifest saved → {out}")

if __name__ == "__main__":
    cfg = yaml.safe_load(open("config.yaml"))
    run(cfg)