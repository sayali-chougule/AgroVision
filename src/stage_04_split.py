import pandas as pd, yaml, json, logging
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

def run(cfg):
    df   = pd.read_csv(cfg["data"]["manifest_processed"])
    seed = cfg["split"]["random_seed"]
    test_ratio = cfg["split"]["test_ratio"] + cfg["split"]["val_ratio"]

    le = LabelEncoder()
    df["label_id"] = le.fit_transform(df["label"])
    class_map = {int(i): c for i, c in enumerate(le.classes_)}

    with open(cfg["data"]["class_map"], "w") as f:
        json.dump(class_map, f, indent=2)
    log.info(f"Classes: {len(class_map)}")

    train, tmp = train_test_split(df, test_size=test_ratio,
                                  stratify=df.label_id, random_state=seed)
    val, test  = train_test_split(tmp, test_size=0.5,
                                  stratify=tmp.label_id, random_state=seed)

    sp = Path(cfg["data"]["splits_dir"])
    sp.mkdir(parents=True, exist_ok=True)
    train.to_csv(sp / "train.csv", index=False)
    val.to_csv(sp   / "val.csv",   index=False)
    test.to_csv(sp  / "test.csv",  index=False)

    log.info(f"Train {len(train)} | Val {len(val)} | Test {len(test)}")

if __name__ == "__main__":
    cfg = yaml.safe_load(open("config.yaml"))
    run(cfg)