import yaml, sys, logging, time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/pipeline.log")
    ]
)
log = logging.getLogger(__name__)

sys.path.insert(0, "src")

from stage_01_ingest     import run as ingest
from stage_02_validate   import run as validate
from stage_03_preprocess import run as preprocess
from stage_04_split      import run as split
from stage_05_train      import run as train
from stage_06_evaluate   import run as evaluate

STAGES = [
    ("ingest",     ingest),
    ("validate",   validate),
    ("preprocess", preprocess),
    ("split",      split),
    ("train",      train),
    ("evaluate",   evaluate),
]

def main():
    Path("logs").mkdir(exist_ok=True)
    cfg        = yaml.safe_load(open("config.yaml"))
    resume     = sys.argv[1] if len(sys.argv) > 1 else None  # e.g. python pipeline.py train
    skip       = resume is not None
    total_start = time.time()

    log.info("=" * 60)
    log.info("   AgroVision MLOps Pipeline Starting")
    log.info("=" * 60)

    for name, stage_fn in STAGES:
        # resume from a specific stage
        if skip:
            if name == resume:
                skip = False
            else:
                log.info(f"Skipping stage: {name}")
                continue

        # checkpoint: skip if already done
        ckpt = Path(f"logs/.done_{name}")
        if ckpt.exists():
            log.info(f"Stage [{name}] already complete — skipping")
            continue

        log.info(f"\n{'='*60}")
        log.info(f"  STAGE: {name.upper()}")
        log.info(f"{'='*60}")

        start = time.time()
        try:
            stage_fn(cfg)
            ckpt.touch()
            elapsed = time.time() - start
            log.info(f"Stage [{name}] done in {elapsed:.1f}s")
        except Exception as e:
            log.error(f"Stage [{name}] FAILED: {e}")
            log.error("Fix the error and re-run — completed stages will be skipped")
            sys.exit(1)

    total = time.time() - total_start
    log.info(f"\n{'='*60}")
    log.info(f"  PIPELINE COMPLETE in {total/60:.1f} min")
    log.info(f"{'='*60}")

if __name__ == "__main__":
    main()