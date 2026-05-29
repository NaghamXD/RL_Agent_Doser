"""
scripts/preprocess_all.py

Run once before any training to preprocess all OpenKBP patients.

What it does per patient:
  - Reads raw CSV files via PatientLoader
  - Resamples CT-space arrays to (64, 64, 64) via SimpleITK
  - Saves ct.npy, masks.npy, possible_dose_mask.npy, dose_gt.npy,
    voxel_dims.npy, present_structures.json

Usage:
    python scripts/preprocess_all.py \
        --data-dir data/raw \
        --out-dir  data/processed \
        --splits   train validation test

    # Process only training patients:
    python scripts/preprocess_all.py --splits train

    # Process a single patient for testing:
    python scripts/preprocess_all.py --patient pt_1
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Allow imports from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import PatientLoader
from src.data.preprocessor import Preprocessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Folder names in data/raw/ for each split
SPLIT_DIRS: dict[str, str] = {
    "train":      "train-pats",
    "validation": "validation-pats",
    "test":       "test-pats",
}


def get_patient_dirs(data_dir: Path, splits: list[str]) -> list[Path]:
    """Return sorted list of patient directories for the requested splits."""
    dirs: list[Path] = []
    for split in splits:
        split_dir = data_dir / SPLIT_DIRS[split]
        if not split_dir.exists():
            log.warning(f"Split directory not found: {split_dir} — skipping")
            continue
        patient_dirs = sorted(split_dir.iterdir())
        log.info(f"  {split}: {len(patient_dirs)} patients in {split_dir}")
        dirs.extend(patient_dirs)
    return dirs


def process_patient(
    patient_dir: Path,
    preprocessor: Preprocessor,
    overwrite: bool,
) -> bool:
    """
    Load and preprocess one patient. Returns True on success, False on error.
    """
    patient_id = patient_dir.name
    out_dir    = preprocessor.out_root / patient_id

    if not overwrite and (out_dir / "ct.npy").exists():
        log.debug(f"[{patient_id}] already processed — skipping")
        return True

    try:
        loader = PatientLoader(patient_dir)
        data   = loader.load()
        preprocessor.process(data, patient_id)
        return True

    except Exception as exc:
        log.error(f"[{patient_id}] FAILED: {exc}", exc_info=True)
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess OpenKBP patients to 64³ numpy arrays."
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/raw"),
        help="Root of raw patient data (default: data/raw)",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("data/processed"),
        help="Output root for processed data (default: data/processed)",
    )
    parser.add_argument(
        "--splits", nargs="+",
        choices=["train", "validation", "test"],
        default=["train", "validation", "test"],
        help="Which splits to process (default: all three)",
    )
    parser.add_argument(
        "--patient", type=str, default=None,
        help="Process a single patient ID, e.g. pt_1 (overrides --splits)",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-process patients even if output already exists",
    )
    args = parser.parse_args()

    preprocessor = Preprocessor(out_root=args.out_dir)

    # ── Single patient mode ────────────────────────────────────────────
    if args.patient is not None:
        # Search all splits for the requested patient
        found = False
        for split_dir_name in SPLIT_DIRS.values():
            patient_dir = args.data_dir / split_dir_name / args.patient
            if patient_dir.exists():
                log.info(f"Processing single patient: {patient_dir}")
                success = process_patient(patient_dir, preprocessor,
                                          overwrite=True)
                sys.exit(0 if success else 1)
                found = True
                break
        if not found:
            log.error(f"Patient '{args.patient}' not found in any split.")
            sys.exit(1)

    # ── Batch mode ────────────────────────────────────────────────────
    log.info(f"Data root:   {args.data_dir}")
    log.info(f"Output root: {args.out_dir}")
    log.info(f"Splits:      {args.splits}")

    patient_dirs = get_patient_dirs(args.data_dir, args.splits)
    total        = len(patient_dirs)
    log.info(f"Total patients to process: {total}")

    n_ok   = 0
    n_fail = 0
    t0     = time.time()

    for i, patient_dir in enumerate(patient_dirs, start=1):
        t_pat = time.time()
        success = process_patient(patient_dir, preprocessor, args.overwrite)

        elapsed_pat  = time.time() - t_pat
        elapsed_total = time.time() - t0
        eta          = (elapsed_total / i) * (total - i)

        status = "✓" if success else "✗"
        if success:
            n_ok += 1
        else:
            n_fail += 1

        log.info(
            f"[{i:3d}/{total}] {status} {patient_dir.name}  "
            f"({elapsed_pat:.1f}s)  ETA {eta/60:.1f} min"
        )

    # ── Summary ───────────────────────────────────────────────────────
    total_time = time.time() - t0
    log.info("─" * 60)
    log.info(f"Done.  {n_ok} succeeded,  {n_fail} failed  "
             f"({total_time/60:.1f} min total)")

    if n_fail > 0:
        log.warning(f"{n_fail} patients failed — check logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
