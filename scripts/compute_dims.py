"""
scripts/compute_dims.py

Compute and save the Dose Influence Matrix (DIM) for each patient
using pyRadPlan's physics engine.

What it does per patient:
  - Loads raw CT + masks from data/raw/ (not processed — pyRadPlan needs
    the native resolution CT, not the 64³ resampled version)
  - Runs the full pyRadPlan pipeline (CT → STF → DIM)
  - Aggregates beamlets per beam: (V_dose, n_bixels) → (V_dose, 9)
  - Saves dim.npy and dose_grid_meta.json to data/processed/patient_id/

Expected runtime: 5–10 min per patient on CPU.
Run overnight for all 200 training patients.

Usage:
    # Single patient (test first):
    python scripts/compute_dims.py --patient pt_1

    # All training patients:
    python scripts/compute_dims.py --split train

    # Training + validation:
    python scripts/compute_dims.py --split train validation

    # Skip already-computed patients:
    python scripts/compute_dims.py --split train  # skips if dim.npy exists
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# pyRadPlan imports — confirmed working API (pyRadPlan 0.4.0)
# ---------------------------------------------------------------------------
try:
    from pyRadPlan import PhotonPlan, generate_stf, calc_dose_influence
    from pyRadPlan.ct import CT
    from pyRadPlan.cst import StructureSet, validate_voi
    PYRADPLAN_AVAILABLE = True
except ImportError as e:
    log.error(f"pyRadPlan not available: {e}")
    PYRADPLAN_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants — must match CLAUDE.md
# ---------------------------------------------------------------------------
NX, NY = 128, 128
GANTRY_ANGLES = [0.0, 40.0, 80.0, 120.0, 160.0, 200.0, 240.0, 280.0, 320.0]

STRUCTURE_CSV_MAP = {
    "PTV70":        ("PTV_70",        "TARGET"),
    "PTV63":        ("PTV_63",        "TARGET"),
    "PTV56":        ("PTV_56",        "TARGET"),
    "Brainstem":    ("brainstem",     "OAR"),
    "SpinalCord":   ("spinal_cord",   "OAR"),
    "RightParotid": ("right_parotid", "OAR"),
    "LeftParotid":  ("left_parotid",  "OAR"),
    "Larynx":       ("larynx",        "OAR"),
    "Esophagus":    ("esophagus",     "OAR"),
    "Mandible":     ("mandible",      "OAR"),
}

SPLIT_DIRS = {
    "train":      "train-pats",
    "validation": "validation-pats",
    "test":       "test-pats",
}


# ---------------------------------------------------------------------------
# Core DIM computation
# ---------------------------------------------------------------------------

def compute_dim_for_patient(raw_dir: Path, out_dir: Path) -> bool:
    """
    Run full pyRadPlan pipeline for one patient and save DIM + metadata.

    Args:
        raw_dir: path to raw patient folder, e.g. data/raw/train-pats/pt_1
        out_dir: path to processed patient folder, e.g. data/processed/pt_1

    Returns:
        True on success, False on error.
    """
    patient_id = raw_dir.name

    # ── Load voxel spacing ────────────────────────────────────────────
    voxel_dims = np.loadtxt(raw_dir / "voxel_dimensions.csv", dtype=np.float32)
    res = {"x": float(voxel_dims[0]),
           "y": float(voxel_dims[1]),
           "z": float(voxel_dims[2])}
    log.info(f"[{patient_id}] voxel spacing: {res}")

    # ── Load CT at native resolution ──────────────────────────────────
    ct_df      = pd.read_csv(raw_dir / "ct.csv", index_col=0)
    mask_df    = pd.read_csv(raw_dir / "possible_dose_mask.csv", index_col=0)

    ct_indices = ct_df.index.values.astype(np.int64)
    ct_values  = ct_df["data"].values.astype(np.float32)

    max_idx = max(int(ct_indices.max()), int(mask_df.index.max()))
    NZ      = max_idx // (NX * NY) + 1
    log.info(f"[{patient_id}] CT shape: ({NZ}, {NY}, {NX})")

    ct_array = np.full((NZ, NY, NX), -1000.0, dtype=np.float32)
    ct_array.flat[ct_indices] = ct_values

    # ── Build pyRadPlan CT object ──────────────────────────────────────
    ct_sitk = sitk.GetImageFromArray(ct_array)
    ct_sitk.SetSpacing([res["x"], res["y"], res["z"]])
    ct_sitk.SetOrigin([0.0, 0.0, 0.0])
    ct_obj  = CT.model_validate({"cube_hu": ct_sitk, "resolution": res})
    log.info(f"[{patient_id}] CT object created")

    # ── Build StructureSet ─────────────────────────────────────────────
    voi_list = []

    # BODY from possible_dose_mask
    ext_mask = np.zeros((NZ, NY, NX), dtype=np.uint8)
    ext_idx  = mask_df.index.values.astype(np.int64)
    ext_mask.flat[ext_idx[ext_idx < NZ * NY * NX]] = 1
    ext_sitk = sitk.GetImageFromArray(ext_mask)
    ext_sitk.CopyInformation(ct_sitk)
    voi_list.append(validate_voi(
        name="BODY", voi_type="EXTERNAL",
        mask=ext_sitk, ct_image=ct_obj
    ))

    # Structures
    for csv_name, (voi_name, voi_type) in STRUCTURE_CSV_MAP.items():
        csv_path = raw_dir / f"{csv_name}.csv"
        if not csv_path.exists():
            continue
        df      = pd.read_csv(csv_path, index_col=0)
        idx     = df.index.values.astype(np.int64)
        s_mask  = np.zeros((NZ, NY, NX), dtype=np.uint8)
        valid   = idx[idx < NZ * NY * NX]
        s_mask.flat[valid] = 1
        s_sitk  = sitk.GetImageFromArray(s_mask)
        s_sitk.CopyInformation(ct_sitk)
        voi_list.append(validate_voi(
            name=voi_name, voi_type=voi_type,
            mask=s_sitk, ct_image=ct_obj
        ))

    cst_obj = StructureSet(vois=voi_list, ct_image=ct_obj)
    log.info(f"[{patient_id}] StructureSet: {len(voi_list)} VOIs")

    # ── Create PhotonPlan ──────────────────────────────────────────────
    pln = PhotonPlan(
        num_of_fractions=35,
        prescribed_dose=70.0,
        machine="Generic",
        prop_stf={
            "generator":     "photonIMRT",
            "gantry_angles": GANTRY_ANGLES,
            "couch_angles":  [0.0] * len(GANTRY_ANGLES),
            "bixel_width":   5.0,
        },
        prop_dose_calc={"engine": "SVDPB"},
    )
    log.info(f"[{patient_id}] PhotonPlan: 9 beams, SVDPB engine")

    # ── Generate STF ───────────────────────────────────────────────────
    log.info(f"[{patient_id}] Generating STF...")
    stf = generate_stf(ct_obj, cst_obj, pln)
    log.info(f"[{patient_id}] STF: {stf.num_of_beams} beams")

    # ── Compute DIM (slow step) ────────────────────────────────────────
    log.info(f"[{patient_id}] Computing DIM (this takes several minutes)...")
    t0  = time.time()
    dij = calc_dose_influence(ct_obj, cst_obj, stf, pln)
    log.info(f"[{patient_id}] DIM computed in {time.time()-t0:.1f}s "
             f"({dij.num_of_voxels:,} voxels × {dij.total_num_of_bixels:,} bixels)")

    # ── Aggregate bixels per beam → (V_dose, 9) ───────────────────────
    sparse_mat   = dij.physical_dose.flat[0]
    beam_columns = []
    for b in range(dij.num_of_beams):
        beam_mask = (dij.beam_num == b)
        col       = np.asarray(sparse_mat[:, beam_mask].sum(axis=1)).ravel()
        beam_columns.append(col)
        log.info(f"[{patient_id}]   beam {b} ({GANTRY_ANGLES[b]}°): "
                 f"{beam_mask.sum()} bixels, max={col.max():.4f} Gy/fraction")

    dim_per_beam = np.column_stack(beam_columns).astype(np.float32)
    log.info(f"[{patient_id}] Per-beam DIM shape: {dim_per_beam.shape}")

    # ── Sanity check: mean PTV_70 dose with uniform weights ───────────
    uniform_dose = dim_per_beam @ np.ones(9, dtype=np.float32)
    dose_dims    = dij.dose_grid.dimensions      # (dx, dy, dz) sitk order
    dose_vol     = uniform_dose.reshape(
        dose_dims[2], dose_dims[1], dose_dims[0]  # (dz, dy, dx) numpy
    )

    # Resample dose back to CT grid for PTV check
    dose_sitk = sitk.GetImageFromArray(dose_vol)
    dose_sitk.SetOrigin(dij.dose_grid.origin)
    dose_sitk.SetSpacing(dij.dose_grid.resolution_vector)
    dose_sitk.SetDirection(dij.dose_grid.direction.ravel())
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputOrigin(ct_sitk.GetOrigin())
    resampler.SetOutputSpacing(ct_sitk.GetSpacing())
    resampler.SetOutputDirection(ct_sitk.GetDirection())
    resampler.SetSize(ct_sitk.GetSize())
    dose_on_ct = sitk.GetArrayFromImage(resampler.Execute(dose_sitk))

    ptv70_path = raw_dir / "PTV70.csv"
    if ptv70_path.exists():
        ptv70_df  = pd.read_csv(ptv70_path, index_col=0)
        ptv70_idx = ptv70_df.index.values.astype(np.int64)
        ptv70_idx = ptv70_idx[ptv70_idx < NZ * NY * NX]
        ptv70_dose = dose_on_ct.flat[ptv70_idx].mean()
        log.info(f"[{patient_id}] Sanity check — mean PTV_70 dose (uniform weights): "
                 f"{ptv70_dose:.4f} Gy/fraction  (expected ~{70/35:.2f})")

    # ── Save DIM and dose grid metadata ───────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    dim_path = out_dir / "dim.npy"
    np.save(dim_path, dim_per_beam)
    log.info(f"[{patient_id}] Saved dim.npy: shape={dim_per_beam.shape}, "
             f"size={dim_path.stat().st_size/1e6:.1f} MB")

    meta = {
        "origin":     list(dij.dose_grid.origin),
        "spacing":    list(dij.dose_grid.resolution_vector),
        "direction":  list(dij.dose_grid.direction.ravel()),
        "dimensions": list(dij.dose_grid.dimensions),   # (dx, dy, dz) sitk order
    }
    with open(out_dir / "dose_grid_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    log.info(f"[{patient_id}] Saved dose_grid_meta.json")

    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_patient_dirs(data_dir: Path, splits: list[str]) -> list[tuple[Path, Path]]:
    """Return list of (raw_dir, processed_dir) tuples."""
    pairs = []
    for split in splits:
        split_dir = data_dir / "raw" / SPLIT_DIRS[split]
        if not split_dir.exists():
            log.warning(f"Split directory not found: {split_dir}")
            continue
        for patient_dir in sorted(split_dir.iterdir()):
            processed_dir = data_dir / "processed" / patient_dir.name
            pairs.append((patient_dir, processed_dir))
        log.info(f"  {split}: {len(list(split_dir.iterdir()))} patients")
    return pairs


def main():
    parser = argparse.ArgumentParser(
        description="Compute Dose Influence Matrices using pyRadPlan."
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data"),
        help="Root data directory containing raw/ and processed/ (default: data)",
    )
    parser.add_argument(
        "--split", nargs="+",
        choices=["train", "validation", "test"],
        default=["train"],
        help="Which splits to process (default: train only)",
    )
    parser.add_argument(
        "--patient", type=str, default=None,
        help="Process a single patient ID, e.g. pt_1 (overrides --split)",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Recompute even if dim.npy already exists",
    )
    args = parser.parse_args()

    if not PYRADPLAN_AVAILABLE:
        log.error("pyRadPlan is not installed. Run: pip install pyRadPlan==0.4.0 --no-deps")
        sys.exit(1)

    # ── Single patient mode ───────────────────────────────────────────
    if args.patient:
        for split_name in SPLIT_DIRS.values():
            raw_dir = args.data_dir / "raw" / split_name / args.patient
            if raw_dir.exists():
                out_dir = args.data_dir / "processed" / args.patient
                log.info(f"Processing single patient: {raw_dir}")
                if not args.overwrite and (out_dir / "dim.npy").exists():
                    log.info(f"dim.npy already exists — skipping (use --overwrite to recompute)")
                    sys.exit(0)
                success = compute_dim_for_patient(raw_dir, out_dir)
                sys.exit(0 if success else 1)
        log.error(f"Patient '{args.patient}' not found in any split.")
        sys.exit(1)

    # ── Batch mode ────────────────────────────────────────────────────
    pairs = get_patient_dirs(args.data_dir, args.split)
    total = len(pairs)
    log.info(f"Total patients: {total}")

    n_ok = n_skip = n_fail = 0
    t_start = time.time()

    for i, (raw_dir, out_dir) in enumerate(pairs, start=1):
        patient_id = raw_dir.name

        if not args.overwrite and (out_dir / "dim.npy").exists():
            log.debug(f"[{i}/{total}] {patient_id}: already done — skipping")
            n_skip += 1
            continue

        t_pat = time.time()
        log.info(f"[{i}/{total}] Starting {patient_id}...")
        try:
            success = compute_dim_for_patient(raw_dir, out_dir)
        except Exception as e:
            log.error(f"[{patient_id}] FAILED: {e}", exc_info=True)
            success = False

        elapsed_pat   = time.time() - t_pat
        elapsed_total = time.time() - t_start
        done          = n_ok + n_fail + 1
        eta           = (elapsed_total / done) * (total - n_skip - done)

        if success:
            n_ok += 1
            log.info(f"[{i}/{total}] ✓ {patient_id} ({elapsed_pat/60:.1f} min)  "
                     f"ETA {eta/3600:.1f} hr")
        else:
            n_fail += 1
            log.error(f"[{i}/{total}] ✗ {patient_id} FAILED")

    total_time = time.time() - t_start
    log.info("─" * 60)
    log.info(f"Done.  {n_ok} computed,  {n_skip} skipped,  {n_fail} failed  "
             f"({total_time/3600:.1f} hr total)")
    if n_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
