"""
Test pyRadPlan with one OpenKBP patient (pt_1).

Reads CSV data -> builds pyRadPlan CT + StructureSet -> computes Dose
Influence Matrix (DIM) -> aggregates to (V, 9) per-beam matrix -> saves dim.npy.

Run from the test_pyRadPlan folder with the venv active:
    python test_dim.py
"""

import numpy as np
import pandas as pd
import SimpleITK as sitk
import scipy.sparse as sp
from pathlib import Path

import pyRadPlan
from pyRadPlan import PhotonPlan, generate_stf, calc_dose_influence
from pyRadPlan.ct import CT
from pyRadPlan.cst import StructureSet, validate_voi

from visualize import (
    show_patient_overview,
    show_beam_geometry,
    show_per_beam_dose_maps,
    show_dose_distribution,
    show_dvh,
    show_beam_statistics,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PT_DIR = Path(__file__).parent / "pt_1"
GANTRY_ANGLES = [0.0, 40.0, 80.0, 120.0, 160.0, 200.0, 240.0, 280.0, 320.0]
NX, NY = 128, 128          # fixed in OpenKBP

# CSV filename -> (display name, pyRadPlan VOI type)
STRUCTURES = {
    "PTV70":        ("PTV_70",        "TARGET"),
    "PTV63":        ("PTV_63",        "TARGET"),
    "PTV56":        ("PTV_56",        "TARGET"),
    "Brainstem":    ("brainstem",     "OAR"),
    "SpinalCord":   ("spinal_cord",   "OAR"),
    "RightParotid": ("right_parotid", "OAR"),
    "LeftParotid":  ("left_parotid",  "OAR"),
    "Mandible":     ("mandible",      "OAR"),
}

print(f"pyRadPlan version: {pyRadPlan.__version__}")
print(f"Patient directory: {PT_DIR}\n")

# ---------------------------------------------------------------------------
# Step 1 - Read voxel spacing
# ---------------------------------------------------------------------------
spacing = np.loadtxt(PT_DIR / "voxel_dimensions.csv")
res = {"x": float(spacing[0]), "y": float(spacing[1]), "z": float(spacing[2])}
print(f"[1] Voxel spacing (mm):  x={res['x']}, y={res['y']}, z={res['z']}")

# ---------------------------------------------------------------------------
# Step 2 - Read CT CSV and determine volume shape
# ---------------------------------------------------------------------------
ct_df   = pd.read_csv(PT_DIR / "ct.csv", index_col=0)
mask_df = pd.read_csv(PT_DIR / "possible_dose_mask.csv", index_col=0)

ct_indices = ct_df.index.values.astype(int)
ct_values  = ct_df["data"].values.astype(np.float32)

# Determine NZ from the largest flat index across CT and dose mask
max_idx = max(int(ct_indices.max()), int(mask_df.index.max()))
NZ = max_idx // (NX * NY) + 1
print(f"[2] CT volume shape: ({NZ}, {NY}, {NX})  - {NZ*NY*NX:,} total voxels")

# Fill 3D HU array (air = -1000 HU everywhere else)
ct_array = np.full((NZ, NY, NX), -1000.0, dtype=np.float32)
ct_array.flat[ct_indices] = ct_values
print(f"    HU range: [{ct_array.min():.0f}, {ct_array.max():.0f}]")

# ---------------------------------------------------------------------------
# Step 3 - Build pyRadPlan CT object
# ---------------------------------------------------------------------------
ct_sitk = sitk.GetImageFromArray(ct_array)    # numpy (NZ,NY,NX) -> sitk (NX,NY,NZ)
ct_sitk.SetSpacing([res["x"], res["y"], res["z"]])
ct_sitk.SetOrigin([0.0, 0.0, 0.0])

ct_obj = CT.model_validate({"cube_hu": ct_sitk, "resolution": res})
print(f"[3] CT object created  (sitk size: {ct_sitk.GetSize()})")

# ---------------------------------------------------------------------------
# Step 4 - Build StructureSet
# ---------------------------------------------------------------------------
voi_list = []

# Body / external contour from possible_dose_mask
ext_indices = mask_df.index.values.astype(int)
ext_mask = np.zeros((NZ, NY, NX), dtype=np.uint8)
ext_mask.flat[ext_indices] = 1
ext_sitk = sitk.GetImageFromArray(ext_mask)
ext_sitk.CopyInformation(ct_sitk)
ext_voi = validate_voi(name="BODY", voi_type="EXTERNAL", mask=ext_sitk, ct_image=ct_obj)
voi_list.append(ext_voi)
print(f"[4] VOIs:")
print(f"    BODY (EXTERNAL)  - {ext_mask.sum():,} voxels")

for csv_name, (voi_name, voi_type) in STRUCTURES.items():
    csv_path = PT_DIR / f"{csv_name}.csv"
    if not csv_path.exists():
        print(f"    {voi_name}: skipped (file not found)")
        continue

    struct_df = pd.read_csv(csv_path, index_col=0)
    struct_indices = struct_df.index.values.astype(int)

    s_mask = np.zeros((NZ, NY, NX), dtype=np.uint8)
    valid = struct_indices[struct_indices < NZ * NY * NX]
    s_mask.flat[valid] = 1

    s_sitk = sitk.GetImageFromArray(s_mask)
    s_sitk.CopyInformation(ct_sitk)

    voi = validate_voi(name=voi_name, voi_type=voi_type, mask=s_sitk, ct_image=ct_obj)
    voi_list.append(voi)
    print(f"    {voi_name} ({voi_type})  - {s_mask.sum():,} voxels")

cst_obj = StructureSet(vois=voi_list, ct_image=ct_obj)
print(f"    -> StructureSet: {len(voi_list)} VOIs total")

# --- VIZ 1: CT slices + structure contours ---
masks_dict = {voi.name: np.array(sitk.GetArrayFromImage(voi.mask), dtype=np.uint8)
              for voi in voi_list}
print("\n[VIZ] Showing patient overview...")
show_patient_overview(ct_array, masks_dict)

# ---------------------------------------------------------------------------
# Step 5 - Create PhotonPlan
# ---------------------------------------------------------------------------
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
    prop_dose_calc={
        "engine": "SVDPB",
    },
)
print(f"\n[5] PhotonPlan: {len(GANTRY_ANGLES)} beams, engine=SVDPB, fractions=35")

# --- VIZ 2: beam geometry polar plot ---
print("\n[VIZ] Showing beam geometry...")
show_beam_geometry(GANTRY_ANGLES)

# ---------------------------------------------------------------------------
# Step 6 - Generate beam geometry (STF)
# ---------------------------------------------------------------------------
print("\n[6] Generating beam geometry (STF)...")
stf = generate_stf(ct_obj, cst_obj, pln)
print(f"    STF ready: {stf.num_of_beams} beams")

# ---------------------------------------------------------------------------
# Step 7 - Compute Dose Influence Matrix
# ---------------------------------------------------------------------------
print("\n[7] Computing Dose Influence Matrix (may take several minutes)...")
dij = calc_dose_influence(ct_obj, cst_obj, stf, pln)

n_bixels = dij.total_num_of_bixels
n_voxels = dij.num_of_voxels
print(f"    DIM shape: ({n_voxels:,} voxels  x  {n_bixels:,} bixels)")
print(f"    Beams: {dij.num_of_beams}")

# Extract dose grid shape here so VIZ 3 can use it
dose_dims_xyz = dij.dose_grid.dimensions          # (x, y, z) in SimpleITK order

# ---------------------------------------------------------------------------
# Step 8 - Aggregate bixels per beam -> (V, 9) dense matrix
# ---------------------------------------------------------------------------
print("\n[8] Aggregating bixels per beam -> per-beam DIM (V x 9)...")
sparse_mat = dij.physical_dose.flat[0]   # (V, total_bixels) sparse

beam_columns = []
for b in range(dij.num_of_beams):
    beam_mask = dij.beam_num == b
    col = np.asarray(sparse_mat[:, beam_mask].sum(axis=1)).ravel()
    beam_columns.append(col)
    print(f"    Beam {b:2d} (gantry {GANTRY_ANGLES[b]:5.1f} deg): "
          f"{beam_mask.sum()} bixels, "
          f"max dose {col.max():.4f} Gy/fraction")

dim_per_beam = np.column_stack(beam_columns).astype(np.float32)
print(f"\n    Per-beam DIM shape: {dim_per_beam.shape}")

# --- VIZ 3: per-beam dose maps ---
print("\n[VIZ] Showing per-beam dose maps...")
show_per_beam_dose_maps(dim_per_beam, dose_dims_xyz[::-1], GANTRY_ANGLES)

# --- VIZ 6: beam statistics bar charts ---
print("\n[VIZ] Showing beam statistics...")
show_beam_statistics(dim_per_beam, GANTRY_ANGLES)

# ---------------------------------------------------------------------------
# Step 9 - Sanity check: uniform beam weights
# ---------------------------------------------------------------------------
print("\n[9] Sanity check - uniform action vector (all ones):")
uniform_action = np.ones(len(GANTRY_ANGLES), dtype=np.float32)
dose_flat = dim_per_beam @ uniform_action      # (V,) on dose grid

# The dose grid may differ from the full CT grid; reshape using dij metadata
dose_dims_xyz = dij.dose_grid.dimensions          # (x, y, z) in SimpleITK order
dose_vol = dose_flat.reshape(dose_dims_xyz[::-1]) # (z, y, x) for numpy

print(f"    Dose grid shape (z,y,x): {dose_vol.shape}")
print(f"    Dose range: [{dose_vol.min():.4f}, {dose_vol.max():.4f}] Gy/fraction")
print(f"    Non-zero voxels: {(dose_vol > 0).sum():,} / {dose_vol.size:,}")

# Resample dose back to CT grid to check PTV_70 coverage
dose_sitk = sitk.GetImageFromArray(dose_vol.astype(np.float32))
dose_sitk.SetOrigin(dij.dose_grid.origin)
dose_sitk.SetSpacing(dij.dose_grid.resolution_vector)
dose_sitk.SetDirection(dij.dose_grid.direction.ravel())

resampler = sitk.ResampleImageFilter()
resampler.SetInterpolator(sitk.sitkLinear)
resampler.SetOutputOrigin(ct_sitk.GetOrigin())
resampler.SetOutputSpacing(ct_sitk.GetSpacing())
resampler.SetOutputDirection(ct_sitk.GetDirection())
resampler.SetSize(ct_sitk.GetSize())
dose_on_ct = resampler.Execute(dose_sitk)
dose_on_ct_np = sitk.GetArrayFromImage(dose_on_ct)  # (NZ, NY, NX)

ptv70_df  = pd.read_csv(PT_DIR / "PTV70.csv", index_col=0)
ptv70_idx = ptv70_df.index.values.astype(int)
ptv70_idx = ptv70_idx[ptv70_idx < NZ * NY * NX]
ptv70_dose = dose_on_ct_np.flat[ptv70_idx]
print(f"    Mean dose in PTV_70: {ptv70_dose.mean():.4f} Gy/fraction")
print(f"    (Expected ~{70.0/35:.2f} Gy/fraction for a perfectly scaled plan)")

# --- VIZ 4: dose distribution overlaid on CT ---
print("\n[VIZ] Showing dose distribution...")
show_dose_distribution(ct_array, dose_on_ct_np, masks_dict)

# --- VIZ 5: DVH ---
print("\n[VIZ] Showing DVH...")
show_dvh(dose_on_ct_np, masks_dict, n_fractions=35)

# ---------------------------------------------------------------------------
# Step 10 - Save
# ---------------------------------------------------------------------------
out_path = PT_DIR / "dim.npy"
np.save(out_path, dim_per_beam)
print(f"\n[10] Saved per-beam DIM to: {out_path}")
print(f"     Shape: {dim_per_beam.shape}, dtype: {dim_per_beam.dtype}")
print(f"     File size: {out_path.stat().st_size / 1e6:.1f} MB")

print("\nAll done!")
