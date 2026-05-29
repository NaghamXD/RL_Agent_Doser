"""
src/data/loader.py

Loads one OpenKBP patient from raw CSV files into clean numpy arrays.

All CSV files are sparse: column 0 is a flat voxel index, column 1 is the value.
NZ (z-dimension) varies per patient and is computed dynamically.
NX = NY = 128 are always fixed.

Usage:
    from src.data.loader import PatientLoader
    loader = PatientLoader(patient_dir=Path("data/raw/train-pats/pt_1"))
    data   = loader.load()
    # data["ct"]             -> (NZ, 128, 128) float32, HU clipped [0, 4095]
    # data["masks"]          -> dict[str, (NZ, 128, 128)] binary float32
    # data["possible_dose_mask"] -> (NZ, 128, 128) binary float32
    # data["dose_gt"]        -> (NZ, 128, 128) float32 Gy, or None if test set
    # data["voxel_dims"]     -> (3,) float32 [x_mm, y_mm, z_mm]
    # data["present_structures"] -> list[str] of structure names that had CSV files
    # data["shape"]          -> tuple (NZ, 128, 128)
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fixed OpenKBP grid constants
# ---------------------------------------------------------------------------
NX: int = 128
NY: int = 128

# CSV filename (without .csv) -> canonical structure name used throughout project
# Must match MASK_IDX keys in CLAUDE.md
STRUCTURE_CSV_MAP: dict[str, str] = {
    "PTV70":        "PTV_70",
    "PTV63":        "PTV_63",
    "PTV56":        "PTV_56",
    "Brainstem":    "brainstem",
    "SpinalCord":   "spinal_cord",
    "RightParotid": "right_parotid",
    "LeftParotid":  "left_parotid",
    "Larynx":       "larynx",
    "Esophagus":    "esophagus",
    "Mandible":     "mandible",
}


class PatientLoader:
    """
    Loads raw OpenKBP CSV files for a single patient.

    Args:
        patient_dir: Path to the patient folder, e.g. data/raw/train-pats/pt_1
    """

    def __init__(self, patient_dir: Path) -> None:
        self.patient_dir = Path(patient_dir)
        self.patient_id  = self.patient_dir.name

        if not self.patient_dir.exists():
            raise FileNotFoundError(
                f"Patient directory not found: {self.patient_dir}"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> dict:
        """
        Load all available data for this patient.

        Returns:
            dict with keys:
                ct                 (NZ, 128, 128) float32  HU clipped to [0, 4095]
                possible_dose_mask (NZ, 128, 128) float32  binary
                dose_gt            (NZ, 128, 128) float32  Gy, or None for test set
                voxel_dims         (3,)           float32  [x_mm, y_mm, z_mm]
                masks              dict[str -> (NZ,128,128) float32 binary]
                                   keys = canonical structure names (MASK_IDX keys)
                                   absent structures -> zero array (not omitted)
                present_structures list[str]  structures that had a CSV file
                shape              tuple (NZ, 128, 128)
        """
        # Step 1 — voxel spacing (must be first — everything else depends on it)
        voxel_dims = self._load_voxel_dims()

        # Step 2 — CT (determines NZ for this patient)
        ct, nz = self._load_ct()

        # Step 3 — possible dose mask
        possible_dose_mask = self._load_sparse_binary(
            self.patient_dir / "possible_dose_mask.csv", nz
        )

        # Step 4 — ground truth dose (absent in test set)
        dose_gt = self._load_dose_gt(nz)

        # Step 5 — structure masks (all optional)
        masks, present_structures = self._load_structures(nz)

        log.info(
            f"[{self.patient_id}] loaded: shape=({nz},{NY},{NX}), "
            f"structures={present_structures}"
        )

        return {
            "ct":                  ct,
            "possible_dose_mask":  possible_dose_mask,
            "dose_gt":             dose_gt,
            "voxel_dims":          voxel_dims,
            "masks":               masks,
            "present_structures":  present_structures,
            "shape":               (nz, NY, NX),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_voxel_dims(self) -> np.ndarray:
        """
        Read voxel_dimensions.csv -> (3,) float32 [x_mm, y_mm, z_mm].
        Typical values: ~3.5 x 3.5 x 2.0 mm but vary per patient.
        """
        path = self.patient_dir / "voxel_dimensions.csv"
        spacing = np.loadtxt(path, dtype=np.float32)
        if spacing.shape != (3,):
            raise ValueError(
                f"[{self.patient_id}] voxel_dimensions.csv has unexpected "
                f"shape {spacing.shape}, expected (3,)"
            )
        return spacing

    def _load_ct(self) -> tuple[np.ndarray, int]:
        """
        Read ct.csv -> (NZ, NY, NX) float32, HU clipped to [0, 4095].

        NZ is computed dynamically from the largest flat index:
            NZ = max_flat_index // (NX * NY) + 1

        Returns:
            ct:  (NZ, 128, 128) float32
            nz:  int
        """
        path = self.patient_dir / "ct.csv"
        df   = pd.read_csv(path, index_col=0)

        indices = df.index.values.astype(np.int64)
        values  = df["data"].values.astype(np.float32)

        # Dynamic NZ — confirmed necessary by proof-of-concept (pt_1: NZ=94)
        nz = int(indices.max()) // (NX * NY) + 1

        # Fill 3D array (air = -1000 HU where not specified)
        ct = np.full((nz, NY, NX), -1000.0, dtype=np.float32)
        ct.flat[indices] = values

        # Clip to 12-bit convention [0, 4095] as specified in data description
        ct = np.clip(ct, 0.0, 4095.0)

        return ct, nz

    def _load_sparse_binary(self, path: Path, nz: int) -> np.ndarray:
        """
        Read a sparse binary CSV (possible_dose_mask or structure mask).

        Args:
            path: CSV file path
            nz:   z-dimension for this patient (from CT)

        Returns:
            (nz, NY, NX) float32 binary array, zeros if file absent
        """
        volume = np.zeros((nz, NY, NX), dtype=np.float32)
        if not path.exists():
            return volume

        df      = pd.read_csv(path, index_col=0)
        indices = df.index.values.astype(np.int64)

        # Guard against any stray indices beyond the volume size
        valid   = indices[indices < nz * NY * NX]
        volume.flat[valid] = 1.0
        return volume

    def _load_dose_gt(self, nz: int) -> Optional[np.ndarray]:
        """
        Read dose.csv -> (nz, NY, NX) float32 in Gy.
        Returns None for test-set patients (file absent).
        """
        path = self.patient_dir / "dose.csv"
        if not path.exists():
            log.debug(f"[{self.patient_id}] dose.csv not found (test set patient)")
            return None

        df      = pd.read_csv(path, index_col=0)
        indices = df.index.values.astype(np.int64)
        values  = df["data"].values.astype(np.float32)

        dose = np.zeros((nz, NY, NX), dtype=np.float32)
        valid = indices[indices < nz * NY * NX]
        dose.flat[valid] = values[indices < nz * NY * NX]
        return dose

    def _load_structures(
        self, nz: int
    ) -> tuple[dict[str, np.ndarray], list[str]]:
        """
        Load all structure masks. Missing files -> zero array (never skipped).

        Returns:
            masks:              dict[canonical_name -> (nz, NY, NX) float32 binary]
            present_structures: list of canonical names that had a CSV file
        """
        masks:              dict[str, np.ndarray] = {}
        present_structures: list[str]             = []

        for csv_name, canonical_name in STRUCTURE_CSV_MAP.items():
            path   = self.patient_dir / f"{csv_name}.csv"
            volume = self._load_sparse_binary(path, nz)
            masks[canonical_name] = volume

            if path.exists():
                present_structures.append(canonical_name)
                log.debug(
                    f"[{self.patient_id}] loaded {canonical_name}: "
                    f"{int(volume.sum()):,} voxels"
                )
            else:
                log.debug(
                    f"[{self.patient_id}] {canonical_name}: not found, "
                    f"using zero mask"
                )

        return masks, present_structures
