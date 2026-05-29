"""
src/data/preprocessor.py

Takes raw numpy arrays from PatientLoader and:
  1. Resamples all CT-space arrays from (NZ, 128, 128) to (64, 64, 64)
     using SimpleITK in physical space (not naive voxel-count halving).
  2. Normalises CT from [0, 4095] HU to [-1, 1].
  3. Saves all processed arrays to data/processed/patient_id/.

Usage (called by scripts/preprocess_all.py):
    from src.data.preprocessor import Preprocessor
    proc = Preprocessor(out_root=Path("data/processed"))
    proc.process(patient_data, patient_id="pt_1")
"""

import json
import logging
from pathlib import Path

import numpy as np
import SimpleITK as sitk

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — must match CLAUDE.md
# ---------------------------------------------------------------------------
CNN_RES: int = 64       # target cubic resolution for all CT-space arrays

# Canonical mask order in masks.npy — must match MASK_IDX in CLAUDE.md
MASK_ORDER: list[str] = [
    "PTV_70", "PTV_63", "PTV_56",
    "brainstem", "spinal_cord",
    "right_parotid", "left_parotid",
    "larynx", "esophagus", "mandible",
]


class Preprocessor:
    """
    Resamples and saves processed patient data for RL training.

    Args:
        out_root: root directory for processed data, e.g. Path("data/processed")
    """

    def __init__(self, out_root: Path) -> None:
        self.out_root = Path(out_root)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, patient_data: dict, patient_id: str) -> Path:
        """
        Process and save one patient's data.

        Args:
            patient_data: dict returned by PatientLoader.load()
            patient_id:   e.g. "pt_1"

        Returns:
            out_dir: Path to the saved patient directory
        """
        out_dir = self.out_root / patient_id
        out_dir.mkdir(parents=True, exist_ok=True)

        voxel_dims = patient_data["voxel_dims"]          # (3,) float32
        nz, ny, nx = patient_data["shape"]

        # Build a reference SimpleITK image from the CT for all resampling ops
        ct_raw   = patient_data["ct"]                    # (NZ,128,128) float32 [0,4095]
        ct_sitk  = self._to_sitk(ct_raw, voxel_dims)    # physical space image

        # Target SimpleITK reference at CNN_RES³
        target_sitk = self._build_target_image(ct_sitk, nz, ny, nx)

        # ── 1. CT — resample + normalise to [-1, 1] ───────────────────
        ct_resampled = self._resample(ct_sitk, target_sitk,
                                      interpolator=sitk.sitkLinear)
        ct_norm      = self._normalise_ct(ct_resampled)
        np.save(out_dir / "ct.npy", ct_norm)
        log.info(f"[{patient_id}] ct.npy saved {ct_norm.shape}")

        # ── 2. Possible dose mask — resample (nearest neighbour) ───────
        pdm_sitk     = self._to_sitk(patient_data["possible_dose_mask"],
                                     voxel_dims)
        pdm          = self._resample(pdm_sitk, target_sitk,
                                      interpolator=sitk.sitkNearestNeighbor)
        pdm          = (pdm > 0.5).astype(np.float32)
        np.save(out_dir / "possible_dose_mask.npy", pdm)
        log.info(f"[{patient_id}] possible_dose_mask.npy saved {pdm.shape}")

        # ── 3. Structure masks -> (10, 64, 64, 64) ─────────────────────
        mask_stack = self._resample_masks(
            patient_data["masks"], voxel_dims, target_sitk, patient_id
        )
        np.save(out_dir / "masks.npy", mask_stack)
        log.info(f"[{patient_id}] masks.npy saved {mask_stack.shape}")

        # ── 4. Ground-truth dose (train/val only) ──────────────────────
        if patient_data["dose_gt"] is not None:
            dose_sitk    = self._to_sitk(patient_data["dose_gt"], voxel_dims)
            dose         = self._resample(dose_sitk, target_sitk,
                                          interpolator=sitk.sitkLinear)
            dose         = np.clip(dose, 0.0, None)     # dose cannot be negative
            np.save(out_dir / "dose_gt.npy", dose)
            log.info(f"[{patient_id}] dose_gt.npy saved {dose.shape}")
        else:
            log.debug(f"[{patient_id}] no dose_gt (test set patient)")

        # ── 5. Voxel dims ──────────────────────────────────────────────
        np.save(out_dir / "voxel_dims.npy", voxel_dims)

        # ── 6. Present structures JSON ─────────────────────────────────
        with open(out_dir / "present_structures.json", "w") as f:
            json.dump(patient_data["present_structures"], f, indent=2)
        log.info(
            f"[{patient_id}] present_structures: "
            f"{patient_data['present_structures']}"
        )

        return out_dir

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _to_sitk(self, array: np.ndarray,
                 voxel_dims: np.ndarray) -> sitk.Image:
        """
        Wrap a (NZ, NY, NX) numpy array in a SimpleITK image with correct
        physical spacing from voxel_dimensions.csv.

        SimpleITK uses (x, y, z) order for spacing; numpy uses (z, y, x).
        voxel_dims from OpenKBP is [x_mm, y_mm, z_mm].
        """
        img = sitk.GetImageFromArray(array.astype(np.float32))
        img.SetSpacing([
            float(voxel_dims[0]),   # x spacing
            float(voxel_dims[1]),   # y spacing
            float(voxel_dims[2]),   # z spacing
        ])
        img.SetOrigin([0.0, 0.0, 0.0])
        return img

    def _build_target_image(
        self,
        reference: sitk.Image,
        nz: int,
        ny: int,
        nx: int,
    ) -> sitk.Image:
        """
        Build a blank CNN_RES³ SimpleITK image in the same physical space
        as the reference CT. All CT-space arrays are resampled to this target.

        The target spacing is chosen so that the physical extent is preserved:
            target_spacing[i] = original_spacing[i] * original_size[i] / CNN_RES
        """
        orig_spacing = reference.GetSpacing()     # (sx, sy, sz) in mm
        orig_size    = (nx, ny, nz)               # (NX, NY, NZ) in sitk order

        target_spacing = [
            orig_spacing[i] * orig_size[i] / CNN_RES
            for i in range(3)
        ]

        target = sitk.Image([CNN_RES, CNN_RES, CNN_RES], sitk.sitkFloat32)
        target.SetSpacing(target_spacing)
        target.SetOrigin(reference.GetOrigin())
        target.SetDirection(reference.GetDirection())
        return target

    def _resample(
        self,
        source: sitk.Image,
        target: sitk.Image,
        interpolator: int,
    ) -> np.ndarray:
        """
        Resample source to the target grid and return a numpy array (Z, Y, X).
        """
        resampler = sitk.ResampleImageFilter()
        resampler.SetInterpolator(interpolator)
        resampler.SetOutputOrigin(target.GetOrigin())
        resampler.SetOutputSpacing(target.GetSpacing())
        resampler.SetOutputDirection(target.GetDirection())
        resampler.SetSize(target.GetSize())
        resampled = resampler.Execute(source)
        return sitk.GetArrayFromImage(resampled).astype(np.float32)

    def _normalise_ct(self, ct: np.ndarray) -> np.ndarray:
        """
        Normalise CT from [0, 4095] to [-1, 1].
        ct / 2047.5 - 1.0  maps 0->-1, 2047.5->0, 4095->1
        """
        return (ct / 2047.5 - 1.0).astype(np.float32)

    def _resample_masks(
        self,
        masks: dict[str, np.ndarray],
        voxel_dims: np.ndarray,
        target: sitk.Image,
        patient_id: str,
    ) -> np.ndarray:
        """
        Resample all structure masks and stack into (10, 64, 64, 64).
        Order is fixed by MASK_ORDER — matches MASK_IDX in CLAUDE.md.
        Absent structures remain zero planes.
        """
        stack = np.zeros((len(MASK_ORDER), CNN_RES, CNN_RES, CNN_RES),
                         dtype=np.float32)

        for ch, name in enumerate(MASK_ORDER):
            raw = masks.get(name)
            if raw is None or raw.sum() == 0:
                # Structure absent or empty — leave channel as zeros
                continue

            sitk_mask   = self._to_sitk(raw, voxel_dims)
            resampled   = self._resample(sitk_mask, target,
                                         interpolator=sitk.sitkNearestNeighbor)
            binary      = (resampled > 0.5).astype(np.float32)
            stack[ch]   = binary

            log.debug(
                f"[{patient_id}] mask[{ch}] {name}: "
                f"{int(binary.sum()):,} voxels after resample"
            )

        return stack
