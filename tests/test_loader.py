"""
tests/test_loader.py

Unit tests for PatientLoader and Preprocessor.

Run from repo root:
    pytest tests/test_loader.py -v

Requires pt_1 raw data to be present at data/raw/train-pats/pt_1/
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import PatientLoader, STRUCTURE_CSV_MAP
from src.data.preprocessor import Preprocessor, MASK_ORDER, CNN_RES

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
PT1_DIR = Path("data/raw/train-pats/pt_1")


@pytest.fixture(scope="module")
def pt1_data():
    """Load pt_1 once for all tests in this module."""
    if not PT1_DIR.exists():
        pytest.skip(f"pt_1 raw data not found at {PT1_DIR}")
    loader = PatientLoader(PT1_DIR)
    return loader.load()


@pytest.fixture(scope="module")
def pt1_processed(tmp_path_factory, pt1_data):
    """Process pt_1 into a temp directory."""
    out_root = tmp_path_factory.mktemp("processed")
    proc     = Preprocessor(out_root=out_root)
    out_dir  = proc.process(pt1_data, patient_id="pt_1")
    return out_dir


# ---------------------------------------------------------------------------
# PatientLoader tests
# ---------------------------------------------------------------------------
class TestPatientLoader:

    def test_patient_dir_not_found(self):
        with pytest.raises(FileNotFoundError):
            PatientLoader(Path("data/raw/train-pats/pt_9999")).load()

    def test_voxel_dims_shape(self, pt1_data):
        vd = pt1_data["voxel_dims"]
        assert vd.shape == (3,), f"Expected (3,), got {vd.shape}"
        assert vd.dtype == np.float32

    def test_voxel_dims_plausible(self, pt1_data):
        vd = pt1_data["voxel_dims"]
        # Typical OpenKBP values ~3.5 x 3.5 x 2.0 mm
        assert all(0.5 < v < 10.0 for v in vd), \
            f"Voxel dims outside plausible range: {vd}"

    def test_ct_shape(self, pt1_data):
        ct = pt1_data["ct"]
        nz, ny, nx = pt1_data["shape"]
        assert ct.shape == (nz, ny, nx)
        assert ny == 128
        assert nx == 128
        assert ct.dtype == np.float32

    def test_ct_nz_dynamic(self, pt1_data):
        # pt_1 proof-of-concept confirmed NZ=94, not 128
        nz = pt1_data["shape"][0]
        assert nz != 128, \
            "NZ should be dynamic (94 for pt_1), not hardcoded 128"
        assert 50 < nz < 200, f"NZ={nz} outside plausible range"

    def test_ct_clipped(self, pt1_data):
        ct = pt1_data["ct"]
        assert ct.min() >= 0.0,    f"CT min {ct.min()} below 0"
        assert ct.max() <= 4095.0, f"CT max {ct.max()} above 4095"

    def test_possible_dose_mask_shape(self, pt1_data):
        pdm = pt1_data["possible_dose_mask"]
        assert pdm.shape == pt1_data["shape"]
        assert pdm.dtype == np.float32

    def test_possible_dose_mask_binary(self, pt1_data):
        pdm = pt1_data["possible_dose_mask"]
        unique = np.unique(pdm)
        assert set(unique).issubset({0.0, 1.0}), \
            f"possible_dose_mask contains non-binary values: {unique}"

    def test_possible_dose_mask_nonempty(self, pt1_data):
        assert pt1_data["possible_dose_mask"].sum() > 0, \
            "possible_dose_mask is all zeros"

    def test_dose_gt_shape(self, pt1_data):
        dose = pt1_data["dose_gt"]
        assert dose is not None, "dose_gt should exist for train patient"
        assert dose.shape == pt1_data["shape"]
        assert dose.dtype == np.float32

    def test_dose_gt_range(self, pt1_data):
        dose = pt1_data["dose_gt"]
        assert dose.min() >= 0.0, f"Dose contains negative values: {dose.min()}"
        assert dose.max() <= 100.0, f"Dose max {dose.max()} unrealistically high"

    def test_masks_keys(self, pt1_data):
        masks = pt1_data["masks"]
        expected = set(STRUCTURE_CSV_MAP.values())
        assert set(masks.keys()) == expected, \
            f"Missing keys: {expected - set(masks.keys())}"

    def test_masks_shapes(self, pt1_data):
        shape = pt1_data["shape"]
        for name, mask in pt1_data["masks"].items():
            assert mask.shape == shape, \
                f"Mask {name} has shape {mask.shape}, expected {shape}"
            assert mask.dtype == np.float32

    def test_masks_binary(self, pt1_data):
        for name, mask in pt1_data["masks"].items():
            unique = np.unique(mask)
            assert set(unique).issubset({0.0, 1.0}), \
                f"Mask {name} contains non-binary values: {unique}"

    def test_present_structures_is_list(self, pt1_data):
        ps = pt1_data["present_structures"]
        assert isinstance(ps, list)
        assert len(ps) > 0, "No structures found for pt_1"

    def test_present_structures_subset_of_all(self, pt1_data):
        all_canonical = set(STRUCTURE_CSV_MAP.values())
        for name in pt1_data["present_structures"]:
            assert name in all_canonical, \
                f"Unknown structure name: {name}"

    def test_absent_structures_are_zero(self, pt1_data):
        present = set(pt1_data["present_structures"])
        for name, mask in pt1_data["masks"].items():
            if name not in present:
                assert mask.sum() == 0, \
                    f"Absent structure {name} has non-zero mask"

    def test_ptv70_always_present(self, pt1_data):
        # Data description: every plan has at least PTV_70
        assert "PTV_70" in pt1_data["present_structures"], \
            "PTV_70 should always be present"


# ---------------------------------------------------------------------------
# Preprocessor tests
# ---------------------------------------------------------------------------
class TestPreprocessor:

    def test_output_files_exist(self, pt1_processed):
        expected = [
            "ct.npy",
            "masks.npy",
            "possible_dose_mask.npy",
            "dose_gt.npy",
            "voxel_dims.npy",
            "present_structures.json",
        ]
        for fname in expected:
            assert (pt1_processed / fname).exists(), \
                f"Missing output file: {fname}"

    def test_ct_shape(self, pt1_processed):
        ct = np.load(pt1_processed / "ct.npy")
        assert ct.shape == (CNN_RES, CNN_RES, CNN_RES), \
            f"ct.npy shape {ct.shape} != ({CNN_RES},{CNN_RES},{CNN_RES})"
        assert ct.dtype == np.float32

    def test_ct_normalised(self, pt1_processed):
        ct = np.load(pt1_processed / "ct.npy")
        assert ct.min() >= -1.01, f"CT min {ct.min()} below -1"
        assert ct.max() <= 1.01,  f"CT max {ct.max()} above 1"

    def test_masks_shape(self, pt1_processed):
        masks = np.load(pt1_processed / "masks.npy")
        assert masks.shape == (len(MASK_ORDER), CNN_RES, CNN_RES, CNN_RES), \
            f"masks.npy shape {masks.shape}"
        assert masks.dtype == np.float32

    def test_masks_binary(self, pt1_processed):
        masks = np.load(pt1_processed / "masks.npy")
        unique = np.unique(masks)
        assert set(unique).issubset({0.0, 1.0}), \
            f"masks.npy contains non-binary values: {unique}"

    def test_masks_channel_order(self, pt1_processed):
        # Channel 0 should be PTV_70 — must be non-zero for every patient
        masks = np.load(pt1_processed / "masks.npy")
        assert masks[0].sum() > 0, \
            "Channel 0 (PTV_70) is all zeros — MASK_ORDER may be wrong"

    def test_possible_dose_mask_shape(self, pt1_processed):
        pdm = np.load(pt1_processed / "possible_dose_mask.npy")
        assert pdm.shape == (CNN_RES, CNN_RES, CNN_RES)
        assert pdm.dtype == np.float32

    def test_possible_dose_mask_binary(self, pt1_processed):
        pdm = np.load(pt1_processed / "possible_dose_mask.npy")
        unique = np.unique(pdm)
        assert set(unique).issubset({0.0, 1.0})

    def test_dose_gt_shape(self, pt1_processed):
        dose = np.load(pt1_processed / "dose_gt.npy")
        assert dose.shape == (CNN_RES, CNN_RES, CNN_RES)
        assert dose.dtype == np.float32

    def test_dose_gt_nonnegative(self, pt1_processed):
        dose = np.load(pt1_processed / "dose_gt.npy")
        assert dose.min() >= 0.0, f"Processed dose contains negatives: {dose.min()}"

    def test_voxel_dims_preserved(self, pt1_processed):
        vd = np.load(pt1_processed / "voxel_dims.npy")
        assert vd.shape == (3,)
        assert vd.dtype == np.float32
        assert all(0.5 < v < 10.0 for v in vd)

    def test_present_structures_json(self, pt1_processed):
        with open(pt1_processed / "present_structures.json") as f:
            ps = json.load(f)
        assert isinstance(ps, list)
        assert len(ps) > 0
        assert "PTV_70" in ps

    def test_idempotent(self, pt1_data, tmp_path):
        """Running preprocessor twice on same patient gives identical results."""
        proc = Preprocessor(out_root=tmp_path)
        proc.process(pt1_data, patient_id="pt_1")
        ct1 = np.load(tmp_path / "pt_1" / "ct.npy")

        proc.process(pt1_data, patient_id="pt_1")
        ct2 = np.load(tmp_path / "pt_1" / "ct.npy")

        np.testing.assert_array_equal(ct1, ct2, err_msg="Preprocessor not idempotent")
