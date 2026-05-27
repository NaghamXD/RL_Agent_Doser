# CLAUDE.md — Adaptive Radiotherapy Planning via Deep Reinforcement Learning

This file is the single source of truth for Claude Code working on this project.
Read it fully before touching any file. Every architectural decision made during
planning is recorded here so you never have to infer intent from code alone.

---

## Project overview

We are building a reinforcement learning agent that adaptively controls radiation
beam intensities — fraction by fraction — for head and neck cancer treatment
planning. The agent operates over 35 treatment fractions (one episode = one
patient), outputting 9 continuous beam fluence weights per fraction. It learns
purely from a reward signal derived from real medical physics, computed via a
pre-calculated Dose Influence Matrix (DIM) from the pyRadPlan engine.

This is NOT a supervised dose prediction project. Do not add any loss terms that
compare the agent's actions to ground-truth beam weights — those do not exist in
the OpenKBP dataset. The ground-truth dose volume is used only for evaluation.

---

## Repository layout

```
adaptive-rt-rl/
│
├── CLAUDE.md                        ← you are here
├── README.md
├── requirements.txt
├── config/
│   └── default.yaml                 ← all hyperparameters live here, not in code
│
├── data/
│   ├── raw/                         ← OpenKBP CSVs, never modified
│   │   ├── train-pats/              ← 200 patient folders
│   │   ├── validation-pats/         ← 40 patient folders
│   │   └── test-pats/               ← 100 patient folders
│   └── processed/                   ← pre-computed numpy arrays + DIM caches
│       ├── patient_003/
│       │   ├── ct.npy               ← shape (64, 64, 64), float32, normalised [−1,1]
│       │   ├── masks.npy            ← shape (10, 64, 64, 64), binary float32
│       │   ├── possible_dose_mask.npy ← shape (64, 64, 64), binary float32
│       │   ├── dose_gt.npy          ← shape (64, 64, 64), float32, Gy (train/val only)
│       │   ├── voxel_dims.npy       ← shape (3,), float32, original spacing in mm
│       │   ├── present_structures.json ← list of which structure CSVs existed
│       │   └── dim.npy              ← shape (262144, 9), float16  [64^3 × 9 beams]
│       └── ...
│
├── src/
│   ├── data/
│   │   ├── loader.py                ← OpenKBP CSV → numpy pipeline
│   │   └── preprocessor.py          ← resample to 64^3, normalise, cache
│   │
│   ├── physics/
│   │   └── dim_engine.py            ← PyRadPlanEngine class (DIM compute + dose query)
│   │
│   ├── env/
│   │   ├── treatment_env.py         ← TreatmentPlanningEnv(gymnasium.Env)
│   │   └── reward.py                ← all reward computation, fully separated
│   │
│   ├── models/
│   │   ├── encoder.py               ← 3D-CNN state encoder (shared actor-critic trunk)
│   │   └── policy.py                ← actor + critic heads, PPO-compatible
│   │
│   ├── training/
│   │   ├── train_ppo.py             ← PPO training loop via Stable-Baselines3
│   │   └── train_sac.py             ← SAC baseline training
│   │
│   └── evaluation/
│       ├── evaluate.py              ← run agent on test patients, collect DVH
│       ├── dvh.py                   ← DVH score + dose score (OpenKBP official)
│       └── plot_dvh.py              ← matplotlib DVH curve comparison
│
├── scripts/
│   ├── preprocess_all.py            ← run once: builds data/processed/
│   ├── compute_dims.py              ← run once: calls pyRadPlan, saves dim.npy
│   └── sanity_check_env.py          ← step through one episode manually, print rewards
│
├── tests/
│   ├── test_loader.py
│   ├── test_dim_engine.py
│   ├── test_env_step.py
│   └── test_reward.py
│
└── notebooks/
    ├── 01_data_exploration.ipynb
    ├── 02_dim_inspection.ipynb
    └── 03_training_curves.ipynb
```

---

## Data

### Source

OpenKBP dataset — 340 head and neck cancer patients.
Download from: https://github.com/ababier/open-kbp

Each patient folder contains these CSV files. Files marked ⚠️ are NOT guaranteed
to exist for every patient — the loader must handle their absence explicitly.

| File | Present in all patients | Description |
|------|------------------------|-------------|
| `ct.csv` | ✅ Always | CT voxel values in Hounsfield Units |
| `possible_dose_mask.csv` | ✅ Always | Binary mask of voxels that are physically reachable by at least one of the nine beams. Voxels outside this mask will always be zero in the dose distribution regardless of beam intensities — they are either outside the patient body contour or outside the geometric reach of the beam configuration. **This mask governs three things: reward computation (only evaluate coverage inside the mask), dose enforcement (zero out any dose outside the mask after each DIM multiplication), and evaluation (the official dose score is computed only over voxels inside this mask).** |
| `voxel_dimensions.csv` | ✅ Always | Physical size of each voxel in mm (x, y, z). **Critical for pyRadPlan DIM computation and for correct SimpleITK resampling — must be read and passed to both.** Typical values ~3.5mm × 3.5mm × 2.0mm but vary per patient. |
| `dose.csv` | ✅ Always (train/val only) | Ground-truth clinical dose in Gy. Absent in test set. |
| `PTV_70.csv` | ⚠️ Not always | Planning target volume at 70 Gy prescription |
| `PTV_63.csv` | ⚠️ Not always | Planning target volume at 63 Gy prescription |
| `PTV_56.csv` | ⚠️ Not always | Planning target volume at 56 Gy prescription |
| `Brainstem.csv` | ⚠️ Not always | OAR mask |
| `SpinalCord.csv` | ⚠️ Not always | OAR mask |
| `RightParotid.csv` | ⚠️ Not always | OAR mask |
| `LeftParotid.csv` | ⚠️ Not always | OAR mask |
| `Larynx.csv` | ⚠️ Not always | OAR mask |
| `Esophagus.csv` | ⚠️ Not always | OAR mask |
| `Mandible.csv` | ⚠️ Not always | OAR mask |

**Not every patient has every structure contoured.** This is real clinical data —
some patients were treated without contouring every possible OAR. The loader must
treat every structure mask as optional and substitute a zero-filled volume when
the file is absent. Never raise a FileNotFoundError on a missing structure CSV;
always fall back silently and record which structures are missing in the patient
metadata.

**voxel_dimensions.csv format:** A single row with three values [x_mm, y_mm, z_mm] representing
the physical spacing of the voxel grid. Typical values are ~3.5mm × 3.5mm × 2.0mm
but differ per patient. This file must be read before any other processing step
because two things depend on it:
1. SimpleITK resampling must know the original physical spacing to correctly map
   voxels to anatomy in physical space — naive voxel-count halving without spacing
   produces geometrically inconsistent inputs across patients.
2. pyRadPlan requires physical voxel dimensions to compute accurate dose deposition
   in the DIM — passing voxel indices without physical spacing produces wrong dose values.

Always load `voxel_dimensions.csv` first and store as a (3,) float32 array (`voxel_dims.npy`)
before any other processing step for that patient.

### Resolution decision — read before writing any processing code

The OpenKBP data is natively at **128 × 128 × 128 voxels**. This is already
downsampled by the challenge organisers from the original clinical resolution.

This project must decide whether to work at native 128³ or downsample further to 64³.
This is a pure **GPU memory tradeoff**, not a data quality decision. The table below
shows why it matters:

| | Native 128³ | Downsampled 64³ |
|--|--|--|
| DIM shape | (2,097,152 × 900) | (262,144 × 900) |
| DIM memory (float32) | ~7.5 GB per patient | ~0.94 GB per patient |
| State tensor (13ch) | ~109 MB per sample | ~13.6 MB per sample |
| Requires GPU VRAM | 40+ GB | 16–24 GB |
| Spatial fidelity | Full | Reduced |

**Decision: use 64³ downsampling** unless a 40+ GB GPU is available. This
makes DIM storage and CNN training feasible on a standard research GPU (A100 40GB
or RTX 4090). The spatial resolution loss is acceptable because the dose gradients
we are optimising operate at the scale of organ volumes (~cm), not individual voxels.

**Why SimpleITK and not scipy.ndimage.zoom:** `scipy.ndimage.zoom` resizes in
voxel space, ignoring physical spacing. Since each patient has different voxel
dimensions, two 128³ volumes represent physically different-sized anatomies.
Naive zoom to 64³ preserves the voxel ratio but not the physical geometry, making
the CNN inputs geometrically inconsistent across patients. SimpleITK resamples in
physical space — it reads the original voxel spacing, maps voxels to real-world
coordinates in mm, then samples the target grid at uniform physical intervals.
The result is that all 64³ outputs represent the same physical resolution
regardless of the source patient's voxel dimensions. Use `sitk.sitkLinear`
interpolation for CT and dose, and `sitk.sitkNearestNeighbor` for all binary masks.

### Pre-processing pipeline

Run `scripts/preprocess_all.py` once before any training. It:
1. Reads `voxel_dimensions.csv` first — stores as `voxel_dims.npy`, shape (3,) float32
2. Reads `ct.csv` — unravels sparse indices to 128³ numpy array
3. Clips CT to [0, 4095] HU (converting any 16-bit values to 12-bit convention
   as specified in the data description), then normalises to [−1, 1]
4. Reads `possible_dose_mask.csv` — unravels to 128³ binary array
5. For each of the 10 structure CSVs: load and unravel if file exists,
   else create a zero-filled 128³ volume — never raise FileNotFoundError
6. Uses SimpleITK to resample all volumes from 128³ at original voxel spacing
   to 64³ at doubled voxel spacing (preserving physical extent)
7. Stacks masks into a single (10, 64, 64, 64) binary array in this fixed order:
   `[PTV_70, PTV_63, PTV_56, brainstem, spinal_cord, right_parotid,
     left_parotid, larynx, esophagus, mandible]`
   — absent structures remain zero planes in their respective channel
8. Saves to `data/processed/patient_XXX/`:
   - `ct.npy` — shape (64, 64, 64), float32, normalised to [−1, 1]
   - `masks.npy` — shape (10, 64, 64, 64), float32 binary
   - `possible_dose_mask.npy` — shape (64, 64, 64), float32 binary
   - `dose_gt.npy` — shape (64, 64, 64), float32 in Gy (train/val only)
   - `voxel_dims.npy` — shape (3,), float32, original spacing in mm
   - `present_structures.json` — list of which structure CSVs were found,
     e.g. `["PTV_70", "brainstem", "right_parotid"]`. Used by the env
     to skip OAR reward terms for structures that were never contoured.

**Data format note:** All CSV files in OpenKBP are sparse — only non-zero voxels
are stored. Column 1 is a flat index into the 128³ grid; column 2 is the value.
Unravel with `np.unravel_index(indices, (128, 128, 128), order='C')`. Do not
treat these as dense CSVs — reading them naively as a matrix will produce garbage.

Run `scripts/compute_dims.py` separately (GPU recommended, ~5 min/patient):
- Loads `voxel_dims.npy` — passes original physical spacing to pyRadPlan
- Calls `PyRadPlanEngine` with the 64³ CT, masks, and physical voxel dimensions
- Computes the DIM for 9 fixed gantry angles: [0, 40, 80, 120, 160, 200, 240, 280, 320] degrees
- Saves `dim.npy` of shape (262144, 9) — i.e. (64³ voxels × 9 beams), float16
- float16 saves ~500 MB per patient vs ~1 GB at float32; cast to float32 inside
  `compute_dose()` only, never store float32 DIM to disk.

### Critical constants — never hardcode these, always import from config

```python
N_FRACTIONS     = 35
N_BEAMS         = 9
VOXEL_RES       = 64          # cubic side length after downsampling
GANTRY_ANGLES   = [0, 40, 80, 120, 160, 200, 240, 280, 320]  # degrees

# Dose prescriptions (Gy, full course)
PRESCRIPTION    = {
    'PTV_70': 70.0,
    'PTV_63': 63.0,
    'PTV_56': 56.0,
}

# OAR tolerance doses (Gy, full course mean-dose limits)
OAR_TOLERANCES  = {
    'brainstem':     54.0,
    'spinal_cord':   45.0,
    'right_parotid': 26.0,
    'left_parotid':  26.0,
    'larynx':        45.0,
    'esophagus':     45.0,
    'mandible':      70.0,
}

# Mask channel indices in masks.npy
MASK_IDX = {
    'PTV_70': 0, 'PTV_63': 1, 'PTV_56': 2,
    'brainstem': 3, 'spinal_cord': 4,
    'right_parotid': 5, 'left_parotid': 6,
    'larynx': 7, 'esophagus': 8, 'mandible': 9,
}
```

---

## RL problem formulation

Refer to this whenever there is any ambiguity. These definitions are final.

### State — S_t ∈ ℝ^(13 × 64 × 64 × 64)

Thirteen channels stacked in this exact order:

| Index | Content | Notes |
|-------|---------|-------|
| 0 | CT scan | Normalised HU ∈ [−1, 1] |
| 1–3 | PTV masks (70, 63, 56) | Binary float32 |
| 4–10 | OAR masks (7 structures) | Binary float32, order = MASK_IDX |
| 11 | Cumulative dose so far | In Gy, unnormalised |
| 12 | Dose gap | prescription_map − cumulative_dose, clipped to [−10, 70] |

`prescription_map` is a 3D volume where each PTV voxel holds its target dose
(70, 63, or 56 Gy) and all other voxels are 0. Pre-compute once per patient in
`reset()`. The dose gap channel is the most informative signal for the agent —
it directly encodes where treatment is behind or ahead of schedule.

Fraction index t/35 is NOT a channel — it is appended as a scalar to the flat
latent vector inside the policy network after the CNN encoder, not before it.

### Action — A_t ∈ [0, 1]^9

Nine continuous fluence weights, one per beam. The agent outputs raw logits;
the policy applies `sigmoid` to bound them to [0, 1]. Do NOT use `tanh` + rescale
here — sigmoid is cleaner for a strictly non-negative fluence domain.

Dose from this fraction:
```python
fractional_dose = dim @ action_vector           # shape: (262144,) → reshape to (64,64,64)
fractional_dose = fractional_dose.reshape(64, 64, 64)
fractional_dose *= possible_dose_mask           # zero out physically unreachable voxels
```

### Reward — R_t

The reward reflects the clinical asymmetry between OARs and tumor:

- **OARs** are penalised at EVERY fraction — radiation damage is irreversible.
  Exceeding a tolerance at fraction 10 cannot be undone in fraction 11.
- **Tumor coverage** is evaluated ONLY at the terminal step (t = 35) — only the
  final cumulative dose matters clinically. The agent is free to reach the
  prescription via any trajectory: linear, front-loaded, back-loaded, or
  patient-specific adaptive. No intermediate pace is enforced.
- **Progress shaping** is added at every step to mitigate the sparse terminal
  signal without constraining the delivery trajectory. It rewards closing the
  remaining prescription gap this fraction, regardless of pace.

**No ground truth beam weights exist in OpenKBP.** The reward is therefore
entirely dose-based — it never compares actions to a reference action.
The only valid comparison at t = 35 is cumulative dose vs. ground truth dose.

```python
def compute_reward(cum_dose, prev_cum_dose, masks, possible_dose_mask,
                   prescription_map, present_structures, t, n_fractions=35):
    """
    Args:
        cum_dose:          (64,64,64) cumulative dose after this fraction, in Gy
        prev_cum_dose:     (64,64,64) cumulative dose before this fraction, in Gy
        masks:             (10,64,64,64) structure masks in MASK_IDX order
        possible_dose_mask:(64,64,64) binary, physically reachable voxels
        prescription_map:  (64,64,64) per-voxel prescription dose in Gy
                           (70/63/56 inside respective PTVs, 0 elsewhere)
        present_structures: list of structure names contoured for this patient
        t:                 current fraction index, 1-indexed (1..35)
        n_fractions:       total fractions in the episode (35)
    Returns:
        scalar float reward
    """
    is_terminal = (t == n_fractions)

    # ── Shared mask: PTV intersected with possible_dose_mask ──────────────
    # PTV voxels outside the possible_dose_mask can never receive dose
    # regardless of the agent's action — exclude them from all computations.
    ptv_combined = (masks[0] + masks[1] + masks[2]).clip(0, 1).bool()
    ptv_mask = ptv_combined & possible_dose_mask.bool()

    # ── 1. OAR penalty (dense — every fraction) ───────────────────────────
    # Compare cumulative dose against the FULL-COURSE tolerance, not a
    # scaled fraction of it. The hard limit applies from fraction 1 onward.
    # Penalise any exceedance immediately — do not wait for the terminal step.
    oar_penalty = 0.0
    n_present_oars = 0
    for name, tol in OAR_TOLERANCES.items():
        idx = MASK_IDX[name]
        if name not in present_structures:       # not contoured for this patient
            continue
        oar_voxels = masks[idx].bool() & possible_dose_mask.bool()
        if oar_voxels.sum() == 0:                # safety check
            continue
        n_present_oars += 1
        oar_dose = cum_dose[oar_voxels].mean()
        excess = (oar_dose - tol).clamp(min=0) / tol   # normalised exceedance
        oar_penalty += excess

    # Normalise by number of present OARs so patients with fewer contours
    # are not unfairly advantaged.
    if n_present_oars > 0:
        oar_penalty /= n_present_oars

    # ── 2. Progress shaping (dense — every fraction) ──────────────────────
    # Reward closing the remaining prescription gap this fraction.
    # Does NOT enforce a linear pace — an agent that delivers 0 Gy in early
    # fractions and compensates later still gets shaping reward when it
    # eventually closes the gap. Only the rate of closure is rewarded.
    remaining_before = (prescription_map[ptv_mask] - prev_cum_dose[ptv_mask]
                        ).clamp(min=0)
    remaining_after  = (prescription_map[ptv_mask] - cum_dose[ptv_mask]
                        ).clamp(min=0)
    gap_closed       = (remaining_before - remaining_after).mean()
    # Normalise by max prescription so shaping is in roughly [0, 1] per step
    progress_shaping = gap_closed / (prescription_map[ptv_mask].mean() + 1e-8)

    # ── 3. Terminal tumor coverage reward (sparse — fraction 35 only) ─────
    # The agent is not penalised for being behind the linear pace at t < 35.
    # Only the final cumulative dose quality matters for tumor control.
    coverage_reward = 0.0
    if is_terminal:
        ptv_dose        = cum_dose[ptv_mask]
        ptv_rx          = prescription_map[ptv_mask]
        # Fraction of PTV voxels that received at least their full prescription
        coverage_reward = (ptv_dose >= ptv_rx).float().mean()
        # Add DVH-based terminal bonus aligned with the official evaluation metric
        dvh_bonus       = compute_dvh_score(cum_dose, masks, dose_gt,
                                            possible_dose_mask) * 2.0
        # dvh_bonus scaled by 2.0 so the terminal signal remains visible
        # after discounting: 0.99^35 * 2.0 ≈ 1.4 — larger than any single
        # dense step reward, ensuring the agent optimises for final outcome.
        coverage_reward += dvh_bonus

    # ── Combined reward ───────────────────────────────────────────────────
    # Weights are initial values — tune empirically using the protocol below.
    LAMBDA_OAR      = 1.0   # scale of OAR penalty relative to shaping
    LAMBDA_SHAPING  = 0.3   # keep shaping weaker than terminal reward

    reward = (coverage_reward
              - LAMBDA_OAR     * oar_penalty
              + LAMBDA_SHAPING * progress_shaping)

    return float(reward)
```

### Reward weight tuning protocol

The weights `LAMBDA_OAR` and `LAMBDA_SHAPING` are defined in `config/default.yaml`,
not hardcoded. They are initial values that require empirical tuning before final
experiments. Follow this protocol:

1. Train for 200k steps with equal weights (LAMBDA_OAR=1.0, LAMBDA_SHAPING=0.3)
2. Plot the three reward components separately over training (not just total reward)
3. If OAR penalty dominates and agent ignores tumor: reduce LAMBDA_OAR to 0.5
4. If agent makes no progress toward prescription: increase LAMBDA_SHAPING to 0.5
5. If agent learns uniform delivery and never adapts: verify terminal bonus is
   propagating — check that value function at t=1 is non-zero for good episodes
6. Report the final tuned values and the tuning process in the methods section

### Discount factor

γ = 0.99. Do not change this without updating this file. Rationale: at γ = 0.99
the terminal bonus at step 35 is discounted by 0.99^35 ≈ 0.70, keeping it
visible from step 1. At γ = 0.95 it would be 0.95^35 ≈ 0.17 — effectively invisible.

### Episode termination

Episodes end at t = 35 (normal) or earlier if any OAR exceeds 110% of its
full-course tolerance (hard clinical constraint violation). In the latter case,
apply a large negative terminal penalty of −5.0 and set `terminated = True`.

---

## Architecture

### PyRadPlanEngine (`src/physics/dim_engine.py`)

```python
class PyRadPlanEngine:
    def __init__(self, ct: np.ndarray, masks: np.ndarray,
                 angles: list, possible_dose_mask: np.ndarray):
        # Calls pyRadPlan to compute DIM — runs once, ~5 min per patient
        self.dim = self._compute_dim(ct, masks, angles)   # shape (V, 9), float16
        self.shape = (64, 64, 64)
        self.possible_dose_mask = possible_dose_mask      # shape (64,64,64), binary

    def compute_dose(self, action: np.ndarray) -> np.ndarray:
        # action: (9,) float32 in [0, 1]
        # returns: (64, 64, 64) float32 dose map in Gy
        flat_dose = self.dim.astype(np.float32) @ action
        dose = flat_dose.reshape(self.shape)
        dose *= self.possible_dose_mask   # enforce zero outside reachable voxels
        return dose
```

If pyRadPlan is unavailable, `dim_engine.py` falls back to loading a pre-saved
`dim.npy` from `data/processed/patient_XXX/`. The fallback must be transparent
to `TreatmentPlanningEnv` — the env always calls `engine.compute_dose(action)`,
never calls pyRadPlan directly.

### TreatmentPlanningEnv (`src/env/treatment_env.py`)

Inherits from `gymnasium.Env`. Key design rules:
- `reset(patient_id=None)` — if None, sample uniformly from the training set
- `step(action)` returns `(obs, reward, terminated, truncated, info)`
- `info` dict must always contain: `{'t': int, 'patient_id': str,
  'ptv_coverage': float, 'oar_max_excess': float}`
- The env owns one `PyRadPlanEngine` instance per patient, recreated on `reset()`
- Do NOT load DIM inside `step()` — it must be loaded during `reset()` only
- Observation is returned as `np.float32` always — SB3 will complain otherwise

### Encoder (`src/models/encoder.py`)

3D CNN with this structure (do not change without updating here):
```
Input: (13, 64, 64, 64)
Conv3d(13→32, k=3, s=2, p=1)   → (32, 32, 32, 32)   + ReLU
Conv3d(32→64, k=3, s=2, p=1)   → (64, 16, 16, 16)   + ReLU
Conv3d(64→128, k=3, s=2, p=1)  → (128, 8, 8, 8)     + ReLU
Conv3d(128→256, k=3, s=2, p=1) → (256, 4, 4, 4)     + ReLU
AdaptiveAvgPool3d(1)            → (256, 1, 1, 1)
Flatten                         → (256,)
Linear(256→256)                 → (256,)              + ReLU
# Append scalar t/35 here
Linear(257→128)                 → (128,)              + ReLU
```
Output latent dim: 128. Actor and critic heads are each a single `Linear(128, ?)`.

### Policy (`src/models/policy.py`)

Uses SB3's `ActorCriticPolicy` with a custom `features_extractor_class` pointing
to the encoder above. The actor head outputs mean + log_std for a 9-dim Gaussian.
Log_std is clamped to [−3, 0] (i.e. std ∈ [0.05, 1.0]) to maintain the
exploration floor throughout training.

---

## Training

### Primary algorithm: PPO

All hyperparameters live in `config/default.yaml`. Canonical starting values:

```yaml
ppo:
  learning_rate: 3.0e-4
  n_steps: 35           # one full episode per rollout per env
  batch_size: 35
  n_epochs: 10
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  ent_coef: 0.01        # entropy bonus — do not set to 0
  vf_coef: 0.5
  max_grad_norm: 0.5
  total_timesteps: 3_500_000   # 35 steps × 100,000 episodes

reward:
  lambda_oar: 1.0       # OAR penalty weight — tune per tuning protocol
  lambda_shaping: 0.3   # progress shaping weight — tune per tuning protocol
  dvh_bonus_scale: 2.0  # terminal DVH bonus multiplier — keep ≥ 2.0
```

`ent_coef: 0.01` is mandatory. Setting it to 0 causes policy collapse in early
training because the flat reward landscape provides no gradient signal until the
agent accidentally discovers a good beam combination.

All reward weights must be loaded from config — never hardcode them in
`src/env/reward.py`. This allows tuning without touching source code.

### Secondary baseline: SAC

Trained with identical encoder and same patient curriculum. Compare final DVH
scores on the test set. SAC hyperparams in `config/sac.yaml`.

### Patient curriculum

Training patients are sampled uniformly without replacement each epoch. A
"curriculum" here means the env sees all 200 training patients before repeating
any of them — implement via a shuffled patient index queue in the env wrapper.

Do NOT train on validation or test patients. The split is:
- Train: `train-pats/` (200 patients)
- Validation: `validation-pats/` (40 patients) — used for checkpoint selection
- Test: `test-pats/` (100 patients) — used only for final reported metrics

### Checkpointing

Save a checkpoint every 10,000 timesteps. Select the best checkpoint by mean
DVH score on the validation set, not by training reward. These are different
things. Training reward is shaped; DVH score is the real metric.

---

## Evaluation

Evaluation runs the policy deterministically: `action = mean_of_policy(state)`,
no sampling. This is SB3's default at `model.predict(obs, deterministic=True)`.

Metrics reported on the test set (100 patients):

1. **DVH score** (primary) — OpenKBP official metric. Lower is better.
   Computed by `src/evaluation/dvh.py`, identical to the challenge scoring script.
2. **Dose score** — mean absolute error of the 3D dose volume vs. ground truth.
3. **PTV D95** — dose received by 95% of PTV volume (clinical standard).
4. **OAR mean doses** — mean dose to each of the 7 OARs.
5. **Adaptivity score** — std deviation of beam weights across the 35 fractions
   for a single patient. A non-adaptive agent will have near-zero std. This is
   a novel metric we report to demonstrate the agent genuinely adapts.

Baseline for comparison: the ground-truth clinical plans from OpenKBP
(i.e., what a human dosimetrist produced). The agent does not need to beat the
clinical plan — matching it while demonstrating adaptive behavior is sufficient
for a course project. Any improvement is a strong result.

---

## Common failure modes and fixes

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Agent receives reward for voxels it cannot control | possible_dose_mask not intersected with PTV in reward | Always AND ptv_mask with possible_dose_mask before computing coverage |
| Dose leaks outside patient body | possible_dose_mask not applied after DIM multiply | Multiply every fractional_dose by possible_dose_mask in compute_dose() |
| DIM dose values physically wrong | voxel_dimensions.csv spacing not passed to pyRadPlan | Always load voxel_dims.npy and pass spacing to PyRadPlanEngine |
| Resampled volume has wrong scale | SimpleITK not using original spacing | Pass voxel_dims when setting image spacing before resampling |
| OAR penalty dominates for one patient | Missing structure counted as zero-dose OAR | Load present_structures.json in env; skip absent structures in reward |
| Reward stuck near 0.0, agent ignores tumor | Terminal reward not propagating | Verify γ=0.99 and dvh_bonus scale factor is 2.0; check value function at t=1 |
| OAR penalty dominates, agent ignores tumor | LAMBDA_OAR too large | Reduce LAMBDA_OAR from 1.0 to 0.5 following the tuning protocol |
| Agent makes no progress toward prescription | LAMBDA_SHAPING too small | Increase LAMBDA_SHAPING from 0.3 to 0.5 |
| Agent delivers uniform dose every fraction | Terminal signal too weak to drive adaptation | Increase dvh_bonus scale factor from 2.0 to 3.0 |
| DVH score worse than random | Log_std clamping too tight | Widen to [−4, 0] in early training |
| DIM compute OOM | 64³ DIM too large at float32 | Save as float16; cast to float32 in `compute_dose` only |
| Episode terminates at t=1 | OAR hard constraint too strict | Raise threshold from 110% to 120% for first 50k steps |
| Training loss NaN | Learning rate too high | Reduce to 1e-4; add gradient clipping at 0.3 |
| Test DVH much worse than val | Checkpoint selected on training reward | Always select checkpoint on validation DVH score |

---

## Code style and conventions

- Python 3.10+
- Type hints on all function signatures
- No magic numbers in src/ — all constants imported from config or the constants
  block in this file
- All file I/O goes through `pathlib.Path`, never string concatenation
- Logging via Python `logging` module, not `print()` — except in scripts/
- One class per file in src/
- Tests in tests/ mirror the src/ structure: `tests/test_dim_engine.py` tests
  `src/physics/dim_engine.py`
- Run `pytest tests/` before committing anything

---

## Environment setup

```bash
# Create environment
conda create -n adaptive-rt python=3.10
conda activate adaptive-rt

# Core dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install stable-baselines3[extra]
pip install gymnasium
pip install SimpleITK
pip install pyRadPlan          # physics engine
pip install numpy pandas matplotlib seaborn
pip install pytest pyyaml

# Verify GPU
python -c "import torch; print(torch.cuda.is_available())"

# Verify pyRadPlan
python -c "import pyradplan; print('pyRadPlan OK')"
```

If pyRadPlan import fails, check https://github.com/e0404/pyRadPlan for
installation instructions. It may require a separate MATLAB runtime or
conda-specific install. If it remains unavailable after 30 minutes of debugging,
fall back to the pre-saved DIM files and note this in the report.

---

## What NOT to do

- Do not use the dose ground truth as a state channel — it leaks the answer
- Do not compare agent actions to ground-truth beam weights — they don't exist
- Do not run evaluation on training patients — results will be meaningless
- Do not set `ent_coef = 0` — policy will collapse silently
- Do not load the DIM inside `step()` — it will make training ~1000x slower
- Do not change γ below 0.97 — the terminal DVH bonus becomes invisible
- Do not skip the validation-set checkpoint selection — test results will not
  reflect the best model
- Do not add supervised loss terms alongside the RL objective — this project
  is pure RL by design
- Do not enforce a linear delivery pace in the reward — the agent must be free
  to discover any trajectory that reaches the prescription by fraction 35
- Do not compare agent beam weights to ground truth beam weights — no such
  ground truth exists in OpenKBP; the only valid comparison is dose vs. dose
- Do not hardcode reward weights in src/env/reward.py — always load LAMBDA_OAR,
  LAMBDA_SHAPING, and dvh_bonus_scale from config/default.yaml
- Do not skip the reward weight tuning protocol before final experiments — the
  initial values are starting points, not final values
- Do not compute PTV coverage over the raw ptv_mask alone — always intersect
  with possible_dose_mask first, or the agent is penalised for voxels it can
  never physically reach regardless of its actions
- Do not skip applying possible_dose_mask after DIM multiplication — the mask
  must be enforced as a hard constraint in compute_dose(), not left to chance
- Do not resample volumes without reading voxel_dimensions.csv first — the
  native voxel grid differs per patient and incorrect spacing corrupts the DIM
- Do not assume all 10 structure channels are non-zero — always check
  present_structures.json before including an OAR in the reward computation

---

## Quick reference: run commands

```bash
# Step 1 — preprocess data (run once)
python scripts/preprocess_all.py --data-dir data/raw --out-dir data/processed

# Step 2 — compute DIMs (run once, GPU recommended)
python scripts/compute_dims.py --processed-dir data/processed --patients train

# Step 3 — sanity check environment (one episode, manual)
python scripts/sanity_check_env.py --patient-id patient_003

# Step 4 — train PPO
python src/training/train_ppo.py --config config/default.yaml

# Step 5 — train SAC baseline
python src/training/train_sac.py --config config/sac.yaml

# Step 6 — evaluate best checkpoint on test set
python src/evaluation/evaluate.py --checkpoint runs/ppo/best_model.zip \
    --patients data/processed/test --out results/ppo_test.json

# Step 7 — plot DVH curves
python src/evaluation/plot_dvh.py --results results/ppo_test.json \
    --patient patient_042
```

---

*Last updated: project planning phase. Update this file whenever an architectural
decision changes. Claude Code should re-read CLAUDE.md at the start of every
session and flag any inconsistency between this file and the actual code.*

**Note on data documentation vs. reality:** The official OpenKBP data description
PDF refers to the voxel spacing file as `voxels.csv`, but the actual filename in
the repository is `voxel_dimensions.csv`. When any discrepancy exists between the
PDF documentation and the actual files in the repo, the actual files always take
precedence. Always verify filenames against the real patient folder before writing
any I/O code.
