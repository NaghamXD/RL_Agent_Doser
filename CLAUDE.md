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
│       │   ├── ct.npy               ← shape (64, 64, 64), float32, normalised HU
│       │   ├── masks.npy            ← shape (10, 64, 64, 64), binary float32
│       │   ├── dose_gt.npy          ← shape (64, 64, 64), float32, Gy
│       │   └── dim.npy              ← shape (262144, 9), float32  [64^3 × 9 beams]
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

Each patient folder contains CSV files:
- `ct.csv` — CT voxel values (Hounsfield Units)
- `possible_dose_mask.csv` — feasible dose region
- `structure_masks/` — one CSV per structure (PTV_70, PTV_63, PTV_56, brainstem,
  spinal_cord, right_parotid, left_parotid, larynx, esophagus, mandible)
- `dose.csv` — ground-truth clinical dose distribution (Gy)

### Pre-processing pipeline

Run `scripts/preprocess_all.py` once before any training. It:
1. Reads each patient's CSV files via `src/data/loader.py`
2. Resamples all volumes from native resolution to (64, 64, 64) via SimpleITK
3. Clips CT to [−1000, 3000] HU then normalises to [−1, 1]
4. Stacks masks into a single (10, 64, 64, 64) binary array in this fixed order:
   `[PTV_70, PTV_63, PTV_56, brainstem, spinal_cord, right_parotid,
     left_parotid, larynx, esophagus, mandible]`
5. Saves ct.npy, masks.npy, dose_gt.npy to `data/processed/patient_XXX/`

Run `scripts/compute_dims.py` separately (GPU recommended, ~5 min/patient):
- Calls `PyRadPlanEngine` with the patient CT and masks
- Computes the DIM for 9 fixed gantry angles: [0, 40, 80, 120, 160, 200, 240, 280, 320] degrees
- Saves dim.npy of shape (262144, 9) — i.e. (64^3 voxels, 9 beams)
- Memory per DIM: ~1 GB float32. If memory is tight, save as float16 (~500 MB).

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
fractional_dose = dim @ action_vector      # shape: (262144,) → reshape to (64,64,64)
```

### Reward — R_t (dense, fraction-aware)

```python
def compute_reward(cum_dose, masks, t, n_fractions=35):
    progress = t / n_fractions          # fraction of treatment completed

    # 1. Tumor coverage (how much of PTV is on track)
    ptv_mask = (masks[0] + masks[1] + masks[2]).clip(0, 1).bool()
    ptv_dose = cum_dose[ptv_mask]
    ptv_prescription = prescription_map[ptv_mask]
    expected = progress * ptv_prescription
    coverage = (ptv_dose >= expected * 0.95).float().mean()   # 5% tolerance

    # 2. Underdose penalty (mean shortfall in PTV, normalised)
    shortfall = (expected - ptv_dose).clamp(min=0).mean() / 70.0

    # 3. OAR excess penalty (cumulative overdose relative to running tolerance)
    oar_penalty = 0.0
    for name, tol in OAR_TOLERANCES.items():
        idx = MASK_IDX[name]
        if masks[idx].sum() == 0:
            continue
        oar_dose = cum_dose[masks[idx].bool()].mean()
        expected_oar = progress * tol
        excess = (oar_dose - expected_oar * 1.05).clamp(min=0) / tol
        oar_penalty += excess

    reward = coverage - 0.4 * shortfall - 0.3 * oar_penalty
    return float(reward)
```

Terminal bonus at t = 35 (appended AFTER the step reward, not instead of it):
```python
terminal_bonus = compute_dvh_score(cum_dose, masks, dose_gt) * 2.0
```

The factor of 2.0 is intentional — the terminal signal must be large enough
to propagate meaningful gradients back through 35 discount steps at γ = 0.99.

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
    def __init__(self, ct: np.ndarray, masks: np.ndarray, angles: list):
        # Calls pyRadPlan to compute DIM — runs once, ~5 min per patient
        self.dim = self._compute_dim(ct, masks, angles)   # shape (V, 9)
        self.shape = (64, 64, 64)

    def compute_dose(self, action: np.ndarray) -> np.ndarray:
        # action: (9,) float32 in [0, 1]
        # returns: (64, 64, 64) float32 dose map in Gy
        flat_dose = self.dim @ action
        return flat_dose.reshape(self.shape)
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
```

`ent_coef: 0.01` is mandatory. Setting it to 0 causes policy collapse in early
training because the flat reward landscape provides no gradient signal until the
agent accidentally discovers a good beam combination.

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
| Reward stuck near −0.3 | Entropy collapse — policy outputting same action | Increase `ent_coef` to 0.05 |
| OAR penalty dominates | λ weights too large | Reduce OAR coefficient from 0.3 to 0.1 |
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
