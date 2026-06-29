# Technical Summary — RL Agent for Fractionated Radiotherapy Planning

Source material for the final report (Intro to RL course, final project). This
document is a factual inventory of the **repository as it currently stands**
(branch `mps-fixes-and-gamma-sweep`), gathered by reading the code, configs,
logs, checkpoints, and git history directly — nothing here is guessed.

**Read this note before writing the report.** The codebase was originally
built around a *contextual-bandit* formulation (each fraction = its own
1-step PPO episode/update) and was later redesigned into a *sequential
35-step MDP* (one episode = one full patient course). The refactor kept the
old bandit code path alive behind a config flag (`sequential: false`) for
regression/reference purposes, and several docstrings/comments still describe
the old design verbatim. **§1 and §13 flag every place this matters.** The
report should describe the **new sequential-MDP design**, since that is the
only thing `configs/default.yaml` actually runs and the only thing every
logged experiment in `runs/` was produced with.

---

## 1. Old design vs. new design — read this first

| | **OLD (legacy "bandit") — `sequential: false`** | **NEW (active) — `sequential: true`, the actual default** |
|---|---|---|
| Episode unit | One fraction = one complete 1-step PPO episode (`done=True` every `step()`) | One **35-fraction patient course** = one episode (`done=True` only on the last fraction) |
| Reward | `reward = -λ_oar·oar_penalty(fraction_dose) + λ_ptv·fractional_ptv_reward(...)`, both evaluated against **per-fraction** targets/limits, no temporal structure | Dense **potential-based shaping** `γ·Φ(s′) − Φ(s)` every fraction **plus** a one-time **terminal** whole-course objective on the last fraction |
| Role of `gamma` / `gae_lambda` | Inert — GAE collapses to `advantage = reward − value` because every step is terminal | **Live** — real GAE bootstraps return-to-go across the 35-step horizon |
| PPO update granularity | Originally (pre-refactor, see `train.py`'s own module docstring) intended **one PPO update per fraction-level rollout**, with `total_episodes` counting those fraction-level rollouts | One PPO update every `batch_n_patients` (8, default) full patient trajectories; `total_episodes` counts **patients** |
| Status in this repo | Reward math (`reward.py`'s `oar_penalty`, `fractional_ptv_reward`) and the `DoseEnv.step` branch are still implemented and reachable via the config flag — kept as "a regression baseline" (README's words) | The only path used by `configs/default.yaml` and by **every** sweep / experiment described in this document |

**Where the old design still leaves fingerprints in the code (do not let
these mislead the report):**
- `train.py`'s top-of-file module docstring and `collect_patient`'s docstring
  still say *"Each fraction is a 1-step PPO episode... one PPO update per
  patient"* and *"cfg.total_episodes therefore counts patients trained on,
  not 35-step rollouts as in the previous design"* — that last clause is the
  explicit admission of the old design. The **code below the docstring**
  already does the new batched-sequential update (`batch_n_patients=8`); the
  docstring is simply stale relative to the current default config.
- `src/env/reward.py` keeps `progress_shaping`, `terminal_bonus`, `dvh_bonus`,
  `coverage` (non-soft) explicitly labelled `.. note:: legacy` — these are
  dead code for the active path, retained only for metrics/evaluation reuse.
- `Config.best_rolling_window` / the rolling-mean reward criterion in
  `train.py` is a leftover from when there was no validation-DVH criterion;
  it now only activates as a fallback if no validation split is found.
- The project's git history is **not** incremental on this point — the
  earliest commit in this repo (`ee3566c "first commit"`, 2026-06-22) already
  contains *both* code paths side by side (the dual-mode design predates the
  available git history; the true single-fraction-online-update version is
  only attested in comments, not in a separate commit).

---

## 2. Project framing (maps to the course's required report sections)

- **Course**: "Introduction to Reinforcement Learning", final project
  (`final_project_instructions.pdf`). Required report sections: Project
  Overview, Problem Formulation (states/actions/rewards/dynamics),
  Methodology (algorithm + implementation + hyperparameter tuning),
  Results (quantitative + qualitative), Discussion (limitations), Conclusion.
  8–20 pages, 1.5 spacing, Arial 12, 1" margins.
- **Domain**: IMRT (Intensity-Modulated Radiation Therapy) treatment planning
  on the public **OpenKBP** head-and-neck cancer dataset
  (https://github.com/ababier/open-kbp). Real anatomical CT scans, organ
  contours, and ground-truth clinically-optimized dose plans for 340 patients
  (`train`: 200, `validation`: 40, `test`: 100 — counts confirmed from
  `data/original/{split}`).
- **Task being learned**: per-fraction beamlet-intensity allocation across a
  35-fraction treatment course, balancing tumour (PTV) dose coverage against
  organ-at-risk (OAR) sparing, evaluated against the planner's real
  ground-truth dose distribution.

---

## 3. Repository map

```
configs/default.yaml          single source of truth for every hyperparameter
src/config.py                 Config dataclass (defaults) + YAML loader + device resolution
src/data/loader.py             OpenKBP sparse-CSV -> dense numpy volumes
src/data/preprocess.py         SimpleITK resample to grid^3, CT normalisation, beam-path channel
src/env/dose_env.py            DoseEnv: state/action/transition, both reward modes
src/env/reward.py              all reward math (bandit + sequential + legacy/metrics-only helpers)
src/models/encoder.py          3-D CNN encoder (Conv3d stack)
src/models/actor_critic.py     Gaussian actor (softplus-squashed) + critic heads
src/agents/ppo.py              PPO: act(), GAE, update_batch(), pretrain_actor() warm-start, save/load
src/utils/metrics.py           DVH/D95/MAE/OAR-mean utilities
src/utils/early_stopping.py    PlateauMonitor (validation-DVH plateau detector)
src/utils/train_logger.py      CSV episode logger (runs/train_log.csv)
src/utils/visualize.py         all matplotlib chart generation (DIM diagnostics, eval charts, trajectory charts)
train.py                       training entrypoint (warm-start -> curriculum -> PPO loop -> best.pt/early-stop)
evaluate.py                    deterministic evaluation + clinical metrics table + chart generation
scripts/preprocess.py          CLI for src/data/preprocess.py
scripts/compute_dose_influence_matrix.py   pyRadPlan dose-influence matrix per patient (+ 6 diagnostic PNGs)
scripts/compute_warmstart_actions.py       FISTA-NNLS per-patient warm-start beamlet plan a*
scripts/sweep_gamma.py         gamma ablation harness (shared warm-start, branches train.py)
scripts/sweep_hparams.py       4-regime hyperparameter-bundle sweep harness
scripts/sweep_beam_paths.py    beam-path state-channel ablation (on/off)
scripts/sweep_reward_shaping.py 5-variant reward-shaping ablation (C / A / B improvements)
scripts/visualize_patient_trajectory.py    per-fraction trajectory diagnostics for trained policies
scripts/benchmark_device.py    CPU vs MPS vs CUDA timing harness (synthetic data)
scripts/mps_warmstart_retry.py isolated MPS-NaN-bug investigation (does not touch production code)
scripts/inspect_numpy_arrays.py quick sanity dump of cached .npy files
runs/                          all checkpoints, CSV logs, sweep outputs (gitignored)
reports/                       reward-shaping ablation writeup + summary CSV + comparison plot
app.py, dashboard.html, *.md (DELIVERABLES/QUICK_START/INDEX/START_HERE/PROJECT_SUMMARY/UI_README)
                                a separate Flask + React demo dashboard for live-simulating the agent;
                                not part of the RL training/evaluation pipeline itself (see §12)
```

---

## 4. Problem formulation (the active, `sequential: true` design)

### State space
`DoseEnv._build_state()` returns a dense tensor of shape `(C, 64, 64, 64)`
(`cfg.grid = 64`), `C = 12` by default:

| Channel(s) | Content |
|---|---|
| 0 | Normalised CT (HU → `[-1, 1]`) |
| 1–8 | Binary structure masks, fixed order: `PTV70, PTV63, PTV56, Brainstem, SpinalCord, Mandible, LeftParotid, RightParotid` (`src/config.py: ALL_STRUCTURES`) |
| 9 | `cumulative_dose / 70` Gy — running dose delivered so far this patient course |
| 10 | Per-voxel **PTV dose gap**: `clip(prescription − cumulative_dose, 0) / 70` |
| 11 | Beam-path channel (coplanar ray-cast fan for 9 gantry angles); **togglable** via `cfg.include_beam_paths` (ablation, see §9.3) — drops `C` to 11 when off |

Plus a scalar **`fraction_index / 35`** ("fraction progress") concatenated
onto the CNN latent before the actor/critic heads (`ActorCritic.features`).
`cumulative_dose` and `fraction_index` make the state **Markov** across the
35-fraction course without needing recurrence — this is explicitly called
out in `README.md` and is why a feed-forward (not recurrent) policy is used.

### Action space
`(n_beams=9, beamlet_h=16, beamlet_w=16)` = **2304-dim continuous vector**
of non-negative beamlet intensities. The actor outputs a diagonal Gaussian
over *raw* logits; the env-facing action is
`softplus(raw_sample) * ACTION_SCALE` (`ActorCritic.to_action`,
`ACTION_SCALE = 1.0`) — guarantees non-negativity without a hard clip.
`fraction_dose = dose_influence_matrix @ action` (a sparse `(64³, 2304)`
matrix specific to each patient, see §5.3), reshaped to `(64,64,64)` Gy.

### Reward (active = sequential mode; `src/env/reward.py`)
Per fraction `t`:
```
Φ(cumulative_dose) = −λ_phi · ptv_gap_fraction(cumulative_dose)
                      − λ_oar · oar_overshoot_fraction(cumulative_dose)
reward_t = γ · Φ(s_{t+1}) − Φ(s_t)                      # dense potential-based shaping
         + [terminal_reward   if t == last fraction]
terminal_reward = λ_ptv · mean_PTV_coverage − terminal_dvh_weight · dvh_score
```
- Potential-based shaping (Ng, Harada & Russell, 1999) is **policy-invariant**
  — it speeds learning without changing the optimal policy — and is computed
  on the *cumulative* dose, so it's a function of the Markov state only.
- `ptv_gap_fraction`: mean over PTV voxels of `clip(Rx − dose, 0)/Rx` (raised
  to `ptv_gap_power`, default 1.0 = linear; ablation "Improvement A" tests
  2.0 = quadratic, see §9.4).
- `oar_overshoot_fraction`: weighted sum over OARs of
  `max(0, mean_dose/tol − 1)` (hard threshold by default; ablation
  "Improvement B" tests a soft exponential barrier, see §9.4).
- `terminal_reward` uses a **hard** `dose ≥ Rx` coverage step by default
  (ablation "Improvement C" tests a sigmoid/soft coverage, see §9.4) minus a
  cheap DVH-score proxy (`utils/metrics.dvh_score`: mean `|D95 + Dmean|`
  diff vs. ground truth per structure).
- A hard numerical safety cap (`_EXCESS_FRACTION_CLIP = 5.0`,
  `_OAR_BARRIER_CLIP = 25.0`) prevents a single wildly over-dosed voxel
  during early random rollouts from producing reward magnitudes the critic
  can't fit (documented cause of a real NaN incident — see §13).

### Transition dynamics
- `fraction_dose = (DIM @ action) * possible_dose_mask`, optionally clipped to
  `max_fraction_dose` (3.0 Gy/voxel by default) — this physical cap is what
  *forces* the agent to spread dose across all 35 fractions instead of
  dumping the whole course in fraction 1; without it the MDP degenerates
  back toward a one-shot allocation problem.
- `cumulative_dose ← cumulative_dose + fraction_dose` (persists across the
  whole patient course; reset to zero only in `DoseEnv.reset()`, i.e. once
  per patient, not once per fraction).
- Episode terminates (`done=True`) exactly when `fraction_index == 35`.

---

## 5. Data pipeline

1. **Raw data** (`data/original/{train,validation,test}/pt_*/*.csv`): OpenKBP
   sparse `(flat_index, value)` CSVs for CT, ground-truth dose, structure
   masks, `possible_dose_mask`, and `voxel_dimensions.csv`. Loaded by
   `src/data/loader.load_patient` into dense `(NZ, 128, 128)` volumes
   (`NZ` inferred per patient from the max flat index).
2. **Preprocessing** (`scripts/preprocess.py` → `src/data/preprocess.py`):
   SimpleITK-resamples every volume to a fixed `64³` cube respecting
   real-world mm spacing (linear interpolation for CT/dose, nearest-neighbour
   for masks), normalises CT to `[-1, 1]`, and casts a simple coplanar
   ray-cast beam-path map (`ray_cast_beam_paths`). Cached as `.npy` per
   patient under `data/processed/<split>/<pt>/`.
3. **Dose-Influence Matrix (DIM)** (`scripts/compute_dose_influence_matrix.py`):
   uses **pyRadPlan** to run a real photon-IMRT SVDPB pencil-beam dose
   calculation at native CT resolution for 9 coplanar gantry angles, bins
   native bixels into a `16×16` BEV fluence grid per beam, then mean-pools
   down to the `64³` training grid via a custom row-stochastic resampling
   matrix (`_downsample_matrix`) that fixes a historical bug where naive
   integer-floor downsampling left ~25% of PTV voxels permanently
   unreachable. Result: a sparse `(64³, 2304)` CSR matrix per patient
   (`dose_influence_matrix.npz`), plus 6 diagnostic PNGs and a `.log` per
   patient under `data/processed/<pt>/charts/`.
   - **Known limitation** (documented in the script and in
     `compute_warmstart_actions.py`): the DIM only covers **~70–85% of body
     voxels** per patient — a structural ceiling on achievable D95 that the
     policy cannot exceed regardless of training quality.
4. **Warm-start actions** (`scripts/compute_warmstart_actions.py`): solves a
   structure-weighted non-negative least squares problem
   `a* = argmin_{a≥0} ||√W (DIM·a − dose_gt/35)||²` via FISTA (400 iters,
   PTV voxels weighted 50×, OARs 5×, body 1×) per training patient. Cached
   as `warmstart_action.npy`, consumed only by `train.py`'s actor-pretraining
   step (train split only — validation/test warm-starts are never read
   during training, only for an optional evaluation chart overlay).

---

## 6. Model architecture

`src/models/encoder.py` — **`CNNEncoder3D`**: 4×`Conv3d` (stride 2, kernel 3,
channels 16→32→64→128) each followed by `ReLU(inplace=False)` — the
`inplace=False` choice is deliberate, not stylistic (see §13) — then
`AdaptiveAvgPool3d(1)` and a `Linear(128, 128)` projection. `64³` input
collapses through `32³→16³→8³→4³` feature maps to a 128-d latent.

`src/models/actor_critic.py` — **`ActorCritic`**:
- `features()` = CNN latent (128) ‖ `fraction_progress` scalar → 129-d.
- **Actor**: 2×`Linear(256)` + ReLU trunk → `Linear(256, 2304)` mean head
  (orthogonal-init gain 0.01, bias initialised to `−1.0` so the *initial*
  policy already delivers a non-trivial fractional dose instead of getting
  stuck at a "do nothing" local optimum) + a **state-independent**
  `log_std` parameter (diagonal Gaussian, standard PPO continuous-control
  recipe), clamped to `[−5, actor_log_std_max]` every forward pass.
- **Critic**: same-shape 2-layer trunk → scalar value head (orthogonal
  init gain 1.0, no squashing).
- All weight inits follow the Engstrom et al. (2020) / Stable-Baselines3 /
  CleanRL orthogonal-init recipe (`gain=√2` before ReLU, `gain=1.0` for the
  unsquashed value head, `gain=0.01` for the policy-mean head).
- On **MPS only**, the encoder + actor-trunk layers get a dampened
  orthogonal-init gain (`0.1` instead of `√2`) — a targeted workaround for a
  device-specific instability (§13), CPU/CUDA are untouched.

---

## 7. RL algorithm — PPO with GAE (`src/agents/ppo.py`)

- **Clipped-surrogate PPO** (Schulman et al., 2017) with a **clipped value
  loss** (PPO2-style, `value_clip = clip_eps`) and global gradient-norm
  clipping (`0.5`); skips the optimizer step entirely (without zeroing
  Adam's moment estimates incorrectly) if any gradient is non-finite, rather
  than letting a NaN propagate through Adam state forever.
- **GAE** (`_compute_gae`): standard recursion
  `δ_t = r_t + γ·V(s_{t+1})·(1−done_t) − V(s_t)`,
  `A_t = δ_t + γ·λ·(1−done_t)·A_{t+1}`. In sequential mode `done` is `True`
  only on fraction 35, so this is **real, live temporal credit assignment**
  over the 35-step horizon; in legacy bandit mode every `done=True` collapses
  it to `A_t = r_t − V(s_t)` (gamma inert) — see §1.
- **Batched update** (`update_batch`): collects `batch_n_patients` (8) full
  patient trajectories, computes GAE **per trajectory** (each with its own
  terminal bootstrap `last_value=0.0`, correct because `done=True` at the
  end of every trajectory), concatenates along time, then normalises
  advantages **once across the whole batch** (lower variance than per-trajectory
  normalisation at only 35 steps/trajectory) before the clipped-PPO
  minibatch-epoch loop (`ppo_epochs=6` epochs over `minibatch=64`-sized
  shuffled chunks of the `8×35=280`-transition batch).
- **Supervised actor warm-start** (`pretrain_actor`): before any PPO update,
  MSE/Huber-regresses the actor's deterministic mean output toward
  `inv_softplus(a*/scale)` for every training patient with a cached
  warm-start action — breaks the symmetric "every beamlet ≈ `softplus(−1)`"
  initial policy that PPO otherwise can't escape from in a reasonable number
  of episodes. Only the encoder + actor-trunk + actor-mu are updated; the
  critic and `log_std` are left untouched so PPO can still explore around
  the warm-started mean.
- **Checkpointing**: `save()`/`load()` persist `net` + optimizer state, plus
  `episode_index` / `best_val_dvh` when given, enabling resume-safe
  continuation of both the OAR curriculum and the best.pt comparison.

---

## 8. Training loop (`train.py`)

```
load_config → seed (np + torch) → build DoseEnv(train split)
→ peek one state to infer in_channels → build PPO agent
→ [optional --resume: restore net/optimizer/episode_index/best_val_dvh]
→ [optional warm-start: pretrain_actor on cached a*, save warmstart.pt]
→ build a second DoseEnv(validation split) for best.pt scoring
→ for episode_index in range(start_episode, start_episode + total_episodes):
      env.lambda_oar ← ramped value (OAR curriculum, see below)
      rollout, total_reward ← collect_patient(env, agent)   # 35-fraction trajectory
      buffer.append(rollout)
      if len(buffer) >= batch_n_patients or last episode:
          stats ← agent.update_batch(buffer);  buffer ← []
      every best_eval_every (25) episodes:
          val_dvh ← deterministic rollout over best_n_val_patients (40) validation patients
          if val_dvh improved → save best.pt
          plateau_monitor.update(...) → may set early_stopped=True
      every eval_every (50) episodes: save a numbered checkpoint epN.pt
      log everything to train_log.csv
→ save last.pt; final validation pass (may still update best.pt)
```

**OAR-weight curriculum** (`_effective_lambda_oar`): `env.lambda_oar` ramps
linearly from `lambda_oar * lambda_oar_ramp_start_factor` (0.10× by default)
up to the full `lambda_oar` over the first `lambda_oar_ramp_episodes` (600)
*patients*. Rationale: at full OAR weight from episode 0, the marginal OAR
penalty exceeds the marginal PTV reward at initialisation and PPO converges
to a near-zero-dose policy; ramping lets the agent learn to actually deliver
dose first.

**best.pt selection**: by **validation DVH score** (lower better), not
training reward — the training-reward rolling mean was found to favour an
early near-zero-dose policy simply because its OAR penalty was small too.
Validation set size for this criterion was increased twice during the
project (3 → 6 → 40 patients) after each smaller sample size was shown to be
noisy enough to lock in a worse checkpoint (see the dated comments in
`configs/default.yaml`).

**Early stopping** (`PlateauMonitor`, opt-in via `early_stop_patience_evals`,
**8** in `configs/default.yaml` — i.e. **enabled** by default in this
project, even though the `Config` dataclass's own fallback default is `0`/
disabled): stops once `val_dvh` shows no `>0.5%` relative improvement for 8
consecutive validation checks, but **only after** the OAR-curriculum ramp has
fully completed (so the curriculum's own transient dip during ramp-up isn't
mistaken for a plateau). Motivated by a real run that peaked at episode 99
and then continued to episode 3499 while the validation DVH score degraded
from 22.67 to 41.23 — see §10.1 and §13.

---

## 9. Hyperparameters (`configs/default.yaml`, with rationale comments preserved in the file)

### 9.1 Geometry / data
`grid=64`, `n_fractions=35`, `n_beams=9`, `beamlet_h=beamlet_w=16` →
`n_beamlets = 2304`.

### 9.2 Clinical targets
| | Gy (full course) |
|---|---|
| PTV70 / PTV63 / PTV56 | 70 / 63 / 56 |
| Brainstem tol (serial) | 54 |
| SpinalCord tol (serial) | 45 |
| Mandible tol | 70 |
| Left/Right Parotid tol | 26 each |

OAR per-organ weights (inside the penalty, not the tolerance):
`Brainstem 0.5, SpinalCord 0.6, Mandible 0.5, LeftParotid 1.6, RightParotid 1.6`
— iteratively raised on the parotids ("Iteration-2" comment, 2026-06-20)
after they remained ~90–110% over tolerance while serial OARs had slack.

### 9.3 Reward weights (production values)
`lambda_oar=0.6`, `lambda_ptv=3.5`, `oar_voxel_subweight=1.0`,
`oar_mean_subweight=0.35`, `oar_dmax_subweight=1.0`,
`lambda_phi=1.0`, `terminal_dvh_weight=0.1`, `max_fraction_dose=3.0`,
`batch_n_patients=8`.

### 9.4 Reward-shaping ablation knobs (all default to "old"/off; see §10.4)
`ptv_gap_power=1.0` (A: 2.0=quadratic), `oar_barrier_steepness=null`
(B: e.g. 2.0=soft exponential barrier), `terminal_use_soft_coverage=false`
(C: true=sigmoid coverage).

### 9.5 Actor exploration
`actor_log_std_init=0.0` (σ=1.0) — deliberately **high**, not the
architecture's own fallback default of `−1.0`, because the actor mean is
warm-started onto a static one-shot plan and needs to actively explore away
from it; `actor_log_std_max=2.0` (σ ≤ e²≈7.4 clamp ceiling).

### 9.6 PPO
`gamma=0.99` (live in sequential mode), `gae_lambda=0.95`, `clip_eps=0.2`,
`ent_coef=0.005`, `vf_coef=0.5`, `lr=1.5e-4`, `ppo_epochs=6`, `minibatch=64`.

### 9.7 Training schedule
`total_episodes=3500` (200 train patients × ~17.5 visits via random sampling),
`eval_every=50`, `seed=42`, **`device=cpu`** (pinned, not "auto" — see §13).

### 9.8 Warm-start
`warmstart_enabled=true`, `warmstart_epochs=30`, `warmstart_minibatch=4`,
`warmstart_lr=3e-4`.

### 9.9 OAR curriculum / best.pt / early stopping
`lambda_oar_ramp_episodes=600`, `lambda_oar_ramp_start_factor=0.10`,
`best_n_val_patients=40`, `best_eval_every=25`, `early_stop_patience_evals=8`,
`early_stop_min_delta=0.005`.

### 9.10 Hyperparameter-tuning process actually performed
Two structured sweeps beyond manual tuning of the values above:
- **Gamma sweep** (`scripts/sweep_gamma.py`): `{0.95, 0.97, 0.99, 0.995, 0.999}`,
  branched from one shared warm-start checkpoint. Compared on validation DVH
  *because* gamma is baked directly into the reward landscape itself in
  sequential mode (it appears inside the shaping term, not just in GAE), so
  raw training reward isn't comparable across gamma values.
- **4-regime hyperparameter bundle sweep** (`scripts/sweep_hparams.py`):
  named presets bundling `batch_n_patients`, `minibatch`, `ppo_epochs`, `lr`,
  `lambda_oar_ramp_episodes` together (a coarse "preset" comparison, not a
  one-variable-at-a-time ablation) — see results in §10.2.

---

## 10. Experimental results actually produced

All numbers below are read directly from `runs/*/train_log.csv`,
`runs/*/eval_results.log`, and `reports/`. "val_dvh" = the training-time
deterministic-rollout metric (`train._validation_dvh_score`, used for
`best.pt` selection); "DVH"/clinical columns = `evaluate.py`'s post-hoc
metrics table on `best.pt` against the full 40-patient validation split.
Lower is better for every DVH/MAE number.

### 10.1 Main production run (`runs/`, gamma=0.99, the production `configs/default.yaml`)
- 200 train patients, 3500 patient-episodes, CPU, sequential mode,
  8-patient batches.
- `best.pt`: **episode 99**, val_dvh **22.67** (verified directly from the
  saved checkpoint's `best_val_dvh` metadata).
- `last.pt` (episode 3499): val_dvh history shows **monotonic-ish
  degradation** after the early peak — 22.67 (ep99) → 30+ by ep1700 →
  41–44 by ep3300–3499, with policy entropy in the log climbing into the
  ~1e5 range late in training (`runs/train_log.csv` columns `entropy`,
  `policy_loss`). This run **predates** the early-stopping feature (see git
  history, §13) and is the concrete motivating case written into
  `src/utils/early_stopping.py`'s own docstring.
- This run is also reused as the gamma=0.99 candidate and as the
  `sweep_hparams` "baseline" regime (the script's `_reuse_historical_baseline`
  copies `runs/train_log.csv` instead of re-training) — so its 41.23
  final-DVH collapse is the same data point surfacing in three places, not
  three independent confirmations.

### 10.2 Gamma sweep (`runs/sweep_gamma/summary.csv`, 900 episodes/candidate)
| gamma | best_val_dvh | final_val_dvh |
|---|---|---|
| **0.99** | **22.67** | 26.57 |
| 0.97 | 22.79 | 32.00 |
| 0.999 | 23.88 | 29.43 |
| 0.95 | 26.06 | 35.92 |
| 0.995 | 27.65 | 29.80 |

`gamma=0.99` (the production default) wins on best-checkpoint quality;
0.97 is close behind. No monotonic trend with gamma — 0.999 and 0.995
underperform both their neighbours, suggesting the result is noisy /
regime-sensitive rather than a clean optimum.

### 10.3 Hyperparameter-bundle sweep — two separate runs of differing length

**(a) `runs/sweep_hparams/` — CPU, script-driven, 600 episodes/regime**
(except `baseline`, which reuses the 3500-episode historical run, §10.1):
| regime | best_val_dvh | final_val_dvh |
|---|---|---|
| baseline (reused, 3500 ep) | 22.67 | 41.23 |
| sweep1_minibatch_active | 23.73 | 31.75 |
| sweep3_aggressive_adaptivity | 23.75 | 26.57 |
| sweep2_heavy_reg | 26.61 | 26.82 |

**(b) `runs/sweep_hparams_mps/` — MPS/GPU, manual full-length reruns (3500
episodes each) after early stopping was implemented** (commit `01b655a`):
| regime | best_val_dvh | episode stopped | outcome |
|---|---|---|---|
| baseline | 26.63 | **800 (early-stopped)** | clean stop, no collapse |
| sweep1_minibatch_active | 24.23 | 3500 (ran full length) | **catastrophic late collapse**: val_dvh climbs from ~24 (ep~1024) to 66.6 by ep3499 |
| **sweep2_heavy_reg** | **23.34** | 3500 (ran full length) | **stable** — ends at 25.15, no collapse |
| sweep3_aggressive_adaptivity | 25.21 | 3500 (ran full length) | **catastrophic late collapse**: val_dvh climbs from ~26 (ep~774) to 122 by ep3499 |

Key finding: **`sweep2_heavy_reg`** (larger minibatch=128, fewer PPO epochs=3,
lower lr=5e-5, slower OAR ramp=300 episodes) is the only non-baseline regime
that trains *stably* across the full 3500 episodes without late-training
divergence, and has the best `best_val_dvh` of all four GPU reruns. It was
selected as the **`control`** regime for the reward-shaping ablation (§10.4).
Early stopping did **not** catch the sweep1/sweep3 collapses — the plateau
monitor's relative-improvement check keeps finding marginal new bests amid
the noisy trajectory before the real divergence sets in, so patience never
accumulates 8 consecutive non-improving checks until after the collapse is
already severe (or training reaches `total_episodes` first). This is a real
limitation, not a configuration error — see §13.

### 10.4 Reward-shaping ablation (`reports/reward_shaping_sweep.md`,
`scripts/sweep_reward_shaping.py`), layered C → C+A → C+A(retuned) → C+A+B,
then a second round fixing a flaw found in B, on top of `sweep2_heavy_reg`,
evaluated via `evaluate.py` on the full 40-patient validation split:

| variant | DVH | best ep | D95_PTV70 | D95_PTV63 | D95_PTV56 | Brainstem(54) | SpinalCord(45) | Mandible(70) | LeftParotid(26) | RightParotid(26) |
|---|---|---|---|---|---|---|---|---|---|---|
| **control (no changes)** | **23.34** | 1699 | 43.97 | 39.31 | 38.92 | 19.65 | 24.71 | 41.07 | 39.32 | 43.21 |
| C (soft terminal coverage) | 26.82 | 500 | 33.76 | 28.08 | 29.00 | 10.09 | 21.03 | 29.53 | 31.12 | 31.44 |
| C+A (+ quadratic PTV gap, λ_phi unchanged) | 32.08 | 500 | 27.70 | 23.16 | 23.00 | 9.39 | 15.23 | 22.66 | 20.70 | 24.41 |
| C+A retuned (λ_phi=2.0) | 26.82 | 650 | 33.09 | 29.33 | 28.33 | 11.93 | 20.29 | 28.52 | 30.46 | 33.27 |
| C+A+B (+ soft OAR barrier, steepness=2.0) | 25.04 | 925 | 35.29 | 29.54 | 32.86 | 15.84 | 29.04 | 34.72 | 40.14 | 39.08 |
| C+A+B v2_barrier (+ activation-threshold floor fix, k=1.5) | 29.40 | 399 | 31.83 | 25.20 | 26.89 | 11.35 | 23.51 | 32.12 | 39.27 | 39.62 |
| C+A+B v2_full (+ terminal_dvh_weight 0.4) | 31.09 | 274 | 26.64 | 21.51 | 22.74 | 9.17 | 18.48 | 25.81 | 32.81 | 30.85 |
| v3_gentle_terminal (terminal_dvh_weight 0.2, full 2000-ep run, no cutoff) | 29.55 | 1949 | 22.87 | 18.61 | 18.31 | 11.55 | 27.12 | 33.58 | 37.28 | 34.16 |
| v3_rebalanced_oar (+ λ_oar 0.4, full 2000-ep run, no cutoff) | 26.91 | 174 (then collapsed) | 33.89 | 27.78 | 28.37 | 10.41 | 21.84 | 31.63 | 32.23 | 35.31 |
| **v3_oar_only (λ_oar 0.4 alone, terminal_dvh_weight back to 0.1)** | **26.22** | 249 | 35.22 | 30.02 | 29.77 | 10.93 | 23.26 | 30.58 | 34.45 | 38.44 |

**Verdict** (full detail in `reports/reward_shaping_sweep.md`): **none of
the 9 shaping variants beat the unmodified control.** Round 1: C alone
over-trades PTV coverage for OAR sparing. C+A unretuned doesn't converge
at all (squaring the gap shrinks Φ's magnitude without compensating
`lambda_phi`). C+A retuned recovers stability but no net improvement.
C+A+B was the best of round 1 (still improving when early-stopped at
ep925) but doesn't fix the parotid overdose it targeted and worsens
SpinalCord dose vs. control. **Round 2** traced this to a real flaw: B's
soft barrier `exp(steepness·excess)` has no floor (`exp(−2.0)=0.135` even
for an organ at zero dose), so `lambda_oar` fights a constant tax
everywhere rather than pressure concentrated near tolerance — visible in
`pt_201`'s per-fraction trajectory as an accelerating reward decline
(-1.55/fraction by fraction 35) despite a near-zero terminal reward
(+0.06, coverage was "paid for" by unchecked OAR dose). The
`oar_barrier_activation_threshold` fix (zero penalty below 80% of
tolerance) is **mechanistically validated twice** — it kills the
acceleration and returns delivered-dose intensity to the same scale every
other regime uses — but `terminal_dvh_weight` increases consistently hurt
(0.1→25.04, 0.2→29.55, 0.4→31.09, confirmed not an early-stopping artifact
since `v3_gentle_terminal` used nearly its full uncapped budget), while
softening `lambda_oar` helped: isolated cleanly in `v3_oar_only`, it's the
best of all 9 variants (26.22) — still short of `control`. **Recommendation**:
keep `control`; this investigation is closed out — no configuration tested
anywhere in the project (9 reward-shaping variants, 4 hyperparameter
regimes) ever got parotid dose under its 26 Gy tolerance, so this has been
redistributing the same coverage/OAR trade-off rather than pushing it.

### 10.5 Beam-paths state-channel ablation (`runs/sweep_beam_paths/summary.csv`,
300 episodes each, independent warm-starts since the channel changes
`in_channels` / network shape):
| variant | best_val_dvh |
|---|---|
| **on** (`include_beam_paths=True`, 12 channels) | **26.95** |
| off (11 channels) | 29.78 |

The beam-path channel measurably helps (production default keeps it on).

### 10.6 Qualitative per-patient trajectory analysis (`scripts/visualize_patient_trajectory.py`,
run against `sweep_hparams_mps/sweep2_heavy_reg/best.pt`,
`runs/sweep_hparams_mps/sweep2_heavy_reg/trajectory_charts/trajectory_summary.csv`,
40 validation patients):
- **`beam_profile_drift`** (coefficient-of-variation of each beam's mean
  intensity *across the 35 fractions*) is **very low for every patient**
  (range ~0.01–0.05). This means the learned policy delivers an almost
  **static** beam-intensity profile every fraction and repeats it 35 times,
  rather than meaningfully re-balancing beam allocation over the course of
  treatment — a notable gap versus the sequential MDP's intended capability
  to adapt fraction-to-fraction (e.g. compensate an under-dosed early
  fraction later).
- **`dominant_beam`** alternates between beam 0 and beam 2 depending on
  patient anatomy (not a single global favourite beam), and
  **`top_beam_share`** (~0.14–0.19) shows fairly even spread across all 9
  beams rather than collapsing to one or two.
- **`cliff_fraction`** (the fraction index after which the dense shaping
  reward's sign permanently flips) varies patient-to-patient, roughly
  fraction 17–30 of 35 — i.e. the per-fraction reward is often net-negative
  for a meaningful trailing portion of the course.
- **`terminal_reward`** (the one-time end-of-course payout) is **negative
  for most patients** in this sample (e.g. −1.9, −2.5, −3.3 Gy-equivalent
  units), implying the `terminal_dvh_weight · dvh_score` penalty term
  usually outweighs the `lambda_ptv · coverage` credit at the policy's
  actual coverage/DVH operating point.

---

## 11. Evaluation pipeline (`evaluate.py`)

Deterministic (mean-action, no sampling) rollout over every patient in a
chosen split. Reports per-patient and mean: `MAE` (mean abs dose error in
body), `dvh_score` (proxy DVH, lower better), `D95_<PTV>` per target volume,
mean dose per OAR — plus a full config snapshot so every number is traceable
to the exact YAML that produced it. Optional `--charts-patient` generates 6
diagnostic figures per patient (`src/utils/visualize.generate_eval_charts`):
agent fluence maps, warm-start fluence overlay, DIM sensitivity, DVH
comparison (predicted vs. ground truth), 3-plane dose-on-CT slices,
per-fraction reward-component curve.

---

## 12. The web dashboard (`app.py`, `dashboard.html`, and the
`DELIVERABLES.md`/`QUICK_START.md`/`INDEX.md`/`START_HERE.md`/
`PROJECT_SUMMARY.md`/`UI_README.md` documents)

This is a **separate deliverable**: a Flask REST API (`app.py`) + a React
(CDN) single-page dashboard (`dashboard.html`) for interactively picking a
patient, running a simulated fraction, and visualising PTV/OAR/beam/reward
state in a browser. It is **not part of the RL training or evaluation
pipeline** described above and its accompanying docs do not describe the
sequential-vs-bandit distinction at all (they're scoped to the UI only) —
safe to mention as a demo/visualization artifact in the report but not as
part of the algorithm or results.

---

## 13. Engineering notes / known issues worth a Discussion-section paragraph

- **MPS (Apple Silicon GPU) instability**: orthogonal weight init at the
  standard gain reliably NaNs the actor's supervised warm-start within one
  epoch on MPS (confirmed reproducible across 7+ trials), traced to two
  independent causes: (1) `inplace=True` ReLUs corrupt the Conv3d backward
  pass on this PyTorch/MPS combination (fixed: all ReLUs are now
  `inplace=False`, verified to match the CPU loss curve); (2) the standard
  `√2`-gain orthogonal init still produces escalating gradients on MPS even
  after fix (1) (mitigated, not fully fixed: a dampened gain of `0.1` is
  applied only to MPS's encoder/actor-trunk layers). `device: cpu` is
  pinned as the production default specifically because of this, with the
  reasoning preserved verbatim in `configs/default.yaml`'s comments.
  `scripts/mps_warmstart_retry.py` is a dedicated, read-only investigation
  harness for this bug that does not touch production code.
- **Policy collapse over long runs**: the main 3500-episode run (§10.1) and
  two of the four MPS hyperparameter-sweep regimes (§10.3b) all show the
  validation DVH score improving early, peaking, then degrading severely
  (up to ~5× worse) over thousands of further episodes with entropy/`log_std`
  growing unchecked. `best.pt` is never at risk (saved independently the
  moment val_dvh improves), but compute is wasted and `last.pt` is
  misleading if used naively. Early stopping (`PlateauMonitor`) was added in
  direct response to this, but §10.3b shows it does **not** reliably catch
  every instance — the plateau-relative-improvement check can keep
  resetting on noisy intermediate "new bests" right up until the point of
  severe divergence.
- **DIM coverage ceiling**: the dose-influence matrix only reaches ~70–85%
  of body voxels per patient (16×16 BEV binning + integer-floor
  downsampling miss the rest) — D95 numbers in every result table are
  bounded above by this physical reachability limit, not purely by policy
  quality. (A downsample-matrix bug that left ~25% of *PTV* voxels
  permanently unreachable was found and fixed during the project — see the
  `_downsample_matrix` docstring in `compute_dose_influence_matrix.py` —
  current numbers reflect the fixed version.)
- **Reward-magnitude safety clipping**: both the legacy-bandit OAR penalty
  and the sequential soft-barrier shaping term include hard numeric clips
  (`_EXCESS_FRACTION_CLIP=5.0`, `_OAR_BARRIER_CLIP=25.0`) added specifically
  because earlier unclipped versions caused reward blow-ups → NaN in the
  actor head during early (near-random) rollouts.

---

## 14. Artifact inventory (for figures/appendix)

- `runs/train_log.png`, `runs/train_log.csv` — main run learning curve (raw data).
- `reports/reward_shaping_comparison.png`, `reports/reward_shaping_summary.csv`,
  `reports/reward_shaping_sweep.md` — reward-shaping ablation, ready to cite directly.
- `runs/sweep_gamma/comparison.png`, `runs/sweep_gamma/summary.csv`.
- `runs/sweep_hparams/comparison.png` (600-episode regimes) and
  `runs/sweep_hparams_mps/*/train_log.csv` (3500-episode GPU reruns, no
  consolidated comparison.png — was a manual run, not script-driven).
- `runs/sweep_beam_paths/comparison.png`, `summary.csv`.
- `runs/sweep_hparams_mps/sweep2_heavy_reg/trajectory_charts/` — per-patient
  beam-intensity heatmaps + reward trajectories + `trajectory_summary.csv`
  (40 patients) for the qualitative analysis in §10.6.
- `runs/sweep_hparams_mps/*/eval_charts/` (where generated) and
  `data/processed/<pt>/charts/` — per-patient DIM/DVH/fluence diagnostic PNGs.
- Checkpoints: `runs/{warmstart,best,last,epN}.pt` for the main run, mirrored
  per-regime under each `runs/sweep_*/<regime>/`.

---

## 15. Suggested mapping onto the required report sections

- **Project Overview** → §2.
- **Problem Formulation** → §4 (state/action/reward/dynamics), with §1's
  old-vs-new table as a "design evolution" subsection if the report wants
  to discuss the refactor as part of the methodology narrative.
- **Methodology** → §5–§9 (architecture, algorithm, training loop,
  hyperparameters + how they were tuned).
- **Results** → §10 (every sub-section has a table of real numbers) + §14
  for figure references.
- **Discussion** → §13 (limitations: MPS instability, policy collapse,
  DIM coverage ceiling) + the "verdict" paragraphs in §10.3/§10.4 (negative
  results are real findings, not failures to hide).
- **Conclusion** → synthesize: sequential MDP + PPO + potential-based
  shaping + NNLS warm-start is implemented and trains a policy that beats a
  near-zero-dose baseline and gets within the DIM's reachability ceiling on
  PTV coverage; the reward-shaping ablation is a clean negative result
  (control already near a local optimum the three tested modifications
  don't escape); the most actionable open lead is a jointly-retuned
  `lambda_oar`/`lambda_phi`/soft-barrier combination, plus addressing why
  the learned policy doesn't meaningfully vary its beam profile across
  fractions despite the MDP supporting it.
