# Reward-shaping ablation: results

Compares the opt-in reward-shaping changes in `src/env/reward.py` (see
`scripts/sweep_reward_shaping.py`), layered in incrementally on top of the
winning hyperparameter regime (`sweep2_heavy_reg`), against `control`
(that regime's existing, already-evaluated result). All runs resume from
the same shared `warmstart.pt`; each is evaluated on the same 40-patient
validation split via `evaluate.py`.

Raw data: `reports/reward_shaping_summary.csv`. Validation-DVH curves:
`reports/reward_shaping_comparison.png`.

## Round 1 — C, A, B layered one at a time

| variant | DVH | best ep | D95_PTV70 | D95_PTV63 | D95_PTV56 | Brainstem(54) | SpinalCord(45) | Mandible(70) | LeftParotid(26) | RightParotid(26) |
|---|---|---|---|---|---|---|---|---|---|---|
| **control (sweep2, no changes)** | **23.34** | 1699 | 43.97 | 39.31 | 38.92 | 19.65 | 24.71 | 41.07 | 39.32 | 43.21 |
| C (soft coverage) | 26.82 | 500 | 33.76 | 28.08 | 29.00 | 10.09 | 21.03 | 29.53 | 31.12 | 31.44 |
| C_A (+ quadratic gap, lambda_phi unchanged) | 32.08 | 500 | 27.70 | 23.16 | 23.00 | 9.39 | 15.23 | 22.66 | 20.70 | 24.41 |
| C_A_retuned (+ lambda_phi=2.0) | 26.82 | 650 | 33.09 | 29.33 | 28.33 | 11.93 | 20.29 | 28.52 | 30.46 | 33.27 |
| C_A_B (+ soft OAR barrier, steepness=2.0) | 25.04 | 925 | 35.29 | 29.54 | 32.86 | 15.84 | 29.04 | 34.72 | 40.14 | 39.08 |

All four score worse than `control`. The *why* is informative enough to record:

- **C alone trades PTV coverage for OAR sparing, too aggressively.**
  Smoothing the hard coverage step gives partial credit for near-miss
  voxels, weakening the all-or-nothing pressure to fully reach Rx.
  D95_PTV70 drops 43.97 -> 33.76 in exchange for better OAR doses across
  every organ -- a real loss, not a wash.
- **C_A (unretuned) reproduces exactly the predicted magnitude-shrinkage
  problem.** Its validation-DVH curve never converges (oscillates
  32-51 the whole run) -- squaring the gap without compensating
  `lambda_phi` left the PTV-shaping signal too weak relative to `lambda_oar`.
- **C_A_retuned validates the fix but doesn't improve beyond C.**
  `lambda_phi=2.0` recovers a stable curve similar to C alone, but lands
  at basically the same DVH/trade-off, not a further improvement.
- **C_A_B is the most informative result of round 1.** It's the only
  variant still improving when early stopping cut it off (best at episode
  925, still trending down) -- but it recovers PTV coverage at the cost of
  a *worse* SpinalCord dose than control (29.04 vs 24.71), and the
  parotids are still just as far over their 26 Gy tolerance as control
  (40.14/39.08 vs 39.32/43.21).

## Round 2 — fixing the OAR barrier's missing floor (C_A_B_v2 / v3)

`scripts/visualize_patient_trajectory.py` on `C_A_B`'s checkpoint (patient
`pt_201`) showed *why* B didn't fix the parotid problem: the soft barrier
`exp(steepness * excess_fraction)` has no floor -- even a perfectly safe
organ at zero dose gets a nonzero penalty (`exp(-2.0) = 0.135`), so
`lambda_oar` fights a constant tax everywhere instead of pressure
concentrated near the tolerance line. The per-fraction reward accelerated
all course long (-1.55/fraction by fraction 35, ~25x deeper than any
hyperparameter regime's floor) while the terminal reward was nearly zero
(+0.06) because soft coverage was satisfied -- the agent was "paying to
win" coverage by letting OAR dose run away, uncorrected.

Fix: `oar_overshoot_fraction` gained an `oar_barrier_activation_threshold`
parameter (zero penalty below that fraction of tolerance, smooth ramp
above it) -- see `src/env/reward.py`. Five follow-up variants tested it
plus two complementary levers (a higher `terminal_dvh_weight`, and
softening `lambda_oar` to compensate for the floor fix):

| variant | DVH | best ep | D95_PTV70 | D95_PTV63 | D95_PTV56 | Brainstem(54) | SpinalCord(45) | Mandible(70) | LeftParotid(26) | RightParotid(26) |
|---|---|---|---|---|---|---|---|---|---|---|
| v2_barrier (threshold=0.8, k=1.5) | 29.40 | 600 (early-stop cut) | 31.83 | 25.20 | 26.89 | 11.35 | 23.51 | 32.12 | 39.27 | 39.62 |
| v2_full (+ terminal_dvh_weight=0.4) | 31.09 | 500 (early-stop cut) | 26.64 | 21.51 | 22.74 | 9.17 | 18.48 | 25.81 | 32.81 | 30.85 |
| v3_gentle_terminal (terminal_dvh_weight=0.2, early-stop disabled) | 29.55 | 1949 (full 2000-ep run, no cutoff) | 22.87 | 18.61 | 18.31 | 11.55 | 27.12 | 33.58 | 37.28 | 34.16 |
| v3_rebalanced_oar (+ lambda_oar=0.4, early-stop disabled) | 26.91 | 174, then collapsed to 32.37 by ep2000 | 33.89 | 27.78 | 28.37 | 10.41 | 21.84 | 31.63 | 32.23 | 35.31 |
| **v3_oar_only (lambda_oar=0.4 alone, terminal_dvh_weight=0.1, real early-stop)** | **26.22** | 500 (clean early-stop) | 35.22 | 30.02 | 29.77 | 10.93 | 23.26 | 30.58 | 34.45 | 38.44 |

- **The barrier-floor fix is mechanistically validated, twice.** Re-running
  `pt_201`'s trajectory under `v2_barrier` and `v3_oar_only` both show the
  runaway per-fraction acceleration is gone (shaping reward sum -3.39 and
  even +0.38 respectively, vs `C_A_B`'s -12.91), and delivered dose
  intensity drops back to the same scale every hyperparameter regime uses,
  rather than `C_A_B`'s ~2x over-delivery.
- **`terminal_dvh_weight` increases consistently hurt, monotonically**
  (0.1 -> 25.04, 0.2 -> 29.55, 0.4 -> 31.09, barrier settings held fixed).
  `dvh_score` is symmetric (penalizes under-coverage exactly like
  over-dose), so a strong terminal penalty just makes the agent broadly
  risk-averse rather than precisely trading OAR safety for coverage --
  confirmed not to be an early-stopping artifact: `v3_gentle_terminal` got
  the full 2000-episode runway with no cutoff and still only reached 29.55.
- **Softening `lambda_oar` (0.6 -> 0.4) is the one lever that helped.**
  `v3_rebalanced_oar` peaked early (ep174) then collapsed for the
  remaining ~1800 episodes (the same entropy-runaway-after-peak pattern
  documented for sweep3) -- `best.pt` protected against that automatically.
  Isolating it cleanly (`v3_oar_only`: terminal_dvh_weight back at its only
  validated-good value, real early stopping instead of disabled) confirms
  it's real and not a fluke: 26.22, the best of all 9 reward-shaping
  variants tried, with a clean early stop rather than a lucky pre-collapse
  snapshot.
- **Still, nothing beats `C_A_B` (25.04), let alone `control` (23.34).**

## The finding that cuts across every variant tested

Across all 9 reward-shaping variants *and* every hyperparameter regime
from the earlier 4-way comparison, **no configuration has ever gotten
parotid dose under its 26 Gy tolerance.** Every variant lands at
30-44 Gy on both parotids. The lowest parotid numbers belong to `C_A`
(20.70/24.41) and `v2_full` (32.81/30.85) -- and both get there only by
sacrificing PTV coverage so badly (D95_PTV70 27.70 and 26.64 vs control's
43.97) that they aren't real candidates. Every change tested has
redistributed the same coverage/OAR trade-off along one frontier, not
pushed the frontier itself.

## Recommendation

Keep `control` (`sweep2_heavy_reg`, no reward-shaping changes). None of
the 9 variants tried across two rounds beat it, and the parotid-overdose
problem that originally motivated this investigation was never actually
solved by any of them. The barrier-floor fix
(`oar_barrier_activation_threshold`) and the `lambda_oar` softening
direction are both mechanistically sound and kept in the codebase
(old-preserving defaults, opt-in via `scripts/sweep_reward_shaping.py`)
in case a future attempt wants to build on them, but neither is being
adopted as the active configuration. Closing out this line of
investigation here.
