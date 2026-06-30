# Training-Run Analysis — Per-Experiment Deep Dive

Companion to `technical_summary.md` (architecture/pipeline reference). This
document is the **results/discussion** source material: every quantity below
was computed directly from `runs/*/train_log.csv` and `runs/*/eval_results.log`
(not estimated), using the analysis defined at the end of this file. All
"sequential mode" framing (§1 of `technical_summary.md`) applies throughout —
every run analyzed here used the active sequential-MDP design.

## 0. Two methodology caveats that affect every comparison below

**(a) A logging-units change splits all runs into two eras.** Commit
`6c72dca` ("updating PPO - normalizing per minibatch", 2026-06-24) changed
`policy_loss`/`value_loss`/`entropy` in `train_log.csv` from a **sum over all
`ppo_epochs × minibatches` per update** (~30 minibatches at the production
config) to a **per-minibatch mean**. The **Production run and the gamma
sweep** (both run before this fix) report entropy in the old, ~30×-inflated
scale (~98,000–133,000); **every other experiment** (hyperparameter sweeps,
beam-path ablation, reward-shaping ablation — all run after the fix) reports
the correct per-minibatch-mean scale (~3,270–6,700). Raw `entropy`/
`policy_loss`/`value_loss` magnitudes are therefore **not comparable across
the two eras**; this analysis instead converts entropy to an interpretable
action standard deviation **σ** (see formula at the bottom) and compares
*relative growth* within each run, which is valid in both eras.

**(b) Early stopping did not exist for every run.** `early_stop_patience_evals`
was added in commit `01b655a` (2026-06-27). The **Production run, gamma
sweep, CPU hyperparameter sweep, and beam-path ablation** all predate it and
ran for a fixed episode budget with no plateau protection. Only the **MPS
hyperparameter reruns** and the **reward-shaping ablation** (both run after
`01b655a`) had it active.

**(c) Clinical metrics (D95, per-OAR mean dose) only exist for the runs
that called `evaluate.py`** — that's the 4 MPS hyperparameter regimes and the
6 reward-shaping variants. The Production run, gamma sweep, CPU hyperparameter
sweep, and beam-path ablation are compared on **validation DVH score only**
(`train._validation_dvh_score`, the same metric used for `best.pt` selection).

---

## Experiment 0 — Production baseline run (`runs/`)

**What changed**: nothing — this is the reference run that
`configs/default.yaml` produces as-is (gamma=0.99, CPU, sequential mode,
3500 episodes, 200 train patients, batch_n_patients=8). It is also reused
verbatim as the gamma=0.99 gamma-sweep candidate and as the CPU
hyperparameter sweep's "baseline" row (see §0).

**Why this configuration**: it's the project's production default after the
warm-start + OAR-curriculum + validation-DVH-selection design (`train.py`)
was finalized; every later experiment branches from it.

**Effect on learning / learning curve**: very fast initial improvement —
val DVH drops from the warm-started starting point to its best value
(**22.67 at episode 99**) within the first **24 episodes** (`conv_ep`, first
point within 5% of eventual best). After that the curve **monotonically
worsens** for the remaining ~3400 episodes, reaching **41.23 at episode
3499** — a **+81.9%** degradation from the best checkpoint. This is the
single clearest example in the whole project of the "peak early, then
silently get much worse" failure mode.

**Evaluation performance**: only the training-time val-DVH metric is
available (no `evaluate.py` clinical breakdown was run for this exact
checkpoint/config combination at the time); `best.pt` (ep99, DVH 22.67) is
the number that propagates into every later comparison (gamma table, CPU
hyperparameter table) as the run to beat.

**Convergence speed**: very fast (24 episodes) — almost entirely attributable
to the supervised NNLS warm-start handing PPO an already-reasonable policy,
not to PPO's own exploration.

**Stability**: poor over the long run. Tail (last 20% of episodes)
`policy_loss` standard deviation is **5.51**, by far the highest of any
experiment in this project (next-highest comparable-era value is ~0.17),
confirming the policy is still being pushed hard by gradient updates very
late in training rather than settling.

**Overfitting / underfitting**: this is **not** classic overfitting (train
reward improving while held-out DVH worsens) — the correlation between
`rolling_mean_reward` and `val_dvh` across the run is **−0.90** (strongly
negative), meaning *training reward also got worse* as the run degraded.
Reward and the clinical proxy moved together. This is better described as
**policy degradation / instability**, not reward-objective mismatch.

**Exploration sufficiency**: action σ (derived from entropy) rises from 1.00
at the start to **~1.3–1.5 through the first 75% of training**, before a
single late reading drops to ~0.77 (a single noisy data point, not a
smoothed trend — read the 0→75% climb as the real signal). A near-50% rise
in action standard deviation over a run that is simultaneously getting worse
is consistent with **the policy becoming progressively more random rather
than more refined** — exploration that never reins itself back in once the
warm-started policy is left behind.

**Reward function behaviour**: behaved *as designed* in the sense that it
faithfully tracked the degrading policy (strong negative correlation, §above)
— it did not reward-hack. The problem is upstream of the reward shaping: the
**policy itself** destabilizes (rising σ + rising `policy_loss` variance),
and the reward correctly reports that as it happens.

**Why it succeeded/failed**: **succeeded** at quickly finding a good policy
(warm-start + PPO reaches DVH 22.67 in 99 episodes) but **failed** to *stay*
there for the other 97% of the configured training budget — this run is the
direct motivating case for the early-stopping feature added later
(`src/utils/early_stopping.py`'s own docstring describes almost exactly this
trajectory). `best.pt` is unaffected (saved independently the moment val_dvh
improves), but ~3400 episodes of compute were wasted with nothing watching
for the regression.

---

## Experiment 1 — Discount-factor (γ) sweep

**What changed**: `gamma ∈ {0.95, 0.97, 0.99, 0.995, 0.999}`, all branched
from one shared warm-start checkpoint, 900 episodes each (enough to clear the
600-episode OAR ramp). Everything else fixed at the production config.

**Why**: γ is **live** in sequential mode (it appears both in `PPO._compute_gae`'s
bootstrapping *and* directly inside the per-fraction potential-shaping reward
itself, since `reward = γ·Φ(s′) − Φ(s)`), so it was treated as a first-class
hyperparameter rather than left at a default guess. Compared on validation
DVH specifically because changing γ changes the reward landscape itself, so
raw training reward isn't comparable across candidates.

**Effect on learning / learning curves**:

| γ | best ep | best DVH | conv. ep (≤5% of best) | final (ep899) DVH | degradation |
|---|---|---|---|---|---|
| 0.95 | 774 | 26.06 | 749 | 35.92 | +37.8% |
| 0.97 | 74 | 22.79 | 74 | 32.00 | +40.4% |
| **0.99** | 99 | **22.67** | 24 | 26.57 | +17.2% |
| 0.995 | 74 | 27.65 | 24 | 29.80 | +7.8% |
| 0.999 | 99 | 23.88 | 49 | 29.43 | +23.3% |

**Evaluation performance**: best-checkpoint quality is **not monotonic in
γ** — 0.99 wins, 0.97 is close second, but 0.995 (between them) is the worst
of all five. This is a strong signal that the result is at least partly
**noise/seed-sensitive** rather than a clean unimodal optimum in γ, since γ
moving slightly past 0.99 should not by itself produce a qualitatively worse
outcome if the landscape were smooth.

**Convergence speed**: γ=0.99 and γ=0.995 both reach their respective
(very different quality) plateaus fastest (24 episodes); γ=0.95 is far the
slowest (749 episodes) — lower γ discounts the terminal whole-course reward
more heavily relative to the per-fraction shaping term, which apparently
slows down how quickly the value function learns to anticipate it.

**Stability**: every candidate degrades after its peak (this experiment
predates early stopping, §0b) — degradation ranges from a mild +7.8%
(γ=0.995) to a severe +40.4% (γ=0.97). Notably, the candidate with the
*mildest* degradation (0.995) also has the *worst* peak quality — a
**stability/quality trade-off** across this sweep, not a free win.

**Overfitting/underfitting**: γ=0.995 and γ=0.999 both converge fast to a
visibly *worse* asymptote (27.65 / 23.88 vs. 22.67 for γ=0.99) and barely
move afterward — a form of **premature convergence to a shallow optimum**
("underfitting" relative to what 0.99 demonstrably achieves), rather than
runaway divergence.

**Exploration sufficiency**: action σ rises through training in every
candidate (consistent with the Production run, since this era shares its
logging/instability characteristics) — γ=0.99 shows the most contained
growth (σ ≈1.0→1.11 through 75%), γ=0.995/0.999 show similarly modest growth,
γ=0.95/0.97 show somewhat more (σ up to ~1.2). No candidate shows
*insufficient* exploration (collapsed-to-zero σ); if anything every candidate
trends toward slightly *too much* over time.

**Reward function behaviour**: negative reward/DVH correlation in every
candidate (−0.69 to −0.84, except γ=0.995's weaker −0.33) — reward tracked
true policy quality directionally in all five, i.e. the shaping formula's
γ-dependence didn't decouple the proxy from the real objective at any tested
value.

**Why it succeeded/failed**: **succeeded** as a sweep — it answered the
question it was designed for (γ=0.99, the production default, is in fact the
best of the tested values, validating that default rather than leaving it
unchecked) and surfaced the non-monotonic, partly-noisy relationship between
γ and outcome quality as a real finding rather than a clean textbook
trend.

---

## Experiment 2 — Hyperparameter-bundle sweep, short CPU runs (600 episodes)

**What changed**: four named presets bundling `batch_n_patients`,
`minibatch`, `ppo_epochs`, `lr`, `lambda_oar_ramp_episodes` together (a
coarse "regime" comparison, not one-variable-at-a-time):

| regime | batch_n_patients | minibatch | ppo_epochs | lr | ramp episodes |
|---|---|---|---|---|---|
| baseline | 8 | 64 | 6 | 1.5e-4 | 600 |
| sweep1_minibatch_active | 4 | 64 | 3 | 1e-4 | 200 |
| sweep2_heavy_reg | 8 | 128 | 3 | 5e-5 | 300 |
| sweep3_aggressive_adaptivity | 2 | 32 | 4 | 1e-4 | 200 |

`baseline` here is **not** independently re-trained — the script reuses the
historical Production run's `train_log.csv` (Experiment 0) verbatim.

**Why**: to see whether a different combination of update-aggressiveness
(batch size, minibatch size, PPO epochs, learning rate) and curriculum speed
could beat the production defaults, as a bundled "preset" search rather than
an expensive full grid.

**Effect on learning / learning curves** (all genuinely-run regimes capped
at 600 episodes — too short to see the late-training collapse Experiment 0
shows):

| regime | best ep | best DVH | conv. ep | final (ep599) DVH | degradation |
|---|---|---|---|---|---|
| sweep1_minibatch_active | 424 | 23.73 | 249 | 31.75 | +33.8% |
| sweep2_heavy_reg | 474 | 26.61 | 424 | 26.82 | **+0.8%** |
| sweep3_aggressive_adaptivity | 174 | 23.75 | 174 | 26.57 | +11.9% |

**Evaluation performance**: only val-DVH is available at this stage (no
`evaluate.py` clinical run for the 600-episode versions). sweep1 and sweep3
reach lower (better) *best* DVH than sweep2 within the 600-episode budget,
but sweep2 is already visibly the most **stable** of the three (smallest
degradation after its peak) even at this short horizon — a preview of the
much sharper version of the same pattern in Experiment 3.

**Convergence speed**: sweep3 (aggressive: small batch, more epochs, fast
ramp) converges fastest (174 episodes) — consistent with "more, smaller,
more-aggressive updates move the policy faster per episode." sweep2 (heavy
regularization: large minibatch, fewer epochs, slow ramp) is the slowest to
even approach its eventual best (424 episodes) — the expected cost of more
conservative updates.

**Stability**: sweep2 is already the standout — its tail `policy_loss`/
`value_loss` standard deviations (0.074 / 0.137) are the smallest of the
three, and its post-peak degradation (+0.8%) is almost negligible at this
horizon.

**Overfitting/underfitting**: no clear overfitting signal at 600 episodes
(reward/DVH correlations are noisy but small-to-moderate, −0.41 to +0.49 —
notably sweep3's is *positive* even at this short horizon, an early hint of
the same reward/DVH decoupling that becomes severe later for the
"aggressive" regimes, see Experiment 3).

**Exploration sufficiency**: σ growth is mild for all three within 600
episodes (sweep1 ≈1.0→1.03, sweep2 ≈1.0→1.02, sweep3 ≈1.0→1.22) — sweep3
already shows the most σ growth of the three, foreshadowing its later
runaway behavior at full length.

**Reward function behaviour**: mostly tracking DVH correctly (negative
correlation for sweep1/sweep2); sweep3's flip to positive correlation even
at 600 episodes is the earliest visible warning sign that this regime's
reward signal would decouple from true quality given more time.

**Why it succeeded/failed**: **partially conclusive** — at this short
horizon sweep1 and sweep3 *look* competitive or better than sweep2 on raw
best-DVH, which is exactly why the project re-ran all four regimes to full
length on MPS (Experiment 3) rather than trusting this 600-episode snapshot;
the short-run comparison alone would have picked the wrong regime.

---

## Experiment 3 — Hyperparameter-bundle sweep, full-length MPS reruns (3500 episodes, early stopping active)

**What changed**: the same four regimes as Experiment 2, this time run to
the full 3500-episode budget on the GPU (MPS) with early stopping enabled
(post-commit `01b655a`) — the regime definitions are identical to Experiment
2's table.

**Why**: 600 episodes (Experiment 2) wasn't long enough to see whether any
regime would suffer the kind of late-training collapse seen in Experiment 0;
this rerun (with early stopping as a safety net this time) tests all four to
the same horizon as the production run.

**Effect on learning / learning curves**:

| regime | best ep | best DVH | conv. ep | stopped at | final DVH | degradation |
|---|---|---|---|---|---|---|
| baseline | 499 | 26.63 | 199 | **800 (early-stopped)** | 27.41 | +3.0% |
| sweep1_minibatch_active | 1199 | 24.23 | 724 | 3500 (ran full length) | **66.56** | **+174.7%** |
| **sweep2_heavy_reg** | 1699 | **23.34** | 1324 | 3500 (ran full length) | 25.15 | +7.7% |
| sweep3_aggressive_adaptivity | 174 | 25.21 | 24 | 3500 (ran full length) | **122.05** | **+384.2%** |

**Evaluation performance** (clinical metrics on `best.pt`, full 40-patient
validation split, `evaluate.py`):

| regime | DVH | D95_PTV70 | D95_PTV63 | D95_PTV56 | Brainstem | SpinalCord | Mandible | L.Parotid | R.Parotid |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 26.63 | 34.31 | 29.12 | 29.24 | 15.88 | 27.14 | 31.54 | 41.80 | 40.62 |
| sweep1 | 24.23 | **44.04** | **39.87** | **39.38** | 18.62 | 30.36 | 43.21 | 44.00 | 48.92 |
| **sweep2** | **23.34** | 43.97 | 39.31 | 38.92 | 19.65 | 24.71 | 41.07 | 39.32 | 43.21 |
| sweep3 | 25.21 | 38.14 | 34.44 | 33.92 | **14.17** | **22.86** | **32.61** | **36.12** | **40.11** |

Important nuance: **sweep3's `best.pt`** (saved at episode 174, *before* its
catastrophic later collapse) sits at a genuinely **different point on the
PTV-coverage/OAR-sparing trade-off** — worst PTV coverage of the four but
**best OAR sparing on every single organ**. It is not simply "worse than the
others"; it found a different, OAR-favoring optimum and then destroyed its
own training run afterward. sweep1's `best.pt` (episode 1199, also pre-collapse)
has the **best D95 numbers of all four** but the worst parotid doses — also a
trade-off point, not a strictly dominant or dominated one.

**Convergence speed**: wildly different across regimes — sweep3 converges
to its peak almost immediately (24 episodes, fastest of any regime in the
whole project), sweep1 takes 724 episodes, baseline 199, and **sweep2 is the
slowest of all four (1324 episodes)**. The slow-converging regime is also
the only stable one — direct evidence that this project's instability is
tied to *how fast* a regime is willing to move the policy, not to the
absolute episode budget available.

**Stability**: this is the headline result of the whole sweep series.
**sweep2_heavy_reg (large minibatch=128, only 3 PPO epochs, low lr=5e-5,
slow 300-episode OAR ramp) is the only regime that trains for 3500 episodes
without collapsing.** sweep1 and sweep3 — both smaller-minibatch /
faster-ramp / more-aggressive-update configurations — degrade by 175% and
384% respectively. Tail policy/value-loss volatility mirrors this exactly
(sweep2 tail `policy_loss` std 0.063 — the lowest of the four — vs. sweep3's
0.068 looking deceptively similar *only* in this one statistic, while its
`value_loss` tail std (0.279) and especially its σ trajectory tell the real
story, next point).

**Overfitting/underfitting**: sweep3 is the clearest case of something
resembling **reward/objective decoupling**: its reward/DVH correlation is
**−0.99** (an almost perfect negative correlation, like Experiment 0 —
reward and DVH both worsen together, so this is faithful-but-catastrophic
degradation, not silent reward hacking). sweep2, by contrast, shows
essentially **zero** correlation (+0.02) — consistent with a regime that has
genuinely *converged* and is fluctuating in a narrow band around a good
solution rather than trending in either direction.

**Exploration sufficiency — the clearest causal signal in this project**:
derived action σ at start → 25% → 50% → 75% → end of training:

| regime | σ(start) | σ(25%) | σ(50%) | σ(75%) | σ(end) |
|---|---|---|---|---|---|
| baseline (stopped ep800) | 1.00 | 1.05 | 1.12 | 1.18 | 1.22 |
| sweep1 | 1.00 | 1.11 | 1.18 | 1.26 | 1.34 |
| **sweep2 (stable)** | 1.00 | 1.03 | 1.06 | 1.08 | **1.10** |
| sweep3 (catastrophic) | 1.00 | 1.35 | 1.84 | 2.87 | **4.40** |

`actor_log_std_max=2.0` caps σ at ~7.4; sweep3 ends within striking distance
of that ceiling. **The size of a regime's σ growth over training predicts
the severity of its eventual collapse almost perfectly across all four
regimes** (sweep3 worst growth → worst collapse; sweep1 moderate growth →
moderate collapse; sweep2 flattest growth → no collapse). The shared,
fixed entropy coefficient (`ent_coef=0.005`) is evidently strong enough to
restrain σ under sweep2's conservative update regime but **not** under
sweep1/sweep3's more aggressive ones — exploration becomes self-reinforcing
(noisier actions → noisier reward/advantage estimates → noisier updates →
even more σ growth) once the update aggressiveness crosses some threshold
this sweep accidentally located.

**Reward function behaviour**: tracked true quality faithfully (strong
negative correlation) in the collapsing regimes, near-zero in the converged
one — both are "expected" behaviours of a working reward signal; the failure
mode here is entirely on the **policy-update side** (runaway σ), not the
reward design.

**Why each succeeded/failed**: **baseline** succeeded cleanly — early
stopping caught its drift at episode 800 before serious damage. **sweep2**
succeeded outright — slowest to converge but the only regime stable across
the full budget, and it has the best overall DVH and best/near-best D95
numbers of the four; it was subsequently adopted as the base regime for the
entire reward-shaping ablation (Experiment 5). **sweep1** and **sweep3**
**partially succeeded** (both found genuinely good, even differently-shaped,
`best.pt` checkpoints early) but **failed** at the training-process level —
both ran the full 3500-episode budget without early stopping ever firing,
despite catastrophic divergence, because the plateau monitor's
relative-improvement check kept resetting on intermittent noisy "new bests"
right up until the real collapse was already underway. This is a genuine
limitation of the plateau-detection logic, not just bad luck.

---

## Experiment 4 — Beam-path state-channel ablation

**What changed**: `include_beam_paths` (one of the 12 state channels — a
precomputed coplanar ray-cast beam-geometry map) toggled `True`/`False`,
which also changes the CNN encoder's input-channel count (12 vs. 11) — each
variant trains an independent network from its own warm-start (checkpoints
aren't interchangeable across the toggle). 300 episodes each, CPU, production
hyperparameters otherwise. Predates early stopping (§0b).

**Why**: to test whether the precomputed geometric prior (which voxels any
beam can physically reach) actually helps the CNN, or whether it's an
unnecessary 9th of the input channels.

**Effect on learning / learning curves**: `on` reaches best DVH **26.95**
(episode 299, i.e. still improving at the 300-episode cutoff); `off` reaches
only **29.78** (also episode 299, still improving). `on` also approaches its
end-of-run value faster (conv. ep 124 vs. 274 for `off`).

**Evaluation performance**: no clinical `evaluate.py` breakdown was run for
this ablation — comparison is val-DVH only.

**Convergence speed**: `on` converges roughly **2.2× faster** (124 vs. 274
episodes to within 5% of its 300-episode value) — the geometric prior gives
the CNN a head start on "where is dose physically achievable" rather than
making it infer that purely from the dose-influence-matrix's effect on
training signal.

**Stability**: inconclusive by design — both variants were still improving
when the 300-episode run ended (degradation = 0% for both, simply because
the last logged point is also each run's best so far). This ablation
**cannot** speak to long-run stability the way Experiments 0/3 can; it only
measures early-training behaviour, well before the 600-episode OAR-ramp
default would even finish.

**Overfitting/underfitting**: no signal available at this horizon (both
runs are still in their improving phase, never reaching a peak to overfit
past).

**Exploration sufficiency**: both show mild σ growth (`on`: 1.00→1.07,
`off`: 1.00→1.11) — `off` shows marginally *more* σ growth despite ending up
worse, consistent with a harder-to-fit problem (no geometric prior) needing
more stochasticity to make any progress at all.

**Reward function behaviour**: both negative reward/DVH correlations (−0.74
`on`, −0.65 `off`) — reward tracked improving quality correctly in both,
exactly as expected for two runs that are simply still learning.

**Why it succeeded/failed**: **succeeded** at answering its narrow question
— the beam-path channel measurably speeds up and improves early training
(production default correctly keeps it on) — but the experiment's 300-episode
budget means this conclusion is only validated for the *early-training*
regime, not for the full 3500-episode horizon where Experiments 0/3 show
behaviour can change substantially.

---

## Experiment 5 — Reward-shaping ablation (Improvements C, A, B, layered)

**What changed**: starting from the `sweep2_heavy_reg` regime (Experiment
3's only stable configuration — batch=8, minibatch=128, ppo_epochs=3,
lr=5e-5, ramp=300 episodes, MPS, early stopping active), three independent
reward-shaping changes were toggled on, layered incrementally:

- **C** — `terminal_use_soft_coverage=True`: replace the terminal reward's
  hard `dose ≥ Rx` coverage step with a differentiable sigmoid.
- **A** (on top of C) — `ptv_gap_power=2.0`: square the PTV-gap term inside
  the dense shaping potential Φ instead of linear.
- **A retuned** — same as C+A but `lambda_phi=2.0` (compensating for A's
  shrunk Φ magnitude).
- **B** (on top of C+A retuned) — `oar_barrier_steepness=2.0`: replace the
  OAR-overshoot term's hard threshold with a soft exponential barrier.
- **B v2 (barrier floor fix)** — same as C+A+B but with
  `oar_barrier_activation_threshold=0.8` added (zeroes the barrier below 80%
  of tolerance instead of taxing every organ everywhere), in two variants:
  `v2_barrier` (fix only) and `v2_full` (fix + `terminal_dvh_weight` raised
  0.1→0.4).
- **B v3 (confound check + lambda_oar rebalance)** — both v2 variants
  early-stopped much sooner than C+A+B did (ep600/500 vs. ep925, where it
  was still improving), raising the question of whether they were judged
  on premature stops. Three follow-ups, all with `early_stop_patience_evals`
  either disabled (`0`) or restored: `v3_gentle_terminal`
  (`terminal_dvh_weight=0.2` instead of 0.4, early stopping disabled, run
  the full 2000-episode budget with no cutoff), `v3_rebalanced_oar` (same +
  `lambda_oar=0.6→0.4`, early stopping disabled), and `v3_oar_only`
  (isolates `lambda_oar=0.4` alone — `terminal_dvh_weight` back to its only
  validated-good value of 0.1, early stopping restored to `patience=8`).
- **control** = `sweep2_heavy_reg`'s own `best.pt` (Experiment 3) — i.e. the
  *unmodified* reward, reused rather than re-trained.

**Why**: a baseline 4-regime clinical-metrics evaluation showed every regime
substantially underdosing every PTV (D95 ~38–44 Gy vs. Rx 56–70 Gy) and
overdosing both parotids 1.4–1.9× past their 26 Gy tolerance. C/A/B were
proposed, independently, to address exactly those two symptoms (a
differentiable terminal target for coverage; steeper anticipatory shaping
near under-dosed PTV voxels; anticipatory rather than reactive OAR pressure).

**Effect on learning / learning curves and evaluation performance**:

| variant | best ep | best DVH | stop ep | final DVH | degradation | D95_PTV70 | LeftParotid(26) | RightParotid(26) |
|---|---|---|---|---|---|---|---|---|
| **control (unmodified)** | 1699 | **23.34** | n/a (Exp. 3) | — | — | **43.97** | 39.32 | 43.21 |
| C | 149 | 26.82 | 500 | 34.33 | +28.0% | 33.76 | 31.12 | 31.44 |
| C+A (unretuned) | 299 | 32.08 | 500 | 39.85 | +24.2% | 27.70 | 20.70 | 24.41 |
| C+A retuned | 499 | 26.82 | 650 | 29.75 | +10.9% | 33.09 | 30.46 | 33.27 |
| C+A+B | 724 | 25.04 | 925 | 25.56 | **+2.1%** | 35.29 | 40.14 | 39.08 |
| C+A+B v2_barrier (barrier-floor fix) | 399 | 29.40 | 600 | 33.63 | +14.4% | 31.83 | 39.27 | 39.62 |
| C+A+B v2_full (+ terminal_dvh_weight 0.4) | 274 | 31.09 | 500 | 33.31 | +7.1% | 26.64 | 32.81 | 30.85 |
| v3_gentle_terminal (terminal_dvh_weight 0.2, early-stop disabled) | 1949 | 29.55 | 1999 (full budget, no cutoff) | 29.67 | +0.4% | 22.87 | 37.28 | 34.16 |
| v3_rebalanced_oar (+ lambda_oar 0.4, early-stop disabled) | 174 | 26.91 | 1999 (full budget, no cutoff) | 32.37 | +20.3% | 33.89 | 32.23 | 35.31 |
| **v3_oar_only (lambda_oar 0.4 alone, terminal_dvh_weight back to 0.1)** | 249 | **26.22** | 499 (clean early-stop) | 31.10 | +18.6% | 35.22 | 34.45 | 38.44 |

**None of the nine shaping variants beat the unmodified control on
aggregate DVH** — every one scores worse than 23.34. Best-to-worst:
control < C+A+B < v3_oar_only < C ≈ C+A-retuned < v3_rebalanced_oar <
v2_barrier < v3_gentle_terminal < v2_full < C+A. The changes layered in the
agreed order **partially recover** what each previous change lost (C+A+B
is the best of the round-1 modified variants; `v3_oar_only` is the best of
the round-2 barrier-floor-fix family) but none reach parity with simply
leaving the reward alone.

**Convergence speed**: every round-1 shaping variant converges to its own
best *faster* than control did (149–724 episodes vs. control's 1699)
simply because all four were early-stopped well before 1699 episodes —
a budget artifact of early stopping triggering, not evidence the modified
reward is intrinsically easier to optimize. Round 2 deliberately tests
this: `v2_barrier`/`v2_full` were *also* cut short (ep600/500), so two
follow-ups (`v3_gentle_terminal`, `v3_rebalanced_oar`) disabled early
stopping entirely and ran the full 2000-episode budget. `v3_gentle_terminal`
genuinely used nearly all of it (best at ep1924) and still only reached
29.55 — ruling out the early-stopping artifact explanation for that
specific lever (`terminal_dvh_weight`). `v3_rebalanced_oar` peaked far
earlier (ep174) even with the cutoff removed, confirming its early peak
was a property of the reward landscape, not a premature stop.

**Stability**: early stopping worked correctly for all round-1 variants
and for `v3_oar_only` (clean trigger within its 8-check patience window),
but "clean trigger" still allows up to 200 episodes of post-peak drift
before firing — `v3_oar_only`'s own degradation (+18.6%) shows this isn't
free. **C+A+B still has by far the mildest post-peak drift (+2.1%)** of
every variant tested in either round; `v3_gentle_terminal`'s near-zero
degradation (+0.4%) is a different phenomenon — it simply never stopped
improving until almost the very end of its (uncapped) 2000-episode budget,
not evidence of a stable plateau. `v3_rebalanced_oar`, run with early
stopping disabled, shows the round's worst degradation (+20.3%) — the same
entropy-runaway-after-peak pattern documented for hyperparameter sweep3
(Experiment 3), here triggered by a reward-shaping change instead of a
hyperparameter regime. `best.pt` protected against it exactly as designed
(see Lesson 4): the checkpoint actually evaluated (ep174) is unaffected by
the later collapse.

**Overfitting/underfitting**: C+A (unretuned) is the clearest underfitting
case in this experiment — `reports/reward_shaping_sweep.md` independently
notes its validation curve "never converges, oscillates 32–51 the whole
run," and its own σ stays essentially flat (≈1.0→1.03) the entire time —
**the policy isn't destabilizing, it's simply receiving too weak a learning
signal** (squaring the PTV gap shrinks Φ's magnitude without compensating
`lambda_phi`), a textbook underfitting signature distinct from Experiment
3's *overfitting-to-noise* collapses.

**Exploration sufficiency**: σ stays essentially flat (1.00→1.01–1.03) for
every one of the round-1 shaping variants and the patience-8 round-2
variants (`v2_barrier`, `v2_full`, `v3_oar_only`) — none show anything
resembling Experiment 3's runaway exploration. The two early-stop-disabled,
full-length round-2 runs are the exception: `v3_gentle_terminal` (1.00→1.10)
and `v3_rebalanced_oar` (1.00→1.10) both show real σ growth, similar in
*size* to the MPS baseline regime (Experiment 3's stable `baseline`, also
~1.10-1.22) — consistent with Lesson 1's finding that σ growth is mostly a
function of *how long a run is allowed to continue*, not of the reward
formulation, since these are the only two reward-shaping variants given
enough episodes to show it. This rules out "insufficient/runaway
exploration" as the explanation for why none of the nine beat control; the
gap is in the **reward landscape itself**, not in how thoroughly the policy
explored it.

**Reward function behaviour — the standout finding of this experiment**: the
tail-window `value_loss` mean/std tells a story the DVH table alone doesn't:

| variant | tail value_loss mean | tail value_loss std | reward/DVH corr |
|---|---|---|---|
| control / C / C+A / C+A-retuned | 0.13 – 0.21 | 0.09 – 0.27 | −0.25 to −0.80 (expected) |
| **C+A+B** | **16.33** | **9.07** | **+0.73 (anomalous)** |
| C+A+B v2_barrier | 11.50 | 10.39 | +0.34 (anomalous) |
| C+A+B v2_full | 4.47 | 4.86 | −0.41 (recovers expected sign) |
| v3_gentle_terminal | 0.93 | 0.89 | +0.17 (still anomalous, but small) |
| v3_rebalanced_oar | 1.49 | 1.88 | −0.24 (recovers expected sign) |
| **v3_oar_only** | **0.47** | **0.31** | +0.33 (mildly anomalous) |

**Every variant that includes Improvement B (the soft exponential OAR
barrier) inflates the critic's value loss relative to every variant that
doesn't**, regardless of whether the activation-threshold floor fix (v2/v3)
is applied. The floor fix (added in commit `7f69e67` specifically because
it was traced to "an accelerating per-fraction OAR penalty… even a fully
safe organ gets a nonzero `exp(−steepness)` tax") *does* shrink the
blow-up monotonically as more of the fix's logic is applied
(16.3 → 11.5 → 4.5 → 0.93/1.49/0.47), and softening `lambda_oar` in
`v3_oar_only` produces the lowest value loss of the entire B-family by a
wide margin. **Caveat**: the round-2 variants also ran far longer than
round-1's before reaching their tail window (500–2000 episodes vs. C+A+B's
925), so part of this drop is plausibly just "the critic has had more
updates to fit a hard target," not purely "the target itself got easier" —
the comparison is suggestive, not fully apples-to-apples across rounds.
What's unambiguous regardless of that caveat: the floor fix and the
`lambda_oar` softening both move the value loss *in the right direction*.
Correlation sign is noisier across the B-family than value loss is —
`v2_full` and `v3_rebalanced_oar` both flip back to the "expected" negative
sign, while `v2_barrier`, `v3_gentle_terminal`, and `v3_oar_only` stay
mildly positive — so value loss, not correlation sign, is the more
reliable diagnostic for this particular failure mode.

**Why it succeeded/failed**: the experiment **succeeded** as a piece of
research — it's a clean, well-instrumented negative result with a specific,
mechanistic explanation per sub-variant (C trades coverage for OAR sparing
too readily; C+A unretuned underfits from a shrunk gradient; C+A+B
destabilizes the critic via an unfloored exponential barrier; the v2/v3
floor fix demonstrably helps the critic, and isolating `lambda_oar=0.4`
alone produces the best reward-shaping variant tried, `v3_oar_only` at
26.22 — without yet closing the gap to control's 23.34). None of the nine
**failed silently** — each failure mode is attributable to a specific,
identified mechanism rather than "it just didn't work," which is exactly
the kind of evidence a Discussion section should be built on. Two
follow-up confound checks (disabling early stopping entirely for
`v3_gentle_terminal`/`v3_rebalanced_oar`) ruled out "premature cutoff" as
the explanation for the `terminal_dvh_weight` lever specifically (it used
nearly its full 2000-episode budget and still underperformed) while
confirming the `lambda_oar` lever's early peak was real, not an artifact —
a useful general methodology: **before concluding a lever doesn't help,
verify the run wasn't simply cut off before it had a chance to.**

The cross-cutting finding that survived all nine variants and both
hyperparameter sweeps: **no configuration tested anywhere in this project
has ever gotten parotid dose under its 26 Gy tolerance.** Every variant
lands at 30–44 Gy on both parotids; the lowest parotid numbers (`C+A`'s
20.70/24.41, `v2_full`'s 32.81/30.85) only get there by sacrificing PTV
coverage so badly they aren't viable candidates. This investigation is
**closed out** with `control` (`sweep2_heavy_reg`, unmodified reward) as
the standing recommendation — the barrier-floor fix and the `lambda_oar`
softening direction are both kept in the codebase (old-preserving
defaults, opt-in) as validated-but-not-yet-sufficient building blocks for
a future attempt, not as the active configuration.

---

## Overall Lessons Learned

1. **Update aggressiveness, not episode budget or reward shape, is the
   dominant cause of training instability in this project.** Every
   catastrophic collapse observed (Production run, gamma-sweep candidates,
   hyperparameter sweep1/sweep3) shares the same fingerprint: a steadily
   **rising action standard deviation σ** that never gets reined back in by
   the fixed entropy coefficient. The one regime that avoided this
   (`sweep2_heavy_reg`: large minibatch=128, only 3 PPO epochs, low
   lr=5e-5, slow 300-episode OAR ramp) was also the *slowest to converge* —
   stability and speed traded directly against each other in every
   experiment that varied update aggressiveness. **A regime should be judged
   by its full-length trajectory, never by an early snapshot** — Experiment
   2's 600-episode read of sweep1/sweep3 looked competitive; Experiment 3's
   3500-episode rerun of the exact same regimes showed both collapsing by
   175–384%.

2. **The reward function itself was almost never the source of failure —
   it faithfully reported policy quality even while that quality collapsed.**
   Reward/DVH correlation was strongly negative (reward and DVH degrading
   together) in every unstable run, which rules out reward hacking as an
   explanation. The one clear exception — Improvement B's soft OAR barrier
   inflating critic value-loss (16.33 at its worst vs. 0.13–0.21 for
   variants without B) and briefly flipping the reward/DVH correlation
   positive — was a **value-function fitting problem caused by an
   unbounded exponential penalty term**, not a misspecified objective; it
   was diagnosable directly from training logs (value-loss magnitude)
   without needing the downstream clinical metrics at all, and the
   project's floor-fix plus softening `lambda_oar` (commit `7f69e67` and
   the `v3_oar_only` follow-up) brought it down to 0.47 — much closer to,
   though still above, the no-B baseline range.

3. **Early stopping is necessary but not sufficient — and "it triggered" is
   not the same as "it triggered at zero cost."** It worked cleanly in 10
   of 12 runs where it was active (the MPS baseline regime and 9 of the 9
   reward-shaping variants that had it on), each time catching a plateau
   within its 8-check (200-episode) patience window. But that 200-episode
   grace period is itself a real cost: `v3_oar_only`'s degradation (+18.6%
   even with a "clean" trigger) shows a clean stop can still cost real
   quality before it fires. Early stopping **failed to fire at all** in the
   two runs that needed it most (hyperparameter sweep1/sweep3 on MPS)
   because their validation curves kept producing noisy intermittent "new
   bests" that reset the patience counter right up until the real, severe
   collapse was already underway — a real limitation of
   relative-improvement-based plateau detection on noisy validation
   signals, not a configuration mistake. A related, generalizable
   methodology point from the reward-shaping ablation's round 2: **when a
   variant's early-stopped result looks bad, disable early stopping and
   give it the full episode budget before concluding the *lever* (not the
   stopping point) is the problem** — doing exactly this for
   `terminal_dvh_weight` confirmed it genuinely underperforms even with no
   cutoff at all (`v3_gentle_terminal` used 1949 of its 2000 episodes), while
   doing it for `lambda_oar` confirmed the opposite risk: removed from its
   safety net, that variant (`v3_rebalanced_oar`) peaked at episode 174 and
   then degraded for the remaining ~1800 episodes — exactly the failure
   mode early stopping exists to catch.

4. **`best.pt`'s independent, improvement-triggered saving is what actually
   protects this project from every instability above** — not early
   stopping, not hyperparameter choice. Every single collapsing run in this
   document (Production, gamma candidates, sweep1/sweep3) still produced a
   usable, often very good, checkpoint, because `best.pt` is written the
   moment validation DVH improves, independent of whatever happens for the
   rest of the run. The lesson generalizes: **for a noisy, occasionally
   unstable training process, decoupling "the checkpoint you keep" from
   "the run you let finish" was the single highest-leverage engineering
   decision in this project.**

5. **Exploration was, if anything, excessive rather than insufficient.**
   No experiment in this project ever showed σ collapsing toward zero
   (premature exploitation) — every instability ran in the *opposite*
   direction (σ growing unchecked). The reward-shaping ablation's flat σ
   traces (1.00→~1.02 across 7 of the 9 variants) show this isn't
   universal, though: the two exceptions (`v3_gentle_terminal`,
   `v3_rebalanced_oar` — the pair deliberately run with early stopping
   disabled for the full 2000-episode budget) *do* show real σ growth
   (1.00→1.10), reinforcing that σ growth is tied to **how long a run is
   allowed to continue**, the same update-aggressiveness/duration factor as
   Lesson 1, not to the reward formulation itself. Separately, the per-patient
   trajectory analysis (`technical_summary.md` §10.6) shows that even a
   *stable*, well-converged policy (`sweep2_heavy_reg`'s `best.pt`) realizes
   very low beam-profile drift (~0.01–0.05) across the 35-fraction course —
   i.e. **σ measured in parameter space does not guarantee meaningfully
   varied behaviour across time-steps**; the agent samples noisily around a
   nearly time-invariant mean policy rather than adapting its strategy
   fraction-to-fraction, despite the sequential MDP being designed to allow
   exactly that.

6. **Every configuration tested, across every experiment, underdoses every
   PTV relative to prescription** (best D95_PTV70 achieved anywhere in this
   project is ~44 Gy against a 70 Gy prescription — about 63%), **and not a
   single one ever got parotid dose under its 26 Gy tolerance** — every
   variant tested, across both hyperparameter regimes and all 9
   reward-shaping variants, lands at 30–44 Gy on both parotids. Some of the
   coverage ceiling is structural (the dose-influence matrix only reaches
   ~70–85% of body voxels, `technical_summary.md` §5.3/§13), but the
   reward-shaping ablation's own attempts to specifically push PTV coverage
   harder (Improvement A, the quadratic gap) made things *worse* on first
   try, and even the eventual best reward-shaping variant found
   (`v3_oar_only`, isolating a softened `lambda_oar`) only reached DVH 26.22
   — better than every other shaping attempt, but still short of `control`'s
   23.34, and still over parotid tolerance. The two lowest parotid-dose
   results in the whole project (`C+A`'s 20.70/24.41, `v2_full`'s
   32.81/30.85) only get there by sacrificing PTV coverage so badly they
   aren't viable. **This is the project's central unresolved tension**: nine
   reward-shaping variants and four hyperparameter regimes have explored
   the same PTV-coverage/OAR-sparing frontier from many directions without
   ever pushing the frontier itself outward — the reward-shaping
   investigation is closed out on this basis, with `control` as the
   standing recommendation and the barrier-floor fix / `lambda_oar`
   softening kept as validated-but-insufficient building blocks for a
   future attempt rather than the active configuration.

7. **Convergence speed and final quality are governed by different knobs.**
   γ and update-aggressiveness controlled *how fast* a run reached its
   plateau (24 episodes for the fastest, 1324+ for the slowest), but the
   *quality* of that plateau was governed by a largely separate set of
   factors (which regime, which reward shaping) — the fastest-converging
   configurations in this project (γ=0.995, hyperparameter sweep3) were
   consistently among the *worst*-quality or least stable, while the
   slowest (`sweep2_heavy_reg`) was the best. **Treat "converges quickly" as
   a yellow flag worth checking against long-run stability, not a
   straightforward virtue, on this task.**

---

### Appendix: how the derived quantities in this document were computed

- **`conv_ep`**: first episode at which logged `val_dvh` is within 5% of that
  run's eventual minimum `val_dvh`.
- **`degradation`**: `100 × (final_val_dvh − best_val_dvh) / best_val_dvh`.
- **action σ**: the actor's `log_std` is a single scalar broadcast across all
  2304 beamlet dimensions, so the logged `entropy` (sum of per-dimension
  Gaussian entropies before the §0(a) normalization fix, or that sum divided
  by ~30 minibatches after it) satisfies
  `entropy ≈ n_beamlets × (ln σ + 0.5·ln(2πe))`. Solving for σ:
  `σ = exp(entropy / n_beamlets / [divisor] − 0.5·ln(2πe))`, with
  `divisor = 30` for pre-fix-era runs (Production, gamma sweep) and `1`
  otherwise. This converts the two incompatible logging scales (§0a) into one
  directly comparable, physically-interpretable quantity.
- **reward/DVH correlation**: Pearson correlation between `rolling_mean_reward`
  and `val_dvh` over every episode in a run where `val_dvh` was logged
  (i.e. every `best_eval_every`-th episode).
- All values were computed by loading every `runs/**/train_log.csv` with
  pandas and are reproducible from the raw CSVs checked into (gitignored,
  locally present) `runs/`.
