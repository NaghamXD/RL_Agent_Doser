"""Visualize one or more patients' full 35-fraction trajectory under a
trained policy.

Two questions this answers, for a single patient course:
  1. Beam usage: does the agent pick some beams and not others, and how does
     the planned intensity per beam evolve fraction-to-fraction?
  2. Reward: how does the dense per-fraction shaping reward evolve, and how
     much does the terminal (end-of-course) reward contribute on top?

Unlike evaluate.py's run_one (which only keeps a running *sum* of the
action across all 35 fractions, and the diagnostic oar_penalty/ptv_reward
pair for charts), this keeps the full per-fraction action and the actual
env reward returned by step() -- the dense potential-based shaping signal
that drives PPO in sequential mode, not just the diagnostics.

When more than one patient is run, also writes a trajectory_summary.csv
(one row per patient: dominant beam, how concentrated/static the beam
profile is, where -- if anywhere -- the shaping reward flips sign and
stays flipped, and the shaping/terminal/total reward split) so the two
patterns noticed on a single patient (a static, non-time-varying beam
profile; a sharp negative "cliff" partway through the course) can be
checked for whether they generalize across the validation set.

Usage
-----
python scripts/visualize_patient_trajectory.py --config ... --ckpt ...        # 1st patient only
python scripts/visualize_patient_trajectory.py --config ... --ckpt ... --patient pt_123
python scripts/visualize_patient_trajectory.py --config ... --ckpt ... --all  # every patient in the split
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.config import load_config
from src.env.dose_env import DoseEnv
from src.agents.ppo import PPO
from src.utils.visualize import (plot_beam_intensity_trajectory,
                                 plot_reward_trajectory)


def _run_patient(cfg, agent: PPO, patient: str, split: str, out_dir: Path) -> dict:
    env = DoseEnv(cfg, patient_ids=[patient], split=split)
    state, fraction_progress = env.reset(patient)

    n_beams, beamlet_h, beamlet_w = cfg.n_beams, cfg.beamlet_h, cfg.beamlet_w
    n_fractions = cfg.n_fractions
    beam_intensity = np.zeros((n_fractions, n_beams), dtype=np.float32)
    rewards = np.zeros(n_fractions, dtype=np.float32)
    terminal_reward = 0.0
    dvh = float("nan")

    patient_done = False
    fraction_index = 0
    while not patient_done:
        action, _raw, _logp, _value = agent.act(
            state, fraction_progress, deterministic=True)
        action_3d = action.reshape(n_beams, beamlet_h, beamlet_w)
        state, fraction_progress, reward, _done, info = env.step(action_3d)
        beam_intensity[fraction_index] = action_3d.mean(axis=(1, 2))
        rewards[fraction_index] = reward
        patient_done = bool(info["patient_done"])
        if patient_done:
            terminal_reward = float(info["terminal_reward"])
            dvh = float(info["dvh_score"])
        fraction_index += 1

    # The last fraction's reward already includes the terminal reward added
    # on top (see DoseEnv.step); subtract it back out so every point in the
    # line plot/summary is the dense shaping signal alone, on the same scale.
    rewards[-1] -= terminal_reward

    out_dir.mkdir(parents=True, exist_ok=True)
    plot_beam_intensity_trajectory(beam_intensity, n_beams, out_dir, patient)
    plot_reward_trajectory(rewards, terminal_reward, dvh, out_dir, patient)

    totals = beam_intensity.sum(axis=0)
    dominant_beam = int(np.argmax(totals))
    top_beam_share = float(totals.max() / (totals.sum() + 1e-8))
    # Coefficient-of-variation of each beam's intensity *across fractions*,
    # averaged over beams: ~0 means every beam keeps the same weight all
    # course (a static plan repeated 35x); higher means the agent actually
    # re-balances beam allocation over time.
    beam_profile_drift = float(
        (beam_intensity.std(axis=0) / (beam_intensity.mean(axis=0) + 1e-8)).mean()
    )

    signs = np.sign(rewards)
    nonzero = signs[signs != 0]
    if nonzero.size and not np.all(nonzero == nonzero[0]):
        # last index (1-based fraction) where the sign changed and never
        # changed back -- the "cliff" fraction.
        last_change = np.where(np.diff(nonzero))[0]
        cliff_fraction = int(last_change[-1]) + 2 if last_change.size else None
    else:
        cliff_fraction = None

    return {
        "patient": patient,
        "dominant_beam": dominant_beam,
        "top_beam_share": top_beam_share,
        "beam_profile_drift": beam_profile_drift,
        "cliff_fraction": cliff_fraction,
        "sum_shaping_reward": float(rewards.sum()),
        "terminal_reward": terminal_reward,
        "total_return": float(rewards.sum() + terminal_reward),
        "dvh_score": dvh,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--split", default=None,
                        help="defaults to cfg.eval_split")
    parser.add_argument("--patient", default=None,
                        help="single patient id (defaults to the first, sorted, "
                             "in the split unless --all is given)")
    parser.add_argument("--all", action="store_true",
                        help="run every patient in the split instead of just one")
    parser.add_argument("--out-dir", default=None,
                        help="parent dir for per-patient subfolders; defaults "
                             "to <ckpt_dir>/trajectory_charts/")
    args = parser.parse_args()

    cfg = load_config(args.config)
    split = args.split or cfg.eval_split

    probe_env = DoseEnv(cfg, split=split)
    all_patients = sorted(probe_env.patient_ids)
    if args.all:
        patients = all_patients
    elif args.patient is not None:
        patients = [args.patient]
    else:
        patients = [all_patients[0]]

    state, _ = probe_env.reset(all_patients[0])
    agent = PPO(cfg, in_channels=state.shape[0])
    agent.load(args.ckpt)
    agent.net.eval()

    out_root = (Path(args.out_dir) if args.out_dir
               else Path(args.ckpt).parent / "trajectory_charts")

    rows = []
    for patient in patients:
        row = _run_patient(cfg, agent, patient, split, out_root / patient)
        rows.append(row)
        print(f"[visualize_patient_trajectory] {patient}: "
              f"dominant_beam=B{row['dominant_beam']} "
              f"top_share={row['top_beam_share']:.2f} "
              f"drift={row['beam_profile_drift']:.3f} "
              f"cliff_fx={row['cliff_fraction']} "
              f"dvh={row['dvh_score']:.2f}")

    if len(rows) > 1:
        summary_df = pd.DataFrame(rows)
        summary_csv = out_root / "trajectory_summary.csv"
        summary_df.to_csv(summary_csv, index=False)
        print(f"\n[visualize_patient_trajectory] saved {summary_csv}")
        print(summary_df.describe(include="all").to_string())


if __name__ == "__main__":
    main()
