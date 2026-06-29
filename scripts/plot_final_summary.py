"""Presentation-ready summary of the reward-shaping investigation.

Consolidates every variant evaluated in reports/reward_shaping_sweep.md
(control + all 9 C/A/B + v2/v3 variants) into one CSV and one 2-panel
figure:

  A. DVH score ranking, lower=better, control highlighted as the winner.
  B. The PTV-coverage / parotid-overdose frontier: every variant plotted
     as (D95_PTV70, mean parotid dose), with the Rx=70Gy and
     tolerance=26Gy reference lines marked, showing visually that no
     variant ever lands in the bottom-right "both targets met" region.

Numbers are taken directly from evaluate.py runs already reported in
reports/reward_shaping_sweep.md (not re-derived here).

Usage
-----
python scripts/plot_final_summary.py
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = Path(__file__).resolve().parents[1] / "reports"

# variant -> (DVH, D95_PTV70, LeftParotid, RightParotid)
DATA = {
    "control":             (23.34, 43.97, 39.32, 43.21),
    "C":                   (26.82, 33.76, 31.12, 31.44),
    "C_A":                 (32.08, 27.70, 20.70, 24.41),
    "C_A_retuned":         (26.82, 33.09, 30.46, 33.27),
    "C_A_B":               (25.04, 35.29, 40.14, 39.08),
    "v2_barrier":          (29.40, 31.83, 39.27, 39.62),
    "v2_full":             (31.09, 26.64, 32.81, 30.85),
    "v3_gentle_terminal":  (29.55, 22.87, 37.28, 34.16),
    "v3_rebalanced_oar":   (26.91, 33.89, 32.23, 35.31),
    "v3_oar_only":         (26.22, 35.22, 34.45, 38.44),
}
PTV70_RX = 70.0
PAROTID_TOLERANCE = 26.0


def main():
    df = pd.DataFrame(
        [(name, *vals) for name, vals in DATA.items()],
        columns=["variant", "DVH", "D95_PTV70", "LeftParotid", "RightParotid"],
    )
    df["mean_parotid"] = df[["LeftParotid", "RightParotid"]].mean(axis=1)
    df = df.sort_values("DVH").reset_index(drop=True)
    csv_path = OUT_DIR / "final_summary.csv"
    df.to_csv(csv_path, index=False)

    fig, (ax_bar, ax_frontier) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Reward-shaping investigation — final results "
                "(40-patient validation split)",
                fontsize=13, fontweight="bold")

    # --- Panel A: DVH ranking ---
    colors = ["#2ca02c" if v == "control" else "#4c72b0" for v in df["variant"]]
    bars = ax_bar.barh(df["variant"], df["DVH"], color=colors)
    ax_bar.invert_yaxis()
    ax_bar.set_xlim(0, df["DVH"].max() * 1.25)
    ax_bar.set_xlabel("DVH score (lower = better)")
    ax_bar.set_title("A. Every variant tried, ranked")
    for bar, value in zip(bars, df["DVH"]):
        ax_bar.text(value + 0.3, bar.get_y() + bar.get_height() / 2,
                   f"{value:.2f}", va="center", fontsize=9)
    ax_bar.text(0.98, 0.5, "control = winner (green)\nnone of the 9 reward-shaping\nvariants beat it",
               transform=ax_bar.transAxes, ha="right", va="center", fontsize=9,
               bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f4ff", edgecolor="#aabbdd"))

    # --- Panel B: coverage / OAR-overdose frontier ---
    # manual label-offset overrides for points that would otherwise collide
    label_offset = {
        "C_A_B": (6, 8), "v2_barrier": (6, -10),
        "C": (6, -10), "C_A_retuned": (8, 6),
    }
    for _, row in df.iterrows():
        is_control = row["variant"] == "control"
        ax_frontier.scatter(row["D95_PTV70"], row["mean_parotid"],
                           s=140 if is_control else 80,
                           color="#2ca02c" if is_control else "#4c72b0",
                           edgecolor="black", linewidth=0.8, zorder=3)
        ax_frontier.annotate(row["variant"], (row["D95_PTV70"], row["mean_parotid"]),
                            textcoords="offset points",
                            xytext=label_offset.get(row["variant"], (6, 4)),
                            fontsize=8)

    ax_frontier.axvline(PTV70_RX, color="gray", ls="--", lw=1)
    ax_frontier.axhline(PAROTID_TOLERANCE, color="gray", ls="--", lw=1)
    ax_frontier.text(PTV70_RX - 1, ax_frontier.get_ylim()[1] * 0.97,
                    "Rx = 70 Gy", rotation=90, va="top", ha="right",
                    fontsize=8, color="gray")
    ax_frontier.text(ax_frontier.get_xlim()[1], PAROTID_TOLERANCE + 0.5,
                    "tolerance = 26 Gy", ha="right", va="bottom",
                    fontsize=8, color="gray")
    # shade the "both targets met" region (full coverage, under tolerance)
    ax_frontier.axvspan(PTV70_RX, ax_frontier.get_xlim()[1] + 5,
                       ymin=0, ymax=(PAROTID_TOLERANCE - ax_frontier.get_ylim()[0])
                       / (ax_frontier.get_ylim()[1] - ax_frontier.get_ylim()[0]),
                       color="#2ca02c", alpha=0.08, zorder=0)
    ax_frontier.set_xlabel("D95_PTV70 (Gy) — higher = better coverage")
    ax_frontier.set_ylabel("mean parotid dose (Gy) — lower = safer")
    ax_frontier.set_title("B. No variant ever reaches the target region\n"
                         "(full coverage AND under tolerance)")
    ax_frontier.grid(True, alpha=0.25)

    fig.tight_layout()
    png_path = OUT_DIR / "final_summary.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[plot_final_summary] saved {csv_path}")
    print(f"[plot_final_summary] saved {png_path}")


if __name__ == "__main__":
    main()
