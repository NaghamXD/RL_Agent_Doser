"""Validation vs. held-out test-set comparison for the best model (control).

Parses both per-patient evaluate.py logs (validation, n=40, and test, n=100)
and produces:

  A. Box plot of per-patient DVH score distribution, validation vs test.
  B. Grouped bar chart of every clinical metric's mean, validation vs test,
     showing the model generalizes (no metric drifts meaningfully worse on
     genuinely unseen patients).

Usage
-----
python scripts/plot_test_set_results.py \
  --val-log <path to evaluate.py log run with --split validation> \
  --test-log <path to evaluate.py log run with --split test>
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = Path(__file__).resolve().parents[1] / "reports"


def _parse_eval_log(path: Path) -> pd.DataFrame:
    rows, header = [], None
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line.startswith("Patient"):
            header = [h.strip() for h in line.split("|")]
        elif header and (line.startswith("pt_") or line.startswith("MEAN")):
            rows.append([v.strip() for v in line.split("|")])
    df = pd.DataFrame(rows, columns=header)
    for col in header[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[df["Patient"] != "MEAN"].reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-log", required=True)
    parser.add_argument("--test-log", required=True)
    args = parser.parse_args()

    val_df = _parse_eval_log(args.val_log)
    test_df = _parse_eval_log(args.test_log)
    val_df.to_csv(OUT_DIR / "test_set_validation_per_patient.csv", index=False)
    test_df.to_csv(OUT_DIR / "test_set_test_per_patient.csv", index=False)

    fig, (ax_box, ax_bar) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("control (sweep2_heavy_reg) — validation (n=40) vs. "
                "held-out test set (n=100)",
                fontsize=13, fontweight="bold")

    # --- Panel A: DVH distribution ---
    box = ax_box.boxplot([val_df["DVH"], test_df["DVH"]],
                        tick_labels=[f"validation\n(n={len(val_df)})",
                                    f"test\n(n={len(test_df)})"],
                        patch_artist=True, widths=0.5)
    for patch, color in zip(box["boxes"], ["#4c72b0", "#2ca02c"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    for i, df in enumerate([val_df, test_df], start=1):
        ax_box.scatter([i] * len(df), df["DVH"], alpha=0.35, s=18,
                      color="black", zorder=3)
        ax_box.text(i, df["DVH"].max() + 1, f"mean={df['DVH'].mean():.2f}",
                   ha="center", fontsize=9, fontweight="bold")
    ax_box.set_ylabel("DVH score (lower = better)")
    ax_box.set_title("A. Per-patient DVH score distribution")
    ax_box.grid(True, alpha=0.25, axis="y")

    # --- Panel B: every clinical metric, mean comparison ---
    metric_cols = ["DVH", "D95_PTV70", "D95_PTV63", "D95_PTV56",
                  "Brainstem(54Gy)", "SpinalCord(45Gy)", "Mandible(70Gy)",
                  "LeftParotid(26Gy)", "RightParotid(26Gy)"]
    metric_cols = [c for c in metric_cols if c in val_df.columns]
    val_means = [val_df[c].mean() for c in metric_cols]
    test_means = [test_df[c].mean() for c in metric_cols]

    x = range(len(metric_cols))
    width = 0.35
    ax_bar.bar([i - width / 2 for i in x], val_means, width,
              label=f"validation (n={len(val_df)})", color="#4c72b0")
    ax_bar.bar([i + width / 2 for i in x], test_means, width,
              label=f"test (n={len(test_df)})", color="#2ca02c")
    ax_bar.set_xticks(list(x))
    ax_bar.set_xticklabels([c.replace("(", "\n(") for c in metric_cols],
                          fontsize=8)
    ax_bar.set_ylabel("Gy (or score)")
    ax_bar.set_title("B. Every clinical metric, mean comparison")
    ax_bar.legend(fontsize=9)
    ax_bar.grid(True, alpha=0.25, axis="y")

    fig.tight_layout()
    png_path = OUT_DIR / "test_set_results.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_test_set_results] saved {png_path}")


if __name__ == "__main__":
    main()
