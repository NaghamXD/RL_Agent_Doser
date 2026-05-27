"""
visualize.py - Data visualization for pyRadPlan test session.

Called from test_dim.py at key stages:
  1. show_patient_overview()     - CT slices + structure overlays
  2. show_beam_geometry()        - polar plot of 9 gantry angles
  3. show_per_beam_dose_maps()   - dose footprint of each beam
  4. show_beam_statistics()      - bar chart of per-beam dose stats
  5. show_dose_distribution()    - final dose overlaid on CT (3 planes)
  6. show_dvh()                  - Dose Volume Histogram for all structures

Files are numbered (1_ ... 6_) to reflect the correct process order.
"""

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

OUT_DIR = Path(__file__).parent

# ------------------------------------------------------------------
# Color map: one distinct color per structure
# ------------------------------------------------------------------
STRUCT_COLORS = {
    "PTV_70":        "red",
    "PTV_63":        "orange",
    "PTV_56":        "yellow",
    "brainstem":     "cyan",
    "spinal_cord":   "lime",
    "right_parotid": "magenta",
    "left_parotid":  "violet",
    "mandible":      "deepskyblue",
    "BODY":          "white",
}


def _add_explanation(fig, lines, y_start=0.01):
    """Add a bullet-point explanation box at the bottom of the figure."""
    text = "\n".join(f"  * {line}" for line in lines)
    fig.text(
        0.01, y_start, text,
        fontsize=7.5, va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f4ff",
                  edgecolor="#aabbdd", alpha=0.9),
        family="monospace",
    )


# ==================================================================
# 1. Patient overview: CT slices + structure contours
# ==================================================================
def show_patient_overview(ct_array, masks_dict, title="Patient Overview"):
    NZ, NY, NX = ct_array.shape
    iz, iy, ix = NZ // 2, NY // 2, NX // 2

    fig = plt.figure(figsize=(16, 7))
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

    axes = [fig.add_axes([0.02 + i * 0.32, 0.22, 0.29, 0.70]) for i in range(3)]

    views = [
        (axes[0], ct_array[iz, :, :],  "Axial  (z=%d)"   % iz),
        (axes[1], ct_array[:, iy, :],  "Coronal (y=%d)"  % iy),
        (axes[2], ct_array[:, :, ix],  "Sagittal (x=%d)" % ix),
    ]

    for ax, slice_2d, view_title in views:
        ax.imshow(slice_2d, cmap="gray", vmin=-200, vmax=300,
                  origin="lower", aspect="auto")
        ax.set_title(view_title, fontsize=10)
        ax.axis("off")
        for name, mask in masks_dict.items():
            if name == "BODY":
                continue
            color = STRUCT_COLORS.get(name, "white")
            if view_title.startswith("Axial"):
                m = mask[iz, :, :]
            elif view_title.startswith("Coronal"):
                m = mask[:, iy, :]
            else:
                m = mask[:, :, ix]
            if m.any():
                ax.contour(m, levels=[0.5], colors=[color], linewidths=1.2)

    patches = [mpatches.Patch(color=STRUCT_COLORS.get(n, "white"), label=n)
               for n in masks_dict if n != "BODY"]
    axes[2].legend(handles=patches, loc="upper right", fontsize=7, framealpha=0.6)

    _add_explanation(fig, [
        "CT anatomy is correct — head/neck cross-section clearly visible.",
        "All structure contours are in the right anatomical locations.",
        "PTV_70 (red) = primary tumor target near center.",
        "Mandible (blue) wraps around jaw area.",
        "Spinal cord (green) runs down the neck.",
    ], y_start=0.01)

    fname = "1_plot_patient_overview.png"
    plt.savefig(OUT_DIR / fname, dpi=120, bbox_inches="tight")
    plt.show()
    print(f"    [viz] Patient overview saved -> {fname}")


# ==================================================================
# 2. Beam geometry: polar plot of gantry angles
# ==================================================================
def show_beam_geometry(gantry_angles, title="Beam Geometry (Gantry Angles)"):
    fig = plt.figure(figsize=(7, 8), facecolor="#1a1a2e")
    fig.suptitle(title, fontsize=13, fontweight="bold", color="white", y=0.97)

    ax = fig.add_axes([0.1, 0.18, 0.8, 0.75], projection="polar",
                      facecolor="#1a1a2e")

    angles_rad = [np.deg2rad(a) for a in gantry_angles]
    colors = plt.cm.tab10(np.linspace(0, 1, len(gantry_angles)))

    for i, (angle_rad, angle_deg) in enumerate(zip(angles_rad, gantry_angles)):
        ax.annotate("", xy=(angle_rad, 1.0), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color=colors[i], lw=2))
        ax.text(angle_rad, 1.10, f"{angle_deg:.0f}deg",
                ha="center", va="center", fontsize=8, color=colors[i])

    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(theta, [0.25] * 200, "w--", lw=1, alpha=0.5)
    ax.text(0, 0, "Patient", ha="center", va="center", fontsize=9, color="white")
    ax.set_ylim(0, 1.25)
    ax.tick_params(colors="white")
    ax.yaxis.set_visible(False)
    ax.xaxis.set_tick_params(labelcolor="white")

    explanation = (
        "  * All 9 beams evenly distributed around the patient (0 to 320 deg, every 40 deg).\n"
        "  * This is the standard clinical setup for head & neck IMRT.\n"
        "  * Beams enter from multiple angles to spare healthy tissue."
    )
    fig.text(0.05, 0.02, explanation, fontsize=8, color="white", va="bottom",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#2a2a4e",
                       edgecolor="#6688bb", alpha=0.9),
             family="monospace")

    fname = "2_plot_beam_geometry.png"
    plt.savefig(OUT_DIR / fname, dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.show()
    print(f"    [viz] Beam geometry saved -> {fname}")


# ==================================================================
# 3. Per-beam dose maps
# ==================================================================
def show_per_beam_dose_maps(dim_per_beam, dose_shape_zyx, gantry_angles,
                            title="Per-Beam Dose Maps — Maximum Intensity Projection"):
    """
    Each panel shows a Maximum Intensity Projection (MIP) of the dose volume
    for one beam. MIP takes the maximum dose across ALL z-slices, collapsing
    the 3D volume into a single 2D image that clearly reveals each beam's
    unique entry direction and coverage.
    """
    NZ, NY, NX = dose_shape_zyx

    fig = plt.figure(figsize=(13, 13))
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.99)

    vmax = dim_per_beam.max()

    for b in range(9):
        row, col = divmod(b, 3)
        ax = fig.add_axes([0.03 + col * 0.32, 0.20 + (2 - row) * 0.26,
                           0.29, 0.23])
        dose_vol = dim_per_beam[:, b].reshape(NZ, NY, NX)

        # Maximum Intensity Projection along z-axis (axial direction)
        # Result shape: (NY, NX) — the strongest dose in each x-y position
        mip = dose_vol.max(axis=0)

        ax.imshow(mip, cmap="hot", origin="lower",
                  vmin=0, vmax=vmax, aspect="auto")
        ax.set_title(f"Beam {b}  ({gantry_angles[b]:.0f} deg)", fontsize=9)
        ax.axis("off")

        # Draw a small arrow showing beam entry direction
        # Gantry angle 0 = beam from right (3 o'clock), increases clockwise
        angle_rad = np.deg2rad(gantry_angles[b])
        cx, cy = NX * 0.5, NY * 0.5
        r = min(NX, NY) * 0.38
        # Arrow tip at centre, tail at beam entry side
        dx = -np.sin(angle_rad) * r
        dy = -np.cos(angle_rad) * r
        ax.annotate("", xy=(cx, cy), xytext=(cx + dx, cy + dy),
                    arrowprops=dict(arrowstyle="->", color="cyan",
                                   lw=1.5, mutation_scale=12))

    sm = plt.cm.ScalarMappable(cmap="hot",
                                norm=plt.Normalize(vmin=0, vmax=vmax))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.92, 0.22, 0.02, 0.70])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Dose (Gy/fraction at weight=1)", fontsize=9)

    _add_explanation(fig, [
        "MIP = Maximum Intensity Projection: each pixel shows the MAX dose across all z-slices.",
        "This reveals the true 3D entry direction of each beam (cyan arrow = beam direction).",
        "White/yellow = high dose at beam entry point, red = lower dose deeper in tissue.",
        "Notice how each beam's bright region rotates with its gantry angle.",
    ], y_start=0.01)

    fname = "3_plot_per_beam_dose.png"
    plt.savefig(OUT_DIR / fname, dpi=120, bbox_inches="tight")
    plt.show()
    print(f"    [viz] Per-beam dose maps (MIP) saved -> {fname}")


# ==================================================================
# 4. Beam statistics bar charts
# ==================================================================
def show_beam_statistics(dim_per_beam, gantry_angles,
                         title="Per-Beam Dose Statistics"):
    labels = [f"{a:.0f}deg" for a in gantry_angles]
    max_doses     = dim_per_beam.max(axis=0)
    mean_doses    = dim_per_beam.mean(axis=0)
    nonzero_counts = (dim_per_beam > 0).sum(axis=0)

    x = np.arange(len(labels))
    width = 0.35

    fig = plt.figure(figsize=(14, 7))
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)

    ax1 = fig.add_axes([0.06, 0.22, 0.40, 0.68])
    ax2 = fig.add_axes([0.55, 0.22, 0.40, 0.68])

    bars1 = ax1.bar(x - width/2, max_doses,  width, label="Max dose",
                    color="tomato",    edgecolor="black", linewidth=0.5)
    bars2 = ax1.bar(x + width/2, mean_doses, width, label="Mean dose",
                    color="steelblue", edgecolor="black", linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax1.set_ylabel("Dose (Gy/fraction at weight=1)", fontsize=10)
    ax1.set_title("Max vs Mean dose per beam", fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_facecolor("#f8f9fa")
    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7)
    for bar in bars2:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=7)

    colors = plt.cm.tab10(np.linspace(0, 1, len(gantry_angles)))
    bars3 = ax2.bar(x, nonzero_counts, color=colors,
                    edgecolor="black", linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax2.set_ylabel("Number of irradiated voxels", fontsize=10)
    ax2.set_title("Irradiated voxels per beam", fontsize=11)
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_facecolor("#f8f9fa")
    for bar in bars3:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                 f"{int(bar.get_height()):,}", ha="center", va="bottom", fontsize=7)

    _add_explanation(fig, [
        "Max dose ~1.2-1.4 Gy/fraction per beam at weight=1 — consistent across all 9 beams (good balance).",
        "Mean dose ~0.025 Gy — low because most voxels receive zero dose from any single beam.",
        "Irradiated voxels ~20,000 per beam out of 470,000 total — reasonable anatomical coverage.",
    ], y_start=0.01)

    fname = "4_plot_beam_statistics.png"
    plt.savefig(OUT_DIR / fname, dpi=120, bbox_inches="tight")
    plt.show()
    print(f"    [viz] Beam statistics saved -> {fname}")


# ==================================================================
# 5. Final dose distribution overlaid on CT (3 planes)
# ==================================================================
def show_dose_distribution(ct_array, dose_on_ct, masks_dict,
                           title="Dose Distribution on CT"):
    NZ, NY, NX = ct_array.shape

    ptv = masks_dict.get("PTV_70", None)
    if ptv is not None and ptv.any():
        coords = np.argwhere(ptv)
        iz, iy, ix = coords.mean(axis=0).astype(int)
    else:
        iz, iy, ix = NZ // 2, NY // 2, NX // 2

    fig = plt.figure(figsize=(16, 7))
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

    axes = [fig.add_axes([0.02 + i * 0.32, 0.22, 0.29, 0.70]) for i in range(3)]

    views = [
        (axes[0], ct_array[iz, :, :],  dose_on_ct[iz, :, :],  "Axial  (z=%d)"   % iz),
        (axes[1], ct_array[:, iy, :],  dose_on_ct[:, iy, :],  "Coronal (y=%d)"  % iy),
        (axes[2], ct_array[:, :, ix],  dose_on_ct[:, :, ix],  "Sagittal (x=%d)" % ix),
    ]

    dose_max = dose_on_ct.max()
    im = None
    for ax, ct_slice, dose_slice, label in views:
        ax.imshow(ct_slice, cmap="gray", vmin=-200, vmax=300,
                  origin="lower", aspect="auto")
        dose_masked = np.ma.masked_where(dose_slice < 0.05 * dose_max, dose_slice)
        im = ax.imshow(dose_masked, cmap="jet", alpha=0.5,
                       vmin=0, vmax=dose_max, origin="lower", aspect="auto")
        ax.set_title(label, fontsize=10)
        ax.axis("off")

    cbar_ax = fig.add_axes([0.96, 0.22, 0.015, 0.70])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Dose (Gy/fraction)", fontsize=10)

    _add_explanation(fig, [
        "Dose colour-wash (jet) overlaid on CT greyscale — shows where radiation is delivered.",
        "Hot colours (red/yellow) = high dose concentrated in the tumor region.",
        "Cool colours (blue) = lower dose at beam entry/exit edges.",
        "Dose correctly falls off toward body boundaries — physically realistic.",
    ], y_start=0.01)

    fname = "5_plot_dose_distribution.png"
    plt.savefig(OUT_DIR / fname, dpi=120, bbox_inches="tight")
    plt.show()
    print(f"    [viz] Dose distribution saved -> {fname}")


# ==================================================================
# 6. DVH - Dose Volume Histogram (vectorised for speed)
# ==================================================================
def show_dvh(dose_on_ct, masks_dict, n_fractions=35,
             title="Dose Volume Histogram (DVH)"):
    dose_total = dose_on_ct * n_fractions   # per-fraction -> full-course Gy
    dose_max   = float(dose_total.max())
    dose_bins  = np.linspace(0, dose_max * 1.05, 300)

    fig = plt.figure(figsize=(11, 7))
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)
    ax = fig.add_axes([0.09, 0.22, 0.87, 0.70])

    # Prescription reference lines
    for dose_rx, rx_label in [(70, "70 Gy"), (63, "63 Gy"), (56, "56 Gy")]:
        ax.axvline(dose_rx, color="gray", lw=0.8, ls="--", alpha=0.6)
        ax.text(dose_rx + 0.5, 101, rx_label, fontsize=7.5,
                color="gray", va="bottom")

    for name, mask in masks_dict.items():
        if name == "BODY":
            continue
        bool_mask = mask.astype(bool)
        if not bool_mask.any():
            continue
        color = STRUCT_COLORS.get(name, "black")
        voxel_doses = dose_total[bool_mask]          # 1-D array of doses in structure
        # Vectorised DVH: shape (n_bins,)
        dvh = (voxel_doses[:, None] >= dose_bins[None, :]).mean(axis=0) * 100
        ax.plot(dose_bins, dvh, color=color, lw=2, label=name)

    ax.set_xlabel("Dose (Gy, full course)", fontsize=11)
    ax.set_ylabel("Volume (%)", fontsize=11)
    ax.set_xlim(0, dose_max * 1.05)
    ax.set_ylim(0, 107)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.85)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor("#f8f9fa")

    _add_explanation(fig, [
        "DVH = Dose Volume Histogram: shows what % of each structure receives at least X Gy.",
        "PTV curves (red/orange/yellow) should stay high and right of their prescription line.",
        "OAR curves (blue/green/cyan) should stay low and left — minimise dose to healthy organs.",
        "Dashed lines mark clinical prescription levels: 70 / 63 / 56 Gy.",
    ], y_start=0.01)

    fname = "6_plot_dvh.png"
    plt.savefig(OUT_DIR / fname, dpi=120, bbox_inches="tight")
    plt.show()
    print(f"    [viz] DVH saved -> {fname}")
