"""Generate the published figures for machine-learning-fmri.html.

The MNI152 template is bundled with Nilearn. The Schaefer atlas is downloaded
by Nilearn into a cache outside this repository. All other plotted data are
deterministic teaching data, not measurements from ABIDE participants.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/machine-learning-fmri-matplotlib")
os.environ.setdefault("NILEARN_DATA", "/tmp/machine-learning-fmri-nilearn")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


OUT = Path(__file__).resolve().parent
INK = "#20201e"
MUTED = "#67655f"
BLUE = "#1d6c9e"
ORANGE = "#b26935"
GREEN = "#287352"
RED = "#a34f4f"
VIOLET = "#6c5b99"
GOLD = "#a87524"
PAPER = "#fffdf9"
COLORS = [BLUE, ORANGE, GREEN, RED, VIOLET, GOLD, "#3f7f89", "#9a5f7a"]


def save(fig: plt.Figure, name: str, dpi: int = 155) -> None:
    fig.savefig(OUT / name, dpi=dpi, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)


def clean_axis(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def brain_views() -> np.ndarray:
    """Create the hero from Nilearn's MNI152 average anatomical template."""
    from nilearn.datasets import load_mni152_template

    image = load_mni152_template(resolution=2)
    data = image.get_fdata()
    cuts = (data.shape[0] // 2, data.shape[1] // 2, data.shape[2] // 2)
    planes = [
        np.rot90(data[cuts[0], :, :]),
        np.rot90(data[:, cuts[1], :]),
        np.rot90(data[:, :, cuts[2]]),
    ]
    labels = ["Sagittal (side)", "Coronal (front)", "Axial (top)"]
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.2), facecolor=PAPER)
    for ax, plane, label in zip(axes, planes, labels):
        ax.imshow(plane, cmap="gray", interpolation="bilinear", vmin=0, vmax=1)
        ax.set_title(label, color=INK, fontsize=13, pad=10, weight="semibold")
        clean_axis(ax)
    fig.suptitle("One anatomical reference, viewed in three perpendicular planes", fontsize=18,
                 color=INK, y=1.01, weight="semibold")
    fig.text(0.5, -0.01, "MNI152 population-average anatomical template distributed with Nilearn",
             ha="center", color=MUTED, fontsize=10)
    fig.tight_layout()
    save(fig, "mni152_brain_views.png")
    return data


def atlas_views(template: np.ndarray) -> None:
    """Overlay a real Schaefer atlas on the MNI152 reference."""
    import nibabel as nib
    from nilearn.datasets import fetch_atlas_schaefer_2018

    atlas = fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=7, resolution_mm=2)
    labels = np.asanyarray(nib.load(atlas.maps).dataobj)
    cuts = (labels.shape[0] // 2, labels.shape[1] // 2, labels.shape[2] // 2)
    anatomy_planes = [
        np.rot90(template[cuts[0], :, :]),
        np.rot90(template[:, cuts[1], :]),
        np.rot90(template[:, :, cuts[2]]),
    ]
    atlas_planes = [
        np.rot90(labels[cuts[0], :, :]),
        np.rot90(labels[:, cuts[1], :]),
        np.rot90(labels[:, :, cuts[2]]),
    ]
    cmap = ListedColormap(COLORS * 13)
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.2), facecolor=PAPER)
    for ax, anatomical, parcel, title in zip(
        axes, anatomy_planes, atlas_planes, ["Sagittal", "Coronal", "Axial"]
    ):
        ax.imshow(anatomical, cmap="gray", vmin=0, vmax=1)
        masked = np.ma.masked_where(parcel == 0, parcel)
        ax.imshow(masked, cmap=cmap, alpha=0.58, interpolation="nearest", vmin=1, vmax=100)
        ax.set_title(title, color=INK, fontsize=13, weight="semibold")
        clean_axis(ax)
    fig.suptitle("Schaefer 2018 cortical atlas: 100 parcels, 7-network ordering",
                 fontsize=18, color=INK, y=1.01, weight="semibold")
    fig.text(0.5, -0.01, "Atlas boundaries are labels, not measured activation",
             ha="center", color=MUTED, fontsize=10)
    fig.tight_layout()
    save(fig, "schaefer_atlas_views.png")


def connectivity_data() -> tuple[np.ndarray, list[str]]:
    """Generate structured synthetic regional signals and their correlations."""
    rng = np.random.default_rng(1960)
    t = np.linspace(0, 8 * np.pi, 220)
    latent = np.vstack([
        np.sin(t + phase) + 0.35 * np.sin(2.7 * t - phase)
        for phase in np.linspace(0, np.pi, 6)
    ])
    signals = []
    names = []
    for network in range(6):
        for region in range(8):
            sign = -1 if network == 5 and region >= 4 else 1
            signal = sign * latent[network] + 0.42 * rng.normal(size=t.size)
            signal += 0.12 * latent[(network + 1) % 6]
            signals.append(signal)
            names.append(f"N{network + 1}-{region + 1}")
    return np.corrcoef(np.asarray(signals)), names


def connectivity_figure(matrix: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6.4), facecolor=PAPER)
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    for boundary in range(8, 48, 8):
        ax.axhline(boundary - 0.5, color=PAPER, linewidth=1.3)
        ax.axvline(boundary - 0.5, color=PAPER, linewidth=1.3)
    centers = np.arange(3.5, 48, 8)
    ax.set_xticks(centers, [f"N{i}" for i in range(1, 7)])
    ax.set_yticks(centers, [f"Network {i}" for i in range(1, 7)])
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_title("A larger functional-connectivity matrix", color=INK, fontsize=18, pad=14)
    cbar = fig.colorbar(image, ax=ax, shrink=0.84, pad=0.03)
    cbar.set_label("Pearson correlation", color=MUTED)
    cbar.ax.tick_params(colors=MUTED)
    fig.text(0.5, 0.01, "Structured synthetic signals, 48 regions; block patterns reflect shared latent signals",
             ha="center", color=MUTED, fontsize=9.5)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, "larger_connectivity_matrix.png")


def roc_points(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    order = np.argsort(-scores)
    y = labels[order]
    tp = np.r_[0, np.cumsum(y == 1)]
    fp = np.r_[0, np.cumsum(y == 0)]
    tpr = tp / np.sum(y == 1)
    fpr = fp / np.sum(y == 0)
    return fpr, tpr, float(np.trapezoid(tpr, fpr))


def evaluation_figure() -> None:
    """Plot one consistent, explicitly synthetic 100-participant example."""
    labels = np.r_[np.ones(40, dtype=int), np.zeros(60, dtype=int)]
    # At threshold 0.5 these scores reproduce TP=36, FN=4, TN=45, FP=15.
    scores = np.r_[
        np.linspace(0.52, 0.96, 36), np.linspace(0.18, 0.47, 4),
        np.linspace(0.06, 0.46, 45), np.linspace(0.51, 0.78, 15),
    ]
    fpr, tpr, auc = roc_points(labels, scores)

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1), facecolor=PAPER)
    counts = np.array([[45, 15], [4, 36]])
    axes[0].imshow(counts, cmap="Blues", vmin=0, vmax=50)
    for row in range(2):
        for col in range(2):
            axes[0].text(col, row, str(counts[row, col]), ha="center", va="center",
                         fontsize=22, color="white" if counts[row, col] > 25 else INK,
                         weight="bold")
    axes[0].set_xticks([0, 1], ["Pred. control", "Pred. group A"])
    axes[0].set_yticks([0, 1], ["True control", "True group A"])
    axes[0].set_title("Confusion matrix", color=INK)

    axes[1].plot(fpr, tpr, color=BLUE, linewidth=2.7, label=f"Synthetic AUC = {auc:.2f}")
    axes[1].plot([0, 1], [0, 1], color="#aaa49a", linestyle="--")
    axes[1].set(xlim=(0, 1), ylim=(0, 1), xlabel="False-positive rate", ylabel="True-positive rate")
    axes[1].legend(frameon=False, loc="lower right", fontsize=9)
    axes[1].set_title("ROC curve", color=INK)

    bins = np.linspace(0, 1, 6)
    centers = (bins[:-1] + bins[1:]) / 2
    calibrated = np.array([0.08, 0.28, 0.50, 0.72, 0.90])
    overconfident = np.array([0.20, 0.34, 0.50, 0.66, 0.79])
    axes[2].plot([0, 1], [0, 1], color="#aaa49a", linestyle="--", label="Ideal")
    axes[2].plot(centers, calibrated, "o-", color=GREEN, linewidth=2, label="Better calibrated")
    axes[2].plot(centers, overconfident, "s-", color=RED, linewidth=2, label="Overconfident")
    axes[2].set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted probability", ylabel="Observed fraction")
    axes[2].legend(frameon=False, fontsize=8)
    axes[2].set_title("Reliability diagram", color=INK)
    for ax in axes[1:]:
        ax.grid(color="#e5dfd4", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Evaluation views use teaching numbers, not ABIDE results", color=INK,
                 fontsize=18, y=1.03, weight="semibold")
    fig.tight_layout()
    save(fig, "synthetic_evaluation_summary.png")


def pipeline_figure() -> None:
    labels = ["4D fMRI", "Clean", "Atlas", "Signals", "Correlate", "Matrix",
              "Graph", "Model", "Probability", "Evaluate"]
    fig, ax = plt.subplots(figsize=(14, 2.4), facecolor=PAPER)
    ax.set_xlim(-0.7, len(labels) - 0.3)
    ax.set_ylim(-0.8, 1.25)
    ax.axis("off")
    for index, label in enumerate(labels):
        color = COLORS[index % len(COLORS)]
        ax.scatter(index, 0.24, s=840, facecolor=PAPER, edgecolor=color, linewidth=3, zorder=3)
        ax.text(index, 0.24, str(index + 1), ha="center", va="center", color=color,
                fontsize=11, weight="bold")
        ax.text(index, -0.48, label, ha="center", color=INK, fontsize=9.5, weight="semibold")
        if index < len(labels) - 1:
            ax.annotate("", xy=(index + 0.67, 0.24), xytext=(index + 0.34, 0.24),
                        arrowprops={"arrowstyle": "->", "color": "#928a7e", "lw": 1.7})
    ax.text(4.5, 1.0, "One participant: from measurements to an evaluated prediction",
            ha="center", color=INK, fontsize=17, weight="semibold")
    fig.tight_layout()
    save(fig, "pipeline_recap.png")


def write_toy_data(matrix: np.ndarray) -> None:
    toy = np.array([
        [1.00, 0.82, 0.44, -0.18, 0.12, -0.52],
        [0.82, 1.00, 0.57, -0.11, 0.26, -0.35],
        [0.44, 0.57, 1.00, 0.08, 0.61, -0.21],
        [-0.18, -0.11, 0.08, 1.00, 0.48, 0.31],
        [0.12, 0.26, 0.61, 0.48, 1.00, -0.64],
        [-0.52, -0.35, -0.21, 0.31, -0.64, 1.00],
    ])
    payload = {
        "regions": ["Visual", "Somatomotor", "Attention", "Limbic", "Control", "Default"],
        "matrix": toy.tolist(),
        "node_positions": [[0.50, 0.08], [0.84, 0.30], [0.76, 0.72], [0.50, 0.92], [0.18, 0.70], [0.13, 0.27]],
        "larger_matrix_shape": list(matrix.shape),
        "note": "All values in this JSON file are deterministic synthetic teaching data.",
    }
    (OUT / "toy_connectivity.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")


def main() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "text.color": INK,
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
    })
    template = brain_views()
    atlas_views(template)
    matrix, _ = connectivity_data()
    connectivity_figure(matrix)
    evaluation_figure()
    pipeline_figure()
    write_toy_data(matrix)
    print("Generated 5 figures and toy_connectivity.json")


if __name__ == "__main__":
    main()
