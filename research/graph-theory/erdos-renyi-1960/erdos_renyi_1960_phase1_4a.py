#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from textwrap import fill
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea


# ============================================================
# Phase 1.4A animation.
# Default output matches the HTML placeholder:
#     erdos_renyi_1960_phase1_4a.mp4
#
# Idea:
#   Above the threshold scale for k-trees, the count of k-vertex tree
#   components is no longer a tiny Poisson count. It should start to
#   wobble around a visible center with spread on the order of sqrt(M_n),
#   where M_n is the theoretical mean from Erdős–Rényi (1960).
#
# Default example:
#   k = 3, N(n) = floor(n^0.6), n = 10,000, with 1,000 runs.
# ============================================================
DEFAULT_N = 10000
DEFAULT_K = 3
DEFAULT_BETA = 0.6
DEFAULT_TRIALS = 1000
DEFAULT_BATCH_SIZE = 20
DEFAULT_SEED = 12345
DEFAULT_FPS = 4
DEFAULT_HOLD_FINAL_FRAMES = 18
DEFAULT_OUTPUT = Path("./erdos_renyi_1960_phase1_4a.mp4")

BAR_COLOR = "#7fb3ff"
BAR_EDGE = "#1565c0"
NORMAL_COLOR = "#c62828"


@dataclass
class SimulationSummary:
    n: int
    k: int
    beta: float
    threshold_alpha: float
    m: int
    trials: int
    counts: List[int]
    predicted_mean: float
    predicted_variance: float
    predicted_sd: float


class UnionFindCounter:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.size = [1] * n
        self.edge_count = [0] * n

    def find(self, x: int) -> int:
        parent = self.parent
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def add_edge(self, u: int, v: int) -> None:
        ru = self.find(u)
        rv = self.find(v)

        if ru == rv:
            self.edge_count[ru] += 1
            return

        if self.size[ru] < self.size[rv]:
            ru, rv = rv, ru

        self.parent[rv] = ru
        self.size[ru] += self.size[rv]
        self.edge_count[ru] += self.edge_count[rv] + 1

    def count_tree_components_of_order(self, k: int) -> int:
        total = 0
        for i, p in enumerate(self.parent):
            if i == p and self.size[i] == k and self.edge_count[i] == k - 1:
                total += 1
        return total


def threshold_alpha_for_k(k: int) -> float:
    return (k - 2) / (k - 1)


def edge_count_above_threshold(n: int, beta: float) -> int:
    return max(1, int(round(n ** beta)))


def predicted_mean_Mn(n: int, k: int, m: int) -> float:
    return (
        n
        * (k ** (k - 2))
        / math.factorial(k)
        * ((2.0 * m / n) ** (k - 1))
        * math.exp(-(2.0 * k * m) / n)
    )


def normal_cdf(x: float, mu: float, sigma: float) -> float:
    if sigma <= 1e-12:
        return 1.0 if x >= mu else 0.0
    z = (x - mu) / (sigma * math.sqrt(2.0))
    return 0.5 * (1.0 + math.erf(z))


def normal_bin_probability(x: int, mu: float, sigma: float) -> float:
    if sigma <= 1e-12:
        return 1.0 if x == int(round(mu)) else 0.0
    lo = x - 0.5
    hi = x + 0.5
    return max(0.0, normal_cdf(hi, mu, sigma) - normal_cdf(lo, mu, sigma))


def sample_unique_edges_sparse(n: int, m: int, rng: np.random.Generator) -> set[Tuple[int, int]]:
    edges: set[Tuple[int, int]] = set()

    while len(edges) < m:
        remaining = m - len(edges)
        batch_size = min(max(64, 3 * remaining), 300_000)

        u = rng.integers(0, n, size=batch_size, dtype=np.int64)
        v = rng.integers(0, n, size=batch_size, dtype=np.int64)

        mask = u != v
        if not np.any(mask):
            continue

        u = u[mask]
        v = v[mask]

        a = np.minimum(u, v)
        b = np.maximum(u, v)

        for x, y in zip(a.tolist(), b.tolist()):
            edges.add((x, y))
            if len(edges) >= m:
                break

    return edges


def trial_tree_count(n: int, m: int, k: int, rng: np.random.Generator) -> int:
    uf = UnionFindCounter(n)
    edges = sample_unique_edges_sparse(n=n, m=m, rng=rng)
    for u, v in edges:
        uf.add_edge(u, v)
    return uf.count_tree_components_of_order(k)


def run_trials(n: int, k: int, beta: float, trials: int, seed: int) -> SimulationSummary:
    rng = np.random.default_rng(seed)
    threshold_alpha = threshold_alpha_for_k(k)
    m = edge_count_above_threshold(n=n, beta=beta)
    counts = [trial_tree_count(n=n, m=m, k=k, rng=rng) for _ in range(trials)]
    mean = predicted_mean_Mn(n=n, k=k, m=m)
    variance = mean
    return SimulationSummary(
        n=n,
        k=k,
        beta=beta,
        threshold_alpha=threshold_alpha,
        m=m,
        trials=trials,
        counts=counts,
        predicted_mean=mean,
        predicted_variance=variance,
        predicted_sd=math.sqrt(max(mean, 1e-12)),
    )


def save_animation(anim: FuncAnimation, output_path: Path, fps: int, dpi: int = 180) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() == ".mp4":
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is not None:
            writer = FFMpegWriter(
                fps=fps,
                codec="libx264",
                bitrate=2600,
                extra_args=[
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    "-vf",
                    "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                ],
            )
            anim.save(output_path, writer=writer, dpi=dpi)
            return output_path

        gif_path = output_path.with_suffix(".gif")
        anim.save(gif_path, writer=PillowWriter(fps=fps), dpi=120)
        return gif_path

    anim.save(output_path, writer=PillowWriter(fps=fps), dpi=120)
    return output_path


def add_runs_line(info_ax, runs_done: int, total_runs: int, m: int, n: int, k: int) -> None:
    left = TextArea("runs shown = ", textprops=dict(color="#111111", fontsize=10))
    middle = TextArea(f"{runs_done:,}/{total_runs:,}", textprops=dict(color="#c62828", fontsize=10, fontweight="bold"))
    right = TextArea(f"   |   n = {n:,}   |   k = {k}   |   m = {m:,}", textprops=dict(color="#111111", fontsize=10))
    packed = HPacker(children=[left, middle, right], align="center", pad=0, sep=0)
    anchored = AnchoredOffsetbox(
        loc="center",
        child=packed,
        pad=0.22,
        frameon=True,
        bbox_to_anchor=(0.5, 0.84),
        bbox_transform=info_ax.transAxes,
        borderpad=0.32,
    )
    anchored.patch.set_facecolor("white")
    anchored.patch.set_edgecolor("0.85")
    info_ax.add_artist(anchored)


def make_phase1_4a_animation(
    n: int,
    k: int,
    beta: float,
    trials: int,
    batch_size: int,
    seed: int,
    fps: int,
    hold_final_frames: int,
    output_path: Path,
) -> Path:
    summary = run_trials(n=n, k=k, beta=beta, trials=trials, seed=seed)

    batches = list(range(0, trials + 1, batch_size))
    if batches[-1] != trials:
        batches.append(trials)
    frame_schedule = batches + [trials] * hold_final_frames

    x_max = max(
        8,
        max(summary.counts),
        int(math.ceil(summary.predicted_mean + 4.5 * summary.predicted_sd)),
    )
    xs = list(range(0, x_max + 1))
    normal_values = [normal_bin_probability(x, summary.predicted_mean, summary.predicted_sd) for x in xs]

    fig = plt.figure(figsize=(10.8, 9.1))
    hist_ax = fig.add_axes([0.08, 0.38, 0.84, 0.43])
    info_ax = fig.add_axes([0.08, 0.05, 0.84, 0.19])
    info_ax.axis("off")
    fig.patch.set_facecolor("white")

    common_handles = [
        Line2D([0], [0], color=BAR_EDGE, lw=8, solid_capstyle="butt", label="Empirical relative frequency"),
        Line2D([0], [0], color=NORMAL_COLOR, linestyle="--", linewidth=2.4, marker="o", markersize=5, label="Normal approximation"),
    ]
    fig.legend(
        handles=common_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.285),
        ncol=2,
        frameon=True,
        edgecolor="#cccccc",
        fontsize=10,
    )

    def update(frame_index: int) -> None:
        runs_done = frame_schedule[frame_index]
        observed = summary.counts[:runs_done]

        hist_ax.clear()
        info_ax.clear()
        info_ax.axis("off")

        empirical = np.zeros(len(xs), dtype=float)
        if runs_done > 0:
            binc = np.bincount(observed, minlength=len(xs))[: len(xs)]
            empirical = binc / runs_done

        hist_ax.bar(
            xs,
            empirical,
            width=0.78,
            color=BAR_COLOR,
            edgecolor=BAR_EDGE,
            linewidth=1.0,
        )
        hist_ax.plot(
            xs,
            normal_values,
            linestyle="--",
            linewidth=2.4,
            marker="o",
            markersize=5,
            color=NORMAL_COLOR,
        )

        hist_ax.set_xlim(-0.5, x_max + 0.5)
        hist_ax.set_ylim(0.0, max(0.52, 1.32 * max(max(empirical, default=0.0), max(normal_values, default=0.0))))
        hist_ax.set_xlabel(rf"Number of $T_{{{summary.k}}}$ tree-components in one run", fontsize=12)
        hist_ax.set_ylabel("Probability / relative frequency", fontsize=12)
        hist_ax.set_title(
            rf"Phase 1.4A — above threshold, many $T_{{{summary.k}}}$ trees and an approximately normal count ({summary.trials:,} runs)",
            fontsize=15,
            pad=40,
        )
        hist_ax.text(
            0.5,
            1.06,
            rf"Using $m = \lfloor n^{{{summary.beta:.1f}}} \rfloor$ while the threshold exponent is $(k-2)/(k-1)={summary.threshold_alpha:.3f}$. The count should wobble around $M_n$ with spread on the order of $\sqrt{{M_n}}$.",
            transform=hist_ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=10.3,
            color="0.25",
        )
        hist_ax.grid(True, axis="y", alpha=0.28)

        add_runs_line(info_ax, runs_done=runs_done, total_runs=summary.trials, m=summary.m, n=summary.n, k=summary.k)

        if runs_done > 0:
            sample_mean = float(np.mean(observed))
            sample_var = float(np.var(observed))
            sample_sd = float(np.std(observed))
        else:
            sample_mean = sample_var = sample_sd = 0.0

        info_ax.text(
            0.5,
            0.54,
            rf"empirical mean = {sample_mean:.3f}   |   empirical variance = {sample_var:.3f}   |   empirical sd = {sample_sd:.3f}   ||   predicted $M_n$ = {summary.predicted_mean:.3f}   |   predicted sd = {summary.predicted_sd:.3f}",
            transform=info_ax.transAxes,
            ha="center",
            va="center",
            fontsize=9.8,
            bbox=dict(facecolor="white", edgecolor="0.85", boxstyle="round,pad=0.24"),
        )

        interpretation = (
            "Above threshold the count is no longer a tiny Poisson count. It should now concentrate around a visible center and look increasingly bell-shaped as n grows. "
            "For small n this may still look rough, but the mean and variance should already be of comparable size."
        )
        info_ax.text(
            0.5,
            0.15,
            fill(interpretation, width=108),
            transform=info_ax.transAxes,
            ha="center",
            va="center",
            fontsize=9.4,
            linespacing=1.28,
            bbox=dict(facecolor="#f9f9f9", edgecolor="0.75", boxstyle="round,pad=0.28"),
        )

    anim = FuncAnimation(fig, update, frames=len(frame_schedule), interval=1000 / fps, repeat=False)
    saved_to = save_animation(anim=anim, output_path=output_path, fps=fps)
    plt.close(fig)
    return saved_to


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create the Phase 1.4A animation showing that above threshold the count of k-vertex tree-components "
            "looks approximately normal across many runs."
        )
    )
    parser.add_argument("--n", type=int, default=DEFAULT_N, help="Number of nodes.")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="Tree order to track.")
    parser.add_argument("--beta", type=float, default=DEFAULT_BETA, help="Exponent in m = round(n^beta), chosen above the threshold exponent but below 1.")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS, help="Number of Monte Carlo runs.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="How many new runs to add per animation frame.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help="Frames per second.")
    parser.add_argument("--hold-final-frames", type=int, default=DEFAULT_HOLD_FINAL_FRAMES, help="Still frames at the end.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output animation path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    n = max(2, args.n)
    k = max(3, args.k)
    beta = min(max(args.beta, 0.05), 0.95)
    trials = max(10, args.trials)
    batch_size = max(1, args.batch_size)

    saved_to = make_phase1_4a_animation(
        n=n,
        k=k,
        beta=beta,
        trials=trials,
        batch_size=batch_size,
        seed=args.seed,
        fps=max(1, args.fps),
        hold_final_frames=max(0, args.hold_final_frames),
        output_path=args.output,
    )

    threshold_alpha = threshold_alpha_for_k(k)
    m = edge_count_above_threshold(n=n, beta=beta)
    Mn = predicted_mean_Mn(n=n, k=k, m=m)
    print(f"n = {n}")
    print(f"k = {k}")
    print(f"beta = {beta:.3f}")
    print(f"threshold exponent = {threshold_alpha:.6f}")
    print(f"m = {m}")
    print(f"predicted M_n = {Mn:.6f}")
    print(f"saved animation to: {saved_to}")


if __name__ == "__main__":
    main()
