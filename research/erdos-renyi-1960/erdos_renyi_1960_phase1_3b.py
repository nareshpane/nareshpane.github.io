#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from textwrap import fill
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.patches import FancyBboxPatch


# ============================================================
# Phase 1.3B comparison animation.
# Default output matches the HTML placeholder:
#     erdos_renyi_1960_phase1_3b.mp4
#
# Idea:
#   Compare n = 100, 1000, and 10,000 for the same threshold choice
#   k = 3, rho = 1, m ~ n^(1/2). Each panel accumulates many runs and
#   shows the empirical distribution of the count of T3 components,
#   against the same Poisson prediction with lambda = 2.
# ============================================================
DEFAULT_NS = [100, 1000, 10000]
DEFAULT_K = 3
DEFAULT_RHO = 1.0
DEFAULT_TRIALS = 1000
DEFAULT_BATCH_SIZE = 20
DEFAULT_SEED = 24680
DEFAULT_FPS = 4
DEFAULT_HOLD_FINAL_FRAMES = 18
DEFAULT_OUTPUT = Path("./erdos_renyi_1960_phase1_3b.mp4")

BAR_COLOR = "#7fb3ff"
BAR_EDGE = "#1565c0"
POISSON_COLOR = "#c62828"


@dataclass
class RunData:
    n: int
    m: int
    counts: List[int]
    predicted_lambda: float


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


def alpha_for_k(k: int) -> float:
    return (k - 2) / (k - 1)


def threshold_edge_count(n: int, k: int, rho: float) -> int:
    return max(1, int(round(rho * (n ** alpha_for_k(k)))))


def theoretical_lambda(k: int, rho: float) -> float:
    return ((2.0 * rho) ** (k - 1)) * (k ** (k - 2)) / math.factorial(k)


def poisson_pmf(x: int, lam: float) -> float:
    return math.exp(-lam) * (lam ** x) / math.factorial(x)


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


def run_trials_for_n(n: int, k: int, rho: float, trials: int, seed: int) -> RunData:
    rng = np.random.default_rng(seed)
    m = threshold_edge_count(n=n, k=k, rho=rho)
    counts = [trial_tree_count(n=n, m=m, k=k, rng=rng) for _ in range(trials)]
    return RunData(n=n, m=m, counts=counts, predicted_lambda=theoretical_lambda(k=k, rho=rho))


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


def empirical_distribution(counts: List[int], runs_done: int, length: int) -> np.ndarray:
    dist = np.zeros(length, dtype=float)
    if runs_done <= 0:
        return dist
    binc = np.bincount(counts[:runs_done], minlength=length)[:length]
    return binc / runs_done


def make_phase1_3b_animation(
    ns: List[int],
    k: int,
    rho: float,
    trials: int,
    batch_size: int,
    seed: int,
    fps: int,
    hold_final_frames: int,
    output_path: Path,
) -> Path:
    clean_ns = [n for n in ns if n >= 2]
    if not clean_ns:
        raise ValueError("Need at least one n >= 2.")

    runs: List[RunData] = []
    for i, n in enumerate(clean_ns):
        runs.append(run_trials_for_n(n=n, k=k, rho=rho, trials=trials, seed=seed + 1009 * i))

    batches = list(range(0, trials + 1, batch_size))
    if batches[-1] != trials:
        batches.append(trials)
    frame_schedule = batches + [trials] * hold_final_frames

    x_max = max(
        6,
        max(max(run.counts) for run in runs),
        int(math.ceil(runs[0].predicted_lambda + 4.0 * math.sqrt(runs[0].predicted_lambda))),
    )
    xs = list(range(0, x_max + 1))
    poisson_values = [poisson_pmf(x, runs[0].predicted_lambda) for x in xs]

    fig = plt.figure(figsize=(18.0, 9.4))
    outer = fig.add_gridspec(
        nrows=2,
        ncols=3,
        left=0.04,
        right=0.96,
        top=0.80,
        bottom=0.15,
        height_ratios=[1.0, 0.42],
        hspace=0.14,
        wspace=0.05,
    )
    panels = []
    for col in range(3):
        hist_ax = fig.add_subplot(outer[0, col])
        info_ax = fig.add_subplot(outer[1, col])
        info_ax.axis("off")
        panels.append((hist_ax, info_ax))

    title_ax = fig.add_axes([0.05, 0.845, 0.90, 0.10])
    title_ax.axis("off")
    title_ax.text(
        0.5,
        0.70,
        rf"Phase 1.3B — comparing the Poisson window for $T_{{{k}}}$ counts across $n=100,1000,10{{,}}000$",
        ha="center",
        va="center",
        fontsize=17,
        fontweight="bold",
    )
    title_ax.text(
        0.5,
        0.30,
        rf"Each panel uses the threshold-scale choice $m = \lfloor \rho n^{{(k-2)/(k-1)}} \rfloor$ with $\rho={rho:.1f}$, so the limiting Poisson mean is the same in every panel.",
        ha="center",
        va="center",
        fontsize=11,
        color="0.25",
    )

    def update(frame_index: int) -> None:
        runs_done = frame_schedule[frame_index]

        for (hist_ax, info_ax), run in zip(panels, runs):
            hist_ax.clear()
            info_ax.clear()
            info_ax.axis("off")

            empirical = empirical_distribution(run.counts, runs_done=runs_done, length=len(xs))
            hist_ax.bar(
                xs,
                empirical,
                width=0.78,
                color=BAR_COLOR,
                edgecolor=BAR_EDGE,
                linewidth=1.0,
                label="Empirical relative frequency",
            )
            hist_ax.plot(
                xs,
                poisson_values,
                linestyle=":",
                linewidth=2.4,
                marker="o",
                markersize=4.8,
                color=POISSON_COLOR,
                label=rf"Poisson($\lambda={run.predicted_lambda:.3f}$)",
            )
            hist_ax.set_xlim(-0.5, x_max + 0.5)
            hist_ax.set_ylim(0.0, max(0.45, 1.18 * max(max(empirical, default=0.0), max(poisson_values, default=0.0))))
            hist_ax.set_title(rf"$n = {run.n:,}$", fontsize=15, pad=10)
            hist_ax.set_xlabel(rf"Count of $T_{{{k}}}$ components", fontsize=11)
            if run.n == clean_ns[0]:
                hist_ax.set_ylabel("Probability / relative frequency", fontsize=11)
            hist_ax.grid(True, axis="y", alpha=0.28)
            hist_ax.legend(loc="upper right", fontsize=9)

            info_ax.add_patch(
                FancyBboxPatch(
                    (0.03, 0.07),
                    0.94,
                    0.86,
                    transform=info_ax.transAxes,
                    boxstyle="round,pad=0.02",
                    linewidth=0.95,
                    edgecolor="#cfcfcf",
                    facecolor="#fcfcfc",
                    zorder=0,
                )
            )
            info_ax.text(
                0.5,
                0.77,
                f"runs shown = {runs_done:,}/{trials:,}   |   m = {run.m:,}",
                transform=info_ax.transAxes,
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                bbox=dict(facecolor="#ffe8e8", edgecolor="#b71c1c", boxstyle="round,pad=0.22"),
            )

            if runs_done > 0:
                observed = run.counts[:runs_done]
                sample_mean = float(np.mean(observed))
                sample_var = float(np.var(observed))
                p0 = float(np.mean(np.array(observed) == 0))
                p1 = float(np.mean(np.array(observed) == 1))
            else:
                sample_mean = sample_var = p0 = p1 = 0.0

            info_ax.text(
                0.5,
                0.49,
                rf"sample mean = {sample_mean:.3f}   |   sample variance = {sample_var:.3f}   |   predicted $\lambda$ = {run.predicted_lambda:.3f}",
                transform=info_ax.transAxes,
                ha="center",
                va="center",
                fontsize=9.8,
                bbox=dict(facecolor="white", edgecolor="0.85", boxstyle="round,pad=0.22"),
            )
            info_ax.text(
                0.5,
                0.18,
                fill(
                    rf"Current empirical read: P(count=0)={p0:.3f}, P(count=1)={p1:.3f}. Larger n should make the histogram sit more cleanly on the same Poisson law.",
                    width=44,
                ),
                transform=info_ax.transAxes,
                ha="center",
                va="center",
                fontsize=9.2,
                linespacing=1.24,
                bbox=dict(facecolor="#f9f9f9", edgecolor="0.75", boxstyle="round,pad=0.24"),
            )

    anim = FuncAnimation(fig, update, frames=len(frame_schedule), interval=1000 / fps, repeat=False)
    saved_to = save_animation(anim=anim, output_path=output_path, fps=fps)
    plt.close(fig)
    return saved_to


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create the Phase 1.3B animation comparing Poisson-like k-tree counts across several n values at the threshold scale."
        )
    )
    parser.add_argument("--ns", type=int, nargs="+", default=DEFAULT_NS, help="Values of n to compare.")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="Tree order to track.")
    parser.add_argument("--rho", type=float, default=DEFAULT_RHO, help="Constant in m = round(rho * n^((k-2)/(k-1))).")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS, help="Number of Monte Carlo runs per n.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="How many new runs to add per animation frame.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Base random seed.")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help="Frames per second.")
    parser.add_argument("--hold-final-frames", type=int, default=DEFAULT_HOLD_FINAL_FRAMES, help="Still frames at the end.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output animation path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    saved_to = make_phase1_3b_animation(
        ns=args.ns,
        k=max(3, args.k),
        rho=max(0.05, args.rho),
        trials=max(10, args.trials),
        batch_size=max(1, args.batch_size),
        seed=args.seed,
        fps=max(1, args.fps),
        hold_final_frames=max(0, args.hold_final_frames),
        output_path=args.output,
    )
    print(f"n values = {[n for n in args.ns if n >= 2]}")
    print(f"saved animation to: {saved_to}")


if __name__ == "__main__":
    main()
