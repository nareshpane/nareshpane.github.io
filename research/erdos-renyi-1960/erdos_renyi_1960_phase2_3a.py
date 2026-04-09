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
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea

# ============================================================
# Phase 2.3A animation.
# Default output matches the HTML placeholder:
#     erdos_renyi_1960_phase2_3a.mp4
#
# Idea:
#   In Phase 2, not only fixed cycles but also connected components with
#   exactly k vertices and k edges (unicyclic components of order k)
#   have a Poisson limit law.
#
#   This animation accumulates many runs for one fixed n and shows the
#   empirical histogram of the number of k-vertex unicyclic components,
#   compared against the Erdős–Rényi Poisson prediction.
# ============================================================
DEFAULT_N = 1000
DEFAULT_K = 4
DEFAULT_C = 0.40
DEFAULT_TRIALS = 1000
DEFAULT_BATCH_SIZE = 20
DEFAULT_SEED = 12345
DEFAULT_FPS = 4
DEFAULT_HOLD_FINAL_FRAMES = 18
DEFAULT_OUTPUT = Path("./erdos_renyi_1960_phase2_3a.mp4")

BAR_COLOR = "#7fb3ff"
BAR_EDGE = "#1565c0"
POISSON_COLOR = "#c62828"


@dataclass
class SimulationSummary:
    n: int
    k: int
    c: float
    m: int
    trials: int
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

    def count_unicyclic_components_of_order(self, k: int) -> int:
        total = 0
        for i, p in enumerate(self.parent):
            if i == p and self.size[i] == k and self.edge_count[i] == k:
                total += 1
        return total


def poisson_mean_unicyclic(c: float, k: int) -> float:
    series = sum((k ** j) / math.factorial(j) for j in range(0, k - 2))
    return ((2.0 * c * math.exp(-2.0 * c)) ** k) * series / math.factorial(k)


def poisson_pmf(x: int, lam: float) -> float:
    return math.exp(-lam) * (lam ** x) / math.factorial(x)


def sample_unique_edges_sparse(n: int, m: int, rng: np.random.Generator) -> List[Tuple[int, int]]:
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
    return list(edges)


def trial_unicyclic_count(n: int, m: int, k: int, rng: np.random.Generator) -> int:
    uf = UnionFindCounter(n)
    edges = sample_unique_edges_sparse(n=n, m=m, rng=rng)
    for u, v in edges:
        uf.add_edge(u, v)
    return uf.count_unicyclic_components_of_order(k)


def run_trials(n: int, k: int, c: float, trials: int, seed: int) -> SimulationSummary:
    rng = np.random.default_rng(seed)
    m = max(1, int(round(c * n)))
    counts = [trial_unicyclic_count(n=n, m=m, k=k, rng=rng) for _ in range(trials)]
    return SimulationSummary(
        n=n,
        k=k,
        c=c,
        m=m,
        trials=trials,
        counts=counts,
        predicted_lambda=poisson_mean_unicyclic(c=c, k=k),
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
                    "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart",
                    "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                ],
            )
            anim.save(output_path, writer=writer, dpi=dpi)
            return output_path

        gif_path = output_path.with_suffix(".gif")
        anim.save(gif_path, writer=PillowWriter(fps=fps), dpi=120)
        return gif_path

    anim.save(output_path, writer=PillowWriter(fps=fps), dpi=120)
    return output_path


def add_runs_line(info_ax, runs_done: int, total_runs: int, m: int, n: int, k: int, c: float) -> None:
    left = TextArea("runs shown = ", textprops=dict(color="#111111", fontsize=10))
    middle = TextArea(f"{runs_done:,}/{total_runs:,}", textprops=dict(color="#c62828", fontsize=10, fontweight="bold"))
    right = TextArea(
        f"   |   n = {n:,}   |   k = {k}   |   c = {c:.2f}   |   m = {m:,}",
        textprops=dict(color="#111111", fontsize=10),
    )
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


def make_phase2_3a_animation(
    n: int,
    k: int,
    c: float,
    trials: int,
    batch_size: int,
    seed: int,
    fps: int,
    hold_final_frames: int,
    output_path: Path,
) -> Path:
    summary = run_trials(n=n, k=k, c=c, trials=trials, seed=seed)

    batches = list(range(0, trials + 1, batch_size))
    if batches[-1] != trials:
        batches.append(trials)
    frame_schedule = batches + [trials] * hold_final_frames

    x_max = max(
        5,
        max(summary.counts) if summary.counts else 0,
        int(math.ceil(summary.predicted_lambda + 4.0 * math.sqrt(max(summary.predicted_lambda, 1e-12)))),
    )
    xs = list(range(0, x_max + 1))
    poisson_values = [poisson_pmf(x, summary.predicted_lambda) for x in xs]

    fig = plt.figure(figsize=(10.8, 8.9))
    hist_ax = fig.add_axes([0.08, 0.35, 0.84, 0.53])
    info_ax = fig.add_axes([0.08, 0.07, 0.84, 0.22])
    info_ax.axis("off")
    fig.patch.set_facecolor("white")

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
            label="Empirical relative frequency",
        )
        hist_ax.plot(
            xs,
            poisson_values,
            linestyle=":",
            linewidth=2.4,
            marker="o",
            markersize=5,
            color=POISSON_COLOR,
            label=rf"Poisson prediction ($\lambda={summary.predicted_lambda:.4f}$)",
        )

        hist_ax.set_xlim(-0.5, x_max + 0.5)
        hist_ax.set_ylim(0.0, max(0.45, 1.20 * max(float(np.max(empirical)) if empirical.size else 0.0, max(poisson_values, default=0.0))))
        hist_ax.set_xlabel(rf"Number of unicyclic components with $v=e={summary.k}$ in one run", fontsize=12)
        hist_ax.set_ylabel("Probability / relative frequency", fontsize=12)
        hist_ax.set_title(
            rf"Phase 2.3A — unicyclic components of order $k$ also have a Poisson law",
            fontsize=15,
            pad=40,
        )
        hist_ax.text(
            0.5,
            0.992,
            fill(
                rf"Tracking connected components with exactly {summary.k} vertices and {summary.k} edges, at the linear scale $m=\lfloor c n \rceil$ with $c={summary.c:.2f}$. For fixed $k$, the limiting count should be Poisson with the Erdős–Rényi mean shown below.",
                width=92,
            ),
            transform=hist_ax.transAxes,
            ha="center",
            va="top",
            fontsize=10.0,
            color="0.25",
            linespacing=1.24,
        )
        hist_ax.grid(True, axis="y", alpha=0.28)
        hist_ax.legend(loc="upper right")

        add_runs_line(info_ax, runs_done=runs_done, total_runs=summary.trials, m=summary.m, n=summary.n, k=summary.k, c=summary.c)

        if runs_done > 0:
            sample_mean = float(np.mean(observed))
            sample_var = float(np.var(observed))
            p0 = float(np.mean(np.array(observed) == 0))
            p1 = float(np.mean(np.array(observed) == 1))
            p2 = float(np.mean(np.array(observed) == 2))
        else:
            sample_mean = sample_var = p0 = p1 = p2 = 0.0

        info_ax.text(
            0.5,
            0.52,
            rf"sample mean = {sample_mean:.4f}   |   sample variance = {sample_var:.4f}   |   predicted $\lambda$ = {summary.predicted_lambda:.4f}   |   rare-event regime at fixed $(c,k)$",
            transform=info_ax.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            bbox=dict(facecolor="white", edgecolor="0.85", boxstyle="round,pad=0.24"),
        )

        interpretation = (
            rf"Empirical reading so far: P(count=0)={p0:.4f}, P(count=1)={p1:.4f}, P(count=2)={p2:.4f}. "
            "Because the mean is finite and typically small, the distribution should stay concentrated near 0 and 1 rather than spreading out like a large bell curve."
        )
        info_ax.text(
            0.5,
            0.14,
            fill(interpretation, width=104),
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
            "Create the Phase 2.3A animation showing that the count of unicyclic components with exactly k vertices and k edges has a Poisson law."
        )
    )
    parser.add_argument("--n", type=int, default=DEFAULT_N, help="Number of nodes.")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="Order k of the unicyclic components to track.")
    parser.add_argument("--c", type=float, default=DEFAULT_C, help="Linear-scale constant in m = round(c n), should stay below 0.5.")
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
    c = min(max(args.c, 0.01), 0.49)
    trials = max(10, args.trials)
    batch_size = max(1, args.batch_size)

    saved_to = make_phase2_3a_animation(
        n=n,
        k=k,
        c=c,
        trials=trials,
        batch_size=batch_size,
        seed=args.seed,
        fps=max(1, args.fps),
        hold_final_frames=max(0, args.hold_final_frames),
        output_path=args.output,
    )

    lam = poisson_mean_unicyclic(c=c, k=k)
    print(f"n = {n}")
    print(f"k = {k}")
    print(f"c = {c:.3f}")
    print(f"m = {int(round(c * n))}")
    print(f"predicted lambda = {lam:.6f}")
    print(f"saved animation to: {saved_to}")


if __name__ == "__main__":
    main()
