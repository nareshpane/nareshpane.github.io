#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# Phase 3.3 plot script.
# Default output matches the HTML placeholder:
#     erdos_renyi_1960_phase3_3.png
#
# Tailored idea:
#   Compare several supercritical choices of c above 1/2. The right scale is
#   no longer n^(2/3), but the linear giant fraction G(c). This script tracks
#   the empirical largest-component fraction against the theoretical value
#   G(c), while also monitoring the second-largest fraction to show that the
#   remaining pieces stay small by comparison.
# ============================================================

DEFAULT_NS = [100, 500, 1000, 5000, 10000, 20000, 50000]
DEFAULT_C_VALUES = [0.51, 0.55, 0.60]
DEFAULT_TRIALS = 1000
DEFAULT_SEED = 24680
DEFAULT_OUTPUT_CSV = Path("./erdos_renyi_1960_phase3_3.csv")
DEFAULT_OUTPUT_PLOT = Path("./erdos_renyi_1960_phase3_3.png")


@dataclass
class SimulationResult:
    n: int
    c: float
    m: int
    trials: int
    mean_largest_size: float
    sd_largest_size: float
    se_largest_size: float
    ci95_low_largest_size: float
    ci95_high_largest_size: float
    mean_second_largest_size: float
    mean_largest_fraction: float
    mean_second_largest_fraction: float
    theory_giant_fraction: float


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

    def largest_two_component_sizes(self) -> tuple[int, int]:
        largest_size = 0
        second_size = 0
        for i, p in enumerate(self.parent):
            if i != p:
                continue
            comp_size = self.size[i]
            if comp_size > largest_size:
                second_size = largest_size
                largest_size = comp_size
            elif comp_size > second_size:
                second_size = comp_size
        return largest_size, second_size


def sample_unique_edges_sparse(n: int, m: int, rng: np.random.Generator) -> List[tuple[int, int]]:
    if m < 0:
        raise ValueError("m must be nonnegative.")
    max_edges = n * (n - 1) // 2
    if m > max_edges:
        raise ValueError("m exceeds the number of simple undirected edges.")

    edges: set[tuple[int, int]] = set()
    while len(edges) < m:
        remaining = m - len(edges)
        batch_size = min(max(128, 3 * remaining), 400_000)

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
            edges.add((int(x), int(y)))
            if len(edges) >= m:
                break

    return list(edges)


def simulate_one_trial(n: int, m: int, rng: np.random.Generator) -> tuple[int, int]:
    uf = UnionFindCounter(n)
    edges = sample_unique_edges_sparse(n=n, m=m, rng=rng)
    for u, v in edges:
        uf.add_edge(u, v)
    return uf.largest_two_component_sizes()


def giant_fraction_theory(c: float, tol: float = 1e-12, max_iter: int = 500) -> float:
    if c <= 0.5:
        return 0.0
    u = math.exp(-2.0 * c)
    for _ in range(max_iter):
        new_u = math.exp(-2.0 * c * (1.0 - u))
        if abs(new_u - u) < tol:
            u = new_u
            break
        u = new_u
    return max(0.0, min(1.0, 1.0 - u))


def simulate_statistics(n: int, c: float, trials: int, rng: np.random.Generator) -> SimulationResult:
    m = max(1, int(round(c * n)))
    theory = giant_fraction_theory(c)

    largest_sizes: List[int] = []
    second_sizes: List[int] = []

    for _ in range(trials):
        largest_size, second_size = simulate_one_trial(n=n, m=m, rng=rng)
        largest_sizes.append(largest_size)
        second_sizes.append(second_size)

    mean_largest = float(np.mean(largest_sizes)) if largest_sizes else 0.0
    sd_largest = float(np.std(largest_sizes, ddof=1)) if len(largest_sizes) >= 2 else 0.0
    se_largest = sd_largest / math.sqrt(len(largest_sizes)) if largest_sizes else 0.0
    mean_second = float(np.mean(second_sizes)) if second_sizes else 0.0

    return SimulationResult(
        n=n,
        c=c,
        m=m,
        trials=trials,
        mean_largest_size=mean_largest,
        sd_largest_size=sd_largest,
        se_largest_size=se_largest,
        ci95_low_largest_size=max(0.0, mean_largest - 1.96 * se_largest),
        ci95_high_largest_size=mean_largest + 1.96 * se_largest,
        mean_second_largest_size=mean_second,
        mean_largest_fraction=(mean_largest / n) if n > 0 else 0.0,
        mean_second_largest_fraction=(mean_second / n) if n > 0 else 0.0,
        theory_giant_fraction=theory,
    )


def save_results_csv(results: Iterable[SimulationResult], output_csv: Path) -> None:
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "n",
            "c",
            "m",
            "trials",
            "mean_largest_size",
            "sd_largest_size",
            "se_largest_size",
            "ci95_low_largest_size",
            "ci95_high_largest_size",
            "mean_second_largest_size",
            "mean_largest_fraction",
            "mean_second_largest_fraction",
            "theory_giant_fraction",
        ])
        for r in results:
            writer.writerow([
                r.n,
                f"{r.c:.6f}",
                r.m,
                r.trials,
                f"{r.mean_largest_size:.6f}",
                f"{r.sd_largest_size:.6f}",
                f"{r.se_largest_size:.6f}",
                f"{r.ci95_low_largest_size:.6f}",
                f"{r.ci95_high_largest_size:.6f}",
                f"{r.mean_second_largest_size:.6f}",
                f"{r.mean_largest_fraction:.6f}",
                f"{r.mean_second_largest_fraction:.6f}",
                f"{r.theory_giant_fraction:.6f}",
            ])


def print_results_table(results: List[SimulationResult]) -> None:
    header = (
        f"{'n':>9}  {'c':>6}  {'m':>8}  {'trials':>8}  "
        f"{'largest/n':>11}  {'second/n':>11}  {'theory G(c)':>11}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.n:>9}  {r.c:>6.2f}  {r.m:>8}  {r.trials:>8}  "
            f"{r.mean_largest_fraction:>11.4f}  {r.mean_second_largest_fraction:>11.4f}  {r.theory_giant_fraction:>11.4f}"
        )


def make_plot(results: List[SimulationResult], output_plot: Path, c_values: List[float]) -> None:
    ns = sorted({r.n for r in results})
    x_positions = list(range(len(ns)))

    fig, axes = plt.subplots(2, 1, figsize=(13.2, 10.2), sharex=True)
    fig.subplots_adjust(top=0.86, bottom=0.12, hspace=0.24)

    color_map = {
        0.51: "#1d4f91",
        0.55: "#b26a00",
        0.60: "#8b1e3f",
    }

    fig.suptitle(
        "Phase 3.3 — above the threshold, the largest component has linear scale $G(c)n$",
        fontsize=16,
        fontweight="bold",
        y=0.97,
    )
    fig.text(
        0.5,
        0.925,
        "Top panel compares the empirical largest-component fraction with the theoretical giant fraction G(c). Bottom panel tracks the second-largest fraction to show that the remaining pieces stay small.",
        ha="center",
        va="center",
        fontsize=10.5,
        color="0.25",
    )

    ax_top, ax_bottom = axes

    for c in c_values:
        subset = [r for r in results if abs(r.c - c) < 1e-12]
        subset.sort(key=lambda r: r.n)
        color = color_map.get(round(c, 2), None)

        mean_largest_frac = [r.mean_largest_fraction for r in subset]
        low = [r.ci95_low_largest_size / r.n for r in subset]
        high = [r.ci95_high_largest_size / r.n for r in subset]
        mean_second_frac = [r.mean_second_largest_fraction for r in subset]
        theory_line = [r.theory_giant_fraction for r in subset]

        ax_top.plot(x_positions, mean_largest_frac, marker="o", linewidth=2.2, label=f"empirical largest / n, c = {c:.2f}", color=color)
        ax_top.fill_between(x_positions, low, high, alpha=0.14, color=color)
        ax_top.plot(x_positions, theory_line, linestyle="--", linewidth=1.9, label=f"theory G(c), c = {c:.2f}", color=color)

        ax_bottom.plot(x_positions, mean_second_frac, marker="s", linewidth=2.0, label=f"second / n, c = {c:.2f}", color=color)

    ax_top.set_ylabel("Largest-component fraction")
    ax_top.set_title("Empirical largest-component fraction versus the theoretical giant fraction G(c)", fontsize=12.7, pad=10)
    ax_top.grid(True, alpha=0.28)
    ax_top.legend(ncol=2, fontsize=9.0, frameon=True)

    ax_bottom.set_ylabel("Second-largest fraction")
    ax_bottom.set_title("Second-largest component stays small compared with the giant", fontsize=12.7, pad=10)
    ax_bottom.grid(True, alpha=0.28)
    ax_bottom.legend(ncol=3, fontsize=9.2, frameon=True)

    ax_bottom.set_xticks(x_positions)
    ax_bottom.set_xticklabels([f"{n:,}" for n in ns], rotation=0)
    ax_bottom.set_xlabel("Number of vertices $n$")

    fig.text(
        0.5,
        0.04,
        "Default experiment: 1,000 simulations for each n and each c in {0.51, 0.55, 0.60}.",
        ha="center",
        va="center",
        fontsize=9.8,
        color="0.30",
    )

    output_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_plot, dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Monte Carlo simulations for Phase 3.3 and generate a CSV plus PNG comparing the largest fraction with G(c)."
    )
    parser.add_argument("--ns", type=int, nargs="+", default=DEFAULT_NS, help="Values of n to simulate.")
    parser.add_argument("--c-values", type=float, nargs="+", default=DEFAULT_C_VALUES, help="Supercritical c values to compare.")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS, help="Number of Monte Carlo trials per (n, c).")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV, help="Output CSV path.")
    parser.add_argument("--output-plot", type=Path, default=DEFAULT_OUTPUT_PLOT, help="Output plot path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ns = sorted({int(n) for n in args.ns if int(n) >= 2})
    c_values = sorted({float(c) for c in args.c_values})
    trials = max(1, int(args.trials))
    rng = np.random.default_rng(args.seed)

    results: List[SimulationResult] = []
    for c in c_values:
        for n in ns:
            results.append(simulate_statistics(n=n, c=c, trials=trials, rng=rng))

    print_results_table(results)
    save_results_csv(results, args.output_csv)
    make_plot(results, args.output_plot, c_values=c_values)
    print(f"saved CSV to: {args.output_csv}")
    print(f"saved PNG to: {args.output_plot}")


if __name__ == "__main__":
    main()
