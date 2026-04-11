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
# Phase 3.4 plot script.
# Default output matches the HTML placeholder:
#     erdos_renyi_1960_phase3_4.png
#
# Tailored idea:
#   In the supercritical regime, most small components are gradually absorbed
#   by the giant. This script estimates two empirical summaries:
#   (1) the fraction of vertices that still live in tree components, and
#   (2) the prevalence of fixed tree sizes 1, 2, and 3.
#   The natural theoretical comparison for tree-vertex mass is 1 - G(c).
# ============================================================

DEFAULT_NS = [100, 500, 1000, 5000, 10000, 20000, 50000]
DEFAULT_C_VALUES = [0.55, 0.70, 1.00]
DEFAULT_TRIALS = 1000
DEFAULT_SEED = 24680
DEFAULT_OUTPUT_CSV = Path("./erdos_renyi_1960_phase3_4.csv")
DEFAULT_OUTPUT_PLOT = Path("./erdos_renyi_1960_phase3_4.png")


@dataclass
class SimulationResult:
    n: int
    c: float
    m: int
    trials: int
    mean_tree_vertices_fraction: float
    sd_tree_vertices_fraction: float
    se_tree_vertices_fraction: float
    ci95_low_tree_vertices_fraction: float
    ci95_high_tree_vertices_fraction: float
    mean_t1_count: float
    mean_t2_count: float
    mean_t3_count: float
    mean_t1_per_1000: float
    mean_t2_per_1000: float
    mean_t3_per_1000: float
    theory_tree_vertices_fraction: float


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

    def tree_statistics(self) -> tuple[int, int, int, int]:
        tree_vertices = 0
        t1 = 0
        t2 = 0
        t3 = 0
        for i, p in enumerate(self.parent):
            if i != p:
                continue
            v = self.size[i]
            e = self.edge_count[i]
            if e == v - 1:
                tree_vertices += v
                if v == 1:
                    t1 += 1
                elif v == 2:
                    t2 += 1
                elif v == 3:
                    t3 += 1
        return tree_vertices, t1, t2, t3


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


def simulate_one_trial(n: int, m: int, rng: np.random.Generator) -> tuple[int, int, int, int]:
    uf = UnionFindCounter(n)
    edges = sample_unique_edges_sparse(n=n, m=m, rng=rng)
    for u, v in edges:
        uf.add_edge(u, v)
    return uf.tree_statistics()


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
    theory_tree_frac = 1.0 - giant_fraction_theory(c)

    tree_fracs: List[float] = []
    t1_counts: List[int] = []
    t2_counts: List[int] = []
    t3_counts: List[int] = []

    for _ in range(trials):
        tree_vertices, t1, t2, t3 = simulate_one_trial(n=n, m=m, rng=rng)
        tree_fracs.append(tree_vertices / n)
        t1_counts.append(t1)
        t2_counts.append(t2)
        t3_counts.append(t3)

    mean_tree_frac = float(np.mean(tree_fracs)) if tree_fracs else 0.0
    sd_tree_frac = float(np.std(tree_fracs, ddof=1)) if len(tree_fracs) >= 2 else 0.0
    se_tree_frac = sd_tree_frac / math.sqrt(len(tree_fracs)) if tree_fracs else 0.0

    mean_t1 = float(np.mean(t1_counts)) if t1_counts else 0.0
    mean_t2 = float(np.mean(t2_counts)) if t2_counts else 0.0
    mean_t3 = float(np.mean(t3_counts)) if t3_counts else 0.0

    scale = 1000.0 / n if n > 0 else 0.0
    return SimulationResult(
        n=n,
        c=c,
        m=m,
        trials=trials,
        mean_tree_vertices_fraction=mean_tree_frac,
        sd_tree_vertices_fraction=sd_tree_frac,
        se_tree_vertices_fraction=se_tree_frac,
        ci95_low_tree_vertices_fraction=max(0.0, mean_tree_frac - 1.96 * se_tree_frac),
        ci95_high_tree_vertices_fraction=mean_tree_frac + 1.96 * se_tree_frac,
        mean_t1_count=mean_t1,
        mean_t2_count=mean_t2,
        mean_t3_count=mean_t3,
        mean_t1_per_1000=mean_t1 * scale,
        mean_t2_per_1000=mean_t2 * scale,
        mean_t3_per_1000=mean_t3 * scale,
        theory_tree_vertices_fraction=theory_tree_frac,
    )


def save_results_csv(results: Iterable[SimulationResult], output_csv: Path) -> None:
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "n",
            "c",
            "m",
            "trials",
            "mean_tree_vertices_fraction",
            "sd_tree_vertices_fraction",
            "se_tree_vertices_fraction",
            "ci95_low_tree_vertices_fraction",
            "ci95_high_tree_vertices_fraction",
            "mean_t1_count",
            "mean_t2_count",
            "mean_t3_count",
            "mean_t1_per_1000",
            "mean_t2_per_1000",
            "mean_t3_per_1000",
            "theory_tree_vertices_fraction",
        ])
        for r in results:
            writer.writerow([
                r.n,
                f"{r.c:.6f}",
                r.m,
                r.trials,
                f"{r.mean_tree_vertices_fraction:.6f}",
                f"{r.sd_tree_vertices_fraction:.6f}",
                f"{r.se_tree_vertices_fraction:.6f}",
                f"{r.ci95_low_tree_vertices_fraction:.6f}",
                f"{r.ci95_high_tree_vertices_fraction:.6f}",
                f"{r.mean_t1_count:.6f}",
                f"{r.mean_t2_count:.6f}",
                f"{r.mean_t3_count:.6f}",
                f"{r.mean_t1_per_1000:.6f}",
                f"{r.mean_t2_per_1000:.6f}",
                f"{r.mean_t3_per_1000:.6f}",
                f"{r.theory_tree_vertices_fraction:.6f}",
            ])


def print_results_table(results: List[SimulationResult]) -> None:
    header = (
        f"{'n':>9}  {'c':>6}  {'m':>8}  {'trials':>8}  {'tree/n':>9}  {'T1/1000':>10}  {'T2/1000':>10}  {'T3/1000':>10}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.n:>9}  {r.c:>6.2f}  {r.m:>8}  {r.trials:>8}  {r.mean_tree_vertices_fraction:>9.4f}  "
            f"{r.mean_t1_per_1000:>10.3f}  {r.mean_t2_per_1000:>10.3f}  {r.mean_t3_per_1000:>10.3f}"
        )


def make_plot(results: List[SimulationResult], output_plot: Path, c_values: List[float]) -> None:
    ns = sorted({r.n for r in results})
    x_positions = list(range(len(ns)))

    fig, axes = plt.subplots(2, 1, figsize=(13.2, 10.4), sharex=True)
    fig.subplots_adjust(top=0.86, bottom=0.12, hspace=0.24)

    color_map = {0.55: "#1d4f91", 0.70: "#b26a00", 1.00: "#8b1e3f"}
    marker_map = {1: "o", 2: "s", 3: "^"}
    linestyle_map = {1: "-", 2: "--", 3: ":"}

    fig.suptitle(
        "Phase 3.4 — small tree components melt into the giant above the threshold",
        fontsize=16,
        fontweight="bold",
        y=0.97,
    )
    fig.text(
        0.5,
        0.925,
        "Top panel compares the empirical fraction of vertices in tree components with the theoretical reference 1-G(c). Bottom panel tracks mean counts of T1, T2, and T3 per 1,000 vertices.",
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

        tree_frac = [r.mean_tree_vertices_fraction for r in subset]
        low = [r.ci95_low_tree_vertices_fraction for r in subset]
        high = [r.ci95_high_tree_vertices_fraction for r in subset]
        theory = [r.theory_tree_vertices_fraction for r in subset]

        ax_top.plot(x_positions, tree_frac, marker="o", linewidth=2.2, label=f"empirical tree-vertex fraction, c = {c:.2f}", color=color)
        ax_top.fill_between(x_positions, low, high, alpha=0.14, color=color)
        ax_top.plot(x_positions, theory, linestyle="--", linewidth=1.9, label=f"theory 1-G(c), c = {c:.2f}", color=color)

        for k in [1, 2, 3]:
            values = [getattr(r, f"mean_t{k}_per_1000") for r in subset]
            ax_bottom.plot(
                x_positions,
                values,
                marker=marker_map[k],
                linestyle=linestyle_map[k],
                linewidth=2.0,
                label=f"T{k} per 1000, c = {c:.2f}",
                color=color,
                alpha=0.92,
            )

    ax_top.set_ylabel("Tree-vertex fraction")
    ax_top.set_title("Empirical tree-vertex mass versus the theoretical fraction outside the giant", fontsize=12.7, pad=10)
    ax_top.grid(True, alpha=0.28)
    ax_top.legend(ncol=2, fontsize=9.0, frameon=True)

    ax_bottom.set_ylabel("Mean count per 1,000 vertices")
    ax_bottom.set_title("Smaller trees remain the most persistent among surviving tree components", fontsize=12.7, pad=10)
    ax_bottom.grid(True, alpha=0.28)
    ax_bottom.legend(ncol=3, fontsize=8.8, frameon=True)

    ax_bottom.set_xticks(x_positions)
    ax_bottom.set_xticklabels([f"{n:,}" for n in ns], rotation=0)
    ax_bottom.set_xlabel("Number of vertices $n$")

    fig.text(
        0.5,
        0.04,
        "Default experiment: 1,000 simulations for each n and each c in {0.55, 0.70, 1.00}.",
        ha="center",
        va="center",
        fontsize=9.8,
        color="0.30",
    )

    output_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_plot, dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Monte Carlo simulations for Phase 3.4 and generate a CSV plus PNG tracking tree decay above the threshold.")
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
    print(f"saved csv to: {args.output_csv}")
    print(f"saved plot to: {args.output_plot}")


if __name__ == "__main__":
    main()
