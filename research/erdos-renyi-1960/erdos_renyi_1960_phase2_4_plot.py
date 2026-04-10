#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from textwrap import fill
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# ============================================================
# Phase 2.4 plot script.
# Default output matches the HTML placeholder:
#     erdos_renyi_1960_phase2_4.png
#
# Idea:
#   At the linear scale m ~ c n with 0 < c < 1/2, most vertices should
#   still lie in tree components, and the mean number of components should
#   remain close to n - m.
# ============================================================
DEFAULT_NS = [100, 500, 1000, 5000, 10000, 20000, 50000]
DEFAULT_C = 0.30
DEFAULT_TRIALS = 400
DEFAULT_SEED = 12345
DEFAULT_OUTPUT_CSV = Path("./erdos_renyi_1960_phase2_4.csv")
DEFAULT_OUTPUT_PLOT = Path("./erdos_renyi_1960_phase2_4.png")

TREE_COLOR = "#1565c0"
UNICYCLIC_COLOR = "#2e7d32"
MULTICYCLIC_COLOR = "#c62828"
EMP_COLOR = "#6a1b9a"
REF_COLOR = "#444444"


@dataclass
class SimulationResult:
    n: int
    c: float
    m: int
    trials: int
    mean_components: float
    mean_tree_vertices_fraction: float
    mean_unicyclic_vertices_fraction: float
    mean_multicyclic_vertices_fraction: float
    mean_abs_component_error: float
    components_standard_error: float


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

    def summarize(self) -> Tuple[int, int, int, int]:
        total_components = 0
        tree_vertices = 0
        unicyclic_vertices = 0
        multicyclic_vertices = 0
        for i, p in enumerate(self.parent):
            if i != p:
                continue
            total_components += 1
            v = self.size[i]
            e = self.edge_count[i]
            if e == v - 1:
                tree_vertices += v
            elif e == v:
                unicyclic_vertices += v
            elif e > v:
                multicyclic_vertices += v
        return total_components, tree_vertices, unicyclic_vertices, multicyclic_vertices


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


def simulate_one_trial(n: int, m: int, rng: np.random.Generator) -> Tuple[int, int, int, int]:
    uf = UnionFindCounter(n)
    edges = sample_unique_edges_sparse(n=n, m=m, rng=rng)
    for u, v in edges:
        uf.add_edge(u, v)
    return uf.summarize()


def simulate_statistics(n: int, c: float, trials: int, rng: np.random.Generator) -> SimulationResult:
    m = max(1, int(round(c * n)))
    ref = n - m
    comps: List[int] = []
    tree_fracs: List[float] = []
    uni_fracs: List[float] = []
    multi_fracs: List[float] = []
    abs_errs: List[float] = []

    for _ in range(trials):
        total_components, tree_vertices, unicyclic_vertices, multicyclic_vertices = simulate_one_trial(n=n, m=m, rng=rng)
        comps.append(total_components)
        tree_fracs.append(tree_vertices / n)
        uni_fracs.append(unicyclic_vertices / n)
        multi_fracs.append(multicyclic_vertices / n)
        abs_errs.append(abs(total_components - ref))

    comp_mean = float(np.mean(comps)) if comps else 0.0
    comp_se = float(np.std(comps, ddof=1) / math.sqrt(len(comps))) if len(comps) >= 2 else 0.0

    return SimulationResult(
        n=n,
        c=c,
        m=m,
        trials=trials,
        mean_components=comp_mean,
        mean_tree_vertices_fraction=float(np.mean(tree_fracs)) if tree_fracs else 0.0,
        mean_unicyclic_vertices_fraction=float(np.mean(uni_fracs)) if uni_fracs else 0.0,
        mean_multicyclic_vertices_fraction=float(np.mean(multi_fracs)) if multi_fracs else 0.0,
        mean_abs_component_error=float(np.mean(abs_errs)) if abs_errs else 0.0,
        components_standard_error=comp_se,
    )


def save_results_csv(results: Iterable[SimulationResult], output_csv: Path) -> None:
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "n",
                "c",
                "m",
                "trials",
                "mean_components",
                "mean_tree_vertices_fraction",
                "mean_unicyclic_vertices_fraction",
                "mean_multicyclic_vertices_fraction",
                "mean_abs_component_error_from_n_minus_m",
                "components_standard_error",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r.n,
                    f"{r.c:.6f}",
                    r.m,
                    r.trials,
                    f"{r.mean_components:.6f}",
                    f"{r.mean_tree_vertices_fraction:.6f}",
                    f"{r.mean_unicyclic_vertices_fraction:.6f}",
                    f"{r.mean_multicyclic_vertices_fraction:.6f}",
                    f"{r.mean_abs_component_error:.6f}",
                    f"{r.components_standard_error:.6f}",
                ]
            )


def print_results_table(results: List[SimulationResult]) -> None:
    header = (
        f"{'n':>9}  {'m':>8}  {'trials':>8}  {'mean comps':>11}  {'tree frac':>10}  {'uni frac':>9}  {'multi frac':>11}  {'mean |comp-(n-m)|':>18}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.n:>9}  {r.m:>8}  {r.trials:>8}  {r.mean_components:>11.3f}  {r.mean_tree_vertices_fraction:>10.4f}  {r.mean_unicyclic_vertices_fraction:>9.4f}  {r.mean_multicyclic_vertices_fraction:>11.4f}  {r.mean_abs_component_error:>18.3f}"
        )


def make_plot(results: List[SimulationResult], output_plot: Path) -> None:
    ns = [r.n for r in results]
    x_positions = list(range(len(ns)))
    trials = results[0].trials if results else DEFAULT_TRIALS
    c = results[0].c if results else DEFAULT_C

    tree_frac = [r.mean_tree_vertices_fraction for r in results]
    uni_frac = [r.mean_unicyclic_vertices_fraction for r in results]
    multi_frac = [r.mean_multicyclic_vertices_fraction for r in results]
    mean_comps = [r.mean_components for r in results]
    se_comps = [1.96 * r.components_standard_error for r in results]
    ref_counts = [r.n - r.m for r in results]
    ms = [r.m for r in results]

    fig, axes = plt.subplots(2, 1, figsize=(12.8, 9.8), sharex=True)
    ax1, ax2 = axes

    ax1.plot(x_positions, tree_frac, marker="o", linewidth=1.9, color=TREE_COLOR, label="Vertices in tree components")
    ax1.plot(x_positions, uni_frac, marker="o", linewidth=1.9, linestyle="--", color=UNICYCLIC_COLOR, label="Vertices in unicyclic components")
    ax1.plot(x_positions, multi_frac, marker="o", linewidth=1.9, linestyle=":", color=MULTICYCLIC_COLOR, label="Vertices in multicyclic components")
    for x, y, m in zip(x_positions, tree_frac, ms):
        ax1.annotate(
            f"m={m}",
            xy=(x, y),
            xytext=(0, -16),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=8.2,
            color="0.20",
            bbox=dict(facecolor="white", edgecolor="0.80", boxstyle="round,pad=0.14"),
        )
    ax1.set_ylabel("Mean vertex fraction", fontsize=12)
    ax1.set_ylim(-0.03, 1.08)
    ax1.grid(True, alpha=0.3)

    ax2.errorbar(
        x_positions,
        mean_comps,
        yerr=se_comps,
        fmt="o-",
        capsize=5,
        linewidth=1.9,
        markersize=6.6,
        color=EMP_COLOR,
        label="Empirical mean number of components",
    )
    ax2.plot(x_positions, ref_counts, linestyle="--", linewidth=1.6, color=REF_COLOR, label=r"Reference line $n-m$")
    ax2.set_ylabel("Mean component count", fontsize=12)
    ax2.set_xlabel("Values of n", fontsize=12, labelpad=10)
    ax2.grid(True, alpha=0.3)

    ax2.set_xticks(x_positions)
    ax2.set_xticklabels([f"{n:,}" for n in ns], fontsize=10)

    fig.suptitle(
        rf"Phase 2.4: most vertices still lie in trees, and the number of components is about $n-m$ with $c={c:.2f}$ ({trials:,} runs per node-size)",
        fontsize=14,
        y=0.975,
    )
    subtitle = fill(
        r"Top panel tracks where vertices live: in tree components, unicyclic components, or multicyclic components. Bottom panel compares the empirical mean number of components against the heuristic $n-m$.",
        width=120,
    )
    fig.text(
        0.5,
        0.935,
        subtitle,
        ha="center",
        va="center",
        fontsize=10.5,
        color="0.25",
        linespacing=1.2,
    )

    top_handles = [
        Line2D([0], [0], color=TREE_COLOR, marker="o", linewidth=1.9, label="Vertices in tree components"),
        Line2D([0], [0], color=UNICYCLIC_COLOR, marker="o", linestyle="--", linewidth=1.9, label="Vertices in unicyclic components"),
        Line2D([0], [0], color=MULTICYCLIC_COLOR, marker="o", linestyle=":", linewidth=1.9, label="Vertices in multicyclic components"),
    ]
    ax1.legend(handles=top_handles, loc="upper right", fontsize=9, ncol=1, frameon=True)

    bottom_handles = [
        Line2D([0], [0], color=EMP_COLOR, marker="o", linewidth=1.9, label="Empirical mean components"),
        Line2D([0], [0], color=REF_COLOR, linestyle="--", linewidth=1.6, label=r"Reference $n-m$"),
    ]
    ax2.legend(handles=bottom_handles, loc="upper right", fontsize=9, ncol=1, frameon=True)

    explanation = (
        r"This subphase is the bridge between Phase 1 and Phase 2. Cycles have appeared, but they still do not capture most of the graph at the vertex level. Most added edges are still doing the forest job of merging components, so the component count remains close to $n-m$ and most vertices remain in tree components."
    )
    caption_lines = [
        fill(f"This figure uses {trials:,} independent runs for each tested node size.", width=128),
        fill(explanation, width=128),
        fill("The empirical expectation is that the tree-vertex fraction stays dominant, while the empirical component count tracks the reference line n-m up to an O(1)-scale discrepancy.", width=128),
    ]
    fig.text(
        0.5,
        0.018,
        "\n".join(caption_lines),
        ha="center",
        va="bottom",
        fontsize=9.6,
        color="0.25",
        wrap=True,
    )

    fig.tight_layout(rect=[0.03, 0.10, 0.98, 0.90])
    fig.savefig(output_plot, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate the Phase 2.4 behavior that, at the linear scale m = round(c n), most vertices still lie in tree components and the number of components stays close to n - m."
        )
    )
    parser.add_argument("--ns", type=int, nargs="+", default=DEFAULT_NS, help="Values of n to test.")
    parser.add_argument("--c", type=float, default=DEFAULT_C, help="Linear-scale constant in m = round(c n), should stay below 0.5.")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS, help="Number of Monte Carlo trials per n.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Base random seed.")
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV, help="CSV file for numerical results.")
    parser.add_argument("--output-plot", type=Path, default=DEFAULT_OUTPUT_PLOT, help="PNG plot file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    c = min(max(args.c, 0.01), 0.49)
    ns = sorted(set(n for n in args.ns if n >= 2))
    rng = np.random.default_rng(args.seed)

    results: List[SimulationResult] = []
    print("Running Phase 2.4 tree-vertex and component-count simulations...")
    for n in ns:
        m = max(1, int(round(c * n)))
        print(f"  n = {n:7d}, m = round(c n) = {m:7d}, trials = {args.trials}")
        results.append(simulate_statistics(n=n, c=c, trials=args.trials, rng=rng))

    print()
    print_results_table(results)
    print()

    save_results_csv(results, args.output_csv)
    make_plot(results, args.output_plot)

    print(f"Saved CSV to:  {args.output_csv}")
    print(f"Saved PNG to:  {args.output_plot}")


if __name__ == "__main__":
    main()
