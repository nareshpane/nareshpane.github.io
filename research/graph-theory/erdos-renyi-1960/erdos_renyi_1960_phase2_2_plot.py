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

# ============================================================
# Phase 2.2 plot script.
# Default output matches the HTML placeholder:
#     erdos_renyi_1960_phase2_2.png
#
# Idea:
#   At the linear scale m ~ c n with 0 < c < 1/2, almost all components
#   should still be trees or unicyclic pieces. This figure estimates how
#   often all nontrivial components are tree-or-unicyclic, and how many
#   unicyclic versus multicyclic components appear on average.
# ============================================================
DEFAULT_NS = [100, 500, 1000, 5000, 10000, 20000, 50000]
DEFAULT_C = 0.30
DEFAULT_TRIALS = 1000
DEFAULT_SEED = 12345
DEFAULT_OUTPUT_CSV = Path("./erdos_renyi_1960_phase2_2.csv")
DEFAULT_OUTPUT_PLOT = Path("./erdos_renyi_1960_phase2_2.png")


@dataclass
class SimulationResult:
    n: int
    c: float
    m: int
    trials: int
    success_tree_or_unicyclic_only: int
    probability_tree_or_unicyclic_only: float
    mean_tree_components: float
    mean_unicyclic_components: float
    mean_multicyclic_components: float
    mean_isolates: float
    standard_error: float
    ci_low: float
    ci_high: float


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
        tree_components = 0
        unicyclic_components = 0
        multicyclic_components = 0
        isolates = 0
        for i, p in enumerate(self.parent):
            if i != p:
                continue
            v = self.size[i]
            e = self.edge_count[i]
            if v == 1:
                isolates += 1
                continue
            if e == v - 1:
                tree_components += 1
            elif e == v:
                unicyclic_components += 1
            elif e > v:
                multicyclic_components += 1
        return tree_components, unicyclic_components, multicyclic_components, isolates


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


def simulate_probability(n: int, c: float, trials: int, rng: np.random.Generator) -> SimulationResult:
    m = max(1, int(round(c * n)))
    success = 0
    tree_counts: List[int] = []
    uni_counts: List[int] = []
    multi_counts: List[int] = []
    isolate_counts: List[int] = []

    for _ in range(trials):
        tree_c, uni_c, multi_c, isolates = simulate_one_trial(n=n, m=m, rng=rng)
        tree_counts.append(tree_c)
        uni_counts.append(uni_c)
        multi_counts.append(multi_c)
        isolate_counts.append(isolates)
        if multi_c == 0:
            success += 1

    p_hat = success / trials
    se = math.sqrt(p_hat * (1.0 - p_hat) / trials) if trials > 0 else 0.0
    ci_low = max(0.0, p_hat - 1.96 * se)
    ci_high = min(1.0, p_hat + 1.96 * se)

    return SimulationResult(
        n=n,
        c=c,
        m=m,
        trials=trials,
        success_tree_or_unicyclic_only=success,
        probability_tree_or_unicyclic_only=p_hat,
        mean_tree_components=float(np.mean(tree_counts)) if tree_counts else 0.0,
        mean_unicyclic_components=float(np.mean(uni_counts)) if uni_counts else 0.0,
        mean_multicyclic_components=float(np.mean(multi_counts)) if multi_counts else 0.0,
        mean_isolates=float(np.mean(isolate_counts)) if isolate_counts else 0.0,
        standard_error=se,
        ci_low=ci_low,
        ci_high=ci_high,
    )


def print_results_table(results: List[SimulationResult]) -> None:
    header = (
        f"{'n':>9}  {'m':>8}  {'trials':>8}  {'tree/unicyclic only':>19}  {'prob':>8}  {'mean trees':>11}  {'mean uni':>10}  {'mean multi':>12}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.n:>9}  {r.m:>8}  {r.trials:>8}  {r.success_tree_or_unicyclic_only:>19}  {r.probability_tree_or_unicyclic_only:>8.4f}  {r.mean_tree_components:>11.3f}  {r.mean_unicyclic_components:>10.3f}  {r.mean_multicyclic_components:>12.3f}"
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
                "runs_with_only_tree_or_unicyclic_components",
                "probability_only_tree_or_unicyclic_components",
                "mean_tree_components",
                "mean_unicyclic_components",
                "mean_multicyclic_components",
                "mean_isolates",
                "standard_error",
                "ci_95_low",
                "ci_95_high",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r.n,
                    f"{r.c:.6f}",
                    r.m,
                    r.trials,
                    r.success_tree_or_unicyclic_only,
                    f"{r.probability_tree_or_unicyclic_only:.6f}",
                    f"{r.mean_tree_components:.6f}",
                    f"{r.mean_unicyclic_components:.6f}",
                    f"{r.mean_multicyclic_components:.6f}",
                    f"{r.mean_isolates:.6f}",
                    f"{r.standard_error:.6f}",
                    f"{r.ci_low:.6f}",
                    f"{r.ci_high:.6f}",
                ]
            )


def make_plot(results: List[SimulationResult], output_plot: Path) -> None:
    ns = [r.n for r in results]
    x_positions = list(range(len(ns)))
    probs = [r.probability_tree_or_unicyclic_only for r in results]
    yerr = [1.96 * r.standard_error for r in results]
    trees = [r.mean_tree_components for r in results]
    unic = [r.mean_unicyclic_components for r in results]
    multi = [r.mean_multicyclic_components for r in results]
    ms = [r.m for r in results]
    trials = results[0].trials if results else DEFAULT_TRIALS
    c = results[0].c if results else DEFAULT_C

    fig, axes = plt.subplots(2, 1, figsize=(12.8, 9.8), sharex=True)
    ax1, ax2 = axes

    ax1.errorbar(
        x_positions,
        probs,
        yerr=yerr,
        fmt="o-",
        capsize=5,
        linewidth=1.9,
        markersize=6.6,
        label="Estimated probability that all nontrivial components are tree-or-unicyclic",
    )
    ax1.axhline(1.0, linestyle="--", linewidth=1.3, label="Phase 2.2 target: probability tends to 1")
    for x, y, m in zip(x_positions, probs, ms):
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
    ax1.set_ylabel("Probability", fontsize=12)
    ax1.set_ylim(0.0, 1.08)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower right", fontsize=9)

    ax2.plot(x_positions, trees, marker="o", linewidth=1.9, label="Mean number of tree components")
    ax2.plot(x_positions, unic, marker="o", linewidth=1.9, linestyle="--", label="Mean number of unicyclic components")
    ax2.plot(x_positions, multi, marker="o", linewidth=1.9, linestyle=":", label="Mean number of multicyclic components")
    ax2.set_ylabel("Mean component count", fontsize=12)
    ax2.set_xlabel("Values of n", fontsize=12, labelpad=10)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right", fontsize=9)

    ax2.set_xticks(x_positions)
    ax2.set_xticklabels([f"{n:,}" for n in ns], fontsize=10)

    fig.suptitle(
        rf"Phase 2.2: trees plus occasional unicyclic components at the linear scale $m=\lfloor c n \rceil$ with $c={c:.2f}$ ({trials:,} runs per node-size)",
        fontsize=14,
        y=0.975,
    )
    fig.text(
        0.5,
        0.94,
        "Top panel asks whether every nontrivial component is either a tree or a single-cycle component. Bottom panel tracks how many tree, unicyclic, and multicyclic components appear on average.",
        ha="center",
        va="center",
        fontsize=10.5,
        color="0.25",
    )

    explanation = (
        r"This subphase is not mainly about the appearance of short cycles themselves. It is about the shape of whole connected components. "
        r"At the linear scale $N(n)\sim c n$ with $0<c<1/2$, the graph should still be dominated by trees, with the non-tree exceptions usually being unicyclic components rather than multi-cycle pieces."
    )
    caption_lines = [
        fill(f"This figure uses {trials:,} independent runs for each tested node size.", width=126),
        fill(explanation, width=126),
        fill("The empirical expectation is that the probability of seeing only tree-or-unicyclic nontrivial components should move upward with n, while the mean number of multicyclic components stays comparatively small.", width=126),
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

    fig.tight_layout(rect=[0.03, 0.10, 0.98, 0.92])
    fig.savefig(output_plot, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate the Phase 2.2 behavior that, at the linear scale m = round(c n), almost all nontrivial components are trees or unicyclic pieces."
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
    print("Running Phase 2.2 tree-versus-unicyclic simulations...")
    for n in ns:
        m = max(1, int(round(c * n)))
        print(f"  n = {n:7d}, m = round(c n) = {m:7d}, trials = {args.trials}")
        results.append(simulate_probability(n=n, c=c, trials=args.trials, rng=rng))

    print()
    print_results_table(results)
    print()

    save_results_csv(results, args.output_csv)
    make_plot(results, args.output_plot)

    print(f"Saved CSV to:  {args.output_csv}")
    print(f"Saved PNG to:  {args.output_plot}")


if __name__ == "__main__":
    main()
