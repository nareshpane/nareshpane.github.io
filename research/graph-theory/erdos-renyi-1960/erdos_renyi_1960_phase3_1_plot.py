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
# Phase 3.1 plot script.
# Default output matches the HTML placeholder:
#     erdos_renyi_1960_phase3_1.png
#
# Idea:
#   Just below n/2, the largest component is still sublinear and usually
#   tree-like. This script estimates that picture empirically over many
#   Monte Carlo trials for several values of n and several subcritical
#   choices of c approaching 1/2 from below.
#
# Default experiment:
#   n = 100, 500, 1000, 5000, 10000, 20000, 50000
#   c = 0.40, 0.45, 0.49
#   trials = 1000
#
# Outputs:
#   - erdos_renyi_1960_phase3_1.csv
#   - erdos_renyi_1960_phase3_1.png
# ============================================================

DEFAULT_NS = [100, 500, 1000, 5000, 10000, 20000, 50000]
DEFAULT_C_VALUES = [0.40, 0.45, 0.49]
DEFAULT_TRIALS = 1000
DEFAULT_SEED = 24680
DEFAULT_OUTPUT_CSV = Path("./erdos_renyi_1960_phase3_1.csv")
DEFAULT_OUTPUT_PLOT = Path("./erdos_renyi_1960_phase3_1.png")


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
    mean_largest_fraction: float
    probability_largest_is_tree: float
    probability_largest_is_unicyclic: float
    probability_largest_is_multicyclic: float
    mean_second_largest_size: float
    theory_largest_size: float


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

    def largest_two_component_stats(self) -> tuple[int, int, str]:
        largest_size = 0
        largest_kind = "tree"
        second_size = 0

        for i, p in enumerate(self.parent):
            if i != p:
                continue
            comp_size = self.size[i]
            comp_edges = self.edge_count[i]
            if comp_edges == comp_size - 1:
                kind = "tree"
            elif comp_edges == comp_size:
                kind = "unicyclic"
            else:
                kind = "multicyclic"

            if comp_size > largest_size:
                second_size = largest_size
                largest_size = comp_size
                largest_kind = kind
            elif comp_size > second_size:
                second_size = comp_size

        return largest_size, second_size, largest_kind


def encode_edge(u: int, v: int, n: int) -> int:
    if u < v:
        return u * n + v
    return v * n + u


def sample_unique_edges_sparse(n: int, m: int, rng: np.random.Generator) -> List[int]:
    if m < 0:
        raise ValueError("m must be nonnegative.")
    max_edges = n * (n - 1) // 2
    if m > max_edges:
        raise ValueError("m exceeds the number of simple undirected edges.")

    edges: set[int] = set()
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
        codes = a * n + b
        for code in codes.tolist():
            edges.add(int(code))
            if len(edges) >= m:
                break

    return list(edges)


def simulate_one_trial(n: int, m: int, rng: np.random.Generator) -> tuple[int, int, str]:
    uf = UnionFindCounter(n)
    encoded_edges = sample_unique_edges_sparse(n=n, m=m, rng=rng)
    for code in encoded_edges:
        u = code // n
        v = code % n
        uf.add_edge(u, v)
    return uf.largest_two_component_stats()


def subcritical_largest_theory(n: int, c: float) -> float:
    if n <= 3 or c <= 0.0 or c >= 0.5:
        return float("nan")
    alpha = 2.0 * c - 1.0 - math.log(2.0 * c)
    if alpha <= 0.0:
        return float("nan")
    logn = math.log(n)
    loglogn = math.log(max(logn, 1.0000001))
    value = (logn - 2.5 * loglogn) / alpha
    return max(0.0, value)


def simulate_statistics(n: int, c: float, trials: int, rng: np.random.Generator) -> SimulationResult:
    m = max(1, int(round(c * n)))

    largest_sizes: List[int] = []
    second_largest_sizes: List[int] = []
    tree_count = 0
    unicyclic_count = 0
    multicyclic_count = 0

    for _ in range(trials):
        largest_size, second_largest_size, largest_kind = simulate_one_trial(n=n, m=m, rng=rng)
        largest_sizes.append(largest_size)
        second_largest_sizes.append(second_largest_size)

        if largest_kind == "tree":
            tree_count += 1
        elif largest_kind == "unicyclic":
            unicyclic_count += 1
        else:
            multicyclic_count += 1

    mean_largest = float(np.mean(largest_sizes)) if largest_sizes else 0.0
    sd_largest = float(np.std(largest_sizes, ddof=1)) if len(largest_sizes) >= 2 else 0.0
    se_largest = sd_largest / math.sqrt(len(largest_sizes)) if largest_sizes else 0.0

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
        mean_largest_fraction=(mean_largest / n) if n > 0 else 0.0,
        probability_largest_is_tree=(tree_count / trials) if trials > 0 else 0.0,
        probability_largest_is_unicyclic=(unicyclic_count / trials) if trials > 0 else 0.0,
        probability_largest_is_multicyclic=(multicyclic_count / trials) if trials > 0 else 0.0,
        mean_second_largest_size=float(np.mean(second_largest_sizes)) if second_largest_sizes else 0.0,
        theory_largest_size=subcritical_largest_theory(n=n, c=c),
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
                "mean_largest_size",
                "sd_largest_size",
                "se_largest_size",
                "ci95_low_largest_size",
                "ci95_high_largest_size",
                "mean_largest_fraction",
                "probability_largest_is_tree",
                "probability_largest_is_unicyclic",
                "probability_largest_is_multicyclic",
                "mean_second_largest_size",
                "theory_largest_size",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r.n,
                    f"{r.c:.6f}",
                    r.m,
                    r.trials,
                    f"{r.mean_largest_size:.6f}",
                    f"{r.sd_largest_size:.6f}",
                    f"{r.se_largest_size:.6f}",
                    f"{r.ci95_low_largest_size:.6f}",
                    f"{r.ci95_high_largest_size:.6f}",
                    f"{r.mean_largest_fraction:.6f}",
                    f"{r.probability_largest_is_tree:.6f}",
                    f"{r.probability_largest_is_unicyclic:.6f}",
                    f"{r.probability_largest_is_multicyclic:.6f}",
                    f"{r.mean_second_largest_size:.6f}",
                    f"{r.theory_largest_size:.6f}" if not math.isnan(r.theory_largest_size) else "nan",
                ]
            )


def print_results_table(results: List[SimulationResult]) -> None:
    header = (
        f"{'n':>9}  {'c':>6}  {'m':>8}  {'trials':>8}  "
        f"{'mean largest':>13}  {'largest/n':>10}  {'P(tree)':>9}  {'P(uni)':>8}  {'P(multi)':>10}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.n:>9}  {r.c:>6.2f}  {r.m:>8}  {r.trials:>8}  "
            f"{r.mean_largest_size:>13.3f}  {r.mean_largest_fraction:>10.4f}  "
            f"{r.probability_largest_is_tree:>9.4f}  {r.probability_largest_is_unicyclic:>8.4f}  "
            f"{r.probability_largest_is_multicyclic:>10.4f}"
        )


def make_plot(results: List[SimulationResult], output_plot: Path, c_values: List[float]) -> None:
    ns = sorted({r.n for r in results})
    x_positions = list(range(len(ns)))

    fig, axes = plt.subplots(2, 1, figsize=(13.0, 10.2), sharex=True)
    fig.subplots_adjust(top=0.86, bottom=0.12, hspace=0.24)

    color_map = {
        0.40: "#1d4f91",
        0.45: "#b26a00",
        0.49: "#8b1e3f",
    }

    style_note = (
        "Solid lines show empirical means over repeated simulations. "
        "Dashed lines show the subcritical benchmark "
        r"$\frac{1}{\alpha}\left(\log n - \frac{5}{2}\log\log n\right)$"
        " with "
        r"$\alpha = 2c - 1 - \log(2c)$."
    )
    fig.suptitle(
        "Phase 3.1 — just below $n/2$, the largest component is still sublinear and usually tree-like",
        fontsize=16,
        fontweight="bold",
        y=0.97,
    )
    fig.text(0.5, 0.925, style_note, ha="center", va="center", fontsize=10.6, color="0.25")

    ax_top, ax_bottom = axes

    for c in c_values:
        subset = [r for r in results if abs(r.c - c) < 1e-12]
        subset.sort(key=lambda r: r.n)
        color = color_map.get(round(c, 2), None)

        mean_largest = [r.mean_largest_size for r in subset]
        theory_largest = [r.theory_largest_size for r in subset]
        low = [r.ci95_low_largest_size for r in subset]
        high = [r.ci95_high_largest_size for r in subset]
        tree_prob = [r.probability_largest_is_tree for r in subset]

        ax_top.plot(
            x_positions,
            mean_largest,
            marker="o",
            linewidth=2.2,
            label=f"empirical mean, c = {c:.2f}",
            color=color,
        )
        ax_top.fill_between(
            x_positions,
            low,
            high,
            alpha=0.14,
            color=color,
        )
        ax_top.plot(
            x_positions,
            theory_largest,
            linestyle="--",
            linewidth=1.9,
            label=f"theory, c = {c:.2f}",
            color=color,
        )

        ax_bottom.plot(
            x_positions,
            tree_prob,
            marker="o",
            linewidth=2.2,
            label=f"c = {c:.2f}",
            color=color,
        )

    ax_top.set_ylabel("Largest component size")
    ax_top.set_title("Empirical largest-component size versus the subcritical logarithmic benchmark", fontsize=12.6, pad=10)
    ax_top.grid(True, alpha=0.28)
    ax_top.legend(ncol=2, fontsize=9.2, frameon=True)

    ax_bottom.set_ylabel("Probability largest component is a tree")
    ax_bottom.set_ylim(-0.02, 1.02)
    ax_bottom.set_title("“Usually tree-like” behavior of the largest component", fontsize=12.6, pad=10)
    ax_bottom.grid(True, alpha=0.28)
    ax_bottom.legend(ncol=3, fontsize=9.4, frameon=True)

    ax_bottom.set_xticks(x_positions)
    ax_bottom.set_xticklabels([f"{n:,}" for n in ns], rotation=0)
    ax_bottom.set_xlabel("Number of vertices $n$")

    fig.text(
        0.5,
        0.04,
        "Default experiment: 1,000 simulations for each n and each c in {0.40, 0.45, 0.49}.",
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
        description=(
            "Run Monte Carlo simulations for Phase 3.1 and generate a CSV plus PNG showing "
            "that just below n/2 the largest component is still sublinear and usually tree-like."
        )
    )
    parser.add_argument("--ns", type=int, nargs="+", default=DEFAULT_NS, help="Values of n to simulate.")
    parser.add_argument("--c-values", type=float, nargs="+", default=DEFAULT_C_VALUES, help="Subcritical c values to compare.")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS, help="Number of Monte Carlo trials per (n, c).")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV, help="Output CSV path.")
    parser.add_argument("--output-plot", type=Path, default=DEFAULT_OUTPUT_PLOT, help="Output plot path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ns = sorted({int(n) for n in args.ns if int(n) >= 2})
    c_values = sorted({float(c) for c in args.c_values if 0.0 < float(c) < 0.5})
    trials = max(1, int(args.trials))

    if not ns:
        raise ValueError("Need at least one n >= 2.")
    if not c_values:
        raise ValueError("Need at least one c with 0 < c < 1/2.")

    print("Running Phase 3.1 simulation.")
    print(f"n values: {ns}")
    print(f"c values: {[round(c, 4) for c in c_values]}")
    print(f"trials per (n, c): {trials}")
    print("Note: the full default run is computationally heavy.")

    rng = np.random.default_rng(args.seed)
    results: List[SimulationResult] = []

    total_jobs = len(ns) * len(c_values)
    completed = 0

    for c in c_values:
        for n in ns:
            completed += 1
            print(f"[{completed}/{total_jobs}] Simulating n = {n:,}, c = {c:.2f} ...")
            result = simulate_statistics(n=n, c=c, trials=trials, rng=rng)
            results.append(result)

    results.sort(key=lambda r: (r.c, r.n))

    print()
    print_results_table(results)
    save_results_csv(results, args.output_csv)
    make_plot(results, args.output_plot, c_values=c_values)

    print()
    print(f"Saved CSV to: {args.output_csv}")
    print(f"Saved plot to: {args.output_plot}")


if __name__ == "__main__":
    main()
