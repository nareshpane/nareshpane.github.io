#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from textwrap import fill
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# Phase 1.2 plot script.
# Default output matches the HTML placeholder:
#     erdos_renyi_1960_phase1_2.png
#
# Idea:
#   For k = 3, 4, 5, 6, 7, 8, test the probability that at least one
#   k-vertex tree component is visible when m is chosen right at the
#   threshold scale m = floor(n^((k-2)/(k-1))).
#
#   Actual values  : solid lines
#   Predicted values: dotted lines, same color as the corresponding k
# ============================================================
DEFAULT_NS = [100, 500, 1000, 5000, 10000]
DEFAULT_KS = [3, 4, 5, 6, 7, 8]
DEFAULT_TRIALS = 1000
DEFAULT_SEED = 12345
DEFAULT_OUTPUT_CSV = Path("./erdos_renyi_1960_phase1_2.csv")
DEFAULT_OUTPUT_PLOT = Path("./erdos_renyi_1960_phase1_2.png")


@dataclass
class SimulationResult:
    k: int
    alpha: float
    n: int
    m: int
    trials: int
    successes: int
    probability_at_least_one: float
    predicted_probability_at_least_one: float
    predicted_lambda: float
    mean_count: float
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

    def count_tree_components_of_order(self, k: int) -> int:
        count = 0
        for i, p in enumerate(self.parent):
            if i == p and self.size[i] == k and self.edge_count[i] == k - 1:
                count += 1
        return count


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


def alpha_for_k(k: int) -> float:
    return (k - 2) / (k - 1)


def theoretical_lambda(k: int, rho: float = 1.0) -> float:
    return ((2.0 * rho) ** (k - 1)) * (k ** (k - 2)) / math.factorial(k)


def theoretical_probability_at_least_one(k: int, rho: float = 1.0) -> float:
    lam = theoretical_lambda(k=k, rho=rho)
    return 1.0 - math.exp(-lam)


def trial_tree_count(n: int, m: int, k: int, rng: np.random.Generator) -> int:
    uf = UnionFindCounter(n)
    edges = sample_unique_edges_sparse(n=n, m=m, rng=rng)
    for u, v in edges:
        uf.add_edge(u, v)
    return uf.count_tree_components_of_order(k)


def simulate_probability(n: int, k: int, trials: int, rng: np.random.Generator) -> SimulationResult:
    alpha = alpha_for_k(k)
    m = max(1, int(round(n ** alpha)))
    predicted_lambda = theoretical_lambda(k=k, rho=1.0)
    predicted_probability = 1.0 - math.exp(-predicted_lambda)

    counts: List[int] = []
    successes = 0
    for _ in range(trials):
        count = trial_tree_count(n=n, m=m, k=k, rng=rng)
        counts.append(count)
        if count >= 1:
            successes += 1

    p_hat = successes / trials
    mean_count = float(np.mean(counts)) if counts else 0.0
    se = math.sqrt(p_hat * (1.0 - p_hat) / trials) if trials > 0 else 0.0
    ci_low = max(0.0, p_hat - 1.96 * se)
    ci_high = min(1.0, p_hat + 1.96 * se)

    return SimulationResult(
        k=k,
        alpha=alpha,
        n=n,
        m=m,
        trials=trials,
        successes=successes,
        probability_at_least_one=p_hat,
        predicted_probability_at_least_one=predicted_probability,
        predicted_lambda=predicted_lambda,
        mean_count=mean_count,
        standard_error=se,
        ci_low=ci_low,
        ci_high=ci_high,
    )


def print_results_table(results: List[SimulationResult]) -> None:
    header = (
        f"{'k':>4}  {'n':>9}  {'alpha':>7}  {'m':>9}  {'trials':>8}  {'runs with ≥1':>12}  "
        f"{'actual p':>10}  {'pred p':>10}  {'mean count':>11}  {'95% CI':>23}"
    )
    print(header)
    print("-" * len(header))

    for r in results:
        ci_text = f"[{r.ci_low:.4f}, {r.ci_high:.4f}]"
        print(
            f"{r.k:>4}  {r.n:>9}  {r.alpha:>7.3f}  {r.m:>9}  {r.trials:>8}  {r.successes:>12}  "
            f"{r.probability_at_least_one:>10.4f}  {r.predicted_probability_at_least_one:>10.4f}  {r.mean_count:>11.4f}  {ci_text:>23}"
        )


def save_results_csv(results: Iterable[SimulationResult], output_csv: Path) -> None:
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "k",
                "alpha",
                "n",
                "m",
                "trials",
                "runs_with_at_least_one_k_tree",
                "estimated_probability_at_least_one_k_tree",
                "predicted_probability_at_least_one_k_tree",
                "predicted_lambda",
                "mean_k_tree_count",
                "standard_error",
                "ci_95_low",
                "ci_95_high",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r.k,
                    f"{r.alpha:.6f}",
                    r.n,
                    r.m,
                    r.trials,
                    r.successes,
                    f"{r.probability_at_least_one:.6f}",
                    f"{r.predicted_probability_at_least_one:.6f}",
                    f"{r.predicted_lambda:.6f}",
                    f"{r.mean_count:.6f}",
                    f"{r.standard_error:.6f}",
                    f"{r.ci_low:.6f}",
                    f"{r.ci_high:.6f}",
                ]
            )


def annotate_actual_points(ax, xs: List[int], ys: List[float], ms: List[int], color: str) -> None:
    for x, y, m in zip(xs, ys, ms):
        ax.annotate(
            f"m={m}",
            xy=(x, y),
            xytext=(0, -16),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=8.2,
            color="0.20",
            bbox=dict(facecolor="white", edgecolor=color, boxstyle="round,pad=0.14", alpha=0.92),
        )


def make_plot(results: List[SimulationResult], output_plot: Path) -> None:
    ks = sorted(set(r.k for r in results))
    ns = sorted(set(r.n for r in results))
    x_positions = list(range(len(ns)))
    x_lookup = {n: i for i, n in enumerate(ns)}
    trials = results[0].trials if results else DEFAULT_TRIALS

    color_map: Dict[int, str] = {
        3: "#d62728",  # red
        4: "#1f77b4",  # blue
        5: "#2ca02c",  # green
        6: "#9467bd",  # purple
        7: "#ff7f0e",  # orange
        8: "#8c564b",  # brown
    }

    fig, ax = plt.subplots(figsize=(13.2, 8.6))

    for k in ks:
        subset = sorted((r for r in results if r.k == k), key=lambda r: r.n)
        xs = [x_lookup[r.n] for r in subset]
        ys_actual = [r.probability_at_least_one for r in subset]
        ys_pred = [r.predicted_probability_at_least_one for r in subset]
        yerr = [1.96 * r.standard_error for r in subset]
        ms = [r.m for r in subset]
        color = color_map.get(k, None)

        ax.errorbar(
            xs,
            ys_actual,
            yerr=yerr,
            fmt="o-",
            capsize=4,
            linewidth=1.9,
            markersize=5.8,
            color=color,
            label=rf"$k={k}$ actual",
        )
        ax.plot(
            xs,
            ys_pred,
            linestyle=":",
            linewidth=2.1,
            color=color,
            label=rf"$k={k}$ predicted",
        )

        annotate_actual_points(ax, xs, ys_actual, ms, color)

    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"{n:,}" for n in ns], fontsize=10)
    ax.set_xlim(-0.25, len(ns) - 0.75 if len(ns) > 1 else 0.25)
    ax.set_ylim(0.0, 1.08)
    ax.set_xlabel("Values of n", fontsize=12, labelpad=12)
    ax.set_ylabel("Probability of seeing at least one k-vertex tree", fontsize=12)
    ax.set_title(
        rf"Phase 1.2: Appearance thresholds for k-trees, k = 3, 4, 5, 6, 7, 8 ({trials:,} runs per node-size)",
        fontsize=13,
        pad=18,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", ncol=2, fontsize=9)

    lambda_parts = [
        rf"$\lambda_{k}={theoretical_lambda(k):.3f}$" for k in ks
    ]
    explanation = (
        "Solid lines are empirical probabilities from simulation, while dotted lines are the Poisson-threshold predictions "
        r"$1-e^{-\lambda_k}$ at $\rho=1$. "
        "The edge count is chosen at the corresponding threshold scale "
        r"$m=\lfloor n^{(k-2)/(k-1)}\rfloor$, so larger k should become visible only at progressively larger sparse scales."
    )

    caption_lines = [
        fill(f"This plot uses {trials:,} independent runs for each (n, k) combination.", width=108),
        fill(explanation, width=108),
        fill("Poisson-threshold means used for the dotted prediction curves: " + ", ".join(lambda_parts) + ".", width=108),
    ]

    fig.text(
        0.5,
        0.015,
        "\n".join(caption_lines),
        ha="center",
        va="bottom",
        fontsize=9.8,
        color="0.25",
        wrap=True,
    )

    fig.tight_layout(rect=[0, 0.19, 1, 1])
    fig.savefig(output_plot, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate the Phase 1.2 probability that a k-vertex tree-component is visible when "
            "m is chosen at the threshold scale m = floor(n^((k-2)/(k-1)))."
        )
    )
    parser.add_argument("--ns", type=int, nargs="+", default=DEFAULT_NS, help="Values of n to test.")
    parser.add_argument("--ks", type=int, nargs="+", default=DEFAULT_KS, help="Values of k to test.")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS, help="Number of Monte Carlo trials per (n, k).")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Base random seed.")
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV, help="CSV file for numerical results.")
    parser.add_argument("--output-plot", type=Path, default=DEFAULT_OUTPUT_PLOT, help="PNG plot file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ns = sorted(set(n for n in args.ns if n >= 2))
    ks = sorted(set(k for k in args.ks if k >= 3))
    rng = np.random.default_rng(args.seed)

    results: List[SimulationResult] = []
    print("Running Phase 1.2 threshold simulations...")
    for k in ks:
        alpha = alpha_for_k(k)
        for n in ns:
            m = max(1, int(round(n ** alpha)))
            print(f"  k = {k}, n = {n:7d}, m = round(n^{alpha:.3f}) = {m:7d}, trials = {args.trials}")
            results.append(simulate_probability(n=n, k=k, trials=args.trials, rng=rng))

    results.sort(key=lambda r: (r.k, r.n))

    print()
    print_results_table(results)
    print()

    save_results_csv(results, args.output_csv)
    make_plot(results, args.output_plot)

    print(f"Saved CSV to:  {args.output_csv}")
    print(f"Saved PNG to:  {args.output_plot}")


if __name__ == "__main__":
    main()
