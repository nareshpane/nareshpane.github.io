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
# Phase 1, subsection 1.1 plot script.
# Default output matches the HTML placeholder:
#     erdos_renyi_1960_phase1_1.png
# ============================================================
DEFAULT_NS = [100, 500, 1000, 5000, 10000, 50000]
DEFAULT_ALPHA = 0.5               # use m = floor(sqrt(n)) by default
DEFAULT_TRIALS = 1000
DEFAULT_SEED = 12345
DEFAULT_OUTPUT_CSV = Path("./erdos_renyi_1960_phase1_1.csv")
DEFAULT_OUTPUT_PLOT = Path("./erdos_renyi_1960_phase1_1.png")


@dataclass
class SimulationResult:
    n: int
    alpha: float
    m: int
    avg_degree: float
    trials: int
    successes: int
    probability: float
    standard_error: float
    ci_low: float
    ci_high: float


class UnionFindForestChecker:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.size = [1] * n
        self.has_cycle = False

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
            self.has_cycle = True
            return

        if self.size[ru] < self.size[rv]:
            ru, rv = rv, ru

        self.parent[rv] = ru
        self.size[ru] += self.size[rv]


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


def trial_is_forest(n: int, m: int, rng: np.random.Generator) -> bool:
    uf = UnionFindForestChecker(n)
    edges = sample_unique_edges_sparse(n=n, m=m, rng=rng)

    for u, v in edges:
        uf.add_edge(u, v)
        if uf.has_cycle:
            return False

    return True


def simulate_probability(n: int, alpha: float, trials: int, rng: np.random.Generator) -> SimulationResult:
    m = max(1, int(n ** alpha))
    successes = 0
    for _ in range(trials):
        if trial_is_forest(n=n, m=m, rng=rng):
            successes += 1

    p_hat = successes / trials
    se = math.sqrt(p_hat * (1.0 - p_hat) / trials) if trials > 0 else 0.0
    ci_low = max(0.0, p_hat - 1.96 * se)
    ci_high = min(1.0, p_hat + 1.96 * se)

    return SimulationResult(
        n=n,
        alpha=alpha,
        m=m,
        avg_degree=2.0 * m / n,
        trials=trials,
        successes=successes,
        probability=p_hat,
        standard_error=se,
        ci_low=ci_low,
        ci_high=ci_high,
    )


def print_results_table(results: List[SimulationResult]) -> None:
    header = (
        f"{'n':>10}  {'m=floor(n^alpha)':>17}  {'avg degree':>11}  {'trials':>8}  {'forest runs':>11}  "
        f"{'p_hat':>10}  {'SE':>10}  {'95% CI':>23}"
    )
    print(header)
    print("-" * len(header))

    for r in results:
        ci_text = f"[{r.ci_low:.4f}, {r.ci_high:.4f}]"
        print(
            f"{r.n:>10}  {r.m:>17}  {r.avg_degree:>11.4f}  {r.trials:>8}  {r.successes:>11}  "
            f"{r.probability:>10.4f}  {r.standard_error:>10.4f}  {ci_text:>23}"
        )


def save_results_csv(results: Iterable[SimulationResult], output_csv: Path) -> None:
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "n",
                "alpha",
                "m",
                "average_degree",
                "trials",
                "forest_successes",
                "estimated_probability_forest",
                "standard_error",
                "ci_95_low",
                "ci_95_high",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r.n,
                    f"{r.alpha:.6f}",
                    r.m,
                    f"{r.avg_degree:.6f}",
                    r.trials,
                    r.successes,
                    f"{r.probability:.6f}",
                    f"{r.standard_error:.6f}",
                    f"{r.ci_low:.6f}",
                    f"{r.ci_high:.6f}",
                ]
            )


def annotate_points(ax, xs: List[int], ys: List[float], ms: List[int]) -> None:
    for x, y, m in zip(xs, ys, ms):
        ax.annotate(
            f"m={m}\np={y:.3f}",
            xy=(x, y),
            xytext=(0, -18),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=8.8,
            color="0.20",
            bbox=dict(facecolor="white", edgecolor="0.80", boxstyle="round,pad=0.18"),
        )


def make_plot(results: List[SimulationResult], output_plot: Path) -> None:
    ns = [r.n for r in results]
    probs = [r.probability for r in results]
    yerr = [1.96 * r.standard_error for r in results]
    ms = [r.m for r in results]
    avg_degrees = [r.avg_degree for r in results]
    trials = results[0].trials if results else DEFAULT_TRIALS

    fig, ax = plt.subplots(figsize=(12.6, 7.6))
    x_positions = list(range(len(ns)))

    ax.errorbar(
        x_positions,
        probs,
        yerr=yerr,
        fmt="o-",
        capsize=5,
        linewidth=1.9,
        markersize=7,
        label="Estimated probability that G(n,m) is a forest",
    )
    ax.axhline(1.0, linestyle="--", linewidth=1.4, label="Phase 1 theoretical target: probability tends to 1")

    annotate_points(ax, x_positions, probs, ms)

    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"{n:,}" for n in ns], fontsize=10)
    ax.set_ylim(0.0, 1.08)
    ax.set_xlim(-0.25, len(ns) - 0.75 if len(ns) > 1 else 0.25)
    ax.set_xlabel("Values of n", fontsize=12)
    ax.set_ylabel("Estimated probability of being a forest", fontsize=12)
    ax.set_title(
        rf"Phase 1.1: Probability that $G(n,m)$ is a forest when $m = \lfloor n^{{1/2}} \rfloor$ ({trials:,} runs per node-size)",
        fontsize=13,
        pad=18,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    avg_degree_text = "The shrinking average degrees are: " + ", ".join(
        f"n={n:,}: {d:.3f}" for n, d in zip(ns, avg_degrees)
    )
    subtitle_lines = [
        fill(f"This plot uses {trials:,} independent simulation runs for each tested node size.", width=110),
        fill("It corresponds to the Phase 1 statement that when N(n)=o(n), the graph is overwhelmingly tree-like.", width=110),
        fill(avg_degree_text, width=110),
    ]
    fig.text(
        0.5,
        0.012,
        "\n".join(subtitle_lines),
        ha="center",
        va="bottom",
        fontsize=10,
        color="0.25",
        wrap=True,
    )

    fig.tight_layout(rect=[0, 0.11, 1, 1])
    fig.savefig(output_plot, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate the Phase 1.1 probability that G(n,m) is a forest when m = floor(n^alpha), "
            "using alpha = 1/2 by default."
        )
    )
    parser.add_argument("--ns", type=int, nargs="+", default=DEFAULT_NS, help="Values of n to test.")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help="Exponent in m = floor(n^alpha), should stay below 1.")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS, help="Number of Monte Carlo trials per n.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Base random seed.")
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV, help="CSV file for numerical results.")
    parser.add_argument("--output-plot", type=Path, default=DEFAULT_OUTPUT_PLOT, help="PNG plot file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    alpha = min(max(args.alpha, 0.05), 0.95)
    ns = sorted(set(n for n in args.ns if n >= 2))
    rng = np.random.default_rng(args.seed)

    results: List[SimulationResult] = []
    print("Running Phase 1.1 forest simulations...")
    for n in ns:
        m = max(1, int(n ** alpha))
        print(f"  n = {n:7d}, m = floor(n^{alpha:.3f}) = {m:7d}, trials = {args.trials}")
        results.append(simulate_probability(n=n, alpha=alpha, trials=args.trials, rng=rng))

    print()
    print_results_table(results)
    print()

    save_results_csv(results, args.output_csv)
    make_plot(results, args.output_plot)

    print(f"Saved CSV to:  {args.output_csv}")
    print(f"Saved PNG to:  {args.output_plot}")


if __name__ == "__main__":
    main()
