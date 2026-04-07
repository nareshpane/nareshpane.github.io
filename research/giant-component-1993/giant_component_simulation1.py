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


@dataclass
class SimulationResult:
    n: int
    m_label: str
    m: int
    trials: int
    successes: int
    probability: float
    standard_error: float
    ci_low: float
    ci_high: float


class UnionFind:
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

    def all_components_have_excess_at_most_two(self) -> bool:
        for i, p in enumerate(self.parent):
            if i == p:
                excess = self.edge_count[i] - self.size[i] + 1
                if excess > 2:
                    return False
        return True


def sample_unique_edges_sparse(
    n: int,
    m: int,
    rng: np.random.Generator,
) -> set[tuple[int, int]]:
    """
    Sample m distinct undirected edges uniformly from K_n.

    In this sparse regime, rejection sampling with a Python set is workable.
    """
    edges: set[tuple[int, int]] = set()

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


def trial_success(
    n: int,
    m: int,
    rng: np.random.Generator,
) -> bool:
    uf = UnionFind(n)
    edges = sample_unique_edges_sparse(n=n, m=m, rng=rng)

    for u, v in edges:
        uf.add_edge(u, v)

    return uf.all_components_have_excess_at_most_two()


def simulate_probability(
    n: int,
    m: int,
    m_label: str,
    trials: int,
    rng: np.random.Generator,
) -> SimulationResult:
    successes = 0
    for _ in range(trials):
        if trial_success(n=n, m=m, rng=rng):
            successes += 1

    p_hat = successes / trials
    se = math.sqrt(p_hat * (1.0 - p_hat) / trials) if trials > 0 else 0.0
    ci_low = max(0.0, p_hat - 1.96 * se)
    ci_high = min(1.0, p_hat + 1.96 * se)

    return SimulationResult(
        n=n,
        m_label=m_label,
        m=m,
        trials=trials,
        successes=successes,
        probability=p_hat,
        standard_error=se,
        ci_low=ci_low,
        ci_high=ci_high,
    )


def build_m_values(n: int) -> list[tuple[str, int]]:
    total_possible = n * (n - 1) // 2
    m_half = n // 2
    return [
        ("m = floor(n/2) - 10", max(0, min(m_half - 10, total_possible))),
        ("m = floor(n/2)", max(0, min(m_half, total_possible))),
        ("m = floor(n/2) + 10", max(0, min(m_half + 10, total_possible))),
    ]


def print_results_table(results: List[SimulationResult]) -> None:
    header = (
        f"{'n':>9}  {'m label':>22}  {'m':>9}  {'trials':>8}  {'successes':>10}  "
        f"{'p_hat':>10}  {'SE':>10}  {'95% CI':>23}"
    )
    print(header)
    print("-" * len(header))

    for r in results:
        ci_text = f"[{r.ci_low:.4f}, {r.ci_high:.4f}]"
        print(
            f"{r.n:>9}  {r.m_label:>22}  {r.m:>9}  {r.trials:>8}  {r.successes:>10}  "
            f"{r.probability:>10.4f}  {r.standard_error:>10.4f}  {ci_text:>23}"
        )


def save_results_csv(results: Iterable[SimulationResult], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "n",
                "m_label",
                "m",
                "trials",
                "successes",
                "estimated_probability",
                "standard_error",
                "ci_95_low",
                "ci_95_high",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r.n,
                    r.m_label,
                    r.m,
                    r.trials,
                    r.successes,
                    f"{r.probability:.6f}",
                    f"{r.standard_error:.6f}",
                    f"{r.ci_low:.6f}",
                    f"{r.ci_high:.6f}",
                ]
            )


def annotate_probabilities(ax, xs: list[int], ys: list[float]) -> None:
    for i, (x, y) in enumerate(zip(xs, ys)):
        offset = 10 if i % 2 == 0 else -14
        va = "bottom" if i % 2 == 0 else "top"

        ax.annotate(
            f"{y:.3f}",
            xy=(x, y),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va=va,
            fontsize=8.6,
            color="0.20",
            bbox=dict(
                facecolor="white",
                edgecolor="0.80",
                boxstyle="round,pad=0.16",
            ),
        )


def make_plot(results: List[SimulationResult], output_plot: Path) -> None:
    output_plot.parent.mkdir(parents=True, exist_ok=True)

    asymptotic_limit = math.sqrt(2.0 / 3.0) * math.cosh(math.sqrt(5.0 / 18.0))

    ns = sorted(set(r.n for r in results))
    x_positions = list(range(len(ns)))
    x_lookup = {n: i for i, n in enumerate(ns)}

    fig, ax = plt.subplots(figsize=(12.4, 7.4))

    series_order = [
        "m = floor(n/2) - 10",
        "m = floor(n/2)",
        "m = floor(n/2) + 10",
    ]

    for label in series_order:
        subset = sorted((r for r in results if r.m_label == label), key=lambda r: r.n)
        xs = [x_lookup[r.n] for r in subset]
        ys = [r.probability for r in subset]
        yerr = [1.96 * r.standard_error for r in subset]

        ax.errorbar(
            xs,
            ys,
            yerr=yerr,
            fmt="o-",
            capsize=4,
            linewidth=1.8,
            markersize=6,
            label=label,
        )

        # Annotate only the central series m = floor(n/2)
        if label == "m = floor(n/2)":
            annotate_probabilities(ax, xs, ys)

    ax.axhline(
        asymptotic_limit,
        linestyle="--",
        linewidth=1.5,
        label=fr"Asymptotic reference $\approx {asymptotic_limit:.4f}$",
    )

    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"{n:,}" for n in ns], fontsize=10)
    ax.set_xlim(-0.35, len(ns) - 0.65)
    ax.set_ylim(0.0, 1.05)

    ax.set_xlabel("Tested values of n (equally spaced plotting positions)", fontsize=12)
    ax.set_ylabel("Estimated probability", fontsize=12)
    ax.set_title(
        r"Probability that Graph $G(n,m)$ Has Only Tree, Unicyclic, and Bicyclic Components",
        fontsize=13,
    )
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_plot, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate the probability that Graph G(n,m) consists entirely of tree, "
            "unicyclic, and bicyclic connected components."
        )
    )
    parser.add_argument(
        "--ns",
        type=int,
        nargs="+",
        default=[50, 100, 200, 500, 1000, 5000, 10000, 20000, 50000],
        help="Values of n to test.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=1000,
        help="Number of Monte Carlo trials per (n, m).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Base random seed.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("./tree_unicyclic_bicyclic_probabilities.csv"),
        help="CSV file for numerical results.",
    )
    parser.add_argument(
        "--output-plot",
        type=Path,
        default=Path("./tree_unicyclic_bicyclic_probabilities.png"),
        help="PNG plot file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ns = sorted(set(n for n in args.ns if n >= 2))
    rng = np.random.default_rng(args.seed)

    results: List[SimulationResult] = []

    print("Running simulations...")
    for n in ns:
        for m_label, m in build_m_values(n):
            print(f"  n = {n:7d}, {m_label:>22}, m = {m:7d}, trials = {args.trials}")
            result = simulate_probability(
                n=n,
                m=m,
                m_label=m_label,
                trials=args.trials,
                rng=rng,
            )
            results.append(result)

    results.sort(key=lambda r: (r.n, r.m))

    print()
    print_results_table(results)
    print()

    save_results_csv(results, args.output_csv)
    make_plot(results, args.output_plot)

    asymptotic_limit = math.sqrt(2.0 / 3.0) * math.cosh(math.sqrt(5.0 / 18.0))
    print(f"Asymptotic reference value: {asymptotic_limit:.6f}")
    print(f"Saved CSV to:  {args.output_csv}")
    print(f"Saved PNG to:  {args.output_plot}")


if __name__ == "__main__":
    main()