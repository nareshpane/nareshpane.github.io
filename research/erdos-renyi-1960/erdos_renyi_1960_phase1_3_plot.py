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
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


# ============================================================
# Phase 1.3 plot script.
# Default output matches the HTML placeholder:
#     erdos_renyi_1960_phase1_3.png
#
# Idea:
#   Static comparison of the empirical count distribution of T3 at the
#   threshold scale m ~ n^(1/2), against the Poisson prediction with
#   lambda = 2, for six graph sizes.
# ============================================================
DEFAULT_NS = [100, 1000, 5000, 10000, 20000, 50000]
DEFAULT_K = 3
DEFAULT_RHO = 1.0
DEFAULT_TRIALS = 1000
DEFAULT_SEED = 12345
DEFAULT_OUTPUT_CSV = Path("./erdos_renyi_1960_phase1_3.csv")
DEFAULT_OUTPUT_PLOT = Path("./erdos_renyi_1960_phase1_3.png")

BAR_COLOR = "#7fb3ff"
BAR_EDGE = "#1565c0"
POISSON_COLOR = "#c62828"


@dataclass
class DistributionRow:
    n: int
    k: int
    rho: float
    m: int
    count_value: int
    empirical_probability: float
    predicted_probability: float


@dataclass
class RunSummary:
    n: int
    k: int
    rho: float
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


def run_trials_for_n(n: int, k: int, rho: float, trials: int, seed: int) -> RunSummary:
    rng = np.random.default_rng(seed)
    m = threshold_edge_count(n=n, k=k, rho=rho)
    counts = [trial_tree_count(n=n, m=m, k=k, rng=rng) for _ in range(trials)]
    return RunSummary(
        n=n,
        k=k,
        rho=rho,
        m=m,
        trials=trials,
        counts=counts,
        predicted_lambda=theoretical_lambda(k=k, rho=rho),
    )


def build_distribution_rows(summary: RunSummary, x_max: int) -> List[DistributionRow]:
    binc = np.bincount(summary.counts, minlength=x_max + 1)[: x_max + 1]
    empirical = binc / summary.trials
    rows: List[DistributionRow] = []
    for x in range(0, x_max + 1):
        rows.append(
            DistributionRow(
                n=summary.n,
                k=summary.k,
                rho=summary.rho,
                m=summary.m,
                count_value=x,
                empirical_probability=float(empirical[x]),
                predicted_probability=poisson_pmf(x, summary.predicted_lambda),
            )
        )
    return rows


def save_results_csv(rows: Iterable[DistributionRow], output_csv: Path) -> None:
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "n",
                "k",
                "rho",
                "m",
                "count_value",
                "empirical_probability",
                "predicted_poisson_probability",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.n,
                    row.k,
                    f"{row.rho:.6f}",
                    row.m,
                    row.count_value,
                    f"{row.empirical_probability:.6f}",
                    f"{row.predicted_probability:.6f}",
                ]
            )


def make_plot(summaries: List[RunSummary], output_plot: Path) -> None:
    x_max = max(
        6,
        max(max(summary.counts) for summary in summaries),
        int(math.ceil(summaries[0].predicted_lambda + 4.0 * math.sqrt(summaries[0].predicted_lambda))),
    )
    xs = np.arange(0, x_max + 1)
    poisson_values = [poisson_pmf(int(x), summaries[0].predicted_lambda) for x in xs]
    trials = summaries[0].trials

    fig, axes = plt.subplots(2, 3, figsize=(18.5, 10.8), sharey=True)
    axes_flat = list(axes.flatten())

    for ax, summary in zip(axes_flat, summaries):
        binc = np.bincount(summary.counts, minlength=x_max + 1)[: x_max + 1]
        empirical = binc / summary.trials

        ax.bar(
            xs,
            empirical,
            width=0.78,
            color=BAR_COLOR,
            edgecolor=BAR_EDGE,
            linewidth=1.0,
        )
        ax.plot(
            xs,
            poisson_values,
            linestyle=":",
            linewidth=2.4,
            marker="o",
            markersize=4.8,
            color=POISSON_COLOR,
        )
        ax.set_title(rf"$n={summary.n:,}$", fontsize=14, pad=10)
        ax.set_xlabel(rf"Count of $T_{{{summary.k}}}$ components", fontsize=10.8, labelpad=8)
        ax.set_xlim(-0.5, x_max + 0.5)
        ax.grid(True, axis="y", alpha=0.28)

        sample_mean = float(np.mean(summary.counts))
        sample_var = float(np.var(summary.counts))
        ax.text(
            0.5,
            0.90,
            rf"$m={summary.m:,}$  |  mean={sample_mean:.3f}  |  var={sample_var:.3f}",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=9.2,
            bbox=dict(facecolor="white", edgecolor="0.82", boxstyle="round,pad=0.20"),
        )

    for ax in axes_flat[len(summaries):]:
        ax.axis("off")

    axes_flat[0].set_ylabel("Probability / relative frequency", fontsize=12)
    axes_flat[3].set_ylabel("Probability / relative frequency", fontsize=12)

    fig.suptitle(
        rf"Phase 1.3: at threshold, the number of $k$-vertex trees should look Poisson ({trials:,} runs per $n$)",
        fontsize=16,
        y=0.985,
    )
    fig.text(
        0.5,
        0.948,
        rf"Here $k={summaries[0].k}$, $\rho={summaries[0].rho:.1f}$, and $m = \lfloor \rho n^{{(k-2)/(k-1)}} \rfloor$. For $k=3$ and $\rho=1$, the limiting mean is $\lambda=2$ for all six graph sizes.",
        ha="center",
        va="center",
        fontsize=11,
        color="0.25",
    )

    common_handles = [
        Patch(facecolor=BAR_COLOR, edgecolor=BAR_EDGE, label="Empirical distribution"),
        Line2D([0], [0], color=POISSON_COLOR, linestyle=":", marker="o", linewidth=2.4, markersize=5, label=rf"Poisson($\lambda={summaries[0].predicted_lambda:.3f}$)"),
    ]
    fig.legend(
        handles=common_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.105),
        ncol=2,
        frameon=True,
        edgecolor="#cccccc",
        fontsize=10,
    )

    caption = (
        f"This figure uses {trials:,} independent runs for each n. The point of Phase 1.3 is not one sample graph, but the whole distribution of the count over many runs. "
        "At threshold the count should stay finite and Poisson-like: many runs with 0, many with 1, fewer with 2, and progressively fewer beyond that. "
        "As n grows, finite-size noise should shrink and the bars should align more cleanly with the same Poisson law."
    )
    fig.text(
        0.5,
        0.018,
        fill(caption, width=155),
        ha="center",
        va="bottom",
        fontsize=9.8,
        color="0.25",
        wrap=True,
    )

    fig.tight_layout(rect=[0.03, 0.17, 0.97, 0.90])
    fig.savefig(output_plot, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create the Phase 1.3 plot comparing empirical k-tree count distributions with the Poisson prediction at the threshold scale."
        )
    )
    parser.add_argument("--ns", type=int, nargs="+", default=DEFAULT_NS, help="Values of n to compare.")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="Tree order to track.")
    parser.add_argument("--rho", type=float, default=DEFAULT_RHO, help="Constant in m = round(rho * n^((k-2)/(k-1))).")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS, help="Number of Monte Carlo runs per n.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Base random seed.")
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV, help="CSV file for probability rows.")
    parser.add_argument("--output-plot", type=Path, default=DEFAULT_OUTPUT_PLOT, help="PNG plot file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ns = [n for n in args.ns if n >= 2]
    k = max(3, args.k)
    rho = max(0.05, args.rho)
    trials = max(10, args.trials)

    summaries: List[RunSummary] = []
    for i, n in enumerate(ns):
        print(f"Running n={n:,}, k={k}, trials={trials:,} ...")
        summaries.append(run_trials_for_n(n=n, k=k, rho=rho, trials=trials, seed=args.seed + 1009 * i))

    x_max = max(
        6,
        max(max(summary.counts) for summary in summaries),
        int(math.ceil(summaries[0].predicted_lambda + 4.0 * math.sqrt(summaries[0].predicted_lambda))),
    )
    rows: List[DistributionRow] = []
    for summary in summaries:
        rows.extend(build_distribution_rows(summary=summary, x_max=x_max))

    save_results_csv(rows, args.output_csv)
    make_plot(summaries, args.output_plot)

    print(f"Saved CSV to:  {args.output_csv}")
    print(f"Saved PNG to:  {args.output_plot}")


if __name__ == "__main__":
    main()
