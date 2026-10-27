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
from matplotlib.patches import Patch


# ============================================================
# Phase 1.4 plot script.
# Default output matches the HTML placeholder:
#     erdos_renyi_1960_phase1_4.png
#
# Idea:
#   Static comparison of the empirical count distribution of T_k above
#   the threshold scale, against the normal approximation with mean M_n
#   and variance about M_n.
#
# Default example:
#   k = 3, beta = 0.6, and n = 100, 1000, 10,000.
# ============================================================
DEFAULT_NS = [100, 1000, 10000]
DEFAULT_K = 3
DEFAULT_BETA = 0.6
DEFAULT_TRIALS = 1000
DEFAULT_SEED = 12345
DEFAULT_OUTPUT_CSV = Path("./erdos_renyi_1960_phase1_4.csv")
DEFAULT_OUTPUT_PLOT = Path("./erdos_renyi_1960_phase1_4.png")

BAR_COLOR = "#7fb3ff"
BAR_EDGE = "#1565c0"
NORMAL_COLOR = "#c62828"


@dataclass
class DistributionRow:
    n: int
    k: int
    beta: float
    m: int
    count_value: int
    empirical_probability: float
    predicted_probability: float
    predicted_mean: float
    predicted_sd: float


@dataclass
class RunSummary:
    n: int
    k: int
    beta: float
    m: int
    trials: int
    counts: List[int]
    predicted_mean: float
    predicted_sd: float


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


def threshold_alpha_for_k(k: int) -> float:
    return (k - 2) / (k - 1)


def edge_count_above_threshold(n: int, beta: float) -> int:
    return max(1, int(round(n ** beta)))


def predicted_mean_Mn(n: int, k: int, m: int) -> float:
    return (
        n
        * (k ** (k - 2))
        / math.factorial(k)
        * ((2.0 * m / n) ** (k - 1))
        * math.exp(-(2.0 * k * m) / n)
    )


def normal_cdf(x: float, mu: float, sigma: float) -> float:
    if sigma <= 1e-12:
        return 1.0 if x >= mu else 0.0
    z = (x - mu) / (sigma * math.sqrt(2.0))
    return 0.5 * (1.0 + math.erf(z))


def normal_bin_probability(x: int, mu: float, sigma: float) -> float:
    if sigma <= 1e-12:
        return 1.0 if x == int(round(mu)) else 0.0
    return max(0.0, normal_cdf(x + 0.5, mu, sigma) - normal_cdf(x - 0.5, mu, sigma))


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


def run_trials_for_n(n: int, k: int, beta: float, trials: int, seed: int) -> RunSummary:
    rng = np.random.default_rng(seed)
    m = edge_count_above_threshold(n=n, beta=beta)
    counts = [trial_tree_count(n=n, m=m, k=k, rng=rng) for _ in range(trials)]
    Mn = predicted_mean_Mn(n=n, k=k, m=m)
    return RunSummary(
        n=n,
        k=k,
        beta=beta,
        m=m,
        trials=trials,
        counts=counts,
        predicted_mean=Mn,
        predicted_sd=math.sqrt(max(Mn, 1e-12)),
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
                beta=summary.beta,
                m=summary.m,
                count_value=x,
                empirical_probability=float(empirical[x]),
                predicted_probability=normal_bin_probability(x, summary.predicted_mean, summary.predicted_sd),
                predicted_mean=summary.predicted_mean,
                predicted_sd=summary.predicted_sd,
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
                "beta",
                "m",
                "count_value",
                "empirical_probability",
                "predicted_normal_probability",
                "predicted_mean_Mn",
                "predicted_sd",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.n,
                    row.k,
                    f"{row.beta:.6f}",
                    row.m,
                    row.count_value,
                    f"{row.empirical_probability:.6f}",
                    f"{row.predicted_probability:.6f}",
                    f"{row.predicted_mean:.6f}",
                    f"{row.predicted_sd:.6f}",
                ]
            )


def make_plot(summaries: List[RunSummary], output_plot: Path) -> None:
    x_max = max(
        8,
        max(max(summary.counts) for summary in summaries),
        max(int(math.ceil(summary.predicted_mean + 4.5 * summary.predicted_sd)) for summary in summaries),
    )
    xs = np.arange(0, x_max + 1)
    trials = summaries[0].trials
    k = summaries[0].k
    beta = summaries[0].beta

    fig, axes = plt.subplots(1, len(summaries), figsize=(17.8, 7.9), sharey=True)
    if len(summaries) == 1:
        axes = [axes]

    for ax, summary in zip(axes, summaries):
        binc = np.bincount(summary.counts, minlength=x_max + 1)[: x_max + 1]
        empirical = binc / summary.trials
        normal_values = [normal_bin_probability(int(x), summary.predicted_mean, summary.predicted_sd) for x in xs]

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
            normal_values,
            linestyle="--",
            linewidth=2.4,
            marker="o",
            markersize=4.8,
            color=NORMAL_COLOR,
        )
        ax.set_title(rf"$n={summary.n:,}$", fontsize=15, pad=12)
        ax.set_xlabel(rf"Count of $T_{{{summary.k}}}$ components", fontsize=11)
        ax.set_xlim(-0.5, x_max + 0.5)
        ax.grid(True, axis="y", alpha=0.28)

        sample_mean = float(np.mean(summary.counts))
        sample_var = float(np.var(summary.counts))
        ax.text(
            0.5,
            0.90,
            rf"$m={summary.m:,}$  |  emp. mean={sample_mean:.3f}  |  emp. var={sample_var:.3f}",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=9.2,
            bbox=dict(facecolor="white", edgecolor="0.82", boxstyle="round,pad=0.20"),
        )
        ax.text(
            0.5,
            0.82,
            rf"predicted $M_n={summary.predicted_mean:.3f}$  |  predicted sd={summary.predicted_sd:.3f}",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=9.0,
            bbox=dict(facecolor="#fffdf9", edgecolor="#d8d0c0", boxstyle="round,pad=0.18"),
        )

    axes[0].set_ylabel("Probability / relative frequency", fontsize=12)
    fig.suptitle(
        rf"Phase 1.4: above threshold, many $k$-vertex trees and an approximately normal count ({trials:,} runs per $n$)",
        fontsize=16,
        y=0.985,
    )
    fig.text(
        0.5,
        0.946,
        rf"Here $k={k}$ and $m = \lfloor n^{{{beta:.1f}}} \rfloor$, which lies above the threshold exponent $(k-2)/(k-1)={threshold_alpha_for_k(k):.3f}$ while still remaining in Phase 1. The theory predicts mean $M_n$ and variance about $M_n$.",
        ha="center",
        va="center",
        fontsize=11,
        color="0.25",
    )

    common_handles = [
        Patch(facecolor=BAR_COLOR, edgecolor=BAR_EDGE, label="Empirical distribution"),
        Line2D([0], [0], color=NORMAL_COLOR, linestyle="--", marker="o", linewidth=2.4, markersize=5, label="Normal approximation"),
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
        f"This figure uses {trials:,} independent runs for each n. The Phase 1.4 point is that once the edge count is well above the k-tree threshold, the count is no longer a tiny Poisson count. "
        "Instead it should fluctuate around a visible center, with a spread comparable to the square root of that center. For n=100 the approximation may still look rough, but for n=1000 and n=10,000 a bell-shaped empirical distribution becomes much more plausible."
    )
    fig.text(
        0.5,
        0.018,
        fill(caption, width=150),
        ha="center",
        va="bottom",
        fontsize=9.8,
        color="0.25",
        wrap=True,
    )

    fig.tight_layout(rect=[0.03, 0.16, 0.97, 0.90])
    fig.savefig(output_plot, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create the Phase 1.4 plot comparing empirical k-tree count distributions with the normal approximation above threshold."
        )
    )
    parser.add_argument("--ns", type=int, nargs="+", default=DEFAULT_NS, help="Values of n to compare.")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="Tree order to track.")
    parser.add_argument("--beta", type=float, default=DEFAULT_BETA, help="Exponent in m = round(n^beta), chosen above threshold but below 1.")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS, help="Number of Monte Carlo runs per n.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Base random seed.")
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV, help="CSV file for probability rows.")
    parser.add_argument("--output-plot", type=Path, default=DEFAULT_OUTPUT_PLOT, help="PNG plot file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ns = [n for n in args.ns if n >= 2]
    k = max(3, args.k)
    beta = min(max(args.beta, 0.05), 0.95)
    trials = max(10, args.trials)

    summaries: List[RunSummary] = []
    for i, n in enumerate(ns):
        print(f"Running n={n:,}, k={k}, trials={trials:,} ...")
        summaries.append(run_trials_for_n(n=n, k=k, beta=beta, trials=trials, seed=args.seed + 1009 * i))

    x_max = max(
        8,
        max(max(summary.counts) for summary in summaries),
        max(int(math.ceil(summary.predicted_mean + 4.5 * summary.predicted_sd)) for summary in summaries),
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
