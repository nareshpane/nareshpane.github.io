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
# Phase 2.3 plot script.
# Default output matches the HTML placeholder:
#     erdos_renyi_1960_phase2_3.png
#
# Idea:
#   Static comparison of the empirical count distribution of k-vertex
#   unicyclic components (size k, edges k) at the linear scale m ~ c n,
#   against the Poisson prediction from Erdős–Rényi (1960).
# ============================================================
DEFAULT_NS = [100, 1000, 10000]
DEFAULT_K = 4
DEFAULT_C = 0.40
DEFAULT_TRIALS = 1000
DEFAULT_SEED = 12345
DEFAULT_OUTPUT_CSV = Path("./erdos_renyi_1960_phase2_3.csv")
DEFAULT_OUTPUT_PLOT = Path("./erdos_renyi_1960_phase2_3.png")

BAR_COLOR = "#7fb3ff"
BAR_EDGE = "#1565c0"
POISSON_COLOR = "#c62828"


@dataclass
class DistributionRow:
    n: int
    k: int
    c: float
    m: int
    count_value: int
    empirical_probability: float
    predicted_probability: float


@dataclass
class RunSummary:
    n: int
    k: int
    c: float
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

    def count_unicyclic_components_of_order(self, k: int) -> int:
        total = 0
        for i, p in enumerate(self.parent):
            if i == p and self.size[i] == k and self.edge_count[i] == k:
                total += 1
        return total


def poisson_mean_unicyclic(c: float, k: int) -> float:
    series = sum((k ** j) / math.factorial(j) for j in range(0, k - 2))
    return ((2.0 * c * math.exp(-2.0 * c)) ** k) * series / math.factorial(k)


def poisson_pmf(x: int, lam: float) -> float:
    return math.exp(-lam) * (lam ** x) / math.factorial(x)


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


def trial_unicyclic_count(n: int, m: int, k: int, rng: np.random.Generator) -> int:
    uf = UnionFindCounter(n)
    edges = sample_unique_edges_sparse(n=n, m=m, rng=rng)
    for u, v in edges:
        uf.add_edge(u, v)
    return uf.count_unicyclic_components_of_order(k)


def run_trials_for_n(n: int, k: int, c: float, trials: int, seed: int) -> RunSummary:
    rng = np.random.default_rng(seed)
    m = max(1, int(round(c * n)))
    counts = [trial_unicyclic_count(n=n, m=m, k=k, rng=rng) for _ in range(trials)]
    return RunSummary(
        n=n,
        k=k,
        c=c,
        m=m,
        trials=trials,
        counts=counts,
        predicted_lambda=poisson_mean_unicyclic(c=c, k=k),
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
                c=summary.c,
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
                "c",
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
                    f"{row.c:.6f}",
                    row.m,
                    row.count_value,
                    f"{row.empirical_probability:.6f}",
                    f"{row.predicted_probability:.6f}",
                ]
            )


def make_plot(summaries: List[RunSummary], output_plot: Path) -> None:
    x_max = max(
        5,
        max(max(summary.counts) for summary in summaries),
        int(math.ceil(summaries[0].predicted_lambda + 4.0 * math.sqrt(max(summaries[0].predicted_lambda, 1e-12)))),
    )
    xs = np.arange(0, x_max + 1)
    poisson_values = [poisson_pmf(int(x), summaries[0].predicted_lambda) for x in xs]
    trials = summaries[0].trials

    fig, axes = plt.subplots(1, len(summaries), figsize=(17.6, 7.9), sharey=True)
    if len(summaries) == 1:
        axes = [axes]

    for ax, summary in zip(axes, summaries):
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
            markersize=5,
            color=POISSON_COLOR,
        )
        ax.set_title(rf"$n={summary.n:,}$", fontsize=15, pad=12)
        ax.set_xlabel(rf"Count of unicyclic $({summary.k},{summary.k})$ components", fontsize=11)
        ax.set_xlim(-0.5, x_max + 0.5)
        ax.grid(True, axis="y", alpha=0.28)

        sample_mean = float(np.mean(summary.counts))
        sample_var = float(np.var(summary.counts))
        ax.text(
            0.5,
            0.90,
            rf"$m={summary.m:,}$  |  mean={sample_mean:.4f}  |  var={sample_var:.4f}",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=9.4,
            bbox=dict(facecolor="white", edgecolor="0.82", boxstyle="round,pad=0.20"),
        )

    axes[0].set_ylabel("Probability / relative frequency", fontsize=12)
    fig.suptitle(
        rf"Phase 2.3: unicyclic components of a fixed size also have a Poisson law ({trials:,} runs per $n$)",
        fontsize=16,
        y=0.98,
    )
    fig.text(
        0.5,
        0.93,
        rf"Here $k={summaries[0].k}$, $c={summaries[0].c:.2f}$, and $m = \,\lfloor c n \rceil$. For fixed $(c,k)$, the limiting Poisson mean is the same across graph sizes: $\lambda={summaries[0].predicted_lambda:.4f}$.",
        ha="center",
        va="center",
        fontsize=11,
        color="0.25",
    )

    common_handles = [
        Patch(facecolor=BAR_COLOR, edgecolor=BAR_EDGE, label="Empirical distribution"),
        Line2D([0], [0], color=POISSON_COLOR, linestyle=":", marker="o", linewidth=2.4, markersize=5, label=rf"Poisson($\lambda={summaries[0].predicted_lambda:.4f}$)"),
    ]
    fig.legend(
        handles=common_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.11),
        ncol=2,
        frameon=True,
        edgecolor="#cccccc",
        fontsize=10,
    )

    caption = (
        f"This figure uses {trials:,} independent runs for each n. The theoretical object here is very specific: connected components with exactly k vertices and k edges, so one is counting rare unicyclic pieces of a fixed size rather than all cycles or all unicyclic components together. "
        "Because the mean stays finite, the natural empirical target is again a small Poisson-like histogram, often with most mass at 0 and a much smaller amount at 1."
    )
    fig.text(
        0.5,
        0.018,
        fill(caption, width=154),
        ha="center",
        va="bottom",
        fontsize=9.7,
        color="0.25",
        wrap=True,
    )

    fig.tight_layout(rect=[0.03, 0.18, 0.98, 0.90])
    fig.savefig(output_plot, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create the Phase 2.3 plot comparing empirical counts of unicyclic components of order k with the Poisson prediction at the linear scale."
        )
    )
    parser.add_argument("--ns", type=int, nargs="+", default=DEFAULT_NS, help="Values of n to compare.")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="Order k of the unicyclic components to track.")
    parser.add_argument("--c", type=float, default=DEFAULT_C, help="Linear-scale constant in m = round(c n), should stay below 0.5.")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS, help="Number of Monte Carlo runs per n.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Base random seed.")
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV, help="CSV file for probability rows.")
    parser.add_argument("--output-plot", type=Path, default=DEFAULT_OUTPUT_PLOT, help="PNG plot file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ns = [n for n in args.ns if n >= 2]
    k = max(3, args.k)
    c = min(max(args.c, 0.01), 0.49)
    trials = max(10, args.trials)

    summaries: List[RunSummary] = []
    for i, n in enumerate(ns):
        print(f"Running n={n:,}, k={k}, c={c:.2f}, trials={trials:,} ...")
        summaries.append(run_trials_for_n(n=n, k=k, c=c, trials=trials, seed=args.seed + 1009 * i))

    x_max = max(
        5,
        max(max(summary.counts) for summary in summaries),
        int(math.ceil(summaries[0].predicted_lambda + 4.0 * math.sqrt(max(summaries[0].predicted_lambda, 1e-12)))),
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
