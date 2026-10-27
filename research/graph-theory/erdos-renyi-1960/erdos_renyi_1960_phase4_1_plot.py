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
# Phase 4.1 plot script.
# Default output matches the HTML placeholder:
#     erdos_renyi_1960_phase4_1.png
#
# Tailored idea:
#   Enter the connectivity window
#       m = (n/2) log n + y n
#   and track how quickly the outside world disappears.  The top panel
#   studies the mean number of vertices outside the dominant component.
#   The bottom panel studies the probability of full connectivity, with
#   the classical approximation exp(-exp(-2y)).
# ============================================================

DEFAULT_NS = [100, 500, 1000, 5000, 10000, 20000, 50000]
DEFAULT_Y_VALUES = [-1.0, 0.0, 1.0]
DEFAULT_TRIALS = 1000
DEFAULT_SEED = 24680
DEFAULT_OUTPUT_CSV = Path("./erdos_renyi_1960_phase4_1.csv")
DEFAULT_OUTPUT_PLOT = Path("./erdos_renyi_1960_phase4_1.png")


@dataclass
class SimulationResult:
    n: int
    y: float
    m: int
    trials: int
    mean_outside_vertices: float
    sd_outside_vertices: float
    se_outside_vertices: float
    ci95_low_outside_vertices: float
    ci95_high_outside_vertices: float
    mean_isolates: float
    probability_connected: float
    theory_mean_isolates: float
    theory_connected_probability: float


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

    def connectivity_window_statistics(self) -> tuple[int, int, bool]:
        largest_size = 0
        isolates = 0
        components = 0
        for i, p in enumerate(self.parent):
            if i != p:
                continue
            components += 1
            v = self.size[i]
            e = self.edge_count[i]
            largest_size = max(largest_size, v)
            if v == 1 and e == 0:
                isolates += 1
        outside_vertices = len(self.parent) - largest_size
        return outside_vertices, isolates, (components == 1)


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


def simulate_one_trial(n: int, m: int, rng: np.random.Generator) -> tuple[int, int, bool]:
    uf = UnionFindCounter(n)
    edges = sample_unique_edges_sparse(n=n, m=m, rng=rng)
    for u, v in edges:
        uf.add_edge(u, v)
    return uf.connectivity_window_statistics()


def connectivity_window_edges(n: int, y: float) -> int:
    log_n = math.log(max(n, 2))
    return max(1, int(round(0.5 * n * log_n + y * n)))


def theory_mean_isolates(y: float) -> float:
    return math.exp(-2.0 * y)


def theory_connected_probability(y: float) -> float:
    return math.exp(-math.exp(-2.0 * y))


def simulate_statistics(n: int, y: float, trials: int, rng: np.random.Generator) -> SimulationResult:
    m = connectivity_window_edges(n=n, y=y)

    outside_values: List[int] = []
    isolate_values: List[int] = []
    connected_count = 0

    for _ in range(trials):
        outside_vertices, isolates, connected = simulate_one_trial(n=n, m=m, rng=rng)
        outside_values.append(outside_vertices)
        isolate_values.append(isolates)
        if connected:
            connected_count += 1

    mean_outside = float(np.mean(outside_values)) if outside_values else 0.0
    sd_outside = float(np.std(outside_values, ddof=1)) if len(outside_values) >= 2 else 0.0
    se_outside = sd_outside / math.sqrt(len(outside_values)) if outside_values else 0.0

    return SimulationResult(
        n=n,
        y=y,
        m=m,
        trials=trials,
        mean_outside_vertices=mean_outside,
        sd_outside_vertices=sd_outside,
        se_outside_vertices=se_outside,
        ci95_low_outside_vertices=max(0.0, mean_outside - 1.96 * se_outside),
        ci95_high_outside_vertices=mean_outside + 1.96 * se_outside,
        mean_isolates=float(np.mean(isolate_values)) if isolate_values else 0.0,
        probability_connected=(connected_count / trials) if trials > 0 else 0.0,
        theory_mean_isolates=theory_mean_isolates(y),
        theory_connected_probability=theory_connected_probability(y),
    )


def save_results_csv(results: Iterable[SimulationResult], output_csv: Path) -> None:
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "n",
            "y",
            "m",
            "trials",
            "mean_outside_vertices",
            "sd_outside_vertices",
            "se_outside_vertices",
            "ci95_low_outside_vertices",
            "ci95_high_outside_vertices",
            "mean_isolates",
            "probability_connected",
            "theory_mean_isolates",
            "theory_connected_probability",
        ])
        for r in results:
            writer.writerow([
                r.n,
                f"{r.y:.6f}",
                r.m,
                r.trials,
                f"{r.mean_outside_vertices:.6f}",
                f"{r.sd_outside_vertices:.6f}",
                f"{r.se_outside_vertices:.6f}",
                f"{r.ci95_low_outside_vertices:.6f}",
                f"{r.ci95_high_outside_vertices:.6f}",
                f"{r.mean_isolates:.6f}",
                f"{r.probability_connected:.6f}",
                f"{r.theory_mean_isolates:.6f}",
                f"{r.theory_connected_probability:.6f}",
            ])


def print_results_table(results: List[SimulationResult]) -> None:
    header = (
        f"{'n':>9}  {'y':>6}  {'m':>10}  {'trials':>8}  "
        f"{'mean outside':>13}  {'mean isolates':>13}  {'P(conn)':>9}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.n:>9}  {r.y:>6.1f}  {r.m:>10}  {r.trials:>8}  "
            f"{r.mean_outside_vertices:>13.3f}  {r.mean_isolates:>13.3f}  {r.probability_connected:>9.4f}"
        )


def make_plot(results: List[SimulationResult], output_plot: Path, y_values: List[float]) -> None:
    ns = sorted({r.n for r in results})
    x_positions = list(range(len(ns)))

    fig, axes = plt.subplots(2, 1, figsize=(13.2, 10.4), sharex=True)
    fig.subplots_adjust(top=0.86, bottom=0.12, hspace=0.24)

    color_map = {-1.0: "#1d4f91", 0.0: "#b26a00", 1.0: "#8b1e3f"}

    fig.suptitle(
        r"Phase 4.1 — entering the connectivity window at $m=\frac{n}{2}\log n + y n$",
        fontsize=16,
        fontweight="bold",
        y=0.97,
    )
    fig.text(
        0.5,
        0.925,
        "Top panel tracks the mean number of vertices outside the dominant component. Bottom panel compares the empirical connectivity probability with the classical approximation exp(-exp(-2y)).",
        ha="center",
        va="center",
        fontsize=10.5,
        color="0.25",
    )

    ax_top, ax_bottom = axes

    for y in y_values:
        subset = [r for r in results if abs(r.y - y) < 1e-12]
        subset.sort(key=lambda r: r.n)
        color = color_map.get(round(y, 1), None)

        mean_outside = [r.mean_outside_vertices for r in subset]
        low = [r.ci95_low_outside_vertices for r in subset]
        high = [r.ci95_high_outside_vertices for r in subset]
        empirical_conn = [r.probability_connected for r in subset]
        theory_conn = [r.theory_connected_probability for r in subset]

        ax_top.plot(
            x_positions,
            mean_outside,
            marker="o",
            linewidth=2.2,
            label=f"empirical mean outside, y = {y:.1f}",
            color=color,
        )
        ax_top.fill_between(x_positions, low, high, alpha=0.14, color=color)

        ax_bottom.plot(
            x_positions,
            empirical_conn,
            marker="o",
            linewidth=2.2,
            label=f"empirical P(conn), y = {y:.1f}",
            color=color,
        )
        ax_bottom.plot(
            x_positions,
            theory_conn,
            linestyle="--",
            linewidth=1.9,
            label=f"theory, y = {y:.1f}",
            color=color,
        )

    ax_top.set_ylabel("Mean outside vertices")
    ax_top.set_title("How quickly the outside world collapses in the connectivity window", fontsize=12.7, pad=10)
    ax_top.grid(True, alpha=0.28)
    ax_top.legend(ncol=2, fontsize=9.0, frameon=True)

    ax_bottom.set_ylabel("Connectivity probability")
    ax_bottom.set_ylim(-0.02, 1.02)
    ax_bottom.set_title("Empirical connectivity probability versus the classical window approximation", fontsize=12.7, pad=10)
    ax_bottom.grid(True, alpha=0.28)
    ax_bottom.legend(ncol=2, fontsize=9.0, frameon=True)

    ax_bottom.set_xticks(x_positions)
    ax_bottom.set_xticklabels([f"{n:,}" for n in ns], rotation=0)
    ax_bottom.set_xlabel("Number of vertices $n$")

    fig.text(
        0.5,
        0.04,
        "Default experiment: 1,000 simulations for each n and each y in {-1, 0, 1}.",
        ha="center",
        va="center",
        fontsize=9.8,
        color="0.30",
    )

    output_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_plot, dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Monte Carlo simulations for Phase 4.1 and generate a CSV plus PNG for the connectivity window.")
    parser.add_argument("--ns", type=int, nargs="+", default=DEFAULT_NS, help="Values of n to simulate.")
    parser.add_argument("--y-values", type=float, nargs="+", default=DEFAULT_Y_VALUES, help="Window y values to compare.")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS, help="Number of Monte Carlo trials per (n, y).")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV, help="Output CSV path.")
    parser.add_argument("--output-plot", type=Path, default=DEFAULT_OUTPUT_PLOT, help="Output plot path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ns = sorted({int(n) for n in args.ns if int(n) >= 2})
    y_values = sorted({float(y) for y in args.y_values})
    trials = max(1, int(args.trials))
    rng = np.random.default_rng(args.seed)

    results: List[SimulationResult] = []
    for y in y_values:
        for n in ns:
            results.append(simulate_statistics(n=n, y=y, trials=trials, rng=rng))

    print_results_table(results)
    save_results_csv(results, args.output_csv)
    make_plot(results, args.output_plot, y_values=y_values)
    print(f"saved csv to: {args.output_csv}")
    print(f"saved plot to: {args.output_plot}")


if __name__ == "__main__":
    main()
