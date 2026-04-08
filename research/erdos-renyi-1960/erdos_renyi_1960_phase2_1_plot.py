#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from textwrap import fill
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


# ============================================================
# Phase 2.1 plot script.
# Default output matches the HTML placeholder:
#     erdos_renyi_1960_phase2_1.png
#
# Idea:
#   At the linear scale m ~ c n with 0 < c < 1/2, short cycles no longer
#   vanish. For fixed cycle length k, the count tends to Poisson with mean
#       lambda_k = (2 c)^k / (2 k).
#   This figure compares the empirical probability of seeing at least one
#   short cycle against the Poisson-limit prediction 1 - exp(-lambda_k),
#   across two side-by-side panels for c = 0.30 and c = 0.45.
# ============================================================
DEFAULT_NS = [100, 500, 1000, 10000, 20000, 50000]
DEFAULT_CS = [0.30, 0.45]
DEFAULT_KS = [3, 4, 5]
DEFAULT_TRIALS = 1000
DEFAULT_SEED = 12345
DEFAULT_OUTPUT_CSV = Path("./erdos_renyi_1960_phase2_1.csv")
DEFAULT_OUTPUT_PLOT = Path("./erdos_renyi_1960_phase2_1.png")


@dataclass
class SimulationResult:
    cycle_len: int
    n: int
    c: float
    m: int
    trials: int
    successes: int
    empirical_probability: float
    predicted_lambda: float
    predicted_probability: float
    standard_error: float
    ci_low: float
    ci_high: float


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


def build_adjacency(n: int, edges: Sequence[Tuple[int, int]]) -> Dict[int, Set[int]]:
    adj: Dict[int, Set[int]] = {i: set() for i in range(n)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def exists_cycle_of_length(adj: Dict[int, Set[int]], cycle_len: int) -> bool:
    nodes = sorted(adj)
    for start in nodes:
        visited = {start}

        def dfs(cur: int, depth: int) -> bool:
            if depth == cycle_len:
                return start in adj[cur]
            for nxt in adj[cur]:
                if nxt <= start or nxt in visited:
                    continue
                visited.add(nxt)
                if dfs(nxt, depth + 1):
                    return True
                visited.remove(nxt)
            return False

        if dfs(start, 1):
            return True
    return False


def poisson_mean_for_cycles(c: float, cycle_len: int) -> float:
    return ((2.0 * c) ** cycle_len) / (2.0 * cycle_len)


def simulate_probability(n: int, cycle_len: int, c: float, trials: int, rng: np.random.Generator) -> SimulationResult:
    m = max(1, int(round(c * n)))
    successes = 0
    for _ in range(trials):
        edges = sample_unique_edges_sparse(n=n, m=m, rng=rng)
        adj = build_adjacency(n, edges)
        if exists_cycle_of_length(adj, cycle_len):
            successes += 1

    p_hat = successes / trials
    se = math.sqrt(p_hat * (1.0 - p_hat) / trials) if trials > 0 else 0.0
    ci_low = max(0.0, p_hat - 1.96 * se)
    ci_high = min(1.0, p_hat + 1.96 * se)
    lam = poisson_mean_for_cycles(c, cycle_len)
    predicted = 1.0 - math.exp(-lam)

    return SimulationResult(
        cycle_len=cycle_len,
        n=n,
        c=c,
        m=m,
        trials=trials,
        successes=successes,
        empirical_probability=p_hat,
        predicted_lambda=lam,
        predicted_probability=predicted,
        standard_error=se,
        ci_low=ci_low,
        ci_high=ci_high,
    )


def print_results_table(results: List[SimulationResult]) -> None:
    header = (
        f"{'c':>6}  {'k':>4}  {'n':>9}  {'m':>8}  {'trials':>8}  {'runs with ≥1':>12}  {'emp p':>10}  {'pred p':>10}  {'95% CI':>23}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        ci_text = f"[{r.ci_low:.4f}, {r.ci_high:.4f}]"
        print(
            f"{r.c:>6.2f}  {r.cycle_len:>4}  {r.n:>9}  {r.m:>8}  {r.trials:>8}  {r.successes:>12}  {r.empirical_probability:>10.4f}  {r.predicted_probability:>10.4f}  {ci_text:>23}"
        )


def save_results_csv(results: Iterable[SimulationResult], output_csv: Path) -> None:
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "cycle_length",
                "n",
                "c",
                "m",
                "trials",
                "runs_with_at_least_one_cycle",
                "empirical_probability",
                "predicted_lambda",
                "predicted_probability",
                "standard_error",
                "ci_95_low",
                "ci_95_high",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r.cycle_len,
                    r.n,
                    f"{r.c:.6f}",
                    r.m,
                    r.trials,
                    r.successes,
                    f"{r.empirical_probability:.6f}",
                    f"{r.predicted_lambda:.6f}",
                    f"{r.predicted_probability:.6f}",
                    f"{r.standard_error:.6f}",
                    f"{r.ci_low:.6f}",
                    f"{r.ci_high:.6f}",
                ]
            )


def annotate_points(ax, xs: List[float], ys: List[float], ms: List[int], color: str, vertical_offset: int) -> None:
    for x, y, m in zip(xs, ys, ms):
        ax.annotate(
            f"m={m}",
            xy=(x, y),
            xytext=(0, vertical_offset),
            textcoords="offset points",
            ha="center",
            va="bottom" if vertical_offset > 0 else "top",
            fontsize=7.6,
            color="0.20",
            bbox=dict(facecolor="white", edgecolor=color, boxstyle="round,pad=0.12", alpha=0.92),
        )


def compact_x_positions(ns: List[int]) -> Dict[int, float]:
    # Compress the visual spacing so the two-panel figure is not unnecessarily wide.
    # Larger n values are still ordered, but the gaps are slightly tighter than equal spacing.
    positions = np.array([0.00, 0.72, 1.38, 2.18, 2.82, 3.46], dtype=float)
    if len(ns) != len(positions):
        positions = np.linspace(0.0, 3.46, len(ns))
    return {n: float(x) for n, x in zip(ns, positions)}


def make_plot(results: List[SimulationResult], output_plot: Path) -> None:
    cycle_lengths = sorted(set(r.cycle_len for r in results))
    cs = sorted(set(r.c for r in results))
    ns = sorted(set(r.n for r in results))
    x_lookup = compact_x_positions(ns)
    x_positions = [x_lookup[n] for n in ns]
    trials = results[0].trials if results else DEFAULT_TRIALS

    color_map = {3: "#d62728", 4: "#1f77b4", 5: "#2ca02c"}
    offset_map = {3: -16, 4: 12, 5: -28}

    fig, axes = plt.subplots(1, len(cs), figsize=(14.6, 8.0), sharey=True)
    if len(cs) == 1:
        axes = [axes]

    max_seen = max(
        max(r.empirical_probability + 1.96 * r.standard_error, r.predicted_probability)
        for r in results
    ) if results else 0.20
    y_top = max(0.20, min(0.30, max_seen * 1.35))

    for ax, c in zip(axes, cs):
        subset_c = [r for r in results if math.isclose(r.c, c, rel_tol=0.0, abs_tol=1e-12)]
        for k in cycle_lengths:
            subset = sorted((r for r in subset_c if r.cycle_len == k), key=lambda r: r.n)
            xs = [x_lookup[r.n] for r in subset]
            ys_actual = [r.empirical_probability for r in subset]
            ys_pred = [r.predicted_probability for r in subset]
            yerr = [1.96 * r.standard_error for r in subset]
            ms = [r.m for r in subset]
            color = color_map.get(k, None)

            ax.errorbar(
                xs,
                ys_actual,
                yerr=yerr,
                fmt="o-",
                capsize=4,
                linewidth=1.8,
                markersize=5.2,
                color=color,
            )
            ax.plot(
                xs,
                ys_pred,
                linestyle=":",
                linewidth=2.0,
                color=color,
            )
            annotate_points(ax, xs, ys_actual, ms, color, offset_map.get(k, -16))

        ax.set_xticks(x_positions)
        ax.set_xticklabels([f"{n:,}" for n in ns], fontsize=9.6)
        ax.set_xlim(min(x_positions) - 0.18, max(x_positions) + 0.18)
        ax.set_ylim(0.0, y_top)
        ax.set_xlabel("Values of n", fontsize=11.5, labelpad=10)
        ax.set_title(rf"$c={c:.2f}$", fontsize=13, pad=10)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Probability of seeing at least one short cycle", fontsize=12)

    fig.suptitle(
        rf"Phase 2.1: short cycles at the linear scale $m=\lfloor c n \rceil$ ({trials:,} runs per node-size)",
        fontsize=14,
        y=0.97,
    )
    fig.text(
        0.5,
        0.925,
        "Left panel uses c = 0.30 and right panel uses c = 0.45. Within each panel, solid curves are empirical and dotted curves are the Poisson-limit predictions.",
        ha="center",
        va="center",
        fontsize=10.5,
        color="0.25",
    )

    common_handles = []
    for k in cycle_lengths:
        color = color_map[k]
        common_handles.append(Line2D([0], [0], color=color, linestyle="-", marker="o", linewidth=1.8, markersize=5.2, label=rf"$C_{k}$ actual"))
        common_handles.append(Line2D([0], [0], color=color, linestyle=":", linewidth=2.0, label=rf"$C_{k}$ predicted"))
    fig.legend(
        handles=common_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.125),
        ncol=3,
        frameon=True,
        edgecolor="#cccccc",
        fontsize=9,
    )

    lambda_bits = []
    for c in cs:
        bits = ", ".join(rf"$\lambda_{k}={poisson_mean_for_cycles(c, k):.4f}$" for k in cycle_lengths)
        lambda_bits.append(f"for c = {c:.2f}: {bits}")

    explanation = (
        r"At the linear scale $N(n)\sim c n$ with $0<c<1/2$, short cycles stop being negligible. "
        r"For fixed cycle length $k$, the count is approximately Poisson with mean $\lambda_k=(2c)^k/(2k)$, "
        r"so the chance of seeing at least one such cycle is about $1-e^{-\lambda_k}$."
    )
    caption_lines = [
        fill(f"This figure uses {trials:,} independent runs for each (c, n, k) combination.", width=140),
        fill(explanation, width=140),
        fill("Poisson means used in the two panels: " + " ; ".join(lambda_bits) + ".", width=140),
    ]
    fig.text(
        0.5,
        0.015,
        "\n".join(caption_lines),
        ha="center",
        va="bottom",
        fontsize=9.6,
        color="0.25",
        wrap=True,
    )

    fig.tight_layout(rect=[0.03, 0.22, 0.98, 0.90], w_pad=1.0)
    fig.savefig(output_plot, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate the Phase 2.1 probability that short cycles are visible when m is chosen at the linear scale m = round(c n)."
    )
    parser.add_argument("--ns", type=int, nargs="+", default=DEFAULT_NS, help="Values of n to test.")
    parser.add_argument("--cs", type=float, nargs="+", default=DEFAULT_CS, help="Linear-scale constants c in m = round(c n), each below 0.5.")
    parser.add_argument("--ks", type=int, nargs="+", default=DEFAULT_KS, help="Cycle lengths to test.")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS, help="Number of Monte Carlo trials per (c, n, k).")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Base random seed.")
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV, help="CSV file for numerical results.")
    parser.add_argument("--output-plot", type=Path, default=DEFAULT_OUTPUT_PLOT, help="PNG plot file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cs = sorted(set(min(max(c, 0.01), 0.49) for c in args.cs))
    ns = sorted(set(n for n in args.ns if n >= 2))
    ks = sorted(set(k for k in args.ks if k >= 3))
    rng = np.random.default_rng(args.seed)

    results: List[SimulationResult] = []
    print("Running Phase 2.1 short-cycle simulations...")
    for c in cs:
        for k in ks:
            for n in ns:
                m = max(1, int(round(c * n)))
                print(f"  c = {c:.2f}, k = {k}, n = {n:7d}, m = round(c n) = {m:7d}, trials = {args.trials}")
                results.append(simulate_probability(n=n, cycle_len=k, c=c, trials=args.trials, rng=rng))

    results.sort(key=lambda r: (r.c, r.cycle_len, r.n))
    print()
    print_results_table(results)
    print()

    save_results_csv(results, args.output_csv)
    make_plot(results, args.output_plot)

    print(f"Saved CSV to:  {args.output_csv}")
    print(f"Saved PNG to:  {args.output_plot}")


if __name__ == "__main__":
    main()
