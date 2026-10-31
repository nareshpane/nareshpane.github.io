"""
Auto-video version of the Theorem 2 script.

Key change:
- running the script normally now generates the MP4 animation automatically
- use --skip-animation only if you want the CSV/plots without the video
"""

from __future__ import annotations

import argparse
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter


@dataclass(frozen=True)
class ResultRow:
    n: int
    c: float
    m: int
    k: int
    theory: float
    empirical: float
    abs_diff: float
    trials: int


@dataclass(frozen=True)
class TailRow:
    n: int
    c: float
    m: int
    tail_prob: float
    trials: int


@dataclass(frozen=True)
class Theorem2AnimationSummary:
    n: int
    c: float
    m: int
    largest_component_size: int
    observed_deficit: int
    theory_probability: float
    empirical_probability: float
    abs_diff: float
    empirical_trials: int


def edge_count_n_c(n: int, c: float) -> int:
    """Return N_c = floor(0.5 * n * log n + c n), clipped to [0, n choose 2]."""
    m = math.floor(0.5 * n * math.log(n) + c * n)
    max_edges = n * (n - 1) // 2
    return max(0, min(m, max_edges))


def theorem2_limit(c: float, k: int) -> float:
    """Theorem 2 limit for P_k(n, N_c): Poisson(e^{-2c}) at k."""
    lam = math.exp(-2.0 * c)
    return math.exp(-lam) * (lam**k) / math.factorial(k)


def estimate_deficit_distribution(
    n: int,
    m: int,
    k_max: int,
    trials: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    """
    Estimate probabilities for k = 0, ..., k_max where
    k = n - size_of_largest_connected_component.

    Returns (probs, tail_prob), where probs has length k_max + 1 and
    tail_prob is the fraction of trials with k > k_max.
    """
    counts = np.zeros(k_max + 1, dtype=np.int64)
    tail = 0

    for _ in range(trials):
        seed = int(rng.integers(0, 2**32 - 1, dtype=np.uint32))
        g = nx.gnm_random_graph(n, m, seed=seed)
        largest_component = max((len(comp) for comp in nx.connected_components(g)), default=0)
        deficit = n - largest_component
        if 0 <= deficit <= k_max:
            counts[deficit] += 1
        else:
            tail += 1

    probs = counts / trials
    tail_prob = tail / trials
    return probs, tail_prob


def estimate_specific_deficit_probability(
    n: int,
    m: int,
    target_deficit: int,
    trials: int,
    rng: np.random.Generator,
) -> float:
    """Estimate P[n - |C_max| = target_deficit] directly by Monte Carlo."""
    hits = 0
    for _ in range(trials):
        seed = int(rng.integers(0, 2**32 - 1, dtype=np.uint32))
        g = nx.gnm_random_graph(n, m, seed=seed)
        largest_component = max((len(comp) for comp in nx.connected_components(g)), default=0)
        deficit = n - largest_component
        if deficit == target_deficit:
            hits += 1
    return hits / trials


def run_experiment(
    n_values: Iterable[int],
    c_values: Iterable[float],
    k_max: int,
    trials: int,
    seed: int,
) -> tuple[list[ResultRow], list[TailRow]]:
    rng = np.random.default_rng(seed)
    rows: list[ResultRow] = []
    tail_rows: list[TailRow] = []

    for n in n_values:
        for c in c_values:
            c_float = float(c)
            m = edge_count_n_c(n, c_float)
            empirical_probs, tail_prob = estimate_deficit_distribution(n, m, k_max, trials, rng)
            tail_rows.append(TailRow(n=n, c=c_float, m=m, tail_prob=tail_prob, trials=trials))

            for k in range(k_max + 1):
                theory = theorem2_limit(c_float, k)
                empirical = float(empirical_probs[k])
                abs_diff = abs(theory - empirical)
                rows.append(
                    ResultRow(
                        n=n,
                        c=c_float,
                        m=m,
                        k=k,
                        theory=theory,
                        empirical=empirical,
                        abs_diff=abs_diff,
                        trials=trials,
                    )
                )

    return rows, tail_rows


def save_csv(rows: list[ResultRow], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("n,c,m,k,theory,empirical,abs_diff,trials\n")
        for row in rows:
            f.write(
                f"{row.n},{row.c:.6f},{row.m},{row.k},{row.theory:.10f},"
                f"{row.empirical:.10f},{row.abs_diff:.10f},{row.trials}\n"
            )


def save_tail_csv(rows: list[TailRow], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("n,c,m,tail_prob_gt_kmax,trials\n")
        for row in rows:
            f.write(f"{row.n},{row.c:.6f},{row.m},{row.tail_prob:.10f},{row.trials}\n")


def plot_results_by_k(rows: list[ResultRow], output_dir: Path) -> None:
    markers = {50: "o", 100: "s", 200: "^", 500: "D"}
    k_values = sorted({r.k for r in rows})
    c_min = min(r.c for r in rows)
    c_max = max(r.c for r in rows)

    for k in k_values:
        plt.figure(figsize=(10.5, 6.2))

        c_curve = np.linspace(c_min, c_max, 400)
        theory_curve = np.array([theorem2_limit(float(c), k) for c in c_curve])
        plt.plot(
            c_curve,
            theory_curve,
            linewidth=2.5,
            label=fr"Theorem 2 limit for $k={k}$",
        )

        for n in sorted({r.n for r in rows}):
            subset = sorted((r for r in rows if r.n == n and r.k == k), key=lambda r: r.c)
            x = [r.c for r in subset]
            y = [r.empirical for r in subset]
            m_labels = [str(r.m) for r in subset]

            plt.plot(
                x,
                y,
                marker=markers.get(n, "o"),
                linewidth=1.4,
                label=fr"Empirical, $n={n}$",
            )

            if subset:
                mid = min(range(len(subset)), key=lambda i: abs(subset[i].c))
                plt.annotate(
                    fr"$m={m_labels[mid]}$",
                    (x[mid], y[mid]),
                    textcoords="offset points",
                    xytext=(8, 8),
                    fontsize=8,
                )

        plt.xlabel(r"$c$ in $N_c = \lfloor \frac{1}{2} n \log n + cn \rfloor$")
        plt.ylabel(fr"Probability that $n-|C_\max| = {k}$")
        plt.title(fr"Erdős–Rényi (1959), Theorem 2 in $G(n,m)$: $P_k$ for $k={k}$")
        plt.ylim(bottom=-0.02)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"theorem2_k{k:02d}_plot.png", dpi=200)
        plt.close()


def print_summary(rows: list[ResultRow], tail_rows: list[TailRow], k_max: int) -> None:
    print("n      c        m       k      theory        empirical      abs_diff")
    print("-" * 78)
    tail_lookup = {(r.n, round(r.c, 6)): r for r in tail_rows}

    current_group: tuple[int, float] | None = None
    for row in rows:
        group = (row.n, row.c)
        if current_group is not None and group != current_group:
            tail = tail_lookup[(current_group[0], round(current_group[1], 6))]
            print(
                f"{'':<5} {'':>6}   {'':<5}   >{k_max:<2d}   {'':>10}   "
                f"{tail.tail_prob:>10.6f}   {'':>10}"
            )
            print()
        print(
            f"{row.n:<5d} {row.c:>6.2f}   {row.m:<5d}   {row.k:<2d}   "
            f"{row.theory:>10.6f}   {row.empirical:>10.6f}   {row.abs_diff:>10.6f}"
        )
        current_group = group

    if current_group is not None:
        tail = tail_lookup[(current_group[0], round(current_group[1], 6))]
        print(
            f"{'':<5} {'':>6}   {'':<5}   >{k_max:<2d}   {'':>10}   "
            f"{tail.tail_prob:>10.6f}   {'':>10}"
        )


def layered_circular_layout(
    n: int,
    rng: np.random.Generator,
    outer_radius: float = 0.94,
    inner_radius: float = 0.58,
) -> Dict[int, Tuple[float, float]]:
    """Mostly circular shell with several interior nodes for a readable 100-node layout."""
    if n <= 0:
        return {}

    if n <= 8:
        outer_count = max(4, n - 2)
    else:
        outer_count = max(10, int(round(0.62 * n)))
        outer_count = min(outer_count, n - 4)

    points: List[Tuple[float, float]] = []
    angle_offset = rng.uniform(0.0, 2.0 * math.pi)
    outer_angles = np.linspace(0.0, 2.0 * math.pi, outer_count, endpoint=False) + angle_offset
    outer_angles += rng.normal(0.0, 0.05, size=outer_count)

    for theta in outer_angles:
        r = outer_radius * (1.0 + rng.normal(0.0, 0.035))
        r = float(np.clip(r, 0.82, 0.99))
        points.append((r * math.cos(theta), r * math.sin(theta)))

    min_dist = 0.085 if n >= 80 else 0.11
    attempts = 0
    while len(points) < n and attempts < 50_000:
        attempts += 1
        theta = rng.uniform(0.0, 2.0 * math.pi)
        r = inner_radius * math.sqrt(rng.uniform(0.0, 1.0))
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        if all((x - px) ** 2 + (y - py) ** 2 >= min_dist**2 for px, py in points):
            points.append((x, y))

    while len(points) < n:
        theta = rng.uniform(0.0, 2.0 * math.pi)
        r = inner_radius * math.sqrt(rng.uniform(0.0, 1.0))
        points.append((r * math.cos(theta), r * math.sin(theta)))

    return {i + 1: points[i] for i in range(n)}


def ordered_random_edges(nodes: List[int], rng: np.random.Generator) -> List[Tuple[int, int]]:
    edges = [(u, v) for i, u in enumerate(nodes) for v in nodes[i + 1 :]]
    rng.shuffle(edges)
    return edges


def build_graph_state(nodes: List[int], edges_in_order: List[Tuple[int, int]], edge_count: int) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges_in_order[:edge_count])
    return g


def component_statistics(g: nx.Graph) -> tuple[int, int, int]:
    """Return (largest_component_size, deficit, num_components)."""
    component_sizes = sorted((len(c) for c in nx.connected_components(g)), reverse=True)
    largest = component_sizes[0] if component_sizes else 0
    deficit = g.number_of_nodes() - largest
    num_components = len(component_sizes)
    return largest, deficit, num_components


def prepare_theorem2_animation_summary(
    n: int,
    c: float,
    m: int,
    displayed_graph: nx.Graph,
    empirical_trials: int,
    seed: int,
) -> Theorem2AnimationSummary:
    largest_component_size, observed_deficit, _ = component_statistics(displayed_graph)
    theory_probability = theorem2_limit(c, observed_deficit)

    empirical_rng = np.random.default_rng(seed)
    empirical_probability = estimate_specific_deficit_probability(
        n=n,
        m=m,
        target_deficit=observed_deficit,
        trials=empirical_trials,
        rng=empirical_rng,
    )

    return Theorem2AnimationSummary(
        n=n,
        c=c,
        m=m,
        largest_component_size=largest_component_size,
        observed_deficit=observed_deficit,
        theory_probability=theory_probability,
        empirical_probability=empirical_probability,
        abs_diff=abs(theory_probability - empirical_probability),
        empirical_trials=empirical_trials,
    )


def save_animation(anim: FuncAnimation, output_path: Path, fps: int) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() == ".mp4":
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is not None:
            writer = FFMpegWriter(
                fps=fps,
                codec="libx264",
                bitrate=2600,
                extra_args=[
                    "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart",
                    "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                ],
            )
            anim.save(output_path, writer=writer, dpi=180)
            return output_path

        gif_path = output_path.with_suffix(".gif")
        anim.save(gif_path, writer=PillowWriter(fps=fps), dpi=120)
        return gif_path

    anim.save(output_path, writer=PillowWriter(fps=fps), dpi=120)
    return output_path


def make_theorem2_animation(
    n: int,
    c: float,
    seed: int,
    fps: int,
    frames_per_edge: int,
    hold_final_frames: int,
    pause_frames: int,
    theorem_trials: int,
    output_path: Path,
) -> tuple[Path, Theorem2AnimationSummary]:
    animation_rng = np.random.default_rng(seed)
    nodes = list(range(1, n + 1))
    positions = layered_circular_layout(n=n, rng=animation_rng)
    all_edges = ordered_random_edges(nodes=nodes, rng=animation_rng)

    theorem_m = edge_count_n_c(n, c)
    theorem_edges = all_edges[:theorem_m]
    checkpoint_graph = build_graph_state(nodes, theorem_edges, theorem_m)
    summary = prepare_theorem2_animation_summary(
        n=n,
        c=c,
        m=theorem_m,
        displayed_graph=checkpoint_graph,
        empirical_trials=theorem_trials,
        seed=seed + 1001,
    )

    total_possible_edges = n * (n - 1) // 2
    reveal_frames = theorem_m * frames_per_edge + 1
    total_frames = reveal_frames + pause_frames + hold_final_frames

    fig = plt.figure(figsize=(10.0, 9.0))
    graph_ax = fig.add_axes([0.05, 0.21, 0.90, 0.70])
    caption_ax = fig.add_axes([0.05, 0.04, 0.90, 0.13])
    fig.patch.set_facecolor("white")
    caption_ax.axis("off")

    info_line_1 = caption_ax.text(
        0.5,
        0.76,
        "",
        ha="center",
        va="center",
        fontsize=10,
        bbox=dict(facecolor="white", edgecolor="0.85", boxstyle="round,pad=0.25"),
    )
    info_line_2 = caption_ax.text(
        0.5,
        0.42,
        "",
        ha="center",
        va="center",
        fontsize=10,
        bbox=dict(facecolor="white", edgecolor="0.85", boxstyle="round,pad=0.25"),
    )
    event_line = caption_ax.text(
        0.5,
        0.10,
        "",
        ha="center",
        va="center",
        fontsize=10.8,
        bbox=dict(facecolor="white", edgecolor="0.72", boxstyle="round,pad=0.32"),
    )

    def update(frame_index: int) -> None:
        graph_ax.clear()

        if frame_index < reveal_frames:
            edge_count = min(frame_index // frames_per_edge, theorem_m)
            at_checkpoint = False
        else:
            edge_count = theorem_m
            at_checkpoint = True

        g = build_graph_state(nodes, theorem_edges, edge_count)
        largest_component, deficit, num_components = component_statistics(g)
        isolates = sum(1 for _, deg in g.degree() if deg == 0)
        density = 0.0 if n <= 1 else 2.0 * g.number_of_edges() / (n * (n - 1))
        avg_degree = 0.0 if n == 0 else 2.0 * g.number_of_edges() / n

        component_sets = list(nx.connected_components(g))
        largest_nodes = max(component_sets, key=len) if component_sets else set()
        node_colors = ["#f4f4f4" if node not in largest_nodes else "#d8e7ff" for node in nodes]

        previous_edges = theorem_edges[: max(0, edge_count - 1)]
        newest_edge = theorem_edges[edge_count - 1 : edge_count] if edge_count > 0 else []

        nx.draw_networkx_edges(
            g,
            pos=positions,
            edgelist=previous_edges,
            ax=graph_ax,
            width=0.85,
            alpha=0.33,
            edge_color="0.50",
        )
        nx.draw_networkx_edges(
            g,
            pos=positions,
            edgelist=newest_edge,
            ax=graph_ax,
            width=1.9,
            alpha=0.92,
            edge_color="black",
        )
        nx.draw_networkx_nodes(
            g,
            pos=positions,
            ax=graph_ax,
            node_size=58,
            linewidths=0.55,
            edgecolors="black",
            node_color=node_colors,
        )

        graph_ax.set_xlim(-1.16, 1.16)
        graph_ax.set_ylim(-1.16, 1.16)
        graph_ax.set_aspect("equal")
        graph_ax.axis("off")
        graph_ax.set_title(
            f"Theorem 2: Vertices Outside the Largest Connected Component with c = {c:.2f}",
            fontsize=15.5,
            pad=28,
        )
        graph_ax.text(
            0.5,
            1.01,
            (
                f"Checkpoint at N_c = floor(0.5 n log n + c n) = {theorem_m} edges; "
                "statistic is k = n - |C_max|"
            ),
            transform=graph_ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=10.3,
            color="0.22",
        )

        info_line_1.set_text(
            f"n = {n}   |   edges shown = {g.number_of_edges()}/{theorem_m}   |   "
            f"possible total = {total_possible_edges}   |   density = {density:.3f}"
        )
        info_line_2.set_text(
            f"largest component = {largest_component}   |   deficit k = {deficit}   |   "
            f"components = {num_components}   |   isolates = {isolates}   |   avg degree = {avg_degree:.2f}"
        )

        if at_checkpoint:
            event_line.set_text(
                "Checkpoint reached: compare observed k with the theorem and Monte Carlo."
            )
            graph_ax.text(
                0.5,
                -0.06,
                (
                    f"Observed k = {summary.observed_deficit}; "
                    f"Theory P_k ≈ {summary.theory_probability:.4f}; "
                    f"Empirical P_k ≈ {summary.empirical_probability:.4f} "
                    f"from {summary.empirical_trials} samples; "
                    f"|diff| = {summary.abs_diff:.4f}"
                ),
                transform=graph_ax.transAxes,
                ha="center",
                va="top",
                fontsize=10.0,
                bbox=dict(facecolor="white", edgecolor="0.55", boxstyle="round,pad=0.32"),
            )
        else:
            edges_left = theorem_m - g.number_of_edges()
            event_line.set_text(
                f"Approaching theorem checkpoint: {edges_left} edges remain before N_c."
            )

    anim = FuncAnimation(fig, update, frames=total_frames, interval=1000 / fps, repeat=False)
    saved_to = save_animation(anim, output_path=output_path, fps=fps)
    plt.close(fig)
    return saved_to, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Monte Carlo test of Erdős–Rényi (1959), Theorem 2, for G(n,m) with "
            "m = floor(0.5*n*log(n) + c*n)."
        )
    )
    parser.add_argument(
        "--n-values",
        type=int,
        nargs="+",
        default=[50, 100, 200, 500],
        help="Vertex counts to test.",
    )
    parser.add_argument(
        "--c-min",
        type=float,
        default=-1.5,
        help="Minimum c value.",
    )
    parser.add_argument(
        "--c-max",
        type=float,
        default=1.5,
        help="Maximum c value.",
    )
    parser.add_argument(
        "--num-c",
        type=int,
        default=13,
        help="Number of equally spaced c values.",
    )
    parser.add_argument(
        "--k-max",
        type=int,
        default=10,
        help="Maximum k value to record explicitly.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=300,
        help="Monte Carlo trials per (n, c) pair.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Random seed.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("./erdos_renyi_theorem2_output"),
        help="Directory for the CSV and PNG outputs.",
    )
    parser.add_argument(
        "--skip-animation",
        action="store_true",
        help="Skip saving the MP4 animation. By default, the script now creates the animation.",
    )
    parser.add_argument(
        "--animation-n",
        type=int,
        default=100,
        help="Number of nodes in the animation.",
    )
    parser.add_argument(
        "--animation-c",
        type=float,
        default=0.0,
        help="Value of c used in the animation checkpoint N_c.",
    )
    parser.add_argument(
        "--animation-seed",
        type=int,
        default=2026,
        help="Random seed for the animation graph and checkpoint estimate.",
    )
    parser.add_argument(
        "--animation-fps",
        type=int,
        default=4,
        help="Frames per second for the animation.",
    )
    parser.add_argument(
        "--animation-frames-per-edge",
        type=int,
        default=3,
        help="How many frames each newly added edge remains on screen before the next edge appears.",
    )
    parser.add_argument(
        "--animation-pause-frames",
        type=int,
        default=28,
        help="How many frames to pause at the theorem checkpoint for the comparison note.",
    )
    parser.add_argument(
        "--animation-hold-final-frames",
        type=int,
        default=16,
        help="Extra still frames after the checkpoint pause.",
    )
    parser.add_argument(
        "--animation-trials",
        type=int,
        default=400,
        help="Monte Carlo samples used to estimate the empirical probability shown in the animation.",
    )
    parser.add_argument(
        "--animation-output",
        type=Path,
        default=Path("./random_graph_100_node_theorem2.mp4"),
        help="Output path for the animation. Use .mp4 or .gif.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    c_values = np.linspace(args.c_min, args.c_max, args.num_c)
    rows, tail_rows = run_experiment(
        n_values=args.n_values,
        c_values=c_values,
        k_max=args.k_max,
        trials=args.trials,
        seed=args.seed,
    )

    csv_path = args.outdir / "theorem2_results.csv"
    tail_csv_path = args.outdir / "theorem2_tail_probabilities.csv"

    save_csv(rows, csv_path)
    save_tail_csv(tail_rows, tail_csv_path)
    plot_results_by_k(rows, args.outdir)
    print_summary(rows, tail_rows, args.k_max)
    print(f"\nSaved CSV to: {csv_path}")
    print(f"Saved tail-summary CSV to: {tail_csv_path}")
    print(f"Saved plots to: {args.outdir}")

    if not args.skip_animation:
        saved_to, summary = make_theorem2_animation(
            n=args.animation_n,
            c=args.animation_c,
            seed=args.animation_seed,
            fps=max(1, args.animation_fps),
            frames_per_edge=max(1, args.animation_frames_per_edge),
            hold_final_frames=max(0, args.animation_hold_final_frames),
            pause_frames=max(0, args.animation_pause_frames),
            theorem_trials=max(1, args.animation_trials),
            output_path=args.animation_output,
        )
        print(f"\nSaved animation to: {saved_to}")
        print(
            "Checkpoint summary: "
            f"n={summary.n}, c={summary.c:.2f}, m={summary.m}, "
            f"observed_k={summary.observed_deficit}, "
            f"theory={summary.theory_probability:.6f}, "
            f"empirical={summary.empirical_probability:.6f}, "
            f"abs_diff={summary.abs_diff:.6f}"
        )


if __name__ == "__main__":
    main()
