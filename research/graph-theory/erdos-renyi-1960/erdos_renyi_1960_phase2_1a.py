#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from textwrap import fill
from typing import Dict, List, Optional, Sequence, Set, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


# ============================================================
# Phase 2.1A animation.
# Default output matches the HTML placeholder:
#     erdos_renyi_1960_phase2_1a.mp4
#
# Idea:
#   In Phase 2, the graph is at the linear scale m ~ c n with 0 < c < 1/2.
#   Short cycles are no longer negligible. This animation grows one graph
#   beyond the linear benchmark and highlights one visible C3, C4, and C5
#   whenever they exist, while showing the Poisson-limit means for these
#   cycle counts.
# ============================================================
DEFAULT_N = 1000
DEFAULT_C = 0.45
DEFAULT_MAX_EDGES = 1000
DEFAULT_SEED = 24680
DEFAULT_FPS = 16                  # 2x faster than the prior faster version
DEFAULT_FRAMES_PER_EDGE = 1
DEFAULT_EVENT_PAUSE_FRAMES = 16   # slightly longer pause at key moments
DEFAULT_HOLD_FINAL_FRAMES = 18
DEFAULT_OUTPUT = Path("./erdos_renyi_1960_phase2_1a.mp4")

CYCLE_COLORS = {
    3: "#c62828",  # red
    4: "#1565c0",  # blue
    5: "#2e7d32",  # green
}
BASE_EDGE_COLOR = "#6b7280"
BENCHMARK_COLOR = "#7b1fa2"
NEUTRAL_EVENT_COLOR = "#444444"


@dataclass(frozen=True)
class Phase2Event:
    edge_index: int
    label: str
    color: str


@dataclass(frozen=True)
class CycleSummary:
    counts: Dict[int, int]
    sample_cycle_nodes: Dict[int, Optional[List[int]]]
    sample_cycle_edges: Dict[int, List[Tuple[int, int]]]
    isolates: int
    components: int
    largest_component: int


def layered_circular_layout(
    n: int,
    rng: np.random.Generator,
    outer_radius: float = 0.96,
    inner_radius: float = 0.55,
) -> Dict[int, Tuple[float, float]]:
    if n <= 0:
        return {}

    outer_count = max(12, int(round(0.62 * n)))
    outer_count = min(outer_count, max(3, n - 2))
    points: List[Tuple[float, float]] = []

    angle_offset = float(rng.uniform(0.0, 2.0 * math.pi))
    outer_angles = np.linspace(0.0, 2.0 * math.pi, outer_count, endpoint=False) + angle_offset
    outer_angles += rng.normal(0.0, 0.07, size=outer_count)

    for theta in outer_angles:
        r = float(np.clip(outer_radius * (1.0 + rng.normal(0.0, 0.05)), 0.82, 1.02))
        points.append((r * math.cos(theta), r * math.sin(theta)))

    min_dist = 0.075 if n >= 100 else 0.12
    attempts = 0
    while len(points) < n and attempts < 25000:
        attempts += 1
        theta = float(rng.uniform(0.0, 2.0 * math.pi))
        r = float(inner_radius * math.sqrt(rng.uniform(0.0, 1.0)))
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        if all((x - px) ** 2 + (y - py) ** 2 >= min_dist**2 for px, py in points):
            points.append((x, y))

    while len(points) < n:
        theta = float(rng.uniform(0.0, 2.0 * math.pi))
        r = float(inner_radius * math.sqrt(rng.uniform(0.0, 1.0)))
        points.append((r * math.cos(theta), r * math.sin(theta)))

    return {i: points[i] for i in range(n)}


def ordered_random_edges(nodes: List[int], rng: np.random.Generator) -> List[Tuple[int, int]]:
    edges = [(u, v) for i, u in enumerate(nodes) for v in nodes[i + 1 :]]
    rng.shuffle(edges)
    return edges


def build_graph_state(nodes: List[int], edges_in_order: List[Tuple[int, int]], edge_count: int) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges_in_order[:edge_count])
    return g


def build_adjacency(g: nx.Graph) -> Dict[int, Set[int]]:
    return {u: set(g.neighbors(u)) for u in g.nodes()}


def find_one_cycle_of_length(adj: Dict[int, Set[int]], cycle_len: int) -> Optional[List[int]]:
    nodes = sorted(adj)
    for start in nodes:
        path = [start]
        visited = {start}

        def dfs(cur: int, depth: int) -> Optional[List[int]]:
            if depth == cycle_len:
                if start in adj[cur]:
                    return path.copy()
                return None

            for nxt in adj[cur]:
                if nxt <= start or nxt in visited:
                    continue
                visited.add(nxt)
                path.append(nxt)
                out = dfs(nxt, depth + 1)
                if out is not None:
                    return out
                path.pop()
                visited.remove(nxt)
            return None

        out = dfs(start, 1)
        if out is not None:
            return out
    return None


def count_cycles_of_length(adj: Dict[int, Set[int]], cycle_len: int) -> int:
    nodes = sorted(adj)
    count = 0
    for start in nodes:
        path = [start]
        visited = {start}

        def dfs(cur: int, depth: int) -> None:
            nonlocal count
            if depth == cycle_len:
                if start in adj[cur]:
                    count += 1
                return
            for nxt in adj[cur]:
                if nxt <= start or nxt in visited:
                    continue
                visited.add(nxt)
                path.append(nxt)
                dfs(nxt, depth + 1)
                path.pop()
                visited.remove(nxt)

        dfs(start, 1)

    return count // 2


def cycle_nodes_to_edges(nodes: Optional[List[int]]) -> List[Tuple[int, int]]:
    if nodes is None or len(nodes) < 3:
        return []
    return [(nodes[i], nodes[(i + 1) % len(nodes)]) for i in range(len(nodes))]


def summarize_cycles(g: nx.Graph, cycle_lengths: Sequence[int]) -> CycleSummary:
    adj = build_adjacency(g)
    counts: Dict[int, int] = {}
    sample_nodes: Dict[int, Optional[List[int]]] = {}
    sample_edges: Dict[int, List[Tuple[int, int]]] = {}

    for k in cycle_lengths:
        sample = find_one_cycle_of_length(adj, k)
        sample_nodes[k] = sample
        sample_edges[k] = cycle_nodes_to_edges(sample)
        counts[k] = count_cycles_of_length(adj, k)

    comps = list(nx.connected_components(g))
    isolates = sum(1 for _, d in g.degree() if d == 0)
    largest = max((len(c) for c in comps), default=0)
    return CycleSummary(
        counts=counts,
        sample_cycle_nodes=sample_nodes,
        sample_cycle_edges=sample_edges,
        isolates=isolates,
        components=len(comps),
        largest_component=largest,
    )


def poisson_mean_for_cycles(c: float, cycle_len: int) -> float:
    return ((2.0 * c) ** cycle_len) / (2.0 * cycle_len)


def collect_events(
    nodes: List[int],
    edges_in_order: List[Tuple[int, int]],
    benchmark_edges: int,
) -> List[Phase2Event]:
    g = nx.Graph()
    g.add_nodes_from(nodes)
    events: List[Phase2Event] = []
    seen = {3: False, 4: False, 5: False}
    benchmark_added = False

    for idx, edge in enumerate(edges_in_order, start=1):
        g.add_edge(*edge)
        summary = summarize_cycles(g, cycle_lengths=[3, 4, 5])
        for k in [3, 4, 5]:
            if not seen[k] and summary.counts[k] > 0:
                seen[k] = True
                events.append(Phase2Event(idx, f"First visible C{k} appears", CYCLE_COLORS[k]))
        if not benchmark_added and idx >= benchmark_edges:
            benchmark_added = True
            events.append(Phase2Event(idx, f"Linear benchmark reached: m ≈ c n = {benchmark_edges:,}", BENCHMARK_COLOR))

    events.append(Phase2Event(len(edges_in_order), f"Stopped at m = {len(edges_in_order):,}", NEUTRAL_EVENT_COLOR))
    return events


def active_event(edge_count: int, events: List[Phase2Event], linger_edges: int) -> Optional[Phase2Event]:
    active = [e for e in events if e.edge_index <= edge_count < e.edge_index + linger_edges]
    if not active:
        return None
    return active[-1]


def build_frame_schedule(max_edges: int, events: List[Phase2Event], pause_frames: int, hold_final_frames: int) -> List[int]:
    event_edges = {e.edge_index for e in events if e.edge_index < max_edges}
    schedule: List[int] = [0]
    for edge_count in range(1, max_edges + 1):
        schedule.append(edge_count)
        if edge_count in event_edges:
            schedule.extend([edge_count] * pause_frames)
    schedule.extend([max_edges] * hold_final_frames)
    return schedule


def save_animation(anim: FuncAnimation, output_path: Path, fps: int, dpi: int = 180) -> Path:
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
            anim.save(output_path, writer=writer, dpi=dpi)
            return output_path

        gif_path = output_path.with_suffix(".gif")
        anim.save(gif_path, writer=PillowWriter(fps=fps), dpi=120)
        return gif_path

    anim.save(output_path, writer=PillowWriter(fps=fps), dpi=120)
    return output_path


def make_phase2_1a_animation(
    n: int,
    c: float,
    max_edges: int,
    seed: int,
    fps: int,
    event_pause_frames: int,
    hold_final_frames: int,
    output_path: Path,
) -> Path:
    rng = np.random.default_rng(seed)
    nodes = list(range(n))
    positions = layered_circular_layout(n=n, rng=rng)
    benchmark_edges = max(1, int(round(c * n)))
    total_possible_edges = n * (n - 1) // 2
    target_edges = max(1, min(max_edges, total_possible_edges))
    edge_order = ordered_random_edges(nodes=nodes, rng=rng)[:target_edges]
    events = collect_events(nodes=nodes, edges_in_order=edge_order, benchmark_edges=benchmark_edges)
    frame_schedule = build_frame_schedule(
        max_edges=target_edges,
        events=events,
        pause_frames=max(0, event_pause_frames),
        hold_final_frames=max(0, hold_final_frames),
    )

    fig = plt.figure(figsize=(10.8, 9.7))
    title_ax = fig.add_axes([0.05, 0.84, 0.90, 0.10])
    title_ax.axis("off")
    graph_ax = fig.add_axes([0.05, 0.33, 0.90, 0.42])
    info_ax = fig.add_axes([0.05, 0.06, 0.90, 0.17])
    info_ax.axis("off")
    fig.patch.set_facecolor("white")

    legend_handles = [
        Line2D([0], [0], color=CYCLE_COLORS[3], linewidth=2.6, label="One visible triangle (C3)"),
        Line2D([0], [0], color=CYCLE_COLORS[4], linewidth=2.6, label="One visible 4-cycle (C4)"),
        Line2D([0], [0], color=CYCLE_COLORS[5], linewidth=2.6, label="One visible 5-cycle (C5)"),
        Line2D([0], [0], color=BENCHMARK_COLOR, linewidth=2.6, label="Linear benchmark highlight"),
        Line2D([0], [0], color=BASE_EDGE_COLOR, linewidth=1.8, label="Other edges"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.245),
        ncol=3,
        frameon=True,
        edgecolor="#cccccc",
        fontsize=9.3,
        handlelength=2.0,
        columnspacing=1.6,
        handletextpad=0.7,
        borderpad=0.7,
        labelspacing=0.8,
    )

    subtitle = fill(
        rf"Using $c={c:.2f}$, so the Phase 2 linear benchmark is $m \approx c n = {benchmark_edges:,}$. "
        rf"This animation continues to $m={target_edges:,}$ edges. The Poisson-limit means are "
        rf"$\lambda_3={poisson_mean_for_cycles(c, 3):.4f}$, $\lambda_4={poisson_mean_for_cycles(c, 4):.4f}$, "
        rf"and $\lambda_5={poisson_mean_for_cycles(c, 5):.4f}$.",
        width=92,
    )

    def update(frame_index: int) -> None:
        edge_count = frame_schedule[frame_index]
        g = build_graph_state(nodes=nodes, edges_in_order=edge_order, edge_count=edge_count)
        summary = summarize_cycles(g, cycle_lengths=[3, 4, 5])
        current_event = active_event(edge_count, events, linger_edges=max(2, event_pause_frames + 1))
        event_color = current_event.color if current_event is not None else NEUTRAL_EVENT_COLOR

        graph_ax.clear()
        info_ax.clear()
        info_ax.axis("off")

        highlighted_sets = {k: {tuple(sorted(e)) for e in summary.sample_cycle_edges[k]} for k in [3, 4, 5]}
        all_highlighted = set().union(*highlighted_sets.values())
        other_edges = [e for e in g.edges() if tuple(sorted(e)) not in all_highlighted]

        nx.draw_networkx_edges(
            g,
            pos=positions,
            edgelist=other_edges,
            ax=graph_ax,
            width=1.0,
            alpha=0.55,
            edge_color=BASE_EDGE_COLOR,
        )
        for k in [3, 4, 5]:
            if summary.sample_cycle_edges[k]:
                nx.draw_networkx_edges(
                    g,
                    pos=positions,
                    edgelist=summary.sample_cycle_edges[k],
                    ax=graph_ax,
                    width=2.6,
                    alpha=0.95,
                    edge_color=CYCLE_COLORS[k],
                )

        non_isolates = [u for u, d in g.degree() if d > 0]
        isolates = [u for u, d in g.degree() if d == 0]
        nx.draw_networkx_nodes(
            g,
            pos=positions,
            nodelist=isolates,
            ax=graph_ax,
            node_size=18 if n >= 1000 else 38,
            linewidths=0.25,
            edgecolors="0.35",
            node_color="white",
        )
        nx.draw_networkx_nodes(
            g,
            pos=positions,
            nodelist=non_isolates,
            ax=graph_ax,
            node_size=28 if n >= 1000 else 48,
            linewidths=0.35,
            edgecolors="#0d3b66",
            node_color="#d9ebff",
        )

        graph_ax.set_xlim(-1.12, 1.12)
        graph_ax.set_ylim(-1.12, 1.12)
        graph_ax.set_aspect("equal")
        graph_ax.axis("off")

        title_ax.clear()
        title_ax.axis("off")
        title_ax.text(
            0.5,
            0.80,
            rf"Phase 2.1A — fixed short cycles appear at the linear scale $m \approx c n$ (n = {n:,})",
            ha="center",
            va="center",
            fontsize=16,
            fontweight="bold",
        )
        title_ax.text(
            0.5,
            0.18,
            subtitle,
            ha="center",
            va="bottom",
            fontsize=10.0,
            color="0.25",
            linespacing=1.25,
        )
        graph_ax.add_patch(
            Rectangle(
                (0.003, 0.003),
                0.994,
                0.994,
                transform=graph_ax.transAxes,
                fill=False,
                linewidth=2.6 if current_event is not None else 1.4,
                edgecolor=event_color if current_event is not None else "#b9b1a5",
                zorder=300,
                clip_on=False,
            )
        )

        avg_degree = 2.0 * edge_count / n
        density = 2.0 * edge_count / total_possible_edges if total_possible_edges > 0 else 0.0
        info_ax.text(
            0.5,
            0.79,
            f"edges shown = {edge_count:,}/{target_edges:,}   |   benchmark m ≈ c n = {benchmark_edges:,}   |   avg degree = {avg_degree:.3f}   |   density = {density:.6f}",
            transform=info_ax.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            bbox=dict(facecolor="#fff6f6" if current_event is not None else "white", edgecolor=event_color if current_event is not None else "0.85", boxstyle="round,pad=0.24"),
        )
        info_ax.text(
            0.5,
            0.49,
            f"current counts: C3 = {summary.counts[3]}   |   C4 = {summary.counts[4]}   |   C5 = {summary.counts[5]}   |   largest component = {summary.largest_component}   |   isolates = {summary.isolates:,}",
            transform=info_ax.transAxes,
            ha="center",
            va="center",
            fontsize=9.8,
            bbox=dict(facecolor="white", edgecolor="0.85", boxstyle="round,pad=0.24"),
        )
        event_text = current_event.label if current_event is not None else ""
        info_ax.text(
            0.5,
            0.13,
            fill(
                ("Interpretation: unlike Phase 1, a few short closed loops can now genuinely appear with stable nonzero probability. " + event_text).strip(),
                width=112,
            ),
            transform=info_ax.transAxes,
            ha="center",
            va="center",
            fontsize=9.1,
            linespacing=1.22,
            bbox=dict(facecolor="#f9f9f9", edgecolor=event_color if current_event is not None else "0.75", boxstyle="round,pad=0.26"),
        )

    anim = FuncAnimation(fig, update, frames=len(frame_schedule), interval=1000 / fps, repeat=False)
    saved_to = save_animation(anim=anim, output_path=output_path, fps=fps)
    plt.close(fig)
    return saved_to


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create the Phase 2.1A animation showing that at the linear scale m ~ c n, short cycles stop being negligible."
        )
    )
    parser.add_argument("--n", type=int, default=DEFAULT_N, help="Number of nodes.")
    parser.add_argument("--c", type=float, default=DEFAULT_C, help="Linear-scale constant in m = round(c n), should stay below 0.5.")
    parser.add_argument("--max-edges", type=int, default=DEFAULT_MAX_EDGES, help="Total number of edges shown in the animation.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help="Frames per second.")
    parser.add_argument("--event-pause-frames", type=int, default=DEFAULT_EVENT_PAUSE_FRAMES, help="Extra pause frames when a cycle first appears or when the linear benchmark is reached.")
    parser.add_argument("--hold-final-frames", type=int, default=DEFAULT_HOLD_FINAL_FRAMES, help="Still frames at the end.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output animation path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    c = min(max(args.c, 0.01), 0.49)
    n = max(2, args.n)
    max_edges = max(1, args.max_edges)
    saved_to = make_phase2_1a_animation(
        n=n,
        c=c,
        max_edges=max_edges,
        seed=args.seed,
        fps=max(1, args.fps),
        event_pause_frames=max(0, args.event_pause_frames),
        hold_final_frames=max(0, args.hold_final_frames),
        output_path=args.output,
    )
    print(f"n = {n}")
    print(f"c = {c:.3f}")
    print(f"linear benchmark round(c n) = {int(round(c * n))}")
    print(f"edges shown up to = {max_edges}")
    print(f"saved animation to: {saved_to}")


if __name__ == "__main__":
    main()
