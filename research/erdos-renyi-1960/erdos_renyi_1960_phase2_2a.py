#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from textwrap import fill
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

# ============================================================
# Phase 2.2A animation.
# Default output matches the HTML placeholder:
#     erdos_renyi_1960_phase2_2a.mp4
#
# Idea:
#   In Phase 2 with m ~ c n and c < 1/2, the graph is no longer a pure
#   forest, but most connected components should still be either trees or
#   unicyclic pieces. This animation grows one graph to the linear scale
#   and highlights one sample tree component, one sample unicyclic component,
#   and one sample multicyclic component whenever they exist.
# ============================================================
DEFAULT_N = 1000
DEFAULT_C = 0.30
DEFAULT_SEED = 13579
DEFAULT_FPS = 6
DEFAULT_EVENT_PAUSE_FRAMES = 12
DEFAULT_HOLD_FINAL_FRAMES = 20
DEFAULT_OUTPUT = Path("./erdos_renyi_1960_phase2_2a.mp4")

TREE_COLOR = "#1565c0"       # blue
UNICYCLIC_COLOR = "#2e7d32"  # green
MULTICYCLIC_COLOR = "#c62828"# red
BASE_EDGE = "#6b7280"
BENCHMARK_COLOR = "#7b1fa2"


@dataclass(frozen=True)
class Phase2Event:
    edge_index: int
    label: str
    color: str


@dataclass(frozen=True)
class ComponentSummary:
    tree_components: int
    unicyclic_components: int
    multicyclic_components: int
    isolates: int
    largest_component: int
    sample_tree_edges: List[Tuple[int, int]]
    sample_unicyclic_edges: List[Tuple[int, int]]
    sample_multicyclic_edges: List[Tuple[int, int]]


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


def summarize_components(g: nx.Graph) -> ComponentSummary:
    tree_components = 0
    unicyclic_components = 0
    multicyclic_components = 0
    isolates = 0
    largest_component = 0

    sample_tree_edges: List[Tuple[int, int]] = []
    sample_unicyclic_edges: List[Tuple[int, int]] = []
    sample_multicyclic_edges: List[Tuple[int, int]] = []

    for comp_nodes in nx.connected_components(g):
        h = g.subgraph(comp_nodes)
        v = h.number_of_nodes()
        e = h.number_of_edges()
        largest_component = max(largest_component, v)
        if v == 1:
            isolates += 1
            continue
        if e == v - 1:
            tree_components += 1
            if not sample_tree_edges:
                sample_tree_edges = list(h.edges())
        elif e == v:
            unicyclic_components += 1
            if not sample_unicyclic_edges:
                sample_unicyclic_edges = list(h.edges())
        elif e > v:
            multicyclic_components += 1
            if not sample_multicyclic_edges:
                sample_multicyclic_edges = list(h.edges())

    return ComponentSummary(
        tree_components=tree_components,
        unicyclic_components=unicyclic_components,
        multicyclic_components=multicyclic_components,
        isolates=isolates,
        largest_component=largest_component,
        sample_tree_edges=sample_tree_edges,
        sample_unicyclic_edges=sample_unicyclic_edges,
        sample_multicyclic_edges=sample_multicyclic_edges,
    )


def collect_events(nodes: List[int], edges_in_order: List[Tuple[int, int]], benchmark_edges: int) -> List[Phase2Event]:
    g = nx.Graph()
    g.add_nodes_from(nodes)
    events: List[Phase2Event] = []
    saw_unicyclic = False
    saw_multicyclic = False
    benchmark_added = False

    for idx, edge in enumerate(edges_in_order, start=1):
        g.add_edge(*edge)
        summary = summarize_components(g)
        if not saw_unicyclic and summary.unicyclic_components > 0:
            saw_unicyclic = True
            events.append(Phase2Event(idx, "First visible unicyclic component appears", UNICYCLIC_COLOR))
        if not saw_multicyclic and summary.multicyclic_components > 0:
            saw_multicyclic = True
            events.append(Phase2Event(idx, "A multicyclic component appears", MULTICYCLIC_COLOR))
        if not benchmark_added and idx >= benchmark_edges:
            benchmark_added = True
            events.append(Phase2Event(idx, f"Linear benchmark reached: m ≈ c n = {benchmark_edges:,}", BENCHMARK_COLOR))

    events.append(Phase2Event(len(edges_in_order), f"Stopped at m = {len(edges_in_order):,}", "#444444"))
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


def make_phase2_2a_animation(
    n: int,
    c: float,
    seed: int,
    fps: int,
    event_pause_frames: int,
    hold_final_frames: int,
    output_path: Path,
) -> Path:
    rng = np.random.default_rng(seed)
    nodes = list(range(n))
    positions = layered_circular_layout(n=n, rng=rng)
    m_target = max(1, int(round(c * n)))
    total_possible_edges = n * (n - 1) // 2
    edge_order = ordered_random_edges(nodes=nodes, rng=rng)[:m_target]
    events = collect_events(nodes=nodes, edges_in_order=edge_order, benchmark_edges=m_target)
    frame_schedule = build_frame_schedule(
        max_edges=m_target,
        events=events,
        pause_frames=max(0, event_pause_frames),
        hold_final_frames=max(0, hold_final_frames),
    )

    fig = plt.figure(figsize=(11.2, 9.7))
    title_ax = fig.add_axes([0.04, 0.85, 0.92, 0.09])
    title_ax.axis("off")
    graph_ax = fig.add_axes([0.03, 0.27, 0.94, 0.50])
    info_ax = fig.add_axes([0.05, 0.05, 0.90, 0.16])
    info_ax.axis("off")
    fig.patch.set_facecolor("white")

    legend_handles = [
        Line2D([0], [0], color=TREE_COLOR, linewidth=1.5, label="One sample tree component"),
        Line2D([0], [0], color=UNICYCLIC_COLOR, linewidth=1.8, label="One sample unicyclic component"),
        Line2D([0], [0], color=MULTICYCLIC_COLOR, linewidth=2.0, label="One sample multicyclic component"),
        Line2D([0], [0], color=BENCHMARK_COLOR, linewidth=1.8, label="Linear benchmark highlight"),
        Line2D([0], [0], color=BASE_EDGE, linewidth=1.0, label="Other edges"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.225),
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
        rf"Using $c={c:.2f}$, so $m = \lfloor c n \rceil = {m_target:,}$ and the average degree is about $2c={2*c:.2f}$. "
        r"The Phase 2.2 claim is that the graph is still mostly made of trees, with the non-tree exceptions usually being unicyclic rather than multi-cycle pieces.",
        width=96,
    )

    def update(frame_index: int) -> None:
        edge_count = frame_schedule[frame_index]
        g = build_graph_state(nodes=nodes, edges_in_order=edge_order, edge_count=edge_count)
        summary = summarize_components(g)
        current_event = active_event(edge_count, events, linger_edges=max(2, event_pause_frames + 1))
        event_color = current_event.color if current_event is not None else "#444444"

        graph_ax.clear()
        info_ax.clear()
        info_ax.axis("off")

        sample_tree = {tuple(sorted(e)) for e in summary.sample_tree_edges}
        sample_uni = {tuple(sorted(e)) for e in summary.sample_unicyclic_edges}
        sample_multi = {tuple(sorted(e)) for e in summary.sample_multicyclic_edges}
        highlighted = sample_tree | sample_uni | sample_multi
        other_edges = [e for e in g.edges() if tuple(sorted(e)) not in highlighted]

        nx.draw_networkx_edges(g, pos=positions, edgelist=other_edges, ax=graph_ax, width=0.65, alpha=0.40, edge_color=BASE_EDGE)
        if summary.sample_tree_edges:
            nx.draw_networkx_edges(g, pos=positions, edgelist=summary.sample_tree_edges, ax=graph_ax, width=1.45, alpha=0.92, edge_color=TREE_COLOR)
        if summary.sample_unicyclic_edges:
            nx.draw_networkx_edges(g, pos=positions, edgelist=summary.sample_unicyclic_edges, ax=graph_ax, width=1.75, alpha=0.95, edge_color=UNICYCLIC_COLOR)
        if summary.sample_multicyclic_edges:
            nx.draw_networkx_edges(g, pos=positions, edgelist=summary.sample_multicyclic_edges, ax=graph_ax, width=1.95, alpha=0.95, edge_color=MULTICYCLIC_COLOR)

        non_isolates = [u for u, d in g.degree() if d > 0]
        isolates = [u for u, d in g.degree() if d == 0]
        nx.draw_networkx_nodes(g, pos=positions, nodelist=isolates, ax=graph_ax, node_size=18 if n >= 1000 else 38, linewidths=0.25, edgecolors="0.35", node_color="white")
        nx.draw_networkx_nodes(g, pos=positions, nodelist=non_isolates, ax=graph_ax, node_size=28 if n >= 1000 else 48, linewidths=0.35, edgecolors="#0d3b66", node_color="#d9ebff")

        graph_ax.set_xlim(-1.12, 1.12)
        graph_ax.set_ylim(-1.12, 1.12)
        graph_ax.set_aspect("equal")
        graph_ax.axis("off")

        title_ax.clear()
        title_ax.axis("off")
        title_ax.text(
            0.5,
            0.78,
            rf"Phase 2.2A — most components are trees or exactly one cycle (n = {n:,})",
            ha="center",
            va="center",
            fontsize=16,
            fontweight="bold",
        )
        title_ax.text(
            0.5,
            0.12,
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
            f"edges shown = {edge_count:,}/{m_target:,}   |   avg degree = {avg_degree:.3f}   |   density = {density:.6f}   |   isolates = {summary.isolates:,}",
            transform=info_ax.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            bbox=dict(facecolor="#fff6f6" if current_event is not None else "white", edgecolor=event_color if current_event is not None else "0.85", boxstyle="round,pad=0.24"),
        )
        info_ax.text(
            0.5,
            0.49,
            f"component counts: tree = {summary.tree_components}   |   unicyclic = {summary.unicyclic_components}   |   multicyclic = {summary.multicyclic_components}   |   largest = {summary.largest_component}",
            transform=info_ax.transAxes,
            ha="center",
            va="center",
            fontsize=9.8,
            bbox=dict(facecolor="white", edgecolor="0.85", boxstyle="round,pad=0.24"),
        )
        predicate = "YES" if summary.multicyclic_components == 0 else "NO"
        event_text = current_event.label if current_event is not None else ""
        info_ax.text(
            0.5,
            0.13,
            fill(
                (f"Are all nontrivial components tree-or-unicyclic right now? {predicate}. "
                 f"Interpretation: Phase 2.2 predicts trees plus occasional single-cycle pieces, not a landscape dominated by multi-cycle components. {event_text}").strip(),
                width=114,
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
            "Create the Phase 2.2A animation showing that at the linear scale m ~ c n, most connected components are still trees or unicyclic pieces."
        )
    )
    parser.add_argument("--n", type=int, default=DEFAULT_N, help="Number of nodes.")
    parser.add_argument("--c", type=float, default=DEFAULT_C, help="Linear-scale constant in m = round(c n), should stay below 0.5.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help="Frames per second.")
    parser.add_argument("--event-pause-frames", type=int, default=DEFAULT_EVENT_PAUSE_FRAMES, help="Extra pause frames when a unicyclic or multicyclic component first appears or when the linear benchmark is reached.")
    parser.add_argument("--hold-final-frames", type=int, default=DEFAULT_HOLD_FINAL_FRAMES, help="Still frames at the end.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output animation path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    c = min(max(args.c, 0.01), 0.49)
    n = max(2, args.n)
    saved_to = make_phase2_2a_animation(
        n=n,
        c=c,
        seed=args.seed,
        fps=max(1, args.fps),
        event_pause_frames=max(0, args.event_pause_frames),
        hold_final_frames=max(0, args.hold_final_frames),
        output_path=args.output,
    )
    print(f"n = {n}")
    print(f"c = {c:.3f}")
    print(f"linear benchmark round(c n) = {int(round(c * n))}")
    print(f"saved animation to: {saved_to}")


if __name__ == "__main__":
    main()
