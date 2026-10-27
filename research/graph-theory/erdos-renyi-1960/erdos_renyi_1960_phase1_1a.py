#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter


# ============================================================
# Forest-viewpoint animation for Phase 1, subsection 1.1A.
# Default output matches the HTML placeholder:
#     erdos_renyi_1960_phase1_1a.mp4
# ============================================================
DEFAULT_N = 100
DEFAULT_ALPHA = 0.5               # N(n) = floor(n^alpha), still inside Phase 1 when alpha < 1
DEFAULT_SEED = 12345
DEFAULT_FPS = 4
DEFAULT_FRAMES_PER_EDGE = 3
DEFAULT_HOLD_FINAL_FRAMES = 18
DEFAULT_OUTPUT = Path("./erdos_renyi_1960_phase1_1a.mp4")


@dataclass(frozen=True)
class Phase1Event:
    edge_index: int
    label: str


@dataclass(frozen=True)
class ForestSummary:
    is_forest: bool
    components: int
    isolates: int
    largest_component: int
    tree_components: int
    cyclic_components: int


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


def summarize_forest_state(g: nx.Graph) -> ForestSummary:
    components = list(nx.connected_components(g))
    isolates = sum(1 for _, degree in g.degree() if degree == 0)
    largest_component = max((len(c) for c in components), default=0)

    tree_components = 0
    cyclic_components = 0
    for comp_nodes in components:
        h = g.subgraph(comp_nodes)
        if h.number_of_edges() == h.number_of_nodes() - 1:
            tree_components += 1
        else:
            cyclic_components += 1

    return ForestSummary(
        is_forest=(cyclic_components == 0),
        components=len(components),
        isolates=isolates,
        largest_component=largest_component,
        tree_components=tree_components,
        cyclic_components=cyclic_components,
    )


def collect_events(nodes: List[int], edges_in_order: List[Tuple[int, int]]) -> List[Phase1Event]:
    g = nx.Graph()
    g.add_nodes_from(nodes)
    events: List[Phase1Event] = []

    first_nontrivial = False
    size_three_component = False
    first_cycle = False

    for idx, edge in enumerate(edges_in_order, start=1):
        g.add_edge(*edge)
        summary = summarize_forest_state(g)

        if not first_nontrivial and g.number_of_edges() >= 1:
            first_nontrivial = True
            events.append(Phase1Event(idx, "The empty graph begins to form tiny tree components"))

        if not size_three_component and summary.largest_component >= 3:
            size_three_component = True
            events.append(Phase1Event(idx, f"A tree component reaches size {summary.largest_component}"))

        if not first_cycle and not summary.is_forest:
            first_cycle = True
            events.append(Phase1Event(idx, "A cycle appears: this run leaves the pure-forest picture"))

    if not first_cycle:
        events.append(Phase1Event(len(edges_in_order), "No cycles appear up to the chosen Phase 1 edge budget"))

    return events


def latest_event_text(edge_count: int, events: List[Phase1Event], linger_edges: int) -> str:
    active = [e for e in events if e.edge_index <= edge_count < e.edge_index + linger_edges]
    if not active:
        return ""
    return active[-1].label


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


def make_phase1_forest_animation(
    n: int,
    alpha: float,
    seed: int,
    fps: int,
    frames_per_edge: int,
    hold_final_frames: int,
    output_path: Path,
) -> Path:
    rng = np.random.default_rng(seed)
    nodes = list(range(n))
    positions = layered_circular_layout(n=n, rng=rng)

    max_edges = max(1, int(n ** alpha))
    edge_order = ordered_random_edges(nodes=nodes, rng=rng)[:max_edges]
    events = collect_events(nodes=nodes, edges_in_order=edge_order)
    total_possible_edges = n * (n - 1) // 2

    frame_schedule: List[int] = [0]
    for edge_count in range(1, max_edges + 1):
        frame_schedule.extend([edge_count] * frames_per_edge)
    frame_schedule.extend([max_edges] * hold_final_frames)

    fig = plt.figure(figsize=(10.0, 9.0))
    graph_ax = fig.add_axes([0.05, 0.20, 0.90, 0.73])
    info_ax = fig.add_axes([0.05, 0.05, 0.90, 0.11])
    info_ax.axis("off")
    fig.patch.set_facecolor("white")

    line1 = info_ax.text(
        0.5, 0.78, "", ha="center", va="center", fontsize=10,
        bbox=dict(facecolor="white", edgecolor="0.85", boxstyle="round,pad=0.25"),
    )
    line2 = info_ax.text(
        0.5, 0.43, "", ha="center", va="center", fontsize=10,
        bbox=dict(facecolor="white", edgecolor="0.85", boxstyle="round,pad=0.25"),
    )
    line3 = info_ax.text(
        0.5, 0.10, "", ha="center", va="center", fontsize=10.5,
        bbox=dict(facecolor="#f9f9f9", edgecolor="0.75", boxstyle="round,pad=0.30"),
    )

    def update(frame_index: int) -> None:
        edge_count = frame_schedule[frame_index]
        g = build_graph_state(nodes=nodes, edges_in_order=edge_order, edge_count=edge_count)
        summary = summarize_forest_state(g)

        graph_ax.clear()
        non_isolates = [u for u, degree in g.degree() if degree > 0]
        isolates = [u for u, degree in g.degree() if degree == 0]

        nx.draw_networkx_edges(
            g,
            pos=positions,
            ax=graph_ax,
            width=1.15,
            alpha=0.75,
            edge_color="#4a5568",
        )
        nx.draw_networkx_nodes(
            g,
            pos=positions,
            nodelist=isolates,
            ax=graph_ax,
            node_size=34 if n >= 100 else 70,
            linewidths=0.35,
            edgecolors="0.30",
            node_color="white",
        )
        nx.draw_networkx_nodes(
            g,
            pos=positions,
            nodelist=non_isolates,
            ax=graph_ax,
            node_size=50 if n >= 100 else 90,
            linewidths=0.50,
            edgecolors="#0d3b66",
            node_color="#b9dcff" if summary.is_forest else "#ffd4d4",
        )

        graph_ax.set_xlim(-1.12, 1.12)
        graph_ax.set_ylim(-1.12, 1.12)
        graph_ax.set_aspect("equal")
        graph_ax.axis("off")
        graph_ax.set_title(f"Phase 1.1A — Forest viewpoint for G(n,m) with n = {n}", fontsize=16, pad=18)
        graph_ax.text(
            0.5,
            1.008,
            f"Using m = floor(n^{alpha:.2f}) = {max_edges}, which stays inside the sparse regime N(n) = o(n).",
            transform=graph_ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=10.3,
            color="0.22",
        )

        avg_degree = 2.0 * edge_count / n
        density = 2.0 * edge_count / total_possible_edges if total_possible_edges > 0 else 0.0
        line1.set_text(
            f"edges shown = {edge_count}/{max_edges}   |   total possible = {total_possible_edges}   |   density = {density:.5f}   |   average degree = {avg_degree:.3f}"
        )
        line2.set_text(
            f"components = {summary.components}   |   isolates = {summary.isolates}   |   largest component = {summary.largest_component}   |   tree components = {summary.tree_components}   |   cyclic components = {summary.cyclic_components}"
        )

        event_text = latest_event_text(edge_count, events, linger_edges=3)
        forest_text = "YES" if summary.is_forest else "NO"
        line3.set_text(f"All connected components are trees right now? {forest_text}. {event_text}".strip())
        line3.set_bbox(
            dict(
                facecolor="#e7f6e7" if summary.is_forest else "#ffe5e5",
                edgecolor="#2e7d32" if summary.is_forest else "#b71c1c",
                boxstyle="round,pad=0.30",
            )
        )

    anim = FuncAnimation(fig, update, frames=len(frame_schedule), interval=1000 / fps, repeat=False)
    saved_to = save_animation(anim=anim, output_path=output_path, fps=fps)
    plt.close(fig)
    return saved_to


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create the Phase 1.1A animation showing that when m = floor(n^alpha) with alpha < 1, "
            "the random graph is typically still a forest or overwhelmingly tree-like."
        )
    )
    parser.add_argument("--n", type=int, default=DEFAULT_N, help="Number of nodes.")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help="Exponent in m = floor(n^alpha), should stay below 1.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help="Frames per second.")
    parser.add_argument("--frames-per-edge", type=int, default=DEFAULT_FRAMES_PER_EDGE, help="Frames to hold each edge addition.")
    parser.add_argument("--hold-final-frames", type=int, default=DEFAULT_HOLD_FINAL_FRAMES, help="Still frames at the end.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output animation path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    alpha = min(max(args.alpha, 0.05), 0.95)

    saved_to = make_phase1_forest_animation(
        n=max(2, args.n),
        alpha=alpha,
        seed=args.seed,
        fps=max(1, args.fps),
        frames_per_edge=max(1, args.frames_per_edge),
        hold_final_frames=max(0, args.hold_final_frames),
        output_path=args.output,
    )
    print(f"n = {max(2, args.n)}")
    print(f"alpha = {alpha:.3f}")
    print(f"m = floor(n^alpha) = {int(max(2, args.n) ** alpha)}")
    print(f"saved animation to: {saved_to}")


if __name__ == "__main__":
    main()
