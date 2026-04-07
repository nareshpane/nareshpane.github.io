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


@dataclass(frozen=True)
class EmergenceEvent:
    frame: int
    label: str


def layered_circular_layout(
    n: int,
    rng: np.random.Generator,
    outer_radius: float = 0.92,
    inner_radius: float = 0.52,
) -> Dict[int, Tuple[float, float]]:
    """Create a mostly circular layout with several nodes clearly placed inside.

    The outer shell is intentionally irregular so the picture is not a perfect circle,
    while the inner nodes are scattered inside the boundary.
    """
    if n <= 0:
        return {}

    if n <= 6:
        outer_count = max(3, n - 1)
    else:
        outer_count = max(8, int(round(0.6 * n)))
        outer_count = min(outer_count, n - 2)

    inner_count = n - outer_count
    points: List[Tuple[float, float]] = []

    angle_offset = rng.uniform(0.0, 2.0 * math.pi)
    outer_angles = np.linspace(0.0, 2.0 * math.pi, outer_count, endpoint=False) + angle_offset
    outer_angles += rng.normal(0.0, 0.08, size=outer_count)

    for theta in outer_angles:
        r = outer_radius * (1.0 + rng.normal(0.0, 0.05))
        r = float(np.clip(r, 0.78, 0.98))
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        points.append((x, y))

    min_dist = 0.18
    attempts = 0
    while len(points) < n and attempts < 20_000:
        attempts += 1
        theta = rng.uniform(0.0, 2.0 * math.pi)
        r = inner_radius * math.sqrt(rng.uniform(0.0, 1.0))
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        if all((x - px) ** 2 + (y - py) ** 2 >= min_dist ** 2 for px, py in points):
            points.append((x, y))

    while len(points) < n:
        theta = rng.uniform(0.0, 2.0 * math.pi)
        r = inner_radius * math.sqrt(rng.uniform(0.0, 1.0))
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        points.append((x, y))

    return {i + 1: points[i] for i in range(n)}



def ordered_random_edges(nodes: List[int], rng: np.random.Generator) -> List[Tuple[int, int]]:
    edges = [(u, v) for i, u in enumerate(nodes) for v in nodes[i + 1 :]]
    rng.shuffle(edges)
    return edges



def has_cycle(g: nx.Graph) -> bool:
    return g.number_of_edges() >= g.number_of_nodes() and len(nx.cycle_basis(g)) > 0



def collect_emergence_events(nodes: List[int], edges_in_order: List[Tuple[int, int]]) -> List[EmergenceEvent]:
    g = nx.Graph()
    g.add_nodes_from(nodes)
    events: List[EmergenceEvent] = []

    first_cycle_at = None
    giant_at = None
    no_isolates_at = None
    connected_at = None

    giant_threshold = math.ceil(len(nodes) / 2)

    for idx, edge in enumerate(edges_in_order, start=1):
        g.add_edge(*edge)

        if first_cycle_at is None and has_cycle(g):
            first_cycle_at = idx
            events.append(EmergenceEvent(idx, "First cycle appears"))

        largest_component = max((len(comp) for comp in nx.connected_components(g)), default=1)
        if giant_at is None and largest_component >= giant_threshold:
            giant_at = idx
            events.append(EmergenceEvent(idx, f"Giant component emerges (size {largest_component})"))

        isolates = sum(1 for _, deg in g.degree() if deg == 0)
        if no_isolates_at is None and isolates == 0:
            no_isolates_at = idx
            events.append(EmergenceEvent(idx, "No isolated vertices remain"))

        if connected_at is None and nx.is_connected(g):
            connected_at = idx
            events.append(EmergenceEvent(idx, "Graph becomes connected"))
            break

    return events



def latest_event_text(edge_count: int, events: List[EmergenceEvent], linger_edges: int) -> str:
    candidates = [e for e in events if e.frame <= edge_count < e.frame + linger_edges]
    if not candidates:
        return ""
    return candidates[-1].label



def build_graph_state(nodes: List[int], edges_in_order: List[Tuple[int, int]], edge_count: int) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges_in_order[:edge_count])
    return g



def save_animation(anim: FuncAnimation, output_path: Path, fps: int) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() == ".mp4":
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is not None:
            writer = FFMpegWriter(
                fps=fps,
                codec="libx264",
                bitrate=2600,
                extra_args=["-pix_fmt", "yuv420p"],
            )
            anim.save(output_path, writer=writer, dpi=180)
            return output_path

        gif_path = output_path.with_suffix(".gif")
        anim.save(gif_path, writer=PillowWriter(fps=fps), dpi=120)
        return gif_path

    anim.save(output_path, writer=PillowWriter(fps=fps), dpi=120)
    return output_path



def make_animation(
    n: int,
    max_edges: int,
    seed: int,
    fps: int,
    hold_final_frames: int,
    frames_per_edge: int,
    output_path: Path,
) -> Path:
    rng = np.random.default_rng(seed)
    nodes = list(range(1, n + 1))
    positions = layered_circular_layout(n=n, rng=rng)
    edge_order = ordered_random_edges(nodes=nodes, rng=rng)[:max_edges]
    events = collect_emergence_events(nodes=nodes, edges_in_order=edge_order)
    total_possible_edges = n * (n - 1) // 2

    fig = plt.figure(figsize=(9.0, 9.1))
    graph_ax = fig.add_axes([0.05, 0.20, 0.90, 0.73])
    caption_ax = fig.add_axes([0.05, 0.04, 0.90, 0.12])
    fig.patch.set_facecolor("white")
    caption_ax.axis("off")

    info_line_1 = caption_ax.text(
        0.5,
        0.78,
        "",
        ha="center",
        va="center",
        fontsize=10,
        bbox=dict(facecolor="white", edgecolor="0.85", boxstyle="round,pad=0.25"),
    )
    info_line_2 = caption_ax.text(
        0.5,
        0.45,
        "",
        ha="center",
        va="center",
        fontsize=10,
        bbox=dict(facecolor="white", edgecolor="0.85", boxstyle="round,pad=0.25"),
    )
    event_line = caption_ax.text(
        0.5,
        0.12,
        "",
        ha="center",
        va="center",
        fontsize=11,
        bbox=dict(facecolor="white", edgecolor="0.70", boxstyle="round,pad=0.32"),
    )

    total_frames = len(edge_order) * frames_per_edge + 1 + hold_final_frames

    def update(frame_index: int) -> None:
        graph_ax.clear()

        edge_count = min(frame_index // frames_per_edge, len(edge_order))
        g = build_graph_state(nodes=nodes, edges_in_order=edge_order, edge_count=edge_count)

        component_sizes = sorted((len(c) for c in nx.connected_components(g)), reverse=True)
        largest_component = component_sizes[0] if component_sizes else 1
        isolates = sum(1 for _, deg in g.degree() if deg == 0)
        avg_degree = 0.0 if n == 0 else 2.0 * g.number_of_edges() / n
        density = 0.0 if n <= 1 else 2.0 * g.number_of_edges() / (n * (n - 1))
        is_connected = nx.is_connected(g) if g.number_of_edges() > 0 else False

        previous_edges = edge_order[: max(0, edge_count - 1)]
        newest_edge = edge_order[edge_count - 1 : edge_count] if edge_count > 0 else []

        nx.draw_networkx_edges(
            g,
            pos=positions,
            edgelist=previous_edges,
            ax=graph_ax,
            width=1.35,
            alpha=0.55,
            edge_color="0.45",
        )
        nx.draw_networkx_edges(
            g,
            pos=positions,
            edgelist=newest_edge,
            ax=graph_ax,
            width=2.8,
            alpha=0.95,
            edge_color="black",
        )
        nx.draw_networkx_nodes(
            g,
            pos=positions,
            ax=graph_ax,
            node_size=430,
            linewidths=1.15,
            edgecolors="black",
            node_color="white",
        )
        nx.draw_networkx_labels(g, pos=positions, ax=graph_ax, font_size=8.5)

        graph_ax.set_xlim(-1.16, 1.16)
        graph_ax.set_ylim(-1.16, 1.16)
        graph_ax.set_aspect("equal")
        graph_ax.axis("off")
        graph_ax.set_title(f"Evolution of a Random Graph on {n} Nodes", fontsize=16, pad=12)

        info_line_1.set_text(
            f"nodes = {n}   |   edges shown = {g.number_of_edges():02d}/{len(edge_order):02d}   |   "
            f"possible total = {total_possible_edges}   |   density = {density:.3f}"
        )
        info_line_2.set_text(
            f"avg degree = {avg_degree:.2f}   |   components = {nx.number_connected_components(g)}   |   "
            f"largest component = {largest_component}   |   isolates = {isolates}   |   connected = {'yes' if is_connected else 'no'}"
        )

        event_text = latest_event_text(edge_count, events, linger_edges=2)
        event_line.set_text(f"Emergence: {event_text}" if event_text else " ")

    anim = FuncAnimation(fig, update, frames=total_frames, interval=1000 / fps, repeat=False)
    saved_to = save_animation(anim=anim, output_path=output_path, fps=fps)
    plt.close(fig)
    return saved_to



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create an MP4 animation showing the evolution of a random graph on 20 nodes "
            "with an irregular outer shell and several interior nodes."
        )
    )
    parser.add_argument("--n", type=int, default=20, help="Number of nodes.")
    parser.add_argument(
        "--max-edges",
        type=int,
        default=40,
        help=(
            "How many random edges to reveal in the animation. "
            "For n nodes, the complete graph has n(n-1)/2 possible edges."
        ),
    )
    parser.add_argument("--seed", type=int, default=12345, help="Random seed.")
    parser.add_argument(
        "--fps",
        type=int,
        default=4,
        help="Frames per second. Default chosen to make the animation 1.5x slower than v2.",
    )
    parser.add_argument(
        "--frames-per-edge",
        type=int,
        default=3,
        help="How many frames each newly added edge should remain on screen before the next one appears.",
    )
    parser.add_argument(
        "--hold-final-frames",
        type=int,
        default=18,
        help="Extra still frames at the end so the final state lingers.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./random_graph_20_nodes_evolution_v3.mp4"),
        help="Output animation path. Use .mp4 or .gif.",
    )
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    max_possible_edges = args.n * (args.n - 1) // 2
    max_edges = max(1, min(args.max_edges, max_possible_edges))
    frames_per_edge = max(1, args.frames_per_edge)

    print(f"Using nodes labeled 1 through {args.n}.")
    print(f"Maximum possible edges for n={args.n}: {max_possible_edges} = {args.n}*({args.n}-1)/2")
    print(f"Animating {max_edges} revealed edges.")

    saved_to = make_animation(
        n=args.n,
        max_edges=max_edges,
        seed=args.seed,
        fps=args.fps,
        hold_final_frames=args.hold_final_frames,
        frames_per_edge=frames_per_edge,
        output_path=args.output,
    )
    print(f"Saved animation to: {saved_to}")


if __name__ == "__main__":
    main()
