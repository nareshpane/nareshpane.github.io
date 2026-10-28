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
class GiantComponentEvent:
    edge_index: int
    label: str


def layered_circular_layout(
    n: int,
    rng: np.random.Generator,
    outer_radius: float = 0.96,
    inner_radius: float = 0.55,
) -> Dict[int, Tuple[float, float]]:
    """Mostly circular shell with several nodes scattered inside."""
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
        if all((x - px) ** 2 + (y - py) ** 2 >= min_dist ** 2 for px, py in points):
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


def giant_component_nodes(g: nx.Graph) -> set[int]:
    components = list(nx.connected_components(g))
    if not components:
        return set()
    return set(max(components, key=len))


def giant_component_edge_lists(
    edges_present: List[Tuple[int, int]],
    giant_nodes: set[int],
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    inside: List[Tuple[int, int]] = []
    outside: List[Tuple[int, int]] = []
    for u, v in edges_present:
        if u in giant_nodes and v in giant_nodes:
            inside.append((u, v))
        else:
            outside.append((u, v))
    return inside, outside


def collect_events(
    nodes: List[int],
    edges_in_order: List[Tuple[int, int]],
    giant_fraction: float,
) -> Tuple[List[GiantComponentEvent], int]:
    g = nx.Graph()
    g.add_nodes_from(nodes)
    events: List[GiantComponentEvent] = []

    giant_threshold = math.ceil(giant_fraction * len(nodes))
    first_giant_edge = len(edges_in_order)

    first_cycle = False
    crossed_quarter = False
    crossed_half = False
    first_giant = False

    for idx, edge in enumerate(edges_in_order, start=1):
        g.add_edge(*edge)
        largest_size = max((len(comp) for comp in nx.connected_components(g)), default=1)

        if not first_cycle and g.number_of_edges() >= g.number_of_nodes() and len(nx.cycle_basis(g)) > 0:
            first_cycle = True
            events.append(GiantComponentEvent(idx, "First cycle appears"))

        if not crossed_quarter and largest_size >= math.ceil(0.25 * len(nodes)):
            crossed_quarter = True
            events.append(GiantComponentEvent(idx, f"Largest component reaches {largest_size} vertices"))

        if not crossed_half and largest_size >= math.ceil(0.50 * len(nodes)):
            crossed_half = True
            events.append(GiantComponentEvent(idx, f"Component exceeds half the graph (size {largest_size})"))

        if not first_giant and largest_size >= giant_threshold:
            first_giant = True
            first_giant_edge = idx
            events.append(
                GiantComponentEvent(
                    idx,
                    f"Giant component emerges at size {largest_size} (threshold {giant_threshold})",
                )
            )
            break

    return events, first_giant_edge


def latest_event_text(edge_count: int, events: List[GiantComponentEvent], linger_edges: int) -> str:
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


def make_giant_component_animation(
    n: int,
    seed: int,
    fps: int,
    frames_per_edge: int,
    hold_final_frames: int,
    pause_frames: int,
    giant_fraction: float,
    output_path: Path,
) -> Path:
    rng = np.random.default_rng(seed)
    nodes = list(range(n))
    positions = layered_circular_layout(n=n, rng=rng)
    full_edge_order = ordered_random_edges(nodes=nodes, rng=rng)

    events, giant_edge_index = collect_events(
        nodes=nodes,
        edges_in_order=full_edge_order,
        giant_fraction=giant_fraction,
    )

    edge_order = full_edge_order[:giant_edge_index]
    total_possible_edges = n * (n - 1) // 2
    giant_threshold = math.ceil(giant_fraction * n)

    pause_start_frame = len(edge_order) * frames_per_edge
    total_frames = pause_start_frame + pause_frames + hold_final_frames + 1

    fig = plt.figure(figsize=(10.0, 9.2))
    graph_ax = fig.add_axes([0.05, 0.19, 0.90, 0.74])
    caption_ax = fig.add_axes([0.05, 0.04, 0.90, 0.12])
    caption_ax.axis("off")
    fig.patch.set_facecolor("white")

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

    def update(frame_index: int) -> None:
        graph_ax.clear()

        if frame_index <= pause_start_frame:
            edge_count = min(frame_index // frames_per_edge, len(edge_order))
        else:
            edge_count = len(edge_order)

        g = build_graph_state(nodes=nodes, edges_in_order=edge_order, edge_count=edge_count)
        giant_nodes = giant_component_nodes(g)
        giant_size = len(giant_nodes)
        component_sizes = sorted((len(comp) for comp in nx.connected_components(g)), reverse=True)
        second_size = component_sizes[1] if len(component_sizes) > 1 else 0
        isolates = sum(1 for _, deg in g.degree() if deg == 0)
        avg_degree = 0.0 if n == 0 else 2.0 * g.number_of_edges() / n
        density = 0.0 if n <= 1 else 2.0 * g.number_of_edges() / (n * (n - 1))

        shown_edges = edge_order[:edge_count]
        giant_edges, other_edges = giant_component_edge_lists(shown_edges, giant_nodes)

        newest_edge = edge_order[edge_count - 1 : edge_count] if edge_count > 0 else []
        newest_is_giant = bool(newest_edge and newest_edge[0] in giant_edges)

        nx.draw_networkx_edges(
            g,
            pos=positions,
            edgelist=other_edges,
            ax=graph_ax,
            width=0.9,
            alpha=0.32,
            edge_color="0.72",
        )
        nx.draw_networkx_edges(
            g,
            pos=positions,
            edgelist=giant_edges,
            ax=graph_ax,
            width=1.9,
            alpha=0.78,
            edge_color="#2a6fbb",
        )
        if newest_edge:
            nx.draw_networkx_edges(
                g,
                pos=positions,
                edgelist=newest_edge,
                ax=graph_ax,
                width=2.8,
                alpha=0.98,
                edge_color="#0a0a0a" if newest_is_giant else "#444444",
            )

        non_giant_nodes = [u for u in g.nodes if u not in giant_nodes]
        nx.draw_networkx_nodes(
            g,
            pos=positions,
            nodelist=non_giant_nodes,
            ax=graph_ax,
            node_size=52 if n >= 100 else 120,
            linewidths=0.6,
            edgecolors="0.25",
            node_color="white",
        )
        nx.draw_networkx_nodes(
            g,
            pos=positions,
            nodelist=sorted(giant_nodes),
            ax=graph_ax,
            node_size=72 if n >= 100 else 145,
            linewidths=0.8,
            edgecolors="#0d3b66",
            node_color="#8ec5ff",
        )

        graph_ax.set_xlim(-1.12, 1.12)
        graph_ax.set_ylim(-1.12, 1.12)
        graph_ax.set_aspect("equal")
        graph_ax.axis("off")
        graph_ax.set_title(
            f"Evolution of the Giant Component in a Random Graph (n = {n})",
            fontsize=16,
            pad=18,
        )
        graph_ax.text(
            0.5,
            1.008,
            f"Tracking the current largest connected component until it first reaches {giant_threshold} vertices.",
            transform=graph_ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=10.3,
            color="0.22",
        )

        info_line_1.set_text(
            f"edges shown = {g.number_of_edges():03d}/{len(edge_order):03d}   |   "
            f"possible total = {total_possible_edges}   |   density = {density:.4f}   |   avg degree = {avg_degree:.2f}"
        )
        info_line_2.set_text(
            f"largest component = {giant_size}   |   second largest = {second_size}   |   "
            f"isolates = {isolates}   |   threshold = {giant_threshold}   |   giant achieved = {'yes' if giant_size >= giant_threshold else 'no'}"
        )

        event_text = latest_event_text(edge_count, events, linger_edges=3)
        if frame_index >= pause_start_frame:
            event_text = f"Giant component locked in at size {giant_size} after {len(edge_order)} edges"
        event_line.set_text(f"Emergence: {event_text}" if event_text else " ")

    anim = FuncAnimation(fig, update, frames=total_frames, interval=1000 / fps, repeat=False)
    saved_to = save_animation(anim=anim, output_path=output_path, fps=fps)
    plt.close(fig)
    return saved_to


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create an MP4 animation that follows the evolution of the largest connected "
            "component and highlights the first emergence of a giant component."
        )
    )
    parser.add_argument("--n", type=int, default=100, help="Number of nodes.")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed.")
    parser.add_argument("--giant-fraction", type=float, default=0.50, help="Fraction of vertices used to define the giant component threshold.")
    parser.add_argument("--fps", type=int, default=4, help="Frames per second.")
    parser.add_argument("--frames-per-edge", type=int, default=3, help="Frames to hold each added edge.")
    parser.add_argument("--pause-frames", type=int, default=18, help="Extra frames to pause when the giant component first appears.")
    parser.add_argument("--hold-final-frames", type=int, default=18, help="Still frames after the pause.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./giant_component_evolution_n100.mp4"),
        help="Output animation path. Use .mp4 or .gif.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    saved_to = make_giant_component_animation(
        n=max(2, args.n),
        seed=args.seed,
        fps=max(1, args.fps),
        frames_per_edge=max(1, args.frames_per_edge),
        hold_final_frames=max(0, args.hold_final_frames),
        pause_frames=max(0, args.pause_frames),
        giant_fraction=min(max(args.giant_fraction, 0.05), 0.95),
        output_path=args.output,
    )
    print(f"Saved animation to: {saved_to}")


if __name__ == "__main__":
    main()
