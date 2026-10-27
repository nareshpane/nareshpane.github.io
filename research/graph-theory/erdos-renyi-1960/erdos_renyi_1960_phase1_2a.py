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
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea


# ============================================================
# Phase 1.2A animation.
# Default output matches the HTML placeholder:
#     erdos_renyi_1960_phase1_2a.mp4
#
# Idea:
#   For one fixed n, grow G(n,m) up to the k=5 threshold scale m ~ n^(3/4).
#   Track when tree-components of orders 3, 4, and 5 first become visible.
#
# Current styling revision:
#   - T3 highlighted in red
#   - T4 highlighted in blue
#   - T5 highlighted in green
#   - legend moved below the graph so it does not overlap nodes
#   - the dynamic "edges shown = .../..." portion is color-highlighted
#   - bottom caption compares predicted thresholds with actual first emergence
# ============================================================
DEFAULT_N = 1000
DEFAULT_SEED = 24680
DEFAULT_FPS = 4
DEFAULT_FRAMES_PER_EDGE = 2
DEFAULT_HOLD_FINAL_FRAMES = 20
DEFAULT_OUTPUT = Path("./erdos_renyi_1960_phase1_2a.mp4")

ORDER_COLORS = {
    3: "#c62828",  # red
    4: "#1565c0",  # blue
    5: "#2e7d32",  # green
}

ORDER_NODE_COLORS = {
    3: "#ffd9d9",
    4: "#d9ebff",
    5: "#dff4df",
}

ORDER_LABELS = {
    3: "T3 tree-components",
    4: "T4 tree-components",
    5: "T5 tree-components",
}


@dataclass(frozen=True)
class ThresholdEvent:
    edge_index: int
    label: str


@dataclass(frozen=True)
class TreeOrderSummary:
    counts: Dict[int, int]
    highlighted_nodes: Dict[int, List[int]]
    highlighted_edges: Dict[int, List[Tuple[int, int]]]
    largest_component: int
    components: int
    isolates: int


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


def summarize_tree_orders(g: nx.Graph, tracked_orders: List[int]) -> TreeOrderSummary:
    counts = {k: 0 for k in tracked_orders}
    highlighted_nodes = {k: [] for k in tracked_orders}
    highlighted_edges = {k: [] for k in tracked_orders}

    components = list(nx.connected_components(g))
    isolates = sum(1 for _, degree in g.degree() if degree == 0)
    largest_component = max((len(comp) for comp in components), default=0)

    for comp_nodes in components:
        h = g.subgraph(comp_nodes)
        v = h.number_of_nodes()
        e = h.number_of_edges()
        if v in counts and e == v - 1:
            counts[v] += 1
            highlighted_nodes[v].extend(list(comp_nodes))
            highlighted_edges[v].extend(list(h.edges()))

    return TreeOrderSummary(
        counts=counts,
        highlighted_nodes=highlighted_nodes,
        highlighted_edges=highlighted_edges,
        largest_component=largest_component,
        components=len(components),
        isolates=isolates,
    )


def threshold_scales(n: int) -> Dict[int, int]:
    return {
        3: max(1, int(round(n ** 0.5))),
        4: max(1, int(round(n ** (2.0 / 3.0)))),
        5: max(1, int(round(n ** 0.75))),
    }


def collect_events(
    nodes: List[int],
    edges_in_order: List[Tuple[int, int]],
    scales: Dict[int, int],
) -> Tuple[List[ThresholdEvent], Dict[int, int | None]]:
    g = nx.Graph()
    g.add_nodes_from(nodes)
    events: List[ThresholdEvent] = []
    seen_first = {3: False, 4: False, 5: False}
    crossed_scale = {3: False, 4: False, 5: False}
    first_appearance: Dict[int, int | None] = {3: None, 4: None, 5: None}

    max_edges = len(edges_in_order)
    for idx, edge in enumerate(edges_in_order, start=1):
        g.add_edge(*edge)
        summary = summarize_tree_orders(g, tracked_orders=[3, 4, 5])

        for k in [3, 4, 5]:
            if not crossed_scale[k] and idx >= scales[k]:
                crossed_scale[k] = True
                events.append(
                    ThresholdEvent(
                        idx,
                        f"Crossed the nominal T{k} threshold scale at about m ≈ {scales[k]:,}",
                    )
                )
            if not seen_first[k] and summary.counts[k] > 0:
                seen_first[k] = True
                first_appearance[k] = idx
                events.append(
                    ThresholdEvent(
                        idx,
                        f"First visible T{k} tree-component appears at m = {idx:,}",
                    )
                )

    events.append(ThresholdEvent(max_edges, f"Stopped at the T5 threshold scale: m = {max_edges:,}"))
    return events, first_appearance


def latest_event_text(edge_count: int, events: List[ThresholdEvent], linger_edges: int) -> str:
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
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    "-vf",
                    "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                ],
            )
            anim.save(output_path, writer=writer, dpi=dpi)
            return output_path

        gif_path = output_path.with_suffix(".gif")
        anim.save(gif_path, writer=PillowWriter(fps=fps), dpi=120)
        return gif_path

    anim.save(output_path, writer=PillowWriter(fps=fps), dpi=120)
    return output_path


def emergence_text(scales: Dict[int, int], first_appearance: Dict[int, int | None]) -> str:
    pieces = []
    for k in [3, 4, 5]:
        actual = first_appearance[k]
        actual_text = f"{actual:,}" if actual is not None else "not yet seen"
        pieces.append(f"T{k}: predicted {scales[k]:,}, actual {actual_text}")
    return "   |   ".join(pieces)


def make_legend_handles() -> List[Line2D]:
    handles = [
        Line2D(
            [0],
            [0],
            color="0.55",
            lw=2.0,
            marker="o",
            markersize=7,
            markerfacecolor="white",
            markeredgecolor="0.35",
            label="Other nodes / edges",
        )
    ]
    for k in [3, 4, 5]:
        handles.append(
            Line2D(
                [0],
                [0],
                color=ORDER_COLORS[k],
                lw=2.6,
                marker="o",
                markersize=7,
                markerfacecolor=ORDER_NODE_COLORS[k],
                markeredgecolor=ORDER_COLORS[k],
                label=ORDER_LABELS[k],
            )
        )
    return handles


def add_colored_edgecount_line(
    info_ax,
    edge_count: int,
    max_edges: int,
    avg_degree: float,
    density: float,
    largest_component: int,
) -> None:
    left = TextArea(
        "edges shown = ",
        textprops=dict(color="#111111", fontsize=10),
    )
    middle = TextArea(
        f"{edge_count:,}/{max_edges:,}",
        textprops=dict(color="#c62828", fontsize=10, fontweight="bold"),
    )
    right = TextArea(
        f"   |   avg degree = {avg_degree:.3f}   |   density = {density:.6f}   |   largest component = {largest_component}",
        textprops=dict(color="#111111", fontsize=10),
    )
    packed = HPacker(children=[left, middle, right], align="center", pad=0, sep=0)
    anchored = AnchoredOffsetbox(
        loc="center",
        child=packed,
        pad=0.22,
        frameon=True,
        bbox_to_anchor=(0.5, 0.82),
        bbox_transform=info_ax.transAxes,
        borderpad=0.32,
    )
    anchored.patch.set_facecolor("white")
    anchored.patch.set_edgecolor("0.85")
    info_ax.add_artist(anchored)


def make_phase1_2a_animation(
    n: int,
    seed: int,
    fps: int,
    frames_per_edge: int,
    hold_final_frames: int,
    output_path: Path,
) -> Path:
    rng = np.random.default_rng(seed)
    nodes = list(range(n))
    positions = layered_circular_layout(n=n, rng=rng)
    scales = threshold_scales(n)

    max_edges = scales[5]
    edge_order = ordered_random_edges(nodes=nodes, rng=rng)[:max_edges]
    events, first_appearance = collect_events(nodes=nodes, edges_in_order=edge_order, scales=scales)
    total_possible_edges = n * (n - 1) // 2

    frame_schedule: List[int] = [0]
    for edge_count in range(1, max_edges + 1):
        frame_schedule.extend([edge_count] * frames_per_edge)
    frame_schedule.extend([max_edges] * hold_final_frames)

    fig = plt.figure(figsize=(10.8, 9.4))
    graph_ax = fig.add_axes([0.05, 0.30, 0.90, 0.58])
    info_ax = fig.add_axes([0.05, 0.05, 0.90, 0.18])
    info_ax.axis("off")
    fig.patch.set_facecolor("white")

    fig.legend(
        handles=make_legend_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.245),
        ncol=4,
        frameon=True,
        fontsize=9.8,
        handlelength=2.2,
        columnspacing=1.5,
    )

    def update(frame_index: int) -> None:
        edge_count = frame_schedule[frame_index]
        g = build_graph_state(nodes=nodes, edges_in_order=edge_order, edge_count=edge_count)
        summary = summarize_tree_orders(g, tracked_orders=[3, 4, 5])

        graph_ax.clear()
        info_ax.clear()
        info_ax.axis("off")

        nodes_k3 = set(summary.highlighted_nodes[3])
        nodes_k4 = set(summary.highlighted_nodes[4])
        nodes_k5 = set(summary.highlighted_nodes[5])
        highlighted_any = nodes_k3 | nodes_k4 | nodes_k5
        others = [u for u in g.nodes if u not in highlighted_any]

        edges_k3 = list({tuple(sorted(e)) for e in summary.highlighted_edges[3]})
        edges_k4 = list({tuple(sorted(e)) for e in summary.highlighted_edges[4]})
        edges_k5 = list({tuple(sorted(e)) for e in summary.highlighted_edges[5]})
        highlighted_edge_set = set(edges_k3) | set(edges_k4) | set(edges_k5)
        other_edges = [e for e in g.edges() if tuple(sorted(e)) not in highlighted_edge_set]

        nx.draw_networkx_edges(
            g,
            pos=positions,
            edgelist=other_edges,
            ax=graph_ax,
            width=1.00,
            alpha=0.55,
            edge_color="#6b7280",
        )
        if edges_k3:
            nx.draw_networkx_edges(
                g,
                pos=positions,
                edgelist=edges_k3,
                ax=graph_ax,
                width=2.35,
                alpha=0.92,
                edge_color=ORDER_COLORS[3],
            )
        if edges_k4:
            nx.draw_networkx_edges(
                g,
                pos=positions,
                edgelist=edges_k4,
                ax=graph_ax,
                width=2.55,
                alpha=0.92,
                edge_color=ORDER_COLORS[4],
            )
        if edges_k5:
            nx.draw_networkx_edges(
                g,
                pos=positions,
                edgelist=edges_k5,
                ax=graph_ax,
                width=2.75,
                alpha=0.92,
                edge_color=ORDER_COLORS[5],
            )

        nx.draw_networkx_nodes(
            g,
            pos=positions,
            nodelist=others,
            ax=graph_ax,
            node_size=20 if n >= 1000 else 42,
            linewidths=0.25,
            edgecolors="0.35",
            node_color="white",
        )
        if nodes_k3:
            nx.draw_networkx_nodes(
                g,
                pos=positions,
                nodelist=sorted(nodes_k3),
                ax=graph_ax,
                node_size=34 if n >= 1000 else 60,
                linewidths=0.55,
                edgecolors=ORDER_COLORS[3],
                node_color=ORDER_NODE_COLORS[3],
            )
        if nodes_k4:
            nx.draw_networkx_nodes(
                g,
                pos=positions,
                nodelist=sorted(nodes_k4),
                ax=graph_ax,
                node_size=40 if n >= 1000 else 68,
                linewidths=0.60,
                edgecolors=ORDER_COLORS[4],
                node_color=ORDER_NODE_COLORS[4],
            )
        if nodes_k5:
            nx.draw_networkx_nodes(
                g,
                pos=positions,
                nodelist=sorted(nodes_k5),
                ax=graph_ax,
                node_size=46 if n >= 1000 else 76,
                linewidths=0.68,
                edgecolors=ORDER_COLORS[5],
                node_color=ORDER_NODE_COLORS[5],
            )

        graph_ax.set_xlim(-1.12, 1.12)
        graph_ax.set_ylim(-1.12, 1.12)
        graph_ax.set_aspect("equal")
        graph_ax.axis("off")
        graph_ax.set_title(
            f"Phase 1.2A — first visible tree-orders in one sparse evolution (n = {n:,})",
            fontsize=16,
            pad=18,
        )
        graph_ax.text(
            0.5,
            1.01,
            f"Growing until m ≈ n^(3/4) = {max_edges:,}. The threshold benchmarks are n^(1/2), n^(2/3), and n^(3/4).",
            transform=graph_ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=10.3,
            color="0.22",
        )

        avg_degree = 2.0 * edge_count / n
        density = 2.0 * edge_count / total_possible_edges if total_possible_edges > 0 else 0.0
        add_colored_edgecount_line(
            info_ax=info_ax,
            edge_count=edge_count,
            max_edges=max_edges,
            avg_degree=avg_degree,
            density=density,
            largest_component=summary.largest_component,
        )

        info_ax.text(
            0.5,
            0.52,
            f"thresholds: T3 → {scales[3]:,}   |   T4 → {scales[4]:,}   |   T5 → {scales[5]:,}   ||   counts now: T3={summary.counts[3]}, T4={summary.counts[4]}, T5={summary.counts[5]}",
            transform=info_ax.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            bbox=dict(facecolor="white", edgecolor="0.85", boxstyle="round,pad=0.25"),
        )

        event_text = latest_event_text(edge_count, events, linger_edges=4)
        bottom_text = (
            f"Interpretation: larger tree-components should arrive later. {event_text}\n"
            f"Predicted vs actual first emergence: {emergence_text(scales, first_appearance)}"
        ).strip()
        info_ax.text(
            0.5,
            0.15,
            bottom_text,
            transform=info_ax.transAxes,
            ha="center",
            va="center",
            fontsize=9.5,
            linespacing=1.30,
            bbox=dict(facecolor="#f9f9f9", edgecolor="0.75", boxstyle="round,pad=0.30"),
        )

    anim = FuncAnimation(fig, update, frames=len(frame_schedule), interval=1000 / fps, repeat=False)
    saved_to = save_animation(anim=anim, output_path=output_path, fps=fps)
    plt.close(fig)
    return saved_to


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create the Phase 1.2A animation showing that 3-, 4-, and 5-vertex tree-components "
            "become visible at progressively larger threshold scales."
        )
    )
    parser.add_argument("--n", type=int, default=DEFAULT_N, help="Number of nodes.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help="Frames per second.")
    parser.add_argument("--frames-per-edge", type=int, default=DEFAULT_FRAMES_PER_EDGE, help="Frames to hold each edge addition.")
    parser.add_argument("--hold-final-frames", type=int, default=DEFAULT_HOLD_FINAL_FRAMES, help="Still frames at the end.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output animation path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    n = max(2, args.n)
    saved_to = make_phase1_2a_animation(
        n=n,
        seed=args.seed,
        fps=max(1, args.fps),
        frames_per_edge=max(1, args.frames_per_edge),
        hold_final_frames=max(0, args.hold_final_frames),
        output_path=args.output,
    )
    scales = threshold_scales(n)
    print(f"n = {n}")
    print(f"T3 threshold ≈ {scales[3]}")
    print(f"T4 threshold ≈ {scales[4]}")
    print(f"T5 threshold ≈ {scales[5]}")
    print(f"saved animation to: {saved_to}")


if __name__ == "__main__":
    main()
