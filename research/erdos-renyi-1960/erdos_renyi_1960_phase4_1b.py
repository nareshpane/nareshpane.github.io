#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from textwrap import fill
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle

# ============================================================
# Phase 4.1B comparison animation.
# Default output matches the HTML placeholder:
#     erdos_renyi_1960_phase4_1b.mp4
#
# Tailored idea:
#   Compare several points in the connectivity window using the same node
#   set, same layout, and same edge order.  The focus is how quickly the
#   number of vertices outside the dominant component collapses.
# ============================================================

DEFAULT_N = 1000
DEFAULT_Y_VALUES = [-1.0, 0.0, 1.0]
DEFAULT_SEED = 13579
DEFAULT_FPS = 5
DEFAULT_NUM_STEPS = 72
DEFAULT_HOLD_FINAL_FRAMES = 24
DEFAULT_OUTPUT = Path("./erdos_renyi_1960_phase4_1b.mp4")

LARGEST_NODE_COLOR = "#d9e8ff"
TREE_NODE_COLOR = "#f2f2f2"
UNICYCLIC_NODE_COLOR = "#fff1dc"
MULTICYCLIC_NODE_COLOR = "#ffe4e4"
ISOLATE_NODE_COLOR = "#ffffff"

LARGEST_EDGE_COLOR = "#1d4f91"
TREE_EDGE_COLOR = "#7a7f87"
UNICYCLIC_EDGE_COLOR = "#c48b18"
MULTICYCLIC_EDGE_COLOR = "#c62828"
NODE_EDGE = "#4f4f4f"


@dataclass(frozen=True)
class PanelSpec:
    y: float
    m_target: int
    label: str


@dataclass(frozen=True)
class ComponentSummary:
    giant_size: int
    second_size: int
    outside_vertices: int
    isolates: int
    components: int
    connected: bool
    mean_degree: float
    tree_counts: Dict[int, int]
    kind_by_node: Dict[int, str]
    edge_kind: Dict[Tuple[int, int], str]
    giant_nodes: List[int]


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
    outer_angles += rng.normal(0.0, 0.07, size=outer_angles.shape[0])

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


def ordered_random_edges(nodes: Sequence[int], rng: np.random.Generator) -> List[Tuple[int, int]]:
    edges = [(u, v) for i, u in enumerate(nodes) for v in nodes[i + 1 :]]
    rng.shuffle(edges)
    return edges


def build_graph_state(nodes: Sequence[int], edges_in_order: Sequence[Tuple[int, int]], edge_count: int) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges_in_order[:edge_count])
    return g


def connectivity_window_edges(n: int, y: float) -> int:
    log_n = math.log(max(n, 2))
    return max(1, int(round(0.5 * n * log_n + y * n)))


def theory_mean_isolates(y: float) -> float:
    return math.exp(-2.0 * y)


def theory_connected_probability(y: float) -> float:
    return math.exp(-math.exp(-2.0 * y))


def summarize_graph(g: nx.Graph) -> ComponentSummary:
    components = []
    kind_by_node: Dict[int, str] = {}
    edge_kind: Dict[Tuple[int, int], str] = {}
    tree_counts: Dict[int, int] = {}

    for comp_nodes in nx.connected_components(g):
        node_list = list(comp_nodes)
        h = g.subgraph(node_list)
        v = h.number_of_nodes()
        e = h.number_of_edges()

        if e == v - 1:
            kind = "tree"
            tree_counts[v] = tree_counts.get(v, 0) + 1
        elif e == v:
            kind = "unicyclic"
        else:
            kind = "multicyclic"

        for node in node_list:
            kind_by_node[node] = kind
        for edge in h.edges():
            edge_kind[tuple(sorted(edge))] = kind

        components.append({"nodes": node_list, "size": v, "kind": kind})

    components.sort(key=lambda item: item["size"], reverse=True)
    giant = components[0] if components else {"nodes": [], "size": 0}
    second_size = components[1]["size"] if len(components) > 1 else 0
    isolates = tree_counts.get(1, 0)
    mean_degree = (2.0 * g.number_of_edges() / g.number_of_nodes()) if g.number_of_nodes() else 0.0

    return ComponentSummary(
        giant_size=giant["size"],
        second_size=second_size,
        outside_vertices=g.number_of_nodes() - giant["size"],
        isolates=isolates,
        components=len(components),
        connected=(len(components) == 1),
        mean_degree=mean_degree,
        tree_counts=tree_counts,
        kind_by_node=kind_by_node,
        edge_kind=edge_kind,
        giant_nodes=list(giant["nodes"]),
    )


def node_colors_from_summary(g: nx.Graph, summary: ComponentSummary) -> List[str]:
    giant_nodes = set(summary.giant_nodes)
    colors = []
    for node in g.nodes():
        if node in giant_nodes:
            colors.append(LARGEST_NODE_COLOR)
        else:
            kind = summary.kind_by_node.get(node, "tree")
            if kind == "tree":
                colors.append(ISOLATE_NODE_COLOR if g.degree(node) == 0 else TREE_NODE_COLOR)
            elif kind == "unicyclic":
                colors.append(UNICYCLIC_NODE_COLOR)
            else:
                colors.append(MULTICYCLIC_NODE_COLOR)
    return colors


def edge_groups_from_summary(g: nx.Graph, summary: ComponentSummary) -> Dict[str, List[Tuple[int, int]]]:
    groups = {"giant": [], "tree": [], "unicyclic": [], "multicyclic": []}
    giant_nodes = set(summary.giant_nodes)
    for u, v in g.edges():
        edge = tuple(sorted((u, v)))
        if u in giant_nodes and v in giant_nodes:
            groups["giant"].append((u, v))
        else:
            groups[summary.edge_kind.get(edge, "tree")].append((u, v))
    return groups


def save_animation(anim: FuncAnimation, output_path: Path, fps: int, dpi: int = 180) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".mp4":
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is not None:
            writer = FFMpegWriter(
                fps=fps,
                codec="libx264",
                bitrate=2600,
                extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"],
            )
            anim.save(output_path, writer=writer, dpi=dpi)
            return output_path
        gif_path = output_path.with_suffix(".gif")
        anim.save(gif_path, writer=PillowWriter(fps=fps), dpi=120)
        return gif_path
    anim.save(output_path, writer=PillowWriter(fps=fps), dpi=120)
    return output_path


def style_graph_box(ax) -> None:
    ax.set_facecolor("white")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.set_frame_on(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2.0)
        spine.set_edgecolor("#4f4f4f")
    ax.add_patch(
        Rectangle((0.003, 0.003), 0.994, 0.994, transform=ax.transAxes, fill=False, linewidth=2.0, edgecolor="#4f4f4f", zorder=250, clip_on=False)
    )


def draw_badge(ax, x: float, y: float, text: str, face: str, edge: str, txt: str = "#111111", fontsize: float = 9.5) -> None:
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold",
        color=txt,
        bbox=dict(facecolor=face, edgecolor=edge, boxstyle="round,pad=0.22"),
    )


def build_panel_specs(n: int, y_values: Sequence[float]) -> List[PanelSpec]:
    return [PanelSpec(y=float(y), m_target=connectivity_window_edges(n=n, y=float(y)), label=f"y = {float(y):.1f}") for y in y_values]


def shared_footer_text(specs: Sequence[PanelSpec], summaries: Sequence[ComponentSummary], n: int) -> List[str]:
    lines: List[str] = []
    for spec, summary in zip(specs, summaries):
        line = (
            f"{spec.label}: outside = {summary.outside_vertices:,}, isolates = {summary.isolates:,}, "
            f"theory mean isolates ≈ {theory_mean_isolates(spec.y):.2f}, theory P(conn) ≈ {theory_connected_probability(spec.y):.3f}."
        )
        lines.append(fill(line, width=46))
    return lines


def make_phase4_1b_animation(
    n: int,
    y_values: Sequence[float],
    seed: int,
    fps: int,
    num_steps: int,
    hold_final_frames: int,
    output_path: Path,
) -> Path:
    specs = build_panel_specs(n=n, y_values=y_values)
    if len(specs) != 3:
        raise ValueError("Please provide exactly three y values.")

    rng = np.random.default_rng(seed)
    positions = layered_circular_layout(n=n, rng=rng)
    nodes = list(range(n))
    max_target = max(spec.m_target for spec in specs)
    edge_order = ordered_random_edges(nodes=nodes, rng=rng)[:max_target]

    frame_schedule = np.linspace(0.0, 1.0, num_steps).tolist() + [1.0] * hold_final_frames

    fig = plt.figure(figsize=(18.0, 10.8))
    outer = fig.add_gridspec(nrows=2, ncols=3, left=0.04, right=0.96, top=0.80, bottom=0.30, height_ratios=[1.0, 0.34], hspace=0.14, wspace=0.03)
    panels = []
    for col in range(3):
        graph_ax = fig.add_subplot(outer[0, col])
        info_ax = fig.add_subplot(outer[1, col])
        info_ax.axis("off")
        panels.append((graph_ax, info_ax))

    caption_ax = fig.add_axes([0.05, 0.835, 0.90, 0.14])
    caption_ax.axis("off")
    caption_ax.text(
        0.5,
        0.82,
        r"Phase 4.1B — entering the connectivity window at $m=\frac{n}{2}\log n + y n$",
        ha="center",
        va="center",
        fontsize=17,
        fontweight="bold",
    )
    caption_ax.text(
        0.5,
        0.44,
        fill(
            "All three panels use the same node set, fixed layout, and edge order. Only the stopping point changes: y = -1, 0, and 1. The aim is to compare how fast the number of vertices outside the dominant component collapses across the connectivity window.",
            width=120,
        ),
        ha="center",
        va="center",
        fontsize=11,
        color="0.25",
    )
    caption_ax.text(
        0.5,
        0.08,
        fill(
            "By this stage the graph should already look like one overwhelming body plus a tiny outside world made mostly of tree remnants.",
            width=118,
        ),
        ha="center",
        va="bottom",
        fontsize=11,
        color="0.25",
    )

    footer_ax = fig.add_axes([0.05, 0.12, 0.90, 0.12])
    footer_ax.axis("off")

    legend_handles = [
        Line2D([0], [0], color=LARGEST_EDGE_COLOR, linewidth=2.6, label="Dominant component"),
        Line2D([0], [0], color=TREE_EDGE_COLOR, linewidth=2.1, label="Outside tree remnants"),
        Line2D([0], [0], color=UNICYCLIC_EDGE_COLOR, linewidth=2.1, label="Outside cyclic remnants"),
        Line2D([0], [0], color=MULTICYCLIC_EDGE_COLOR, linewidth=2.1, label="Other outside pieces"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, 0.03), ncol=4, frameon=True, edgecolor="#cccccc", fontsize=10.2)

    def update(frame_index: int) -> None:
        progress = frame_schedule[frame_index]
        frame_summaries: List[ComponentSummary] = []

        for (graph_ax, info_ax), spec in zip(panels, specs):
            graph_ax.clear()
            info_ax.clear()
            info_ax.axis("off")

            edge_count = min(int(round(progress * spec.m_target)), spec.m_target)
            g = build_graph_state(nodes=nodes, edges_in_order=edge_order, edge_count=edge_count)
            summary = summarize_graph(g)
            frame_summaries.append(summary)
            edge_groups = edge_groups_from_summary(g, summary)

            nx.draw_networkx_edges(g, pos=positions, ax=graph_ax, edgelist=edge_groups["tree"], width=0.72 if n >= 1000 else 1.0, alpha=0.72, edge_color=TREE_EDGE_COLOR)
            if edge_groups["unicyclic"]:
                nx.draw_networkx_edges(g, pos=positions, ax=graph_ax, edgelist=edge_groups["unicyclic"], width=0.86 if n >= 1000 else 1.15, alpha=0.90, edge_color=UNICYCLIC_EDGE_COLOR)
            if edge_groups["multicyclic"]:
                nx.draw_networkx_edges(g, pos=positions, ax=graph_ax, edgelist=edge_groups["multicyclic"], width=0.95 if n >= 1000 else 1.25, alpha=0.92, edge_color=MULTICYCLIC_EDGE_COLOR)
            if edge_groups["giant"]:
                nx.draw_networkx_edges(g, pos=positions, ax=graph_ax, edgelist=edge_groups["giant"], width=1.30 if n >= 1000 else 1.75, alpha=0.98, edge_color=LARGEST_EDGE_COLOR)

            isolates = [u for u, degree in g.degree() if degree == 0]
            non_isolates = [u for u, degree in g.degree() if degree > 0]
            giant_set = set(summary.giant_nodes)
            other_nonisolates = [u for u in non_isolates if u not in giant_set]
            giant_nodes = [u for u in non_isolates if u in giant_set]
            all_node_colors = node_colors_from_summary(g, summary)

            nx.draw_networkx_nodes(g, pos=positions, nodelist=isolates, ax=graph_ax, node_size=4 if n >= 10000 else (10 if n >= 1000 else 34), linewidths=0.15 if n >= 10000 else 0.25, edgecolors="0.35", node_color=ISOLATE_NODE_COLOR)
            nx.draw_networkx_nodes(g, pos=positions, nodelist=other_nonisolates, ax=graph_ax, node_size=6 if n >= 10000 else (14 if n >= 1000 else 48), linewidths=0.18 if n >= 10000 else 0.35, edgecolors=NODE_EDGE, node_color=[all_node_colors[node] for node in other_nonisolates])
            nx.draw_networkx_nodes(g, pos=positions, nodelist=giant_nodes, ax=graph_ax, node_size=7 if n >= 10000 else (16 if n >= 1000 else 54), linewidths=0.20 if n >= 10000 else 0.40, edgecolors=LARGEST_EDGE_COLOR, node_color=LARGEST_NODE_COLOR)

            graph_ax.set_xlim(-1.12, 1.12)
            graph_ax.set_ylim(-1.12, 1.12)
            graph_ax.set_aspect("equal")
            style_graph_box(graph_ax)
            graph_ax.set_title(f"{spec.label}   |   n = {n:,}", fontsize=14.5, pad=8)

            info_ax.add_patch(FancyBboxPatch((0.03, 0.10), 0.94, 0.80, transform=info_ax.transAxes, boxstyle="round,pad=0.02", linewidth=0.95, edgecolor="#cfcfcf", facecolor="#fcfcfc", zorder=0))
            draw_badge(info_ax, 0.50, 0.76, f"edges shown = {edge_count:,}/{spec.m_target:,}", face="#edf4ff", edge="#1d4f91", txt="#123864", fontsize=9.8)
            info_ax.text(0.50, 0.52, f"avg degree = {summary.mean_degree:.3f}   |   outside = {summary.outside_vertices:,}   |   second = {summary.second_size:,}", transform=info_ax.transAxes, ha="center", va="center", fontsize=9.2)
            draw_badge(info_ax, 0.27, 0.28, f"isolates = {summary.isolates:,}", "#edf4ff", "#1d4f91", fontsize=8.8)
            draw_badge(info_ax, 0.74, 0.28, f"T1/T2/T3 = {summary.tree_counts.get(1,0)}/{summary.tree_counts.get(2,0)}/{summary.tree_counts.get(3,0)}", "#f6f6f6", "#8a8a8a", fontsize=8.0)

        footer_ax.clear()
        footer_ax.axis("off")
        footer_ax.add_patch(FancyBboxPatch((0.01, 0.08), 0.98, 0.84, transform=footer_ax.transAxes, boxstyle="round,pad=0.02", linewidth=1.0, edgecolor="#cfcfcf", facecolor="#fbfbfb"))
        footer_lines = shared_footer_text(specs=specs, summaries=frame_summaries, n=n)
        footer_ax.text(0.5, 0.52, "\n".join(footer_lines), ha="center", va="center", fontsize=9.2, linespacing=1.30)

    anim = FuncAnimation(fig, update, frames=len(frame_schedule), interval=1000 / fps, repeat=False)
    saved_to = save_animation(anim=anim, output_path=output_path, fps=fps)
    plt.close(fig)
    return saved_to


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create the Phase 4.1B comparison animation across the connectivity window.")
    parser.add_argument("--n", type=int, default=DEFAULT_N, help="Number of nodes.")
    parser.add_argument("--y-values", type=float, nargs="+", default=DEFAULT_Y_VALUES, help="Three y values to compare.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help="Frames per second.")
    parser.add_argument("--num-steps", type=int, default=DEFAULT_NUM_STEPS, help="Number of progress steps before the final hold.")
    parser.add_argument("--hold-final-frames", type=int, default=DEFAULT_HOLD_FINAL_FRAMES, help="Still frames at the end.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output animation path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    y_values = [float(y) for y in args.y_values][:3]
    if len(y_values) != 3:
        raise ValueError("Please provide exactly three y values.")
    saved_to = make_phase4_1b_animation(
        n=max(50, args.n),
        y_values=y_values,
        seed=args.seed,
        fps=max(1, args.fps),
        num_steps=max(12, args.num_steps),
        hold_final_frames=max(0, args.hold_final_frames),
        output_path=args.output,
    )
    print(f"saved animation to: {saved_to}")


if __name__ == "__main__":
    main()
