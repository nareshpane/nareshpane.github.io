#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from textwrap import fill
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle

# ============================================================
# Phase 2.4B comparison animation.
# Default output matches the HTML placeholder:
#     erdos_renyi_1960_phase2_4b.mp4
#
# Idea:
#   Compare n = 100, 1000, and 10,000 at the same linear rule m ≈ c n.
#   The emphasis is on the vertex-level picture: most vertices should still
#   lie in tree components, and the total component count should stay close
#   to n - m.
# ============================================================
DEFAULT_NS = [100, 1000, 10000]
DEFAULT_C = 0.30
DEFAULT_SEED = 13579
DEFAULT_FPS = 5
DEFAULT_NUM_STEPS = 72
DEFAULT_HOLD_FINAL_FRAMES = 20
DEFAULT_OUTPUT = Path("./erdos_renyi_1960_phase2_4b.mp4")

TREE_VERTEX_COLOR = "#d9ebff"
UNICYCLIC_VERTEX_COLOR = "#e6f6e6"
MULTICYCLIC_VERTEX_COLOR = "#ffe3e3"
TREE_EDGE = "#1565c0"
UNICYCLIC_EDGE = "#2e7d32"
MULTICYCLIC_EDGE = "#c62828"
BASE_EDGE = "#6b7280"
NODE_EDGE = "#4f4f4f"


@dataclass
class RunData:
    n: int
    m_target: int
    positions: Dict[int, Tuple[float, float]]
    edge_order: List[Tuple[int, int]]


@dataclass(frozen=True)
class VertexSummary:
    tree_vertices: int
    unicyclic_vertices: int
    multicyclic_vertices: int
    components: int
    largest_component: int
    tree_nodes: List[int]
    unicyclic_nodes: List[int]
    multicyclic_nodes: List[int]


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


def summarize_vertices(g: nx.Graph) -> VertexSummary:
    tree_vertices = 0
    unicyclic_vertices = 0
    multicyclic_vertices = 0
    tree_nodes: List[int] = []
    unicyclic_nodes: List[int] = []
    multicyclic_nodes: List[int] = []
    components = list(nx.connected_components(g))
    largest_component = max((len(comp) for comp in components), default=0)

    for comp_nodes in components:
        h = g.subgraph(comp_nodes)
        v = h.number_of_nodes()
        e = h.number_of_edges()
        comp_list = list(comp_nodes)
        if e == v - 1:
            tree_vertices += v
            tree_nodes.extend(comp_list)
        elif e == v:
            unicyclic_vertices += v
            unicyclic_nodes.extend(comp_list)
        elif e > v:
            multicyclic_vertices += v
            multicyclic_nodes.extend(comp_list)

    return VertexSummary(
        tree_vertices=tree_vertices,
        unicyclic_vertices=unicyclic_vertices,
        multicyclic_vertices=multicyclic_vertices,
        components=len(components),
        largest_component=largest_component,
        tree_nodes=tree_nodes,
        unicyclic_nodes=unicyclic_nodes,
        multicyclic_nodes=multicyclic_nodes,
    )


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
        Rectangle(
            (0.003, 0.003),
            0.994,
            0.994,
            transform=ax.transAxes,
            fill=False,
            linewidth=2.0,
            edgecolor="#4f4f4f",
            zorder=250,
            clip_on=False,
        )
    )


def make_phase2_4b_animation(
    ns: List[int],
    c: float,
    seed: int,
    fps: int,
    num_steps: int,
    hold_final_frames: int,
    output_path: Path,
) -> Path:
    clean_ns = [n for n in ns if n >= 2]
    if not clean_ns:
        raise ValueError("Need at least one n >= 2.")

    runs: List[RunData] = []
    for i, n in enumerate(clean_ns):
        rng = np.random.default_rng(seed + 1009 * i)
        positions = layered_circular_layout(n=n, rng=rng)
        m_target = max(1, int(round(c * n)))
        edge_order = ordered_random_edges(nodes=list(range(n)), rng=rng)[:m_target]
        runs.append(RunData(n=n, m_target=m_target, positions=positions, edge_order=edge_order))

    frame_schedule = np.linspace(0.0, 1.0, num_steps).tolist() + [1.0] * hold_final_frames

    fig = plt.figure(figsize=(18.2, 10.0))
    outer = fig.add_gridspec(
        nrows=2,
        ncols=3,
        left=0.04,
        right=0.96,
        top=0.82,
        bottom=0.14,
        height_ratios=[1.0, 0.42],
        hspace=0.12,
        wspace=0.04,
    )
    panels = []
    for col in range(3):
        graph_ax = fig.add_subplot(outer[0, col])
        info_ax = fig.add_subplot(outer[1, col])
        info_ax.axis("off")
        panels.append((graph_ax, info_ax))

    title_ax = fig.add_axes([0.05, 0.855, 0.90, 0.10])
    title_ax.axis("off")
    title_ax.text(
        0.5,
        0.72,
        rf"Phase 2.4B — most vertices still lie in trees, and components are about $n-m$ with $c={c:.2f}$",
        ha="center",
        va="center",
        fontsize=17,
        fontweight="bold",
    )
    title_ax.text(
        0.5,
        0.22,
        fill(
            r"Each panel grows to $m=\lfloor c n \rceil$. Node colors show whether a vertex lies in a tree component, a unicyclic component, or a multicyclic component. The point is that cycles appear, but tree vertices still dominate and the component count remains close to the heuristic $n-m$.",
            width=122,
        ),
        ha="center",
        va="center",
        fontsize=11,
        color="0.25",
    )

    legend_handles = [
        Line2D([0], [0], color=TREE_EDGE, marker="o", markersize=7, linewidth=1.2, label="Vertices in tree components"),
        Line2D([0], [0], color=UNICYCLIC_EDGE, marker="o", markersize=7, linewidth=1.2, label="Vertices in unicyclic components"),
        Line2D([0], [0], color=MULTICYCLIC_EDGE, marker="o", markersize=7, linewidth=1.2, label="Vertices in multicyclic components"),
        Line2D([0], [0], color=BASE_EDGE, linewidth=1.0, label="Edges"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, 0.05), ncol=4, frameon=True, edgecolor="#cccccc", fontsize=10)

    def update(frame_index: int) -> None:
        progress = frame_schedule[frame_index]
        for (graph_ax, info_ax), run in zip(panels, runs):
            graph_ax.clear()
            info_ax.clear()
            info_ax.axis("off")

            edge_count = min(int(round(progress * run.m_target)), run.m_target)
            g = build_graph_state(list(range(run.n)), run.edge_order, edge_count)
            summary = summarize_vertices(g)

            nx.draw_networkx_edges(g, pos=run.positions, ax=graph_ax, width=0.50 if run.n >= 1000 else 0.80, alpha=0.42, edge_color=BASE_EDGE)
            if summary.tree_nodes:
                nx.draw_networkx_nodes(g, pos=run.positions, nodelist=summary.tree_nodes, ax=graph_ax, node_size=4 if run.n >= 10000 else (10 if run.n >= 1000 else 34), linewidths=0.15 if run.n >= 10000 else 0.25, edgecolors=TREE_EDGE, node_color=TREE_VERTEX_COLOR)
            if summary.unicyclic_nodes:
                nx.draw_networkx_nodes(g, pos=run.positions, nodelist=summary.unicyclic_nodes, ax=graph_ax, node_size=4 if run.n >= 10000 else (10 if run.n >= 1000 else 34), linewidths=0.15 if run.n >= 10000 else 0.25, edgecolors=UNICYCLIC_EDGE, node_color=UNICYCLIC_VERTEX_COLOR)
            if summary.multicyclic_nodes:
                nx.draw_networkx_nodes(g, pos=run.positions, nodelist=summary.multicyclic_nodes, ax=graph_ax, node_size=4 if run.n >= 10000 else (10 if run.n >= 1000 else 34), linewidths=0.15 if run.n >= 10000 else 0.25, edgecolors=MULTICYCLIC_EDGE, node_color=MULTICYCLIC_VERTEX_COLOR)

            graph_ax.set_xlim(-1.12, 1.12)
            graph_ax.set_ylim(-1.12, 1.12)
            graph_ax.set_aspect("equal")
            style_graph_box(graph_ax)
            graph_ax.set_title(rf"$n = {run.n:,}$", fontsize=15, pad=10)

            ref_count = run.n - edge_count
            diff = summary.components - ref_count
            tree_fraction = summary.tree_vertices / run.n
            cyc_fraction = (summary.unicyclic_vertices + summary.multicyclic_vertices) / run.n

            info_ax.add_patch(FancyBboxPatch((0.03, 0.06), 0.94, 0.88, transform=info_ax.transAxes, boxstyle="round,pad=0.02", linewidth=0.95, edgecolor="#cfcfcf", facecolor="#fcfcfc", zorder=0))
            info_ax.text(0.5, 0.78, f"edges shown = {edge_count:,}/{run.m_target:,}   |   components = {summary.components:,}   |   reference n-m = {ref_count:,}", transform=info_ax.transAxes, ha="center", va="center", fontsize=10, fontweight="bold", bbox=dict(facecolor="#ffe8e8", edgecolor="#b71c1c", boxstyle="round,pad=0.22"))
            info_ax.text(0.5, 0.54, f"tree vertices = {summary.tree_vertices:,} ({tree_fraction:.3f})   |   cycle-bearing vertices = {summary.unicyclic_vertices + summary.multicyclic_vertices:,} ({cyc_fraction:.3f})   |   largest = {summary.largest_component}", transform=info_ax.transAxes, ha="center", va="center", fontsize=9.4, bbox=dict(facecolor="white", edgecolor="0.85", boxstyle="round,pad=0.22"))
            info_ax.text(
                0.5,
                0.22,
                fill(
                    f"Difference from n-m: {diff:+d}. The visual claim of Phase 2.4 is that even after short cycles appear, most vertices still sit in tree components, and most edges are still doing the forest job of merging components.",
                    width=52,
                ),
                transform=info_ax.transAxes,
                ha="center",
                va="center",
                fontsize=8.9,
                bbox=dict(facecolor="#f9f9f9", edgecolor="0.75", boxstyle="round,pad=0.24"),
            )

    anim = FuncAnimation(fig, update, frames=len(frame_schedule), interval=1000 / fps, repeat=False)
    saved_to = save_animation(anim=anim, output_path=output_path, fps=fps)
    plt.close(fig)
    return saved_to


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the Phase 2.4B comparison animation across n = 100, 1000, and 10,000 at the linear scale m ~ c n."
    )
    parser.add_argument("--ns", type=int, nargs="+", default=DEFAULT_NS, help="Values of n to compare.")
    parser.add_argument("--c", type=float, default=DEFAULT_C, help="Linear-scale constant in m = round(c n), should stay below 0.5.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Base random seed.")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help="Frames per second.")
    parser.add_argument("--num-steps", type=int, default=DEFAULT_NUM_STEPS, help="Number of progress steps before the final hold.")
    parser.add_argument("--hold-final-frames", type=int, default=DEFAULT_HOLD_FINAL_FRAMES, help="Still frames at the end.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output animation path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    c = min(max(args.c, 0.01), 0.49)
    saved_to = make_phase2_4b_animation(ns=args.ns, c=c, seed=args.seed, fps=max(1, args.fps), num_steps=max(8, args.num_steps), hold_final_frames=max(0, args.hold_final_frames), output_path=args.output)
    print(f"n values = {[n for n in args.ns if n >= 2]}")
    print(f"c = {c:.3f}")
    print(f"saved animation to: {saved_to}")


if __name__ == "__main__":
    main()
