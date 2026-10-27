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
# Phase 2.2B comparison animation.
# Default output matches the HTML placeholder:
#     erdos_renyi_1960_phase2_2b.mp4
#
# Idea:
#   Compare n = 100, 1000, and 10,000 at the same linear rule m = round(c n)
#   with c = 0.30. The point is not merely that cycles appear, but that the
#   component picture is still dominated by trees and unicyclic components.
# ============================================================
DEFAULT_NS = [100, 1000, 10000]
DEFAULT_C = 1.00
DEFAULT_SEED = 24680
DEFAULT_FPS = 5
DEFAULT_NUM_STEPS = 72
DEFAULT_HOLD_FINAL_FRAMES = 20
DEFAULT_OUTPUT = Path("./erdos_renyi_1960_phase2_2b.mp4")

TREE_COLOR = "#1565c0"
UNICYCLIC_COLOR = "#2e7d32"
MULTICYCLIC_COLOR = "#c62828"
BASE_EDGE = "#6b7280"


@dataclass
class RunData:
    n: int
    m_target: int
    positions: Dict[int, Tuple[float, float]]
    edge_order: List[Tuple[int, int]]


@dataclass(frozen=True)
class ComponentSummary:
    tree_components: int
    unicyclic_components: int
    multicyclic_components: int
    sample_tree_edges: List[Tuple[int, int]]
    sample_unicyclic_edges: List[Tuple[int, int]]
    sample_multicyclic_edges: List[Tuple[int, int]]
    isolates: int
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


def summarize_components(g: nx.Graph) -> ComponentSummary:
    tree_components = 0
    unicyclic_components = 0
    multicyclic_components = 0
    sample_tree_edges: List[Tuple[int, int]] = []
    sample_unicyclic_edges: List[Tuple[int, int]] = []
    sample_multicyclic_edges: List[Tuple[int, int]] = []
    isolates = 0
    largest_component = 0

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
        sample_tree_edges=sample_tree_edges,
        sample_unicyclic_edges=sample_unicyclic_edges,
        sample_multicyclic_edges=sample_multicyclic_edges,
        isolates=isolates,
        largest_component=largest_component,
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


def make_phase2_2b_animation(
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
        m_target = max(1, n)
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
        r"Phase 2.2B — tree versus unicyclic component picture when each panel is grown to $m=n$",
        ha="center",
        va="center",
        fontsize=17,
        fontweight="bold",
    )
    title_ax.text(
        0.5,
        0.22,
        fill(
            fill(
            r"Each panel now grows until the number of displayed edges matches the number of vertices, so the final totals are 100, 1,000, and 10,000 respectively. The claim here is subtler than in Phase 2.1: the graph should still be dominated by trees, with the non-tree exceptions usually being unicyclic components rather than multicyclic ones.",
            width=120,
        ),
            width=120,
        ),
        ha="center",
        va="center",
        fontsize=11,
        color="0.25",
    )

    legend_handles = [
        Line2D([0], [0], color=TREE_COLOR, linewidth=2.6, label="One sample tree component"),
        Line2D([0], [0], color=UNICYCLIC_COLOR, linewidth=2.6, label="One sample unicyclic component"),
        Line2D([0], [0], color=MULTICYCLIC_COLOR, linewidth=2.6, label="One sample multicyclic component"),
        Line2D([0], [0], color=BASE_EDGE, linewidth=1.8, label="Other edges"),
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
            summary = summarize_components(g)

            sample_tree = {tuple(sorted(e)) for e in summary.sample_tree_edges}
            sample_uni = {tuple(sorted(e)) for e in summary.sample_unicyclic_edges}
            sample_multi = {tuple(sorted(e)) for e in summary.sample_multicyclic_edges}
            highlighted = sample_tree | sample_uni | sample_multi
            other_edges = [e for e in g.edges() if tuple(sorted(e)) not in highlighted]

            nx.draw_networkx_edges(g, pos=run.positions, edgelist=other_edges, ax=graph_ax, width=0.50 if run.n >= 1000 else 0.80, alpha=0.60, edge_color=BASE_EDGE)
            if summary.sample_tree_edges:
                nx.draw_networkx_edges(g, pos=run.positions, edgelist=summary.sample_tree_edges, ax=graph_ax, width=0.85 if run.n >= 1000 else 1.15, alpha=0.92, edge_color=TREE_COLOR)
            if summary.sample_unicyclic_edges:
                nx.draw_networkx_edges(g, pos=run.positions, edgelist=summary.sample_unicyclic_edges, ax=graph_ax, width=0.95 if run.n >= 1000 else 1.25, alpha=0.95, edge_color=UNICYCLIC_COLOR)
            if summary.sample_multicyclic_edges:
                nx.draw_networkx_edges(g, pos=run.positions, edgelist=summary.sample_multicyclic_edges, ax=graph_ax, width=1.05 if run.n >= 1000 else 1.35, alpha=0.95, edge_color=MULTICYCLIC_COLOR)
            isolates = [u for u, d in g.degree() if d == 0]
            non_isolates = [u for u, d in g.degree() if d > 0]
            nx.draw_networkx_nodes(g, pos=run.positions, nodelist=isolates, ax=graph_ax, node_size=4 if run.n >= 10000 else (10 if run.n >= 1000 else 34), linewidths=0.15 if run.n >= 10000 else 0.25, edgecolors="0.35", node_color="white")
            nx.draw_networkx_nodes(g, pos=run.positions, nodelist=non_isolates, ax=graph_ax, node_size=6 if run.n >= 10000 else (14 if run.n >= 1000 else 48), linewidths=0.18 if run.n >= 10000 else 0.35, edgecolors="#4f4f4f", node_color="#d9ebff")

            graph_ax.set_xlim(-1.12, 1.12)
            graph_ax.set_ylim(-1.12, 1.12)
            graph_ax.set_aspect("equal")
            style_graph_box(graph_ax)
            graph_ax.set_title(rf"$n = {run.n:,}$", fontsize=15, pad=10)

            info_ax.add_patch(FancyBboxPatch((0.03, 0.06), 0.94, 0.88, transform=info_ax.transAxes, boxstyle="round,pad=0.02", linewidth=0.95, edgecolor="#cfcfcf", facecolor="#fcfcfc", zorder=0))
            info_ax.text(0.5, 0.78, f"edges shown = {edge_count:,}/{run.m_target:,}   |   avg degree = {2.0 * edge_count / run.n:.3f}", transform=info_ax.transAxes, ha="center", va="center", fontsize=10, fontweight="bold", bbox=dict(facecolor="#ffe8e8", edgecolor="#b71c1c", boxstyle="round,pad=0.22"))
            info_ax.text(0.5, 0.53, f"tree comps = {summary.tree_components}   |   unicyclic = {summary.unicyclic_components}   |   multicyclic = {summary.multicyclic_components}   |   isolates = {summary.isolates:,}", transform=info_ax.transAxes, ha="center", va="center", fontsize=9.4, bbox=dict(facecolor="white", edgecolor="0.85", boxstyle="round,pad=0.22"))
            predicate = "YES" if summary.multicyclic_components == 0 else "NO"
            info_ax.text(
                0.5,
                0.21,
                fill(
                    f"Are all nontrivial components tree-or-unicyclic right now? {predicate}. Larger n should make the tree-or-unicyclic picture clearer, with multicyclic components becoming comparatively rare.",
                    width=50,
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
        description="Create the Phase 2.2B comparison animation across n = 100, 1000, and 10,000, with each panel grown until the displayed edge count matches the number of vertices."
    )
    parser.add_argument("--ns", type=int, nargs="+", default=DEFAULT_NS, help="Values of n to compare.")
    parser.add_argument("--c", type=float, default=DEFAULT_C, help="Retained for compatibility; the default animation now grows each panel until m = n.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Base random seed.")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help="Frames per second.")
    parser.add_argument("--num-steps", type=int, default=DEFAULT_NUM_STEPS, help="Number of progress steps before the final hold.")
    parser.add_argument("--hold-final-frames", type=int, default=DEFAULT_HOLD_FINAL_FRAMES, help="Still frames at the end.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output animation path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    c = min(max(args.c, 0.01), 0.49)
    saved_to = make_phase2_2b_animation(ns=args.ns, c=c, seed=args.seed, fps=max(1, args.fps), num_steps=max(8, args.num_steps), hold_final_frames=max(0, args.hold_final_frames), output_path=args.output)
    print(f"n values = {[n for n in args.ns if n >= 2]}")
    print(f"c (retained for compatibility) = {c:.3f}")
    print(f"saved animation to: {saved_to}")


if __name__ == "__main__":
    main()
