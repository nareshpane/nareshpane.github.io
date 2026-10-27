#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle


# ============================================================
# Phase 1.2B comparison animation.
# Default output matches the HTML placeholder:
#     erdos_renyi_1960_phase1_2b.mp4
#
# Idea:
#   Compare n = 100, 1000, 10,000 while each panel grows up to its own
#   k=5 threshold scale m ~ n^(3/4). This makes the table in the HTML page
#   visually concrete.
# ============================================================
DEFAULT_NS = [100, 1000, 10000]
DEFAULT_SEED = 13579
DEFAULT_FPS = 4
DEFAULT_NUM_STEPS = 72
DEFAULT_HOLD_FINAL_FRAMES = 20
DEFAULT_OUTPUT = Path("./erdos_renyi_1960_phase1_2b.mp4")


T3_COLOR = "#c62828"   # red
T4_COLOR = "#1565c0"   # blue
T5_COLOR = "#2e7d32"   # green
BASE_EDGE = "#7a7f87"
NODE_EDGE = "#4f4f4f"


@dataclass
class RunData:
    n: int
    thresholds: Dict[int, int]
    positions: Dict[int, Tuple[float, float]]
    edge_order: List[Tuple[int, int]]
    first_emergence: Dict[int, Optional[int]]


@dataclass(frozen=True)
class TreeOrderSummary:
    counts: Dict[int, int]
    highlighted_edges: Dict[int, List[Tuple[int, int]]]
    largest_component: int
    isolates: int
    components: int


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


def threshold_scales(n: int) -> Dict[int, int]:
    return {
        3: max(1, int(round(n ** 0.5))),
        4: max(1, int(round(n ** (2.0 / 3.0)))),
        5: max(1, int(round(n ** 0.75))),
    }


def summarize_tree_orders(g: nx.Graph, tracked_orders: List[int]) -> TreeOrderSummary:
    counts = {k: 0 for k in tracked_orders}
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
            highlighted_edges[v].extend(list(h.edges()))

    return TreeOrderSummary(
        counts=counts,
        highlighted_edges=highlighted_edges,
        largest_component=largest_component,
        isolates=isolates,
        components=len(components),
    )


def detect_first_emergence(nodes: List[int], edges_in_order: List[Tuple[int, int]], tracked_orders: List[int]) -> Dict[int, Optional[int]]:
    g = nx.Graph()
    g.add_nodes_from(nodes)
    first_seen: Dict[int, Optional[int]] = {k: None for k in tracked_orders}

    for idx, edge in enumerate(edges_in_order, start=1):
        g.add_edge(*edge)
        summary = summarize_tree_orders(g, tracked_orders=tracked_orders)
        for k in tracked_orders:
            if first_seen[k] is None and summary.counts[k] > 0:
                first_seen[k] = idx
        if all(first_seen[k] is not None for k in tracked_orders):
            break

    return first_seen


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


def draw_badge(ax, x: float, y: float, text: str, face: str, edge: str, txt: str = "#111111", fontsize: float = 9.7) -> None:
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


def emergence_piece(actual: Optional[int], predicted: int) -> str:
    if actual is None:
        return f"seen — / pred {predicted:,}"
    return f"seen {actual:,} / pred {predicted:,}"


def make_phase1_2b_animation(
    ns: List[int],
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
        thresholds = threshold_scales(n)
        edge_order = ordered_random_edges(nodes=list(range(n)), rng=rng)[:thresholds[5]]
        first_emergence = detect_first_emergence(list(range(n)), edge_order, tracked_orders=[3, 4, 5])
        runs.append(
            RunData(
                n=n,
                thresholds=thresholds,
                positions=positions,
                edge_order=edge_order,
                first_emergence=first_emergence,
            )
        )

    progress_values = np.linspace(0.0, 1.0, num_steps).tolist()
    frame_schedule = progress_values + [1.0] * hold_final_frames

    fig = plt.figure(figsize=(18.0, 10.0))
    outer = fig.add_gridspec(
        nrows=2,
        ncols=3,
        left=0.04,
        right=0.96,
        top=0.82,
        bottom=0.16,
        height_ratios=[1.0, 0.48],
        hspace=0.12,
        wspace=0.03,
    )
    panels = []
    for col in range(3):
        graph_ax = fig.add_subplot(outer[0, col])
        info_ax = fig.add_subplot(outer[1, col])
        info_ax.axis("off")
        panels.append((graph_ax, info_ax))

    caption_ax = fig.add_axes([0.05, 0.865, 0.90, 0.11])
    caption_ax.axis("off")
    caption_ax.text(
        0.5,
        0.72,
        "Phase 1.2B — comparing appearance scales for 3-, 4-, and 5-vertex trees",
        ha="center",
        va="center",
        fontsize=17,
        fontweight="bold",
    )
    caption_ax.text(
        0.5,
        0.30,
        "Each panel grows only up to its own m ≈ n^(3/4). Colored edges show where T3, T4, and T5 components appear.",
        ha="center",
        va="center",
        fontsize=11,
        color="0.25",
    )
    caption_ax.text(
        0.5,
        0.02,
        "Bottom badge compares predicted threshold scales with the actual first emergence of T3, T4, and T5 in that run.",
        ha="center",
        va="bottom",
        fontsize=11,
        color="0.25",
    )

    legend_handles = [
        Line2D([0], [0], color=T3_COLOR, linewidth=2.8, label="T3 edges"),
        Line2D([0], [0], color=T4_COLOR, linewidth=2.8, label="T4 edges"),
        Line2D([0], [0], color=T5_COLOR, linewidth=2.8, label="T5 edges"),
        Line2D([0], [0], color=BASE_EDGE, linewidth=1.8, label="Other edges"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.05),
        ncol=4,
        frameon=True,
        edgecolor="#cccccc",
        fontsize=10.2,
    )

    def update(frame_index: int) -> None:
        progress = frame_schedule[frame_index]

        for panel_index, run in enumerate(runs):
            graph_ax, info_ax = panels[panel_index]
            graph_ax.clear()
            info_ax.clear()
            info_ax.axis("off")

            edge_count = min(int(round(progress * len(run.edge_order))), len(run.edge_order))
            nodes = list(range(run.n))
            g = build_graph_state(nodes=nodes, edges_in_order=run.edge_order, edge_count=edge_count)
            summary = summarize_tree_orders(g, tracked_orders=[3, 4, 5])

            isolates = [u for u, degree in g.degree() if degree == 0]
            non_isolates = [u for u, degree in g.degree() if degree > 0]

            nx.draw_networkx_edges(
                g,
                pos=run.positions,
                ax=graph_ax,
                edgelist=list(g.edges()),
                width=0.65 if run.n >= 1000 else 1.0,
                alpha=0.60,
                edge_color=BASE_EDGE,
            )
            if summary.highlighted_edges[3]:
                nx.draw_networkx_edges(
                    g,
                    pos=run.positions,
                    ax=graph_ax,
                    edgelist=summary.highlighted_edges[3],
                    width=1.4 if run.n >= 1000 else 2.2,
                    alpha=0.95,
                    edge_color=T3_COLOR,
                )
            if summary.highlighted_edges[4]:
                nx.draw_networkx_edges(
                    g,
                    pos=run.positions,
                    ax=graph_ax,
                    edgelist=summary.highlighted_edges[4],
                    width=1.6 if run.n >= 1000 else 2.4,
                    alpha=0.95,
                    edge_color=T4_COLOR,
                )
            if summary.highlighted_edges[5]:
                nx.draw_networkx_edges(
                    g,
                    pos=run.positions,
                    ax=graph_ax,
                    edgelist=summary.highlighted_edges[5],
                    width=1.8 if run.n >= 1000 else 2.6,
                    alpha=0.95,
                    edge_color=T5_COLOR,
                )

            nx.draw_networkx_nodes(
                g,
                pos=run.positions,
                nodelist=isolates,
                ax=graph_ax,
                node_size=4 if run.n >= 10000 else (10 if run.n >= 1000 else 34),
                linewidths=0.15 if run.n >= 10000 else 0.25,
                edgecolors="0.35",
                node_color="white",
            )
            nx.draw_networkx_nodes(
                g,
                pos=run.positions,
                nodelist=non_isolates,
                ax=graph_ax,
                node_size=6 if run.n >= 10000 else (14 if run.n >= 1000 else 48),
                linewidths=0.18 if run.n >= 10000 else 0.35,
                edgecolors=NODE_EDGE,
                node_color="white",
            )

            graph_ax.set_xlim(-1.12, 1.12)
            graph_ax.set_ylim(-1.12, 1.12)
            graph_ax.set_aspect("equal")
            style_graph_box(graph_ax)
            graph_ax.set_title(f"n = {run.n:,}", fontsize=15, pad=8)

            info_ax.add_patch(
                FancyBboxPatch(
                    (0.03, 0.05),
                    0.94,
                    0.88,
                    transform=info_ax.transAxes,
                    boxstyle="round,pad=0.02",
                    linewidth=0.95,
                    edgecolor="#cfcfcf",
                    facecolor="#fcfcfc",
                    zorder=0,
                )
            )

            draw_badge(
                info_ax,
                0.50,
                0.82,
                f"edges shown = {edge_count:,}/{run.thresholds[5]:,}",
                face="#ffe8e8",
                edge="#b71c1c",
                txt="#8b0000",
                fontsize=10.0,
            )
            info_ax.text(
                0.50,
                0.63,
                f"predicted scales: T3≈{run.thresholds[3]:,}   |   T4≈{run.thresholds[4]:,}   |   T5≈{run.thresholds[5]:,}   |   avg degree = {2.0 * edge_count / run.n:.3f}",
                transform=info_ax.transAxes,
                ha="center",
                va="center",
                fontsize=9.8,
            )

            draw_badge(info_ax, 0.22, 0.43, f"T3 count = {summary.counts[3]}", "#ffe8e8", T3_COLOR)
            draw_badge(info_ax, 0.50, 0.43, f"T4 count = {summary.counts[4]}", "#edf4ff", T4_COLOR)
            draw_badge(info_ax, 0.78, 0.43, f"T5 count = {summary.counts[5]}", "#edfbe8", T5_COLOR)

            emergence_line = (
                f"T3 {emergence_piece(run.first_emergence[3], run.thresholds[3])}   |   "
                f"T4 {emergence_piece(run.first_emergence[4], run.thresholds[4])}   |   "
                f"T5 {emergence_piece(run.first_emergence[5], run.thresholds[5])}"
            )
            draw_badge(
                info_ax,
                0.50,
                0.17,
                emergence_line,
                face="#f6f4ff",
                edge="#8f86c7",
                txt="#222222",
                fontsize=8.9,
            )

        for extra_panel in panels[len(runs):]:
            extra_panel[0].clear()
            extra_panel[0].axis("off")
            extra_panel[1].clear()
            extra_panel[1].axis("off")

    anim = FuncAnimation(fig, update, frames=len(frame_schedule), interval=1000 / fps, repeat=False)
    saved_to = save_animation(anim=anim, output_path=output_path, fps=fps)
    plt.close(fig)
    return saved_to


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create the Phase 1.2B comparison animation across n = 100, 1000, and 10,000, "
            "showing that larger tree-orders become visible at larger edge scales."
        )
    )
    parser.add_argument("--ns", type=int, nargs="+", default=DEFAULT_NS, help="Values of n to compare.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Base random seed.")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help="Frames per second.")
    parser.add_argument("--num-steps", type=int, default=DEFAULT_NUM_STEPS, help="Number of progress steps before the final hold.")
    parser.add_argument("--hold-final-frames", type=int, default=DEFAULT_HOLD_FINAL_FRAMES, help="Still frames at the end.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output animation path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    saved_to = make_phase1_2b_animation(
        ns=args.ns,
        seed=args.seed,
        fps=max(1, args.fps),
        num_steps=max(8, args.num_steps),
        hold_final_frames=max(0, args.hold_final_frames),
        output_path=args.output,
    )
    print(f"n values = {[n for n in args.ns if n >= 2]}")
    print(f"saved animation to: {saved_to}")


if __name__ == "__main__":
    main()
