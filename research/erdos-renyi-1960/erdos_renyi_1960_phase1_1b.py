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
from matplotlib.patches import FancyBboxPatch, Rectangle


# ============================================================
# Comparison animation for Phase 1, subsection 1.1B.
# Default output matches the HTML placeholder:
#     erdos_renyi_1960_phase1_1b.mp4
# ============================================================
DEFAULT_NS = [100, 1000, 10000]
DEFAULT_ALPHA = 0.5               # use m = floor(sqrt(n)) by default
DEFAULT_SEED = 31415
DEFAULT_FPS = 4
DEFAULT_HOLD_FINAL_FRAMES = 16
DEFAULT_OUTPUT = Path("./erdos_renyi_1960_phase1_1b.mp4")


@dataclass
class RunData:
    n: int
    m_target: int
    positions: Dict[int, Tuple[float, float]]
    edge_order: List[Tuple[int, int]]


@dataclass(frozen=True)
class ForestSummary:
    is_forest: bool
    components: int
    isolates: int
    largest_component: int
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
    cyclic_components = 0
    for comp_nodes in components:
        h = g.subgraph(comp_nodes)
        if h.number_of_edges() != h.number_of_nodes() - 1:
            cyclic_components += 1
    return ForestSummary(
        is_forest=(cyclic_components == 0),
        components=len(components),
        isolates=isolates,
        largest_component=largest_component,
        cyclic_components=cyclic_components,
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


def make_phase1_comparison_animation(
    ns: List[int],
    alpha: float,
    seed: int,
    fps: int,
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
        m_target = max(1, int(n ** alpha))
        edge_order = ordered_random_edges(nodes=list(range(n)), rng=rng)[:m_target]
        runs.append(RunData(n=n, m_target=m_target, positions=positions, edge_order=edge_order))

    max_target = max(run.m_target for run in runs)
    frame_schedule = list(range(0, max_target + 1)) + [max_target] * hold_final_frames

    fig = plt.figure(figsize=(18.0, 8.8))
    outer = fig.add_gridspec(nrows=2, ncols=3, height_ratios=[1.0, 0.35], hspace=0.12, wspace=0.03)
    panels = []
    for col in range(3):
        graph_ax = fig.add_subplot(outer[0, col])
        info_ax = fig.add_subplot(outer[1, col])
        info_ax.axis("off")
        panels.append((graph_ax, info_ax))

    caption_ax = fig.add_axes([0.05, 0.93, 0.90, 0.05])
    caption_ax.axis("off")
    caption_ax.text(
        0.5,
        0.72,
        "Phase 1.1B — comparing the same sparse rule m = floor(sqrt(n)) for n = 100, 1000, and 10,000",
        ha="center",
        va="center",
        fontsize=17,
        fontweight="bold",
    )
    caption_ax.text(
        0.5,
        0.18,
        "As n grows, the average degree 2m/n shrinks toward 0, so the graph should look increasingly like a random forest.",
        ha="center",
        va="center",
        fontsize=11,
        color="0.25",
    )

    def update(frame_index: int) -> None:
        current_global = frame_schedule[frame_index]

        for panel_index, run in enumerate(runs):
            graph_ax, info_ax = panels[panel_index]
            graph_ax.clear()
            info_ax.clear()
            info_ax.axis("off")

            edge_count = min(current_global, run.m_target)
            nodes = list(range(run.n))
            g = build_graph_state(nodes=nodes, edges_in_order=run.edge_order, edge_count=edge_count)
            summary = summarize_forest_state(g)

            non_isolates = [u for u, degree in g.degree() if degree > 0]
            isolates = [u for u, degree in g.degree() if degree == 0]

            nx.draw_networkx_edges(
                g,
                pos=run.positions,
                ax=graph_ax,
                width=0.65 if run.n >= 1000 else 1.0,
                alpha=0.70,
                edge_color="#4a5568",
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
                edgecolors="#0d3b66",
                node_color="#b9dcff" if summary.is_forest else "#ffd4d4",
            )

            graph_ax.set_xlim(-1.12, 1.12)
            graph_ax.set_ylim(-1.12, 1.12)
            graph_ax.set_aspect("equal")
            style_graph_box(graph_ax)
            graph_ax.set_title(f"n = {run.n:,}", fontsize=15, pad=14)

            avg_degree = 2.0 * edge_count / run.n
            info_ax.add_patch(
                FancyBboxPatch(
                    (0.03, 0.08),
                    0.94,
                    0.84,
                    transform=info_ax.transAxes,
                    boxstyle="round,pad=0.02",
                    linewidth=0.95,
                    edgecolor="#cfcfcf",
                    facecolor="#fcfcfc",
                    zorder=0,
                )
            )
            info_ax.text(
                0.5,
                0.73,
                f"m = {edge_count}/{run.m_target} = floor(n^{alpha:.2f}) at completion",
                transform=info_ax.transAxes,
                ha="center",
                va="center",
                fontsize=10.4,
                fontweight="bold",
            )
            info_ax.text(
                0.5,
                0.48,
                f"avg degree = {avg_degree:.3f}   |   isolates = {summary.isolates:,}   |   largest component = {summary.largest_component}",
                transform=info_ax.transAxes,
                ha="center",
                va="center",
                fontsize=9.8,
            )
            info_ax.text(
                0.5,
                0.21,
                f"all components are trees? {'YES' if summary.is_forest else 'NO'}   |   cyclic components = {summary.cyclic_components}",
                transform=info_ax.transAxes,
                ha="center",
                va="center",
                fontsize=10.2,
                fontweight="bold",
                color="#1b5e20" if summary.is_forest else "#8b0000",
                bbox=dict(
                    facecolor="#e7f6e7" if summary.is_forest else "#ffe5e5",
                    edgecolor="#2e7d32" if summary.is_forest else "#b71c1c",
                    boxstyle="round,pad=0.24",
                ),
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
            "Create the Phase 1.1B comparison animation across n = 100, 1000, and 10,000 using "
            "the sparse Phase 1 rule m = floor(n^alpha)."
        )
    )
    parser.add_argument("--ns", type=int, nargs="+", default=DEFAULT_NS, help="Values of n to compare.")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help="Exponent in m = floor(n^alpha), should stay below 1.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Base random seed.")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help="Frames per second.")
    parser.add_argument("--hold-final-frames", type=int, default=DEFAULT_HOLD_FINAL_FRAMES, help="Still frames at the end.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output animation path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    alpha = min(max(args.alpha, 0.05), 0.95)

    saved_to = make_phase1_comparison_animation(
        ns=args.ns,
        alpha=alpha,
        seed=args.seed,
        fps=max(1, args.fps),
        hold_final_frames=max(0, args.hold_final_frames),
        output_path=args.output,
    )
    print(f"n values = {[n for n in args.ns if n >= 2]}")
    print(f"alpha = {alpha:.3f}")
    print(f"saved animation to: {saved_to}")


if __name__ == "__main__":
    main()
