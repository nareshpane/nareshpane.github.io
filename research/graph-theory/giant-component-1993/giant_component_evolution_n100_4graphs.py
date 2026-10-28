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
from matplotlib.patches import FancyBboxPatch, Rectangle


# ============================================================
# EDIT THESE DEFAULTS, THEN RUN:
#     python giant_component_evolution_n100.py
# ============================================================
DEFAULT_N = 100
DEFAULT_NUM_RUNS = 4
DEFAULT_NUM_COLS = 2
DEFAULT_BASE_SEED = 12345
DEFAULT_FPS = 4
DEFAULT_FRAMES_PER_EDGE = 3
DEFAULT_CHECKPOINT_PAUSE_FRAMES = 16
DEFAULT_HOLD_FINAL_FRAMES = 18
DEFAULT_STOP_EDGES = 75
DEFAULT_OUTPUT = Path("./components_up_to_m75.mp4")


@dataclass(frozen=True)
class AnimationEvent:
    edge_index: int
    label: str


@dataclass(frozen=True)
class ComponentTypeSummary:
    tree_count: int
    unicyclic_count: int
    bicyclic_count: int
    complex_count: int  # excess >= 3

    @property
    def total_components(self) -> int:
        return self.tree_count + self.unicyclic_count + self.bicyclic_count + self.complex_count

    @property
    def all_components_simple(self) -> bool:
        return self.complex_count == 0

    @property
    def has_unicyclic(self) -> bool:
        return self.unicyclic_count > 0

    @property
    def has_bicyclic(self) -> bool:
        return self.bicyclic_count > 0

    @property
    def has_complex(self) -> bool:
        return self.complex_count > 0


@dataclass
class RunData:
    seed: int
    positions: Dict[int, Tuple[float, float]]
    edge_order: List[Tuple[int, int]]
    events: List[AnimationEvent]


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


def classify_component_types(g: nx.Graph) -> ComponentTypeSummary:
    """
    For each connected component H:
        excess(H) = |E(H)| - |V(H)| + 1

    excess = 0  -> tree
    excess = 1  -> unicyclic
    excess = 2  -> bicyclic
    excess >= 3 -> complex

    Isolated vertices count as tree components.
    """
    tree_count = 0
    unicyclic_count = 0
    bicyclic_count = 0
    complex_count = 0

    for component_nodes in nx.connected_components(g):
        h = g.subgraph(component_nodes)
        v = h.number_of_nodes()
        e = h.number_of_edges()
        excess = e - v + 1

        if excess == 0:
            tree_count += 1
        elif excess == 1:
            unicyclic_count += 1
        elif excess == 2:
            bicyclic_count += 1
        else:
            complex_count += 1

    return ComponentTypeSummary(
        tree_count=tree_count,
        unicyclic_count=unicyclic_count,
        bicyclic_count=bicyclic_count,
        complex_count=complex_count,
    )


def latest_event_text(edge_count: int, events: List[AnimationEvent], linger_edges: int) -> str:
    active = [e for e in events if e.edge_index <= edge_count < e.edge_index + linger_edges]
    if not active:
        return ""
    return active[-1].label


def collect_events_until(
    nodes: List[int],
    edges_in_order: List[Tuple[int, int]],
    max_edges: int,
    checkpoint_edge: int,
) -> List[AnimationEvent]:
    g = nx.Graph()
    g.add_nodes_from(nodes)
    events: List[AnimationEvent] = []

    first_cycle = False
    first_unicyclic = False
    first_bicyclic = False
    first_complex = False
    checkpoint_marked = False

    for idx, edge in enumerate(edges_in_order[:max_edges], start=1):
        g.add_edge(*edge)
        summary = classify_component_types(g)

        if not first_cycle and len(nx.cycle_basis(g)) > 0:
            first_cycle = True
            events.append(AnimationEvent(idx, "First cycle appears"))

        if not first_unicyclic and summary.has_unicyclic:
            first_unicyclic = True
            events.append(AnimationEvent(idx, "First unicyclic component appears"))

        if not first_bicyclic and summary.has_bicyclic:
            first_bicyclic = True
            events.append(AnimationEvent(idx, "First bicyclic component appears"))

        if not first_complex and summary.has_complex:
            first_complex = True
            events.append(AnimationEvent(idx, "A component with excess ≥ 3 appears"))

        if not checkpoint_marked and idx == checkpoint_edge:
            checkpoint_marked = True
            events.append(AnimationEvent(idx, f"Checkpoint reached: m = n/2 = {checkpoint_edge}"))

        if idx == max_edges:
            events.append(AnimationEvent(idx, f"Stopped at m = {max_edges}"))

    return events


def choose_grid(num_runs: int, requested_cols: int) -> Tuple[int, int]:
    ncols = max(1, min(requested_cols, num_runs))
    nrows = math.ceil(num_runs / ncols)
    return nrows, ncols


def build_frame_schedule(
    max_edges: int,
    checkpoint_edge: int,
    frames_per_edge: int,
    checkpoint_pause_frames: int,
    hold_final_frames: int,
) -> List[int]:
    schedule: List[int] = [0]
    for edge_count in range(1, max_edges + 1):
        schedule.extend([edge_count] * frames_per_edge)
        if edge_count == checkpoint_edge:
            schedule.extend([edge_count] * checkpoint_pause_frames)
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


def draw_badge_center(ax, x: float, y: float, label: str, count: int, face: str, edge: str) -> None:
    present = count > 0
    ax.text(
        x,
        y,
        f"{label}: {count}",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10.4,
        fontweight="bold",
        color="#111111" if present else "#666666",
        bbox=dict(
            facecolor=face if present else "#f3f3f3",
            edgecolor=edge if present else "#aaaaaa",
            boxstyle="round,pad=0.22",
        ),
        zorder=5,
    )


def style_graph_box(ax, checkpoint_now: bool) -> None:
    base_color = "#4f4f4f"
    checkpoint_color = "#c62828"
    color = checkpoint_color if checkpoint_now else base_color
    lw = 2.5 if checkpoint_now else 2.1

    ax.set_facecolor("white")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.set_frame_on(True)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(lw)
        spine.set_edgecolor(color)
        spine.set_zorder(200)

    ax.patch.set_edgecolor(color)
    ax.patch.set_linewidth(lw)

    ax.add_patch(
        Rectangle(
            (0.003, 0.003),
            0.994,
            0.994,
            transform=ax.transAxes,
            fill=False,
            linewidth=lw,
            edgecolor=color,
            zorder=250,
            clip_on=False,
        )
    )


def make_multi_run_animation(
    n: int,
    num_runs: int,
    num_cols: int,
    base_seed: int,
    fps: int,
    frames_per_edge: int,
    checkpoint_pause_frames: int,
    hold_final_frames: int,
    stop_edges: int,
    output_path: Path,
) -> Path:
    nodes = list(range(n))
    checkpoint_edge = n // 2
    max_edges = min(stop_edges, n * (n - 1) // 2)
    max_edges = max(max_edges, checkpoint_edge)
    nrows, ncols = choose_grid(num_runs, num_cols)

    runs: List[RunData] = []
    for i in range(num_runs):
        run_seed = base_seed + 1009 * i
        rng = np.random.default_rng(run_seed)
        positions = layered_circular_layout(n=n, rng=rng)
        full_edge_order = ordered_random_edges(nodes=nodes, rng=rng)
        edge_order = full_edge_order[:max_edges]
        events = collect_events_until(
            nodes=nodes,
            edges_in_order=edge_order,
            max_edges=max_edges,
            checkpoint_edge=checkpoint_edge,
        )
        runs.append(
            RunData(
                seed=run_seed,
                positions=positions,
                edge_order=edge_order,
                events=events,
            )
        )

    frame_schedule = build_frame_schedule(
        max_edges=max_edges,
        checkpoint_edge=checkpoint_edge,
        frames_per_edge=frames_per_edge,
        checkpoint_pause_frames=checkpoint_pause_frames,
        hold_final_frames=hold_final_frames,
    )

    # Landscape-first figure sizing
    fig_width = max(15.5, 7.8 * ncols)
    fig_height = max(8.8, 4.7 * nrows + 3.1)

    fig = plt.figure(figsize=(fig_width, fig_height))
    outer = fig.add_gridspec(
        nrows=nrows + 1,
        ncols=ncols,
        height_ratios=[1.0] * nrows + [0.40],
        hspace=0.12,
        wspace=0.001,
    )

    panels = []
    for idx in range(nrows * ncols):
        r = idx // ncols
        c = idx % ncols
        inner = outer[r, c].subgridspec(2, 1, height_ratios=[8.6, 3.4], hspace=0.004)

        graph_ax = fig.add_subplot(inner[0, 0])
        legend_ax = fig.add_subplot(inner[1, 0])
        legend_ax.axis("off")

        panels.append({"graph": graph_ax, "legend": legend_ax})

    common_ax = fig.add_subplot(outer[-1, :])
    common_ax.axis("off")

    fig.patch.set_facecolor("white")

    fig.suptitle(
        "Evolution of Giant Components in Random Graphs",
        fontsize=19.5,
        fontweight="bold",
        y=0.992,
    )
    fig.text(
        0.5,
        0.962,
        f"{num_runs} independent runs, with checkpoint at m = n/2 = {checkpoint_edge} and stop at m = {max_edges}",
        ha="center",
        va="center",
        fontsize=12.8,
        color="0.28",
    )

    common_caption_1 = (
        "Each panel is a separate random-graph run. The largest connected component "
        "in that panel is highlighted in blue."
    )
    common_caption_2 = (
        "For a connected component H, excess(H) = |E(H)| - |V(H)| + 1. "
        "Tree: 0, unicyclic: 1, bicyclic: 2, complex: ≥ 3. "
        "The key checkpoint here is m = n/2 = 50, while the animation continues to m = 75."
    )

    common_ax.text(
        0.5,
        0.82,
        fill(common_caption_1, width=max(76, 38 * ncols)),
        ha="center",
        va="center",
        fontsize=13.0,
        wrap=True,
        bbox=dict(facecolor="white", edgecolor="0.82", boxstyle="round,pad=0.34"),
    )
    common_ax.text(
        0.5,
        0.42,
        fill(common_caption_2, width=max(76, 38 * ncols)),
        ha="center",
        va="center",
        fontsize=12.7,
        color="0.20",
        wrap=True,
        bbox=dict(facecolor="#fafafa", edgecolor="0.84", boxstyle="round,pad=0.34"),
    )

    edge_counter_text = common_ax.text(
        0.5,
        0.03,
        "",
        ha="center",
        va="bottom",
        fontsize=14.2,
        fontweight="bold",
        color="#c62828",
        bbox=dict(
            facecolor="#ffe8e8",
            edgecolor="#c62828",
            boxstyle="round,pad=0.38",
        ),
    )

    def update(frame_index: int) -> None:
        edge_count = frame_schedule[frame_index]
        checkpoint_now = edge_count == checkpoint_edge

        if checkpoint_now:
            edge_counter_text.set_text(f"m = {edge_count}/{max_edges}   |   checkpoint m = n/2 reached")
            edge_counter_text.set_bbox(
                dict(facecolor="#ffd6d6", edgecolor="#c62828", boxstyle="round,pad=0.40")
            )
        else:
            edge_counter_text.set_text(f"m = {edge_count}/{max_edges}")
            edge_counter_text.set_bbox(
                dict(facecolor="#ffe8e8", edgecolor="#c62828", boxstyle="round,pad=0.38")
            )

        for idx, panel in enumerate(panels):
            graph_ax = panel["graph"]
            legend_ax = panel["legend"]

            if idx >= num_runs:
                graph_ax.clear()
                graph_ax.axis("off")
                legend_ax.clear()
                legend_ax.axis("off")
                continue

            run = runs[idx]
            graph_ax.clear()
            legend_ax.clear()
            legend_ax.axis("off")

            g = build_graph_state(nodes=nodes, edges_in_order=run.edge_order, edge_count=edge_count)
            summary = classify_component_types(g)
            giant_nodes = giant_component_nodes(g)
            giant_size = len(giant_nodes)
            component_sizes = sorted((len(comp) for comp in nx.connected_components(g)), reverse=True)
            second_size = component_sizes[1] if len(component_sizes) > 1 else 0
            isolates = sum(1 for _, deg in g.degree() if deg == 0)

            shown_edges = run.edge_order[:edge_count]
            giant_edges, other_edges = giant_component_edge_lists(shown_edges, giant_nodes)
            newest_edge = run.edge_order[edge_count - 1 : edge_count] if edge_count > 0 else []
            newest_is_giant = bool(newest_edge and newest_edge[0] in giant_edges)

            nx.draw_networkx_edges(
                g,
                pos=run.positions,
                edgelist=other_edges,
                ax=graph_ax,
                width=0.82,
                alpha=0.30,
                edge_color="0.72",
            )
            nx.draw_networkx_edges(
                g,
                pos=run.positions,
                edgelist=giant_edges,
                ax=graph_ax,
                width=1.85,
                alpha=0.78,
                edge_color="#2a6fbb",
            )
            if newest_edge:
                nx.draw_networkx_edges(
                    g,
                    pos=run.positions,
                    edgelist=newest_edge,
                    ax=graph_ax,
                    width=2.55,
                    alpha=0.98,
                    edge_color="#111111" if newest_is_giant else "#4a4a4a",
                )

            non_giant_nodes = [u for u in g.nodes if u not in giant_nodes]
            nx.draw_networkx_nodes(
                g,
                pos=run.positions,
                nodelist=non_giant_nodes,
                ax=graph_ax,
                node_size=30 if n >= 100 else 72,
                linewidths=0.44,
                edgecolors="0.25",
                node_color="white",
            )
            nx.draw_networkx_nodes(
                g,
                pos=run.positions,
                nodelist=sorted(giant_nodes),
                ax=graph_ax,
                node_size=44 if n >= 100 else 88,
                linewidths=0.62,
                edgecolors="#0d3b66",
                node_color="#8ec5ff",
            )

            graph_ax.set_xlim(-1.12, 1.12)
            graph_ax.set_ylim(-1.12, 1.12)
            graph_ax.set_aspect("equal")
            style_graph_box(graph_ax, checkpoint_now=checkpoint_now)

            graph_ax.text(
                0.02,
                1.013,
                f"Run {idx + 1}",
                transform=graph_ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=13.6,
                fontweight="bold",
                color="#c62828",
                bbox=dict(
                    facecolor="#ffe8e8",
                    edgecolor="#c62828",
                    boxstyle="round,pad=0.22",
                ),
            )

            legend_ax.add_patch(
                FancyBboxPatch(
                    (0.028, 0.04),
                    0.944,
                    0.91,
                    transform=legend_ax.transAxes,
                    boxstyle="round,pad=0.02",
                    linewidth=0.95,
                    edgecolor="#cfcfcf",
                    facecolor="#fcfcfc",
                    zorder=0,
                )
            )

            # Tightly grouped, centered badges
            draw_badge_center(legend_ax, 0.24, 0.77, "Trees", summary.tree_count, "#dff4df", "#2e7d32")
            draw_badge_center(legend_ax, 0.42, 0.77, "Unicyclic", summary.unicyclic_count, "#fff4cc", "#9c7b00")
            draw_badge_center(legend_ax, 0.60, 0.77, "Bicyclic", summary.bicyclic_count, "#ffe2cc", "#b85c00")
            draw_badge_center(legend_ax, 0.78, 0.77, "Complex", summary.complex_count, "#ffd9d9", "#b71c1c")

            latest = latest_event_text(edge_count, run.events, linger_edges=4)
            if summary.all_components_simple:
                status_text = "All connected components currently have excess ≤ 2"
                status_face = "#e5f6e5"
                status_edge = "#2e7d32"
                status_color = "#1b5e20"
            else:
                status_text = "A higher-complexity component is present"
                status_face = "#ffe5e5"
                status_edge = "#b71c1c"
                status_color = "#8b0000"

            if latest:
                status_text = f"{status_text} | {latest}"

            combined_caption = (
                f"largest={giant_size} | second={second_size} | isolates={isolates} | comps={summary.total_components}\n"
                f"{fill(status_text, width=46)}"
            )

            legend_ax.text(
                0.5,
                0.23,
                combined_caption,
                transform=legend_ax.transAxes,
                ha="center",
                va="center",
                fontsize=10.1,
                fontweight="bold",
                linespacing=1.34,
                color=status_color,
                bbox=dict(facecolor=status_face, edgecolor=status_edge, boxstyle="round,pad=0.28"),
            )

    anim = FuncAnimation(fig, update, frames=len(frame_schedule), interval=1000 / fps, repeat=False)
    saved_to = save_animation(anim=anim, output_path=output_path, fps=fps)
    plt.close(fig)
    return saved_to


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a multi-panel animation of independent random-graph runs, with checkpoint at "
            "m = n/2 and a later stopping point."
        )
    )
    parser.add_argument("--n", type=int, default=DEFAULT_N, help="Number of nodes.")
    parser.add_argument("--num-runs", type=int, default=DEFAULT_NUM_RUNS, help="How many runs to display.")
    parser.add_argument("--num-cols", type=int, default=DEFAULT_NUM_COLS, help="How many columns of panels.")
    parser.add_argument("--seed", type=int, default=DEFAULT_BASE_SEED, help="Base random seed.")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help="Frames per second.")
    parser.add_argument("--frames-per-edge", type=int, default=DEFAULT_FRAMES_PER_EDGE, help="Frames per revealed edge.")
    parser.add_argument(
        "--checkpoint-pause-frames",
        type=int,
        default=DEFAULT_CHECKPOINT_PAUSE_FRAMES,
        help="Extra pause frames exactly at m = n/2.",
    )
    parser.add_argument("--hold-final-frames", type=int, default=DEFAULT_HOLD_FINAL_FRAMES, help="Still frames at the end.")
    parser.add_argument(
        "--stop-edges",
        type=int,
        default=DEFAULT_STOP_EDGES,
        help="Final stopping edge count. Default is 75.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output animation path. Use .mp4 or .gif.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    saved_to = make_multi_run_animation(
        n=max(2, args.n),
        num_runs=max(1, args.num_runs),
        num_cols=max(1, args.num_cols),
        base_seed=args.seed,
        fps=max(1, args.fps),
        frames_per_edge=max(1, args.frames_per_edge),
        checkpoint_pause_frames=max(0, args.checkpoint_pause_frames),
        hold_final_frames=max(0, args.hold_final_frames),
        stop_edges=max(1, args.stop_edges),
        output_path=args.output,
    )

    print(f"n = {max(2, args.n)}")
    print(f"num_runs = {max(1, args.num_runs)}")
    print(f"num_cols = {max(1, args.num_cols)}")
    print(f"checkpoint at m = floor(n/2) = {max(2, args.n) // 2}")
    print(f"stopping at m = {max(1, args.stop_edges)}")
    print(f"saved animation to: {saved_to}")


if __name__ == "__main__":
    main()
