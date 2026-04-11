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
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea


# ============================================================
# Phase 4.1A animation.
# Default output matches the HTML placeholder:
#     erdos_renyi_1960_phase4_1a.mp4
#
# Tailored idea:
#   Enter the connectivity window at
#       m = (n/2) log n + y n
#   and watch one overwhelmingly large component remain while the outside
#   world thins down to a few tiny tree remnants.  The checkpoints y=-1,0,1
#   make the window concrete.
# ============================================================

DEFAULT_N = 1000
DEFAULT_SEED = 24680
DEFAULT_FPS = 4
DEFAULT_FRAMES_PER_EDGE = 2
DEFAULT_HOLD_FINAL_FRAMES = 24
DEFAULT_OUTPUT = Path("./erdos_renyi_1960_phase4_1a.mp4")

WINDOW_Y_VALUES = [-1.0, 0.0, 1.0]
WINDOW_START_Y = -1.5
WINDOW_END_Y = 1.5

CATEGORY_EDGE_COLORS = {
    "giant": "#1565c0",
    "tree": "#6b7280",
    "unicyclic": "#c48b18",
    "multicyclic": "#c62828",
}
CATEGORY_NODE_COLORS = {
    "giant": "#d9ebff",
    "tree": "#eceff3",
    "unicyclic": "#fff0cf",
    "multicyclic": "#ffe1e1",
    "isolate": "#ffffff",
}
CATEGORY_LABELS = {
    "giant": "Dominant component",
    "tree": "Outside tree remnants",
    "unicyclic": "Outside cyclic remnants",
    "multicyclic": "Other outside pieces",
}


@dataclass(frozen=True)
class PhaseEvent:
    edge_index: int
    label: str


@dataclass(frozen=True)
class Snapshot:
    edge_index: int
    y_value: float
    observed_outside_vertices: int
    observed_isolates: int
    observed_connected: bool
    theory_isolates_mean: float
    theory_conn_prob: float


@dataclass(frozen=True)
class GraphSummary:
    giant_nodes: List[int]
    giant_size: int
    second_size: int
    components: int
    isolates: int
    outside_vertices: int
    kind_by_node: Dict[int, str]
    edge_kind: Dict[Tuple[int, int], str]
    tree_counts: Dict[int, int]
    connected: bool


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


def summarize_graph(g: nx.Graph) -> GraphSummary:
    components_raw = []
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

        edges = list(h.edges())
        for node in node_list:
            kind_by_node[node] = kind
        for edge in edges:
            edge_kind[tuple(sorted(edge))] = kind

        components_raw.append({"nodes": node_list, "size": v, "kind": kind})

    components_raw.sort(key=lambda item: item["size"], reverse=True)
    giant_nodes = components_raw[0]["nodes"] if components_raw else []
    giant_size = components_raw[0]["size"] if components_raw else 0
    second_size = components_raw[1]["size"] if len(components_raw) > 1 else 0
    isolates = tree_counts.get(1, 0)

    return GraphSummary(
        giant_nodes=list(giant_nodes),
        giant_size=giant_size,
        second_size=second_size,
        components=len(components_raw),
        isolates=isolates,
        outside_vertices=g.number_of_nodes() - giant_size,
        kind_by_node=kind_by_node,
        edge_kind=edge_kind,
        tree_counts=tree_counts,
        connected=(len(components_raw) == 1),
    )


def component_groups(
    g: nx.Graph,
    summary: GraphSummary,
) -> Tuple[Dict[str, List[int]], Dict[str, List[Tuple[int, int]]]]:
    node_groups = {"giant": [], "tree": [], "unicyclic": [], "multicyclic": []}
    edge_groups = {"giant": [], "tree": [], "unicyclic": [], "multicyclic": []}

    giant_node_set = set(summary.giant_nodes)
    for node in g.nodes():
        if node in giant_node_set:
            node_groups["giant"].append(node)
        else:
            node_groups[summary.kind_by_node.get(node, "tree")].append(node)

    for u, v in g.edges():
        edge = tuple(sorted((u, v)))
        if u in giant_node_set and v in giant_node_set:
            edge_groups["giant"].append((u, v))
        else:
            edge_groups[summary.edge_kind.get(edge, "tree")].append((u, v))

    return node_groups, edge_groups


def benchmark_edges(n: int) -> Dict[float, int]:
    return {float(y): connectivity_window_edges(n=n, y=float(y)) for y in WINDOW_Y_VALUES}


def collect_events_and_snapshots(
    nodes: Sequence[int],
    edges_in_order: Sequence[Tuple[int, int]],
    n: int,
) -> Tuple[List[PhaseEvent], Dict[int, Snapshot]]:
    g = nx.Graph()
    g.add_nodes_from(nodes)

    bench_map = benchmark_edges(n)
    bench_edges = set(bench_map.values())
    snapshots: Dict[int, Snapshot] = {}
    events: List[PhaseEvent] = []

    first_connected_logged = False
    outside_below_ten_logged = False

    for idx, edge in enumerate(edges_in_order, start=1):
        g.add_edge(*edge)
        summary = summarize_graph(g)

        if (not outside_below_ten_logged) and summary.outside_vertices <= 10:
            outside_below_ten_logged = True
            events.append(
                PhaseEvent(
                    idx,
                    f"Outside world first shrinks to at most 10 vertices at m = {idx:,}",
                )
            )

        if (not first_connected_logged) and summary.connected:
            first_connected_logged = True
            events.append(
                PhaseEvent(
                    idx,
                    f"Graph first becomes connected at m = {idx:,}",
                )
            )

        if idx in bench_edges:
            log_n = math.log(max(n, 2))
            y_now = (idx - 0.5 * n * log_n) / n
            snapshots[idx] = Snapshot(
                edge_index=idx,
                y_value=y_now,
                observed_outside_vertices=summary.outside_vertices,
                observed_isolates=summary.isolates,
                observed_connected=summary.connected,
                theory_isolates_mean=theory_mean_isolates(y_now),
                theory_conn_prob=theory_connected_probability(y_now),
            )
            events.append(
                PhaseEvent(
                    idx,
                    f"Checkpoint y ≈ {y_now:.1f}: outside = {summary.outside_vertices}, isolates = {summary.isolates}",
                )
            )

    if edges_in_order:
        events.append(PhaseEvent(len(edges_in_order), f"Stopped after the connectivity-window sweep at m = {len(edges_in_order):,}"))

    return events, snapshots


def latest_event_text(edge_count: int, events: List[PhaseEvent], linger_edges: int) -> str:
    active = [e for e in events if e.edge_index <= edge_count < e.edge_index + linger_edges]
    return active[-1].label if active else ""


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


def make_legend_handles() -> List[Line2D]:
    order = ["giant", "tree", "unicyclic", "multicyclic"]
    handles: List[Line2D] = []
    for key in order:
        handles.append(
            Line2D(
                [0],
                [0],
                color=CATEGORY_EDGE_COLORS[key],
                lw=2.5,
                marker="o",
                markersize=7,
                markerfacecolor=CATEGORY_NODE_COLORS["isolate" if key == "tree" else key],
                markeredgecolor=CATEGORY_EDGE_COLORS[key],
                label=CATEGORY_LABELS[key],
            )
        )
    return handles


def add_colored_edgecount_line(
    info_ax,
    edge_count: int,
    max_edges: int,
    avg_degree: float,
    density: float,
    giant_size: int,
) -> None:
    left = TextArea("edges shown = ", textprops=dict(color="#111111", fontsize=10))
    middle = TextArea(f"{edge_count:,}/{max_edges:,}", textprops=dict(color="#c62828", fontsize=10, fontweight="bold"))
    right = TextArea(
        f"   |   avg degree = {avg_degree:.3f}   |   density = {density:.6f}   |   dominant component = {giant_size}",
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


def snapshot_text(current_edge_count: int, snapshots: Dict[int, Snapshot], n: int) -> str:
    pieces = []
    for y in WINDOW_Y_VALUES:
        m = connectivity_window_edges(n=n, y=y)
        snap = snapshots.get(m)
        if snap is None or current_edge_count < m:
            pieces.append(f"y={y:.1f}: pending")
        else:
            conn_word = "yes" if snap.observed_connected else "no"
            pieces.append(
                f"y={y:.1f}: outside={snap.observed_outside_vertices}, isolates={snap.observed_isolates}, conn={conn_word}"
            )
    return "   |   ".join(pieces)


def make_phase4_1a_animation(
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

    m_start = connectivity_window_edges(n=n, y=WINDOW_START_Y)
    m_end = connectivity_window_edges(n=n, y=WINDOW_END_Y)
    total_possible_edges = n * (n - 1) // 2

    edge_order = ordered_random_edges(nodes=nodes, rng=rng)[:m_end]
    events, snapshots = collect_events_and_snapshots(nodes=nodes, edges_in_order=edge_order, n=n)

    frame_schedule: List[int] = [m_start]
    for edge_count in range(m_start + 1, m_end + 1):
        frame_schedule.extend([edge_count] * frames_per_edge)
    frame_schedule.extend([m_end] * hold_final_frames)

    fig = plt.figure(figsize=(10.8, 9.4))
    title_ax = fig.add_axes([0.05, 0.885, 0.90, 0.085])
    title_ax.axis("off")
    graph_ax = fig.add_axes([0.05, 0.32, 0.90, 0.52])
    info_ax = fig.add_axes([0.05, 0.05, 0.90, 0.21])
    info_ax.axis("off")
    fig.patch.set_facecolor("white")

    title_ax.text(
        0.5,
        0.78,
        f"Phase 4.1A — entering the connectivity window (n = {n:,})",
        ha="center",
        va="center",
        fontsize=15.0,
    )
    subtitle_text = fill(
        r"Grow one fixed $G(n,m)$ process through the window $m=\frac{n}{2}\log n + y n$, from "
        + f"$y\\approx {WINDOW_START_Y:.1f}$ to $y\\approx {WINDOW_END_Y:.1f}$. The picture should be one dominant component plus a very thin outside world made of tiny remnants.",
        width=88,
    )
    title_ax.text(
        0.5,
        0.18,
        subtitle_text,
        ha="center",
        va="center",
        fontsize=9.5,
        color="0.22",
        linespacing=1.25,
    )

    fig.legend(
        handles=make_legend_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.255),
        ncol=4,
        frameon=True,
        fontsize=9.8,
        handlelength=2.2,
        columnspacing=1.5,
    )

    def update(frame_index: int) -> None:
        edge_count = frame_schedule[frame_index]
        g = build_graph_state(nodes=nodes, edges_in_order=edge_order, edge_count=edge_count)
        summary = summarize_graph(g)
        node_groups, edge_groups = component_groups(g, summary)

        graph_ax.clear()
        info_ax.clear()
        info_ax.axis("off")

        for key, width, alpha in [
            ("tree", 1.05, 0.82),
            ("unicyclic", 1.20, 0.90),
            ("multicyclic", 1.25, 0.92),
            ("giant", 2.55, 0.94),
        ]:
            if edge_groups[key]:
                nx.draw_networkx_edges(
                    g,
                    pos=positions,
                    edgelist=edge_groups[key],
                    ax=graph_ax,
                    width=width,
                    alpha=alpha,
                    edge_color=CATEGORY_EDGE_COLORS[key],
                )

        if node_groups["tree"]:
            isolates = [u for u in node_groups["tree"] if g.degree(u) == 0]
            non_isolates_tree = [u for u in node_groups["tree"] if g.degree(u) > 0]
        else:
            isolates = []
            non_isolates_tree = []

        if isolates:
            nx.draw_networkx_nodes(
                g,
                pos=positions,
                nodelist=isolates,
                ax=graph_ax,
                node_size=20 if n >= 1000 else 42,
                linewidths=0.45,
                edgecolors=CATEGORY_EDGE_COLORS["tree"],
                node_color=CATEGORY_NODE_COLORS["isolate"],
            )
        if non_isolates_tree:
            nx.draw_networkx_nodes(
                g,
                pos=positions,
                nodelist=sorted(non_isolates_tree),
                ax=graph_ax,
                node_size=20 if n >= 1000 else 42,
                linewidths=0.45,
                edgecolors=CATEGORY_EDGE_COLORS["tree"],
                node_color=CATEGORY_NODE_COLORS["tree"],
            )

        for key in ["unicyclic", "multicyclic", "giant"]:
            if node_groups[key]:
                nx.draw_networkx_nodes(
                    g,
                    pos=positions,
                    nodelist=sorted(node_groups[key]),
                    ax=graph_ax,
                    node_size=20 if n >= 1000 else 42,
                    linewidths=0.45,
                    edgecolors=CATEGORY_EDGE_COLORS[key],
                    node_color=CATEGORY_NODE_COLORS[key],
                )

        graph_ax.set_xlim(-1.12, 1.12)
        graph_ax.set_ylim(-1.12, 1.12)
        graph_ax.set_aspect("equal")
        graph_ax.axis("off")

        avg_degree = 2.0 * edge_count / n
        density = 2.0 * edge_count / total_possible_edges if total_possible_edges > 0 else 0.0
        add_colored_edgecount_line(info_ax, edge_count=edge_count, max_edges=m_end, avg_degree=avg_degree, density=density, giant_size=summary.giant_size)

        log_n = math.log(max(n, 2))
        y_now = (edge_count - 0.5 * n * log_n) / n
        mid_text = fill(
            (
                f"y ≈ {y_now:.2f}   |   outside vertices = {summary.outside_vertices:,}   |   isolates = {summary.isolates:,}   |   "
                f"outside tree counts T1/T2/T3 = {summary.tree_counts.get(1, 0):,} / {summary.tree_counts.get(2, 0):,} / {summary.tree_counts.get(3, 0):,}   |   "
                f"connected = {'yes' if summary.connected else 'no'}"
            ),
            width=104,
        )
        info_ax.text(
            0.5,
            0.51,
            mid_text,
            transform=info_ax.transAxes,
            ha="center",
            va="center",
            fontsize=9.4,
            linespacing=1.18,
            bbox=dict(facecolor="white", edgecolor="0.85", boxstyle="round,pad=0.28"),
        )

        event_text = latest_event_text(edge_count=edge_count, events=events, linger_edges=8)
        bottom_text = (
            f"The connectivity-window heuristic says the outside world should shrink to just a few tiny remnants. "
            f"Theory at current y gives mean isolates ≈ {theory_mean_isolates(y_now):.2f} and connectivity probability ≈ {theory_connected_probability(y_now):.3f}. "
            f"{event_text}\n"
            f"Checkpoint summary: {snapshot_text(edge_count, snapshots, n)}"
        ).strip()
        info_ax.text(
            0.5,
            0.16,
            fill(bottom_text, width=112),
            transform=info_ax.transAxes,
            ha="center",
            va="center",
            fontsize=9.0,
            linespacing=1.28,
            bbox=dict(facecolor="#f9f9f9", edgecolor="0.76", boxstyle="round,pad=0.30"),
        )

    anim = FuncAnimation(fig, update, frames=len(frame_schedule), interval=1000 / fps, repeat=False)
    saved_to = save_animation(anim=anim, output_path=output_path, fps=fps)
    plt.close(fig)
    return saved_to


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create the Phase 4.1A animation showing one dominant component plus a few tiny remnants in the connectivity window.")
    parser.add_argument("--n", type=int, default=DEFAULT_N, help="Number of nodes.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help="Frames per second.")
    parser.add_argument("--frames-per-edge", type=int, default=DEFAULT_FRAMES_PER_EDGE, help="Frames to hold each edge addition.")
    parser.add_argument("--hold-final-frames", type=int, default=DEFAULT_HOLD_FINAL_FRAMES, help="Still frames at the end.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output animation path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    saved_to = make_phase4_1a_animation(
        n=max(50, args.n),
        seed=args.seed,
        fps=max(1, args.fps),
        frames_per_edge=max(1, args.frames_per_edge),
        hold_final_frames=max(0, args.hold_final_frames),
        output_path=args.output,
    )
    print(f"saved animation to: {saved_to}")


if __name__ == "__main__":
    main()
