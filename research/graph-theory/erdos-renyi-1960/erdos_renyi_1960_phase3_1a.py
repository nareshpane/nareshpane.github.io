#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from textwrap import fill
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea


# ============================================================
# Phase 3.1A animation.
# Default output matches the HTML placeholder:
#     erdos_renyi_1960_phase3_1a.mp4
#
# Styling goal for this revision:
#   Match the visual rhythm and layout of phase1_2a.py more closely:
#   - same layered node placement style
#   - same overall figure layout
#   - same bottom legend placement
#   - same colored edge-count line in the information panel
#   - same slower edge-by-edge feel in the near-critical window
#
# Mathematical goal for Phase 3.1:
#   Stay just below the critical threshold m ~ n/2 and show that the
#   largest component grows noticeably, but is still sublinear and usually
#   tree-like rather than already a giant component.
# ============================================================
DEFAULT_N = 1000
DEFAULT_SEED = 24680
DEFAULT_FPS = 4
DEFAULT_FRAMES_PER_EDGE = 2
DEFAULT_HOLD_FINAL_FRAMES = 20
DEFAULT_OUTPUT = Path("./erdos_renyi_1960_phase3_1a.mp4")

WINDOW_C_VALUES = [0.40, 0.45, 0.49]
WINDOW_START_C = 0.00
WINDOW_END_C = 0.49

CATEGORY_EDGE_COLORS = {
    "largest": "#1565c0",      # blue
    "tree": "#6b7280",         # grey
    "unicyclic": "#c48b18",    # warm gold
    "multicyclic": "#c62828",  # red
}

CATEGORY_NODE_COLORS = {
    "largest": "#d9ebff",
    "tree": "#eceff3",         # light grey
    "unicyclic": "#fff0cf",
    "multicyclic": "#ffe1e1",
}

CATEGORY_LABELS = {
    "largest": "Largest component",
    "tree": "Other tree components",
    "unicyclic": "Other unicyclic components",
    "multicyclic": "Other multicyclic components",
}


@dataclass(frozen=True)
class PhaseEvent:
    edge_index: int
    label: str


@dataclass(frozen=True)
class Snapshot:
    edge_index: int
    c_value: float
    predicted_largest: Optional[float]
    observed_largest: int
    observed_kind: str


@dataclass(frozen=True)
class GraphSummary:
    largest_nodes: List[int]
    largest_edges: List[Tuple[int, int]]
    largest_size: int
    largest_kind: str
    second_size: int
    components: int
    isolates: int
    kind_by_node: Dict[int, str]
    edge_kind: Dict[Tuple[int, int], str]


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


def subcritical_largest_theory(n: int, c: float) -> Optional[float]:
    if c <= 0.0 or c >= 0.5 or n <= 3:
        return None
    alpha = 2.0 * c - 1.0 - math.log(2.0 * c)
    if alpha <= 0:
        return None
    return max(0.0, (math.log(n) - 2.5 * math.log(max(math.log(n), 1.0001))) / alpha)


def summarize_graph(g: nx.Graph) -> GraphSummary:
    components_raw = []
    kind_by_node: Dict[int, str] = {}
    edge_kind: Dict[Tuple[int, int], str] = {}

    for comp_nodes in nx.connected_components(g):
        node_list = list(comp_nodes)
        h = g.subgraph(node_list)
        v = h.number_of_nodes()
        e = h.number_of_edges()

        if e == v - 1:
            kind = "tree"
        elif e == v:
            kind = "unicyclic"
        else:
            kind = "multicyclic"

        edges = list(h.edges())
        for node in node_list:
            kind_by_node[node] = kind
        for edge in edges:
            edge_kind[tuple(sorted(edge))] = kind

        components_raw.append(
            {
                "nodes": node_list,
                "edges": edges,
                "size": v,
                "kind": kind,
            }
        )

    components_raw.sort(key=lambda item: item["size"], reverse=True)

    if components_raw:
        largest = components_raw[0]
        second_size = components_raw[1]["size"] if len(components_raw) > 1 else 0
    else:
        largest = {"nodes": [], "edges": [], "size": 0, "kind": "tree"}
        second_size = 0

    isolates = sum(1 for _, degree in g.degree() if degree == 0)

    return GraphSummary(
        largest_nodes=list(largest["nodes"]),
        largest_edges=list(largest["edges"]),
        largest_size=int(largest["size"]),
        largest_kind=str(largest["kind"]),
        second_size=int(second_size),
        components=len(components_raw),
        isolates=isolates,
        kind_by_node=kind_by_node,
        edge_kind=edge_kind,
    )


def component_groups(g: nx.Graph, summary: GraphSummary) -> Tuple[Dict[str, List[int]], Dict[str, List[Tuple[int, int]]]]:
    node_groups = {"largest": [], "tree": [], "unicyclic": [], "multicyclic": []}
    edge_groups = {"largest": [], "tree": [], "unicyclic": [], "multicyclic": []}

    largest_node_set = set(summary.largest_nodes)
    for node in g.nodes():
        if node in largest_node_set:
            node_groups["largest"].append(node)
        else:
            node_groups[summary.kind_by_node.get(node, "tree")].append(node)

    for u, v in g.edges():
        edge = tuple(sorted((u, v)))
        if u in largest_node_set and v in largest_node_set:
            edge_groups["largest"].append((u, v))
        else:
            edge_groups[summary.edge_kind.get(edge, "tree")].append((u, v))

    return node_groups, edge_groups


def benchmark_edges(n: int) -> Dict[float, int]:
    return {c: max(1, int(round(c * n))) for c in WINDOW_C_VALUES}


def collect_events_and_snapshots(
    nodes: List[int],
    edges_in_order: List[Tuple[int, int]],
    n: int,
) -> Tuple[List[PhaseEvent], Dict[int, Snapshot]]:
    g = nx.Graph()
    g.add_nodes_from(nodes)

    bench_map = benchmark_edges(n)
    bench_edges = set(bench_map.values())
    snapshots: Dict[int, Snapshot] = {}
    events: List[PhaseEvent] = []

    saw_non_tree_largest = False

    for idx, edge in enumerate(edges_in_order, start=1):
        g.add_edge(*edge)
        summary = summarize_graph(g)

        if not saw_non_tree_largest and summary.largest_kind != "tree":
            saw_non_tree_largest = True
            events.append(
                PhaseEvent(
                    idx,
                    f"Largest component first stops being tree-like at m = {idx:,}",
                )
            )

        if idx in bench_edges:
            c_now = idx / n
            pred = subcritical_largest_theory(n=n, c=min(c_now, 0.499999))
            snapshots[idx] = Snapshot(
                edge_index=idx,
                c_value=c_now,
                predicted_largest=pred,
                observed_largest=summary.largest_size,
                observed_kind=summary.largest_kind,
            )
            events.append(
                PhaseEvent(
                    idx,
                    f"Crossed the benchmark c ≈ {c_now:.2f}; largest size is now {summary.largest_size:,}",
                )
            )

    if edges_in_order:
        events.append(PhaseEvent(len(edges_in_order), f"Stopped just below the threshold at m = {len(edges_in_order):,}"))

    return events, snapshots


def latest_event_text(edge_count: int, events: List[PhaseEvent], linger_edges: int) -> str:
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


def make_legend_handles() -> List[Line2D]:
    order = ["largest", "tree", "unicyclic", "multicyclic"]
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
                markerfacecolor=CATEGORY_NODE_COLORS[key],
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


def snapshot_text(current_edge_count: int, snapshots: Dict[int, Snapshot], n: int) -> str:
    pieces = []
    for c in WINDOW_C_VALUES:
        m = max(1, int(round(c * n)))
        snap = snapshots.get(m)
        if snap is None or current_edge_count < m:
            pieces.append(f"c={c:.2f}: pending")
        else:
            pred_text = "—" if snap.predicted_largest is None else f"{snap.predicted_largest:.0f}"
            pieces.append(
                f"c={c:.2f}: pred {pred_text}, obs {snap.observed_largest} ({snap.observed_kind})"
            )
    return "   |   ".join(pieces)


def make_phase3_1a_animation(
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

    m_start = max(0, int(round(WINDOW_START_C * n)))
    m_end = max(m_start, int(round(WINDOW_END_C * n)))
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
        f"Phase 3.1A — just below $n/2$, the largest component is still sublinear (n = {n:,})",
        ha="center",
        va="center",
        fontsize=15.2,
    )
    subtitle_text = fill(
        rf"Growing one $G(n,m)$ sample from $m = 0$ up to $m \approx {WINDOW_END_C:.2f}n = {m_end:,}$. The near-critical checkpoints $c \approx 0.40$, $0.45$, and $0.49$ are preserved, but the largest component should still remain below giant-component scale and usually tree-like.",
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

        nx.draw_networkx_edges(
            g,
            pos=positions,
            edgelist=edge_groups["tree"],
            ax=graph_ax,
            width=1.05,
            alpha=0.82,
            edge_color=CATEGORY_EDGE_COLORS["tree"],
        )
        if edge_groups["unicyclic"]:
            nx.draw_networkx_edges(
                g,
                pos=positions,
                edgelist=edge_groups["unicyclic"],
                ax=graph_ax,
                width=1.20,
                alpha=0.90,
                edge_color=CATEGORY_EDGE_COLORS["unicyclic"],
            )
        if edge_groups["multicyclic"]:
            nx.draw_networkx_edges(
                g,
                pos=positions,
                edgelist=edge_groups["multicyclic"],
                ax=graph_ax,
                width=1.30,
                alpha=0.92,
                edge_color=CATEGORY_EDGE_COLORS["multicyclic"],
            )
        if edge_groups["largest"]:
            nx.draw_networkx_edges(
                g,
                pos=positions,
                edgelist=edge_groups["largest"],
                ax=graph_ax,
                width=2.55,
                alpha=0.94,
                edge_color=CATEGORY_EDGE_COLORS["largest"],
            )

        for key in ["tree", "unicyclic", "multicyclic", "largest"]:
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
        add_colored_edgecount_line(
            info_ax=info_ax,
            edge_count=edge_count,
            max_edges=m_end,
            avg_degree=avg_degree,
            density=density,
            largest_component=summary.largest_size,
        )

        benchmark_line = fill(
            (
                f"benchmarks: c≈0.40 → {int(round(0.40 * n)):,}   |   "
                f"c≈0.45 → {int(round(0.45 * n)):,}   |   "
                f"c≈0.49 → {int(round(0.49 * n)):,}   ||   "
                f"largest kind = {summary.largest_kind}   |   "
                f"second largest = {summary.second_size:,}   |   "
                f"components = {summary.components:,}   |   "
                f"isolates = {summary.isolates:,}"
            ),
            width=105,
        )
        info_ax.text(
            0.5,
            0.51,
            benchmark_line,
            transform=info_ax.transAxes,
            ha="center",
            va="center",
            fontsize=9.5,
            linespacing=1.18,
            bbox=dict(facecolor="white", edgecolor="0.85", boxstyle="round,pad=0.28"),
        )

        event_text = latest_event_text(edge_count, events, linger_edges=4)
        interpretation_line = fill(
            (
                "Interpretation: below $n/2$, the largest component can look noticeably bigger than in Phase 2, "
                "but it still should not capture a fixed fraction of all vertices. "
                f"{event_text}"
            ).strip(),
            width=110,
        )
        checkpoint_line = fill(
            f"Predicted vs observed largest size at checkpoints: {snapshot_text(edge_count, snapshots, n)}",
            width=104,
        )
        bottom_text = interpretation_line + "\n" + checkpoint_line
        info_ax.text(
            0.5,
            0.13,
            bottom_text,
            transform=info_ax.transAxes,
            ha="center",
            va="center",
            fontsize=9.1,
            linespacing=1.22,
            bbox=dict(facecolor="#f9f9f9", edgecolor="0.75", boxstyle="round,pad=0.32"),
        )

    anim = FuncAnimation(fig, update, frames=len(frame_schedule), interval=1000 / fps, repeat=False)
    saved_to = save_animation(anim=anim, output_path=output_path, fps=fps)
    plt.close(fig)
    return saved_to


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create the Phase 3.1A animation for the near-critical subcritical regime, "
            "styled to match phase1_2a.py more closely."
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
    saved_to = make_phase3_1a_animation(
        n=max(50, args.n),
        seed=args.seed,
        fps=max(1, args.fps),
        frames_per_edge=max(1, args.frames_per_edge),
        hold_final_frames=max(0, args.hold_final_frames),
        output_path=args.output,
    )
    print(f"Saved animation to: {saved_to}")


if __name__ == "__main__":
    main()
