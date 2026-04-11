#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from textwrap import fill
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea

# ============================================================
# Phase 3.4A animation.
# Default output matches the HTML placeholder:
#     erdos_renyi_1960_phase3_4a.mp4
#
# Tailored idea:
#   Start once the giant is already present, tag one sample tree of order
#   1, 2, and 3, and follow the process as edges continue to arrive.
#   The tagged trees die once they are no longer exactly the same tree
#   component, illustrating the paper's qualitative claim that small trees
#   can survive for a while before melting into the giant.
# ============================================================

DEFAULT_N = 1000
DEFAULT_SEED = 24680
DEFAULT_FPS = 4
DEFAULT_FRAMES_PER_EDGE = 2
DEFAULT_HOLD_FINAL_FRAMES = 24
DEFAULT_OUTPUT = Path("./erdos_renyi_1960_phase3_4a.mp4")

WINDOW_START_C = 0.55
WINDOW_END_C = 1.00
CHECKPOINT_C_VALUES = [0.55, 0.70, 1.00]
TRACKED_SIZES = [1, 2, 3]

CATEGORY_EDGE_COLORS = {
    "giant": "#1565c0",
    "tree": "#6b7280",
    "unicyclic": "#c48b18",
    "multicyclic": "#c62828",
    "tag1": "#8e24aa",
    "tag2": "#2e7d32",
    "tag3": "#ad1457",
}
CATEGORY_NODE_COLORS = {
    "giant": "#d9ebff",
    "tree": "#eceff3",
    "unicyclic": "#fff0cf",
    "multicyclic": "#ffe1e1",
    "tag1": "#f0ddff",
    "tag2": "#e1f5df",
    "tag3": "#ffdce8",
}


@dataclass(frozen=True)
class TaggedTree:
    order: int
    nodes: Tuple[int, ...]
    birth_edge: int


@dataclass(frozen=True)
class TaggedStatus:
    alive: bool
    death_edge: Optional[int]
    current_component_size: int
    current_kind: str


@dataclass(frozen=True)
class GraphSummary:
    giant_nodes: List[int]
    giant_size: int
    second_size: int
    components: int
    isolates: int
    tree_vertices: int
    tree_counts: Dict[int, int]
    kind_by_node: Dict[int, str]
    edge_kind: Dict[Tuple[int, int], str]
    component_by_node: Dict[int, int]
    nodes_by_component: Dict[int, List[int]]
    edges_by_component: Dict[int, List[Tuple[int, int]]]


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


def giant_fraction_theory(c: float, tol: float = 1e-12, max_iter: int = 500) -> float:
    if c <= 0.5:
        return 0.0
    u = math.exp(-2.0 * c)
    for _ in range(max_iter):
        new_u = math.exp(-2.0 * c * (1.0 - u))
        if abs(new_u - u) < tol:
            u = new_u
            break
        u = new_u
    return max(0.0, min(1.0, 1.0 - u))


def summarize_graph(g: nx.Graph) -> GraphSummary:
    components_raw: List[dict] = []
    kind_by_node: Dict[int, str] = {}
    edge_kind: Dict[Tuple[int, int], str] = {}
    component_by_node: Dict[int, int] = {}
    nodes_by_component: Dict[int, List[int]] = {}
    edges_by_component: Dict[int, List[Tuple[int, int]]] = {}
    tree_counts: Dict[int, int] = {}
    tree_vertices = 0

    for idx, comp_nodes in enumerate(nx.connected_components(g)):
        node_list = sorted(comp_nodes)
        h = g.subgraph(node_list)
        v = h.number_of_nodes()
        e = h.number_of_edges()
        if e == v - 1:
            kind = "tree"
            tree_vertices += v
            tree_counts[v] = tree_counts.get(v, 0) + 1
        elif e == v:
            kind = "unicyclic"
        else:
            kind = "multicyclic"

        edges = list(h.edges())
        for node in node_list:
            kind_by_node[node] = kind
            component_by_node[node] = idx
        for edge in edges:
            edge_kind[tuple(sorted(edge))] = kind
        nodes_by_component[idx] = node_list
        edges_by_component[idx] = edges
        components_raw.append({"idx": idx, "nodes": node_list, "size": v, "kind": kind})

    components_raw.sort(key=lambda item: item["size"], reverse=True)
    giant_nodes = components_raw[0]["nodes"] if components_raw else []
    giant_size = components_raw[0]["size"] if components_raw else 0
    second_size = components_raw[1]["size"] if len(components_raw) > 1 else 0
    isolates = tree_counts.get(1, 0)

    return GraphSummary(
        giant_nodes=giant_nodes,
        giant_size=giant_size,
        second_size=second_size,
        components=len(components_raw),
        isolates=isolates,
        tree_vertices=tree_vertices,
        tree_counts=tree_counts,
        kind_by_node=kind_by_node,
        edge_kind=edge_kind,
        component_by_node=component_by_node,
        nodes_by_component=nodes_by_component,
        edges_by_component=edges_by_component,
    )


def pick_initial_tracked_trees(
    nodes: Sequence[int],
    edges_in_order: Sequence[Tuple[int, int]],
    start_edge: int,
    end_edge: int,
) -> Dict[int, TaggedTree]:
    tracked: Dict[int, TaggedTree] = {}
    for edge_count in range(start_edge, end_edge + 1):
        g = build_graph_state(nodes=nodes, edges_in_order=edges_in_order, edge_count=edge_count)
        summary = summarize_graph(g)
        giant_set = set(summary.giant_nodes)
        for comp_id, comp_nodes in summary.nodes_by_component.items():
            comp_nodes_tuple = tuple(sorted(comp_nodes))
            v = len(comp_nodes_tuple)
            if v not in TRACKED_SIZES or v in tracked:
                continue
            if any(node in giant_set for node in comp_nodes_tuple):
                continue
            edges = summary.edges_by_component[comp_id]
            if len(edges) == v - 1:
                tracked[v] = TaggedTree(order=v, nodes=comp_nodes_tuple, birth_edge=edge_count)
        if all(k in tracked for k in TRACKED_SIZES):
            break
    return tracked


def tracked_status(summary: GraphSummary, tag: TaggedTree, edge_count: int, known_death: Optional[int]) -> TaggedStatus:
    if known_death is not None:
        return TaggedStatus(False, known_death, 0, "dead")
    comp_ids = {summary.component_by_node.get(node, -1) for node in tag.nodes}
    if len(comp_ids) != 1 or -1 in comp_ids:
        return TaggedStatus(False, edge_count, 0, "dead")
    comp_id = next(iter(comp_ids))
    current_nodes = tuple(sorted(summary.nodes_by_component[comp_id]))
    current_kind = summary.kind_by_node.get(tag.nodes[0], "tree")
    if current_nodes == tag.nodes and current_kind == "tree":
        return TaggedStatus(True, None, len(current_nodes), current_kind)
    return TaggedStatus(False, edge_count, len(current_nodes), current_kind)


def component_groups(
    g: nx.Graph,
    summary: GraphSummary,
    tracked: Dict[int, TaggedTree],
    death_edges: Dict[int, Optional[int]],
    edge_count: int,
) -> Tuple[Dict[str, List[int]], Dict[str, List[Tuple[int, int]]], Dict[int, TaggedStatus]]:
    node_groups = {"giant": [], "tree": [], "unicyclic": [], "multicyclic": [], "tag1": [], "tag2": [], "tag3": []}
    edge_groups = {"giant": [], "tree": [], "unicyclic": [], "multicyclic": [], "tag1": [], "tag2": [], "tag3": []}
    statuses: Dict[int, TaggedStatus] = {}

    tagged_alive_nodes: set[int] = set()
    tagged_alive_edges: set[Tuple[int, int]] = set()
    for k, tag in tracked.items():
        status = tracked_status(summary=summary, tag=tag, edge_count=edge_count, known_death=death_edges.get(k))
        if not status.alive and death_edges.get(k) is None:
            death_edges[k] = status.death_edge
        statuses[k] = TaggedStatus(
            alive=status.alive,
            death_edge=death_edges.get(k),
            current_component_size=status.current_component_size,
            current_kind=status.current_kind,
        )
        if status.alive:
            node_groups[f"tag{k}"] = list(tag.nodes)
            tagged_alive_nodes.update(tag.nodes)
            comp_id = summary.component_by_node[tag.nodes[0]]
            comp_edges = summary.edges_by_component[comp_id]
            edge_groups[f"tag{k}"] = list(comp_edges)
            tagged_alive_edges.update(tuple(sorted(e)) for e in comp_edges)

    giant_set = set(summary.giant_nodes)
    for node in g.nodes():
        if node in tagged_alive_nodes:
            continue
        if node in giant_set:
            node_groups["giant"].append(node)
        else:
            node_groups[summary.kind_by_node.get(node, "tree")].append(node)

    for u, v in g.edges():
        edge = tuple(sorted((u, v)))
        if edge in tagged_alive_edges:
            continue
        if u in giant_set and v in giant_set:
            edge_groups["giant"].append((u, v))
        else:
            edge_groups[summary.edge_kind.get(edge, "tree")].append((u, v))

    return node_groups, edge_groups, statuses


def latest_event_text(edge_count: int, events: List[Tuple[int, str]], linger_edges: int) -> str:
    active = [label for idx, label in events if idx <= edge_count < idx + linger_edges]
    return active[-1] if active else ""


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


def make_legend_handles() -> List[Line2D]:
    return [
        Line2D([0], [0], color=CATEGORY_EDGE_COLORS["giant"], lw=2.4, marker="o", markersize=7, markerfacecolor=CATEGORY_NODE_COLORS["giant"], markeredgecolor=CATEGORY_EDGE_COLORS["giant"], label="Giant component"),
        Line2D([0], [0], color=CATEGORY_EDGE_COLORS["tree"], lw=2.0, marker="o", markersize=7, markerfacecolor=CATEGORY_NODE_COLORS["tree"], markeredgecolor=CATEGORY_EDGE_COLORS["tree"], label="Other tree components"),
        Line2D([0], [0], color=CATEGORY_EDGE_COLORS["tag1"], lw=2.2, marker="o", markersize=7, markerfacecolor=CATEGORY_NODE_COLORS["tag1"], markeredgecolor=CATEGORY_EDGE_COLORS["tag1"], label="Tagged T1"),
        Line2D([0], [0], color=CATEGORY_EDGE_COLORS["tag2"], lw=2.2, marker="o", markersize=7, markerfacecolor=CATEGORY_NODE_COLORS["tag2"], markeredgecolor=CATEGORY_EDGE_COLORS["tag2"], label="Tagged T2"),
        Line2D([0], [0], color=CATEGORY_EDGE_COLORS["tag3"], lw=2.2, marker="o", markersize=7, markerfacecolor=CATEGORY_NODE_COLORS["tag3"], markeredgecolor=CATEGORY_EDGE_COLORS["tag3"], label="Tagged T3"),
        Line2D([0], [0], color=CATEGORY_EDGE_COLORS["multicyclic"], lw=2.0, marker="o", markersize=7, markerfacecolor=CATEGORY_NODE_COLORS["multicyclic"], markeredgecolor=CATEGORY_EDGE_COLORS["multicyclic"], label="Other cyclic pieces"),
    ]


def add_colored_edgecount_line(info_ax, edge_count: int, max_edges: int, avg_degree: float, density: float, giant_size: int) -> None:
    left = TextArea("edges shown = ", textprops=dict(color="#111111", fontsize=10))
    middle = TextArea(f"{edge_count:,}/{max_edges:,}", textprops=dict(color="#c62828", fontsize=10, fontweight="bold"))
    right = TextArea(
        f"   |   avg degree = {avg_degree:.3f}   |   density = {density:.6f}   |   giant = {giant_size}",
        textprops=dict(color="#111111", fontsize=10),
    )
    packed = HPacker(children=[left, middle, right], align="center", pad=0, sep=0)
    anchored = AnchoredOffsetbox(loc="center", child=packed, pad=0.22, frameon=True, bbox_to_anchor=(0.5, 0.82), bbox_transform=info_ax.transAxes, borderpad=0.32)
    anchored.patch.set_facecolor("white")
    anchored.patch.set_edgecolor("0.85")
    info_ax.add_artist(anchored)


def format_status(k: int, status: TaggedStatus) -> str:
    if status.alive:
        return f"T{k}: alive"
    if status.death_edge is None:
        return f"T{k}: not found"
    return f"T{k}: died at m={status.death_edge:,}"


def make_phase3_4a_animation(
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

    tracked = pick_initial_tracked_trees(nodes=nodes, edges_in_order=edge_order, start_edge=m_start, end_edge=m_end)
    death_edges: Dict[int, Optional[int]] = {k: None for k in TRACKED_SIZES}

    events: List[Tuple[int, str]] = []
    for c in CHECKPOINT_C_VALUES:
        m = int(round(c * n))
        events.append((m, f"Checkpoint c ≈ {c:.2f}: track the remaining tree vertices and the tagged T1/T2/T3 components."))

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
        f"Phase 3.4A — small tree components melt into the giant (n = {n:,})",
        ha="center",
        va="center",
        fontsize=14.9,
    )
    subtitle_text = fill(
        rf"Starting from a supercritical graph at $m \approx {WINDOW_START_C:.2f}n$ and continuing to $m \approx {WINDOW_END_C:.2f}n$. One sample tree of order 1, 2, and 3 is tagged and followed until it stops being exactly that tree component.",
        width=88,
    )
    title_ax.text(0.5, 0.18, subtitle_text, ha="center", va="center", fontsize=9.5, color="0.22", linespacing=1.25)

    fig.legend(handles=make_legend_handles(), loc="lower center", bbox_to_anchor=(0.5, 0.255), ncol=3, frameon=True, fontsize=9.5, handlelength=2.2, columnspacing=1.2)

    def update(frame_index: int) -> None:
        edge_count = frame_schedule[frame_index]
        g = build_graph_state(nodes=nodes, edges_in_order=edge_order, edge_count=edge_count)
        summary = summarize_graph(g)
        node_groups, edge_groups, statuses = component_groups(g, summary, tracked, death_edges, edge_count)

        graph_ax.clear()
        info_ax.clear()
        info_ax.axis("off")

        for key, width, alpha in [
            ("tree", 1.05, 0.82),
            ("unicyclic", 1.20, 0.90),
            ("multicyclic", 1.25, 0.92),
            ("giant", 2.55, 0.94),
            ("tag1", 2.20, 0.98),
            ("tag2", 2.20, 0.98),
            ("tag3", 2.20, 0.98),
        ]:
            if edge_groups[key]:
                nx.draw_networkx_edges(g, pos=positions, edgelist=edge_groups[key], ax=graph_ax, width=width, alpha=alpha, edge_color=CATEGORY_EDGE_COLORS[key])

        for key in ["tree", "unicyclic", "multicyclic", "giant", "tag1", "tag2", "tag3"]:
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

        theory_g = giant_fraction_theory(edge_count / n)
        mid_text = fill(
            (
                f"tree vertices = {summary.tree_vertices:,} ({summary.tree_vertices / n:.3f} of n)   |   "
                f"tree counts T1/T2/T3 = {summary.tree_counts.get(1, 0):,} / {summary.tree_counts.get(2, 0):,} / {summary.tree_counts.get(3, 0):,}   |   "
                f"giant fraction ≈ {summary.giant_size / n:.3f}   |   theory G(c) ≈ {theory_g:.3f}"
            ),
            width=104,
        )
        info_ax.text(0.5, 0.51, mid_text, transform=info_ax.transAxes, ha="center", va="center", fontsize=9.4, linespacing=1.18, bbox=dict(facecolor="white", edgecolor="0.85", boxstyle="round,pad=0.28"))

        status_line = "   |   ".join(format_status(k, statuses.get(k, TaggedStatus(False, None, 0, "missing"))) for k in TRACKED_SIZES)
        event_text = latest_event_text(edge_count=edge_count, events=events, linger_edges=8)
        bottom_text = (
            f"Interpretation: as the giant expands, the total mass in tree components should decrease, and tagged small trees should eventually disappear. {event_text}\n"
            f"Tracked trees: {status_line}"
        ).strip()
        info_ax.text(0.5, 0.16, fill(bottom_text, width=112), transform=info_ax.transAxes, ha="center", va="center", fontsize=9.0, linespacing=1.28, bbox=dict(facecolor="#f9f9f9", edgecolor="0.76", boxstyle="round,pad=0.30"))

    anim = FuncAnimation(fig, update, frames=len(frame_schedule), interval=1000 / fps, repeat=False)
    saved_to = save_animation(anim=anim, output_path=output_path, fps=fps)
    plt.close(fig)
    return saved_to


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create the Phase 3.4A animation tracking small trees as they melt into the giant component.")
    parser.add_argument("--n", type=int, default=DEFAULT_N, help="Number of nodes.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help="Frames per second.")
    parser.add_argument("--frames-per-edge", type=int, default=DEFAULT_FRAMES_PER_EDGE, help="Frames to hold each edge addition.")
    parser.add_argument("--hold-final-frames", type=int, default=DEFAULT_HOLD_FINAL_FRAMES, help="Still frames at the end.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output animation path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    saved_to = make_phase3_4a_animation(
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
