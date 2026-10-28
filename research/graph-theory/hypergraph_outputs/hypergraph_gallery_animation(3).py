#!/usr/bin/env python3
"""
hypergraph_gallery_animation.py

Creates 10 undirected hypergraphs of increasing complexity and renders each one
in four complementary representations:

1. Region-style hypergraph view
2. Incidence bipartite graph
3. PAOH-like vertical incidence view
4. Simple-graph translation via the 2-section / clique expansion

Outputs:
- 10 PNG files, one per hypergraph
- 1 summary PNG with thumbnails of all 10 hypergraphs
- 1 animation (MP4 if ffmpeg is available, otherwise GIF)

Dependencies:
    pip install matplotlib networkx numpy pillow
"""
from __future__ import annotations

import itertools
import math
import os
import shutil
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib import animation
from matplotlib.patches import Circle, Polygon


@dataclass
class HypergraphExample:
    slug: str
    title: str
    vertices: List[str]
    edges: Dict[str, List[str]]
    note: str = ""


PANEL_FACE = "#fafafa"
PANEL_EDGE = "#b8b8b8"

MAIN_TITLE_SIZE = 22
SUBTITLE_SIZE = 12.8
META_SIZE = 13.2
NOTE_SIZE = 12.2
PANEL_TITLE_SIZE = 14.5
LABEL_SIZE = 12.5
SMALL_LABEL_SIZE = 11.5
SUMMARY_TITLE_SIZE = 18.5
SUMMARY_SUBTITLE_SIZE = 12.0


def build_examples() -> List[HypergraphExample]:
    """Ten hand-crafted examples ordered by increasing structural complexity."""
    return [
        HypergraphExample(
            "single_pair",
            "Single Pair",
            ["v1", "v2", "v3"],
            {"e1": ["v1", "v2"]},
            "Start with one ordinary edge inside a 3-vertex hypergraph.",
        ),
        HypergraphExample(
            "two_pairs_chain",
            "Two-Pair Chain",
            ["v1", "v2", "v3", "v4"],
            {"e1": ["v1", "v2"], "e2": ["v2", "v3"]},
            "Two size-2 hyperedges; still graph-like, but already a hypergraph formally.",
        ),
        HypergraphExample(
            "first_true_hyperedge",
            "First True Hyperedge",
            ["v1", "v2", "v3", "v4"],
            {"e1": ["v1", "v2", "v3"], "e2": ["v3", "v4"]},
            "The first genuine hyperedge: e1 touches three vertices at once.",
        ),
        HypergraphExample(
            "overlap_and_singleton",
            "Overlap and Singleton",
            ["v1", "v2", "v3", "v4", "v5"],
            {"e1": ["v1", "v2", "v3"], "e2": ["v3", "v4"], "e3": ["v5"]},
            "Overlap appears at v3, and e3 is a singleton hyperedge.",
        ),
        HypergraphExample(
            "two_overlapping_triples",
            "Two Overlapping Triples",
            ["v1", "v2", "v3", "v4", "v5"],
            {"e1": ["v1", "v2", "v3"], "e2": ["v2", "v3", "v4"], "e3": ["v4", "v5"]},
            "Two 3-edges overlap in two vertices; the 2-section starts to densify.",
        ),
        HypergraphExample(
            "size_four_edge",
            "A Size-Four Hyperedge",
            ["v1", "v2", "v3", "v4", "v5", "v6"],
            {"e1": ["v1", "v2", "v3", "v4"], "e2": ["v3", "v5"], "e3": ["v4", "v6"]},
            "A 4-vertex hyperedge creates a clique on four vertices in the simple-graph translation.",
        ),
        HypergraphExample(
            "three_way_overlap",
            "Three-Way Overlap",
            ["v1", "v2", "v3", "v4", "v5", "v6"],
            {"e1": ["v1", "v2", "v3"], "e2": ["v2", "v3", "v4"], "e3": ["v3", "v4", "v5", "v6"]},
            "Several hyperedges overlap, with v3 playing a central role.",
        ),
        HypergraphExample(
            "with_isolated_vertex",
            "Mixed Sizes with an Isolated Vertex",
            ["v1", "v2", "v3", "v4", "v5", "v6", "v7"],
            {"e1": ["v1", "v2", "v3"], "e2": ["v2", "v4", "v5"], "e3": ["v3", "v5", "v6"], "e4": ["v4"]},
            "A larger example with mixed sizes and one singleton edge; v7 remains isolated.",
        ),
        HypergraphExample(
            "dense_overlap",
            "Dense Overlap",
            ["v1", "v2", "v3", "v4", "v5", "v6", "v7"],
            {
                "e1": ["v1", "v2", "v3", "v4"],
                "e2": ["v2", "v3", "v5"],
                "e3": ["v3", "v4", "v6"],
                "e4": ["v4", "v5", "v6", "v7"],
            },
            "Several medium-to-large hyperedges overlap, producing a noticeably denser 2-section.",
        ),
        HypergraphExample(
            "largest_mixed_example",
            "Largest Mixed Example",
            ["v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8"],
            {
                "e1": ["v1", "v2", "v3", "v4"],
                "e2": ["v2", "v4", "v5", "v6"],
                "e3": ["v3", "v6", "v7"],
                "e4": ["v1", "v7", "v8"],
                "e5": ["v5", "v8"],
            },
            "The final example mixes larger and smaller hyperedges with multiple overlaps.",
        ),
    ]


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def clear_generated_files(path: Path) -> None:
    """Remove previously generated outputs so the folder stays clean after reruns."""
    for pattern in [
        "hypergraph_*.png",
        "hypergraph_*.mp4",
        "hypergraph_*.gif",
        "README.txt",
    ]:
        for item in path.glob(pattern):
            item.unlink(missing_ok=True)


def algebraic_formulation(example: HypergraphExample) -> str:
    """Compact textual formulation shown beneath the main title."""
    vertex_text = ", ".join(example.vertices)
    edge_names = ", ".join(example.edges.keys())
    edge_parts = [f"{e}={{" + ", ".join(members) + "}}" for e, members in example.edges.items()]
    edge_text = "; ".join(edge_parts)
    return f"X = {{{vertex_text}}},   E = {{{edge_names}}} with {edge_text}"


def hypergraph_to_simple_graph(example: HypergraphExample) -> nx.Graph:
    """2-section / clique expansion."""
    g = nx.Graph()
    g.add_nodes_from(example.vertices)
    for members in example.edges.values():
        for u, v in itertools.combinations(sorted(members), 2):
            if g.has_edge(u, v):
                g[u][v]["multiplicity"] += 1
            else:
                g.add_edge(u, v, multiplicity=1)
    return g


def compute_vertex_layout(example: HypergraphExample, seed: int = 42) -> Dict[str, np.ndarray]:
    """Compute a stable layout and normalize it into a compact square."""
    g_simple = hypergraph_to_simple_graph(example)
    if g_simple.number_of_edges() == 0:
        layout = nx.circular_layout(g_simple)
    else:
        layout = nx.spring_layout(g_simple, seed=seed, k=1.1 / math.sqrt(max(1, len(example.vertices))))

    coords = np.array([layout[v] for v in example.vertices], dtype=float)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    span = np.where(maxs - mins < 1e-9, 1.0, maxs - mins)
    coords = (coords - mins) / span
    coords = coords * 1.58 - 0.79
    return {v: coords[i] for i, v in enumerate(example.vertices)}


def _monotone_chain_convex_hull(points: np.ndarray) -> np.ndarray:
    """Convex hull using the monotone chain algorithm."""
    pts = np.unique(points, axis=0)
    if len(pts) <= 1:
        return pts.copy()
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(tuple(p))

    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(tuple(p))

    hull = np.array(lower[:-1] + upper[:-1], dtype=float)
    return hull


def _chaikin(points: np.ndarray, iterations: int = 2) -> np.ndarray:
    """Smooth a closed polygon with Chaikin corner cutting."""
    pts = points.copy()
    for _ in range(iterations):
        new_pts = []
        n = len(pts)
        for i in range(n):
            p = pts[i]
            q = pts[(i + 1) % n]
            new_pts.append(0.75 * p + 0.25 * q)
            new_pts.append(0.25 * p + 0.75 * q)
        pts = np.array(new_pts, dtype=float)
    return pts


def _capsule_polygon(p1: np.ndarray, p2: np.ndarray, radius: float = 0.16, n_arc: int = 24) -> np.ndarray:
    """Rounded capsule around a segment, better than a thin ellipse for 2-edges."""
    delta = p2 - p1
    length = float(np.linalg.norm(delta))
    if length < 1e-9:
        theta = np.linspace(0, 2 * np.pi, 2 * n_arc, endpoint=False)
        return np.column_stack([p1[0] + radius * np.cos(theta), p1[1] + radius * np.sin(theta)])

    u = delta / length
    n = np.array([-u[1], u[0]])
    angle = math.atan2(u[1], u[0])

    arc1 = np.linspace(angle - np.pi / 2, angle + np.pi / 2, n_arc)
    arc2 = np.linspace(angle + np.pi / 2, angle + 3 * np.pi / 2, n_arc)
    pts1 = np.column_stack([p2[0] + radius * np.cos(arc1), p2[1] + radius * np.sin(arc1)])
    pts2 = np.column_stack([p1[0] + radius * np.cos(arc2), p1[1] + radius * np.sin(arc2)])
    return np.vstack([pts1, pts2])


def _expanded_blob_polygon(points: np.ndarray, pad: float = 0.18) -> np.ndarray:
    """Blob-like polygon around a point cloud for region-style hyperedges."""
    n = len(points)
    if n == 1:
        theta = np.linspace(0, 2 * np.pi, 40, endpoint=False)
        return np.column_stack([
            points[0, 0] + pad * np.cos(theta),
            points[0, 1] + pad * np.sin(theta),
        ])

    if n == 2:
        return _capsule_polygon(points[0], points[1], radius=pad * 0.95)

    hull = _monotone_chain_convex_hull(points)
    if len(hull) < 3:
        return _capsule_polygon(points[0], points[-1], radius=pad)

    centroid = hull.mean(axis=0)
    expanded = []
    for p in hull:
        vec = p - centroid
        norm = float(np.linalg.norm(vec))
        if norm < 1e-9:
            expanded.append(p)
        else:
            expanded.append(p + pad * vec / norm)
    expanded = np.array(expanded, dtype=float)
    return _chaikin(expanded, iterations=2)


def style_panel(ax) -> None:
    ax.set_facecolor(PANEL_FACE)
    ax.set_box_aspect(1)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(PANEL_EDGE)
        spine.set_linewidth(1.1)


def draw_region_view(ax, example: HypergraphExample, pos, edge_colors) -> None:
    ax.set_title("A. Region-style hypergraph view", fontsize=PANEL_TITLE_SIZE, pad=7)
    ax.set_aspect("equal")
    ax.set_xlim(-1.02, 1.02)
    ax.set_ylim(-1.02, 1.02)
    ax.set_xticks([])
    ax.set_yticks([])
    style_panel(ax)

    for i, (e_name, members) in enumerate(example.edges.items()):
        points = np.array([pos[v] for v in members], dtype=float)
        poly = _expanded_blob_polygon(points, pad=0.22 if len(points) >= 3 else 0.20)
        patch = Polygon(
            poly,
            closed=True,
            facecolor=edge_colors[i],
            edgecolor="black",
            linewidth=1.2,
            alpha=0.36,
            joinstyle="round",
            zorder=1,
        )
        ax.add_patch(patch)
        center = points.mean(axis=0)
        ax.text(center[0], center[1], e_name, fontsize=LABEL_SIZE, weight="bold", ha="center", va="center", zorder=2)

    for v in example.vertices:
        x, y = pos[v]
        ax.scatter([x], [y], s=90, c="white", edgecolors="black", zorder=3)
        ax.text(x + 0.035, y + 0.028, v, fontsize=LABEL_SIZE, zorder=4)


def draw_incidence_bipartite(ax, example: HypergraphExample, edge_colors) -> None:
    ax.set_title("B. Incidence bipartite graph", fontsize=PANEL_TITLE_SIZE, pad=7)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    style_panel(ax)

    vertices = example.vertices
    edges = list(example.edges.keys())

    y_vertices = np.linspace(0.85, 0.15, len(vertices))
    y_edges = np.linspace(0.85, 0.15, max(len(edges), 1))

    x_v = 0.23
    x_e = 0.77

    v_pos = {v: (x_v, y_vertices[i]) for i, v in enumerate(vertices)}
    e_pos = {e: (x_e, y_edges[i]) for i, e in enumerate(edges)}

    for i, e in enumerate(edges):
        for v in example.edges[e]:
            x1, y1 = v_pos[v]
            x2, y2 = e_pos[e]
            ax.plot([x1, x2], [y1, y2], color=edge_colors[i], lw=2.1, alpha=0.95, zorder=1)

    for v in vertices:
        x, y = v_pos[v]
        ax.scatter([x], [y], s=180, c="white", edgecolors="black", zorder=3)
        ax.text(x - 0.08, y, v, fontsize=LABEL_SIZE, va="center", ha="right")

    for i, e in enumerate(edges):
        x, y = e_pos[e]
        ax.scatter([x], [y], s=215, c=[edge_colors[i]], edgecolors="black", marker="s", zorder=3)
        ax.text(x + 0.08, y, e, fontsize=LABEL_SIZE, va="center", ha="left")


def draw_paoh(ax, example: HypergraphExample, edge_colors) -> None:
    ax.set_title("C. PAOH-like incidence view", fontsize=PANEL_TITLE_SIZE, pad=7)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    style_panel(ax)

    vertices = example.vertices
    edges = list(example.edges.keys())
    y_vals = np.linspace(0.85, 0.15, len(vertices))
    y_map = {v: y_vals[i] for i, v in enumerate(vertices)}
    x_positions = np.linspace(0.34, 0.80, max(len(edges), 1))
    x_map = {e: x_positions[i] for i, e in enumerate(edges)}

    for i, v in enumerate(vertices):
        y = y_map[v]
        if i % 2 == 0:
            ax.axhspan(y - 0.055, y + 0.055, xmin=0.05, xmax=0.95, color="#f1f1f1", zorder=0)
        ax.text(0.14, y, v, va="center", ha="center", fontsize=LABEL_SIZE)

    for i, e in enumerate(edges):
        x = x_map[e]
        members = example.edges[e]
        ys = [y_map[v] for v in members]
        if len(ys) >= 2:
            ax.plot([x, x], [min(ys), max(ys)], color="#777777", lw=2.0, zorder=1)
        for y in ys:
            ax.scatter([x], [y], s=110, c=[edge_colors[i]], edgecolors="#555555", zorder=2)
        ax.text(x, 0.93, e, fontsize=LABEL_SIZE, ha="center", va="top", color="#333333")


def draw_simple_graph(ax, example: HypergraphExample, pos) -> None:
    ax.set_title("D. Simple graph translation (2-section)", fontsize=PANEL_TITLE_SIZE, pad=7)
    ax.set_aspect("equal")
    ax.set_xlim(-1.02, 1.02)
    ax.set_ylim(-1.02, 1.02)
    ax.set_xticks([])
    ax.set_yticks([])
    style_panel(ax)

    g_simple = hypergraph_to_simple_graph(example)

    for u, v, data in g_simple.edges(data=True):
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        mult = data.get("multiplicity", 1)
        ax.plot([x1, x2], [y1, y2], color="#4b61a8", lw=1.1 + 0.7 * (mult - 1), alpha=0.85, zorder=1)

    for v in example.vertices:
        x, y = pos[v]
        ax.scatter([x], [y], s=92, c="white", edgecolors="black", zorder=3)
        ax.text(x + 0.035, y + 0.028, v, fontsize=LABEL_SIZE, zorder=4)

    ax.text(
        0.03,
        0.03,
        f"|V|={g_simple.number_of_nodes()}, |E|={g_simple.number_of_edges()}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=SMALL_LABEL_SIZE,
        bbox=dict(boxstyle="round,pad=0.26", facecolor="white", edgecolor="#cccccc"),
    )


def draw_full_figure(fig, axes, example: HypergraphExample, index: int, total: int) -> None:
    palette = plt.cm.Set3(np.linspace(0.05, 0.95, max(len(example.edges), 3)))
    pos = compute_vertex_layout(example, seed=42 + index)

    for ax in axes.ravel():
        ax.clear()

    for artist in list(fig.texts):
        try:
            artist.remove()
        except ValueError:
            pass
    fig._suptitle = None

    draw_region_view(axes[0, 0], example, pos, palette)
    draw_incidence_bipartite(axes[0, 1], example, palette)
    draw_paoh(axes[1, 0], example, palette)
    draw_simple_graph(axes[1, 1], example, pos)

    incidence_count = sum(len(m) for m in example.edges.values())
    fig.suptitle(example.title, fontsize=MAIN_TITLE_SIZE, y=0.98)
    fig.text(
        0.5,
        0.948,
        textwrap.fill("Algebraic form: " + algebraic_formulation(example), width=110),
        ha="center",
        va="top",
        fontsize=SUBTITLE_SIZE,
        color="#333333",
    )
    fig.text(
        0.5,
        0.913,
        f"Example {index + 1} of {total}  •  {len(example.vertices)} vertices  •  {len(example.edges)} hyperedges  •  {incidence_count} incidences",
        ha="center",
        va="top",
        fontsize=META_SIZE,
        color="#333333",
    )
    fig.text(
        0.5,
        0.892,
        textwrap.fill(example.note, width=96),
        ha="center",
        va="top",
        fontsize=NOTE_SIZE,
        color="#444444",
    )
    fig.subplots_adjust(top=0.82, bottom=0.06, left=0.06, right=0.97, wspace=0.14, hspace=0.20)


def save_pngs(examples: List[HypergraphExample], out_dir: Path) -> List[Path]:
    png_paths: List[Path] = []
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    for i, example in enumerate(examples):
        draw_full_figure(fig, axes, example, i, len(examples))
        path = out_dir / f"hypergraph_{i + 1:02d}_{example.slug}.png"
        fig.savefig(path, dpi=220)
        png_paths.append(path)
        print(f"Saved {path}")
    plt.close(fig)
    return png_paths


def save_summary_sheet(examples: List[HypergraphExample], out_dir: Path) -> Path:
    fig, axes = plt.subplots(5, 2, figsize=(12, 22))
    axes = axes.ravel()
    for i, example in enumerate(examples):
        ax = axes[i]
        palette = plt.cm.Set3(np.linspace(0.05, 0.95, max(len(example.edges), 3)))
        pos = compute_vertex_layout(example, seed=42 + i)
        draw_region_view(ax, example, pos, palette)
        ax.set_title(f"{i + 1}. {example.title}", fontsize=META_SIZE, pad=5)
    for j in range(len(examples), len(axes)):
        axes[j].axis("off")
    fig.suptitle("Ten Hypergraphs in Increasing Complexity", fontsize=SUMMARY_TITLE_SIZE, y=0.992)
    fig.text(0.5, 0.975, "Region-style snapshots only", ha="center", va="top", fontsize=SUMMARY_SUBTITLE_SIZE, color="#444444")
    fig.subplots_adjust(top=0.96, bottom=0.03, left=0.06, right=0.97, wspace=0.16, hspace=0.24)
    path = out_dir / "hypergraph_summary_sheet.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print(f"Saved {path}")
    return path


def save_animation(examples: List[HypergraphExample], out_dir: Path, fps: int = 1) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    def update(frame_index: int):
        draw_full_figure(fig, axes, examples[frame_index], frame_index, len(examples))
        return []

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(examples),
        interval=1000 // max(fps, 1),
        blit=False,
        repeat=True,
    )

    mp4_path = out_dir / "hypergraph_evolution.mp4"
    gif_path = out_dir / "hypergraph_evolution.gif"

    ffmpeg_available = shutil.which("ffmpeg") is not None
    if ffmpeg_available:
        writer = animation.FFMpegWriter(fps=fps, bitrate=2300)
        ani.save(mp4_path, writer=writer, dpi=170)
        plt.close(fig)
        print(f"Saved {mp4_path}")
        return mp4_path

    writer = animation.PillowWriter(fps=fps)
    ani.save(gif_path, writer=writer, dpi=150)
    plt.close(fig)
    print(f"Saved {gif_path}")
    return gif_path


def main() -> None:
    out_dir = Path.cwd() / "hypergraph_outputs"
    ensure_output_dir(out_dir)
    clear_generated_files(out_dir)

    examples = build_examples()
    save_pngs(examples, out_dir)
    save_summary_sheet(examples, out_dir)
    save_animation(examples, out_dir, fps=1)

    readme = out_dir / "README.txt"
    readme.write_text(
        textwrap.dedent(
            """
            Files created by hypergraph_gallery_animation.py

            - hypergraph_01_*.png ... hypergraph_10_*.png
                One composite PNG per hypergraph, each with four representations.

            - hypergraph_summary_sheet.png
                A one-page sheet showing all ten region-style hypergraphs.

            - hypergraph_evolution.mp4 or hypergraph_evolution.gif
                Animation stepping through the ten examples in order of complexity.

            Representation notes:
            - "Region-style hypergraph view" shows each hyperedge as a translucent colored region.
            - "Incidence bipartite graph" introduces one node per hyperedge.
            - "PAOH-like incidence view" shows membership vertically.
            - "Simple graph translation" means the 2-section/clique expansion:
              vertices become adjacent whenever they co-occur in a hyperedge.
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved {readme}")
    print("\nDone. Output folder:", out_dir)


if __name__ == "__main__":
    main()
