#!/usr/bin/env python3

import json
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.animation import FuncAnimation


OUTPUT_DIR = "2026-03-30-basic-graph-animations"
SAVE_MP4 = False
SAVE_GIF = True
DPI = 220
FIGSIZE = (6.4, 7.2)

BG = "#fbfaf7"
EDGE = "#777777"
NODE = "#4c78a8"
NODE_ALT = "#72b7b2"
NODE_HI = "#e45756"
NODE_ACCENT = "#f2cf5b"
TEXT = "#1f1f1f"
SUBTEXT = "#585858"


@dataclass
class VisualSpec:
    slug: str
    title: str
    description: str
    kind: str
    builder: Callable[[], Dict]


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def circular_pos(nodes):
    G = nx.Graph()
    G.add_nodes_from(nodes)
    return nx.circular_layout(G)


def bipartite_pos(left: List[int], right: List[int]) -> Dict[int, Tuple[float, float]]:
    pos = {}
    left_y = list(reversed(range(len(left))))
    right_y = list(reversed(range(len(right))))
    for i, u in enumerate(left):
        pos[u] = (-1.0, left_y[i])
    for i, v in enumerate(right):
        pos[v] = (1.0, right_y[i])
    return pos


def shell_pos(shells: List[List[int]]) -> Dict[int, Tuple[float, float]]:
    G = nx.Graph()
    for shell in shells:
        G.add_nodes_from(shell)
    return nx.shell_layout(G, shells)


def draw_base(
    ax_graph,
    G: nx.Graph,
    pos: Dict,
    title: str,
    description: str,
    node_colors: Optional[List[str]] = None,
    edge_colors: Optional[List[str]] = None,
    edge_widths: Optional[List[float]] = None,
    node_size: int = 520,
    font_size: int = 10,
):
    ax_graph.clear()
    ax_graph.set_facecolor(BG)

    if node_colors is None:
        node_colors = [NODE for _ in G.nodes()]
    if edge_colors is None:
        edge_colors = [EDGE for _ in G.edges()]
    if edge_widths is None:
        edge_widths = [1.8 for _ in G.edges()]

    nx.draw_networkx_nodes(
        G, pos, ax=ax_graph,
        node_color=node_colors,
        node_size=node_size,
        linewidths=0.7,
        edgecolors="#ffffff"
    )
    nx.draw_networkx_edges(
        G, pos, ax=ax_graph,
        edge_color=edge_colors,
        width=edge_widths
    )
    nx.draw_networkx_labels(
        G, pos, ax=ax_graph,
        labels={u: str(u) for u in G.nodes()},
        font_size=font_size,
        font_color="#ffffff"
    )

    ax_graph.set_title(title, fontsize=16, color=TEXT, pad=14)
    ax_graph.set_axis_off()

    ax_graph.text(
        0.5, -0.08, description,
        transform=ax_graph.transAxes,
        ha="center", va="top",
        fontsize=10.2,
        color=SUBTEXT,
        wrap=True
    )


def save_static(spec: VisualSpec, payload: Dict):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.patch.set_facecolor(BG)

    draw_base(
        ax_graph=ax,
        G=payload["G"],
        pos=payload["pos"],
        title=spec.title,
        description=spec.description,
        node_colors=payload.get("node_colors"),
        edge_colors=payload.get("edge_colors"),
        edge_widths=payload.get("edge_widths"),
        node_size=payload.get("node_size", 520),
        font_size=payload.get("font_size", 10),
    )

    out_png = os.path.join(OUTPUT_DIR, f"{spec.slug}.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=DPI, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return {"png": out_png}


def save_dynamic(spec: VisualSpec, payload: Dict):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.patch.set_facecolor(BG)

    G_base: nx.Graph = payload["G"]
    pos = payload["pos"]
    frames = payload["frames"]
    node_size = payload.get("node_size", 520)
    font_size = payload.get("font_size", 10)

    def update(frame_index: int):
        frame = frames[frame_index]
        G_now = nx.Graph()
        G_now.add_nodes_from(G_base.nodes())
        G_now.add_edges_from(frame["edges"])

        node_colors = frame.get("node_colors")
        if node_colors is None:
            node_colors = [NODE for _ in G_now.nodes()]

        edge_colors = frame.get("edge_colors")
        if edge_colors is None:
            edge_colors = [EDGE for _ in G_now.edges()]

        edge_widths = frame.get("edge_widths")
        if edge_widths is None:
            edge_widths = [1.8 for _ in G_now.edges()]

        desc = spec.description
        if "caption" in frame and frame["caption"]:
            desc = frame["caption"]

        draw_base(
            ax_graph=ax,
            G=G_now,
            pos=pos,
            title=spec.title,
            description=desc,
            node_colors=node_colors,
            edge_colors=edge_colors,
            edge_widths=edge_widths,
            node_size=node_size,
            font_size=font_size,
        )

    anim = FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=900,
        repeat=True
    )

    outputs = {}

    if SAVE_GIF:
        out_gif = os.path.join(OUTPUT_DIR, f"{spec.slug}.gif")
        anim.save(out_gif, writer="pillow", fps=1)
        outputs["gif"] = out_gif

    if SAVE_MP4:
        out_mp4 = os.path.join(OUTPUT_DIR, f"{spec.slug}.mp4")
        try:
            anim.save(out_mp4, writer="ffmpeg", fps=1)
            outputs["mp4"] = out_mp4
        except Exception:
            pass

    plt.close(fig)
    return outputs


def manifest_entry(spec: VisualSpec, outputs: Dict):
    return {
        "slug": spec.slug,
        "title": spec.title,
        "description": spec.description,
        "kind": spec.kind,
        "files": outputs,
    }


def b_empty():
    G = nx.empty_graph(8)
    return {"G": G, "pos": circular_pos(G.nodes())}


def b_single_edge():
    G = nx.empty_graph(8)
    G.add_edge(0, 1)
    edge_colors = [NODE_HI]
    return {"G": G, "pos": circular_pos(G.nodes()), "edge_colors": edge_colors}


def b_path():
    G = nx.path_graph(8)
    return {"G": G, "pos": circular_pos(G.nodes())}


def b_cycle():
    G = nx.cycle_graph(8)
    return {"G": G, "pos": circular_pos(G.nodes())}


def b_complete():
    G = nx.complete_graph(8)
    return {"G": G, "pos": circular_pos(G.nodes()), "node_size": 430, "font_size": 9}


def b_star():
    G = nx.star_graph(7)
    pos = shell_pos([[0], list(range(1, 8))])
    node_colors = [NODE_HI] + [NODE for _ in range(7)]
    return {"G": G, "pos": pos, "node_colors": node_colors}


def b_tree():
    G = nx.Graph()
    G.add_nodes_from(range(8))
    G.add_edges_from([(0,1),(0,2),(1,3),(1,4),(2,5),(5,6),(5,7)])
    pos = {
        0:(0,3), 1:(-1.2,2), 2:(1.2,2),
        3:(-1.8,1), 4:(-0.6,1), 5:(1.2,1),
        6:(0.6,0), 7:(1.8,0)
    }
    return {"G": G, "pos": pos}


def b_connected_not_complete():
    G = nx.Graph()
    G.add_nodes_from(range(8))
    G.add_edges_from([(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,0),(0,4)])
    return {"G": G, "pos": circular_pos(G.nodes())}


def b_disconnected():
    G = nx.Graph()
    G.add_nodes_from(range(8))
    G.add_edges_from([(0,1),(1,2),(2,3),(4,5),(6,7)])
    return {"G": G, "pos": circular_pos(G.nodes())}


def b_complete_bipartite():
    G = nx.complete_bipartite_graph(4, 4)
    left = [0,1,2,3]
    right = [4,5,6,7]
    pos = bipartite_pos(left, right)
    node_colors = [NODE for _ in left] + [NODE_ALT for _ in right]
    return {"G": G, "pos": pos, "node_colors": node_colors, "node_size": 480}


def b_bipartite_not_complete():
    G = nx.Graph()
    G.add_nodes_from(range(8))
    left = [0,1,2,3]
    right = [4,5,6,7]
    G.add_edges_from([(0,4),(0,5),(1,5),(1,6),(2,6),(2,7),(3,4)])
    pos = bipartite_pos(left, right)
    node_colors = [NODE for _ in left] + [NODE_ALT for _ in right]
    return {"G": G, "pos": pos, "node_colors": node_colors, "node_size": 480}


def b_triangle_free():
    G = nx.cycle_graph(8)
    return {"G": G, "pos": circular_pos(G.nodes())}


def b_has_triangle():
    G = nx.Graph()
    G.add_nodes_from(range(8))
    G.add_edges_from([(0,1),(1,2),(2,0),(2,3),(3,4),(4,5),(5,6),(6,7)])
    edge_colors = [NODE_HI if e in [(0,1),(1,2),(0,2)] or (e[1],e[0]) in [(0,1),(1,2),(0,2)] else EDGE for e in G.edges()]
    return {"G": G, "pos": circular_pos(G.nodes()), "edge_colors": edge_colors}


def b_bridge():
    G = nx.Graph()
    G.add_nodes_from(range(8))
    G.add_edges_from([(0,1),(1,2),(2,3),(3,0),(3,4),(4,5),(5,6),(6,7),(7,4)])
    pos = circular_pos(G.nodes())
    edge_colors = [NODE_HI if set(e) == {3,4} else EDGE for e in G.edges()]
    edge_widths = [3.4 if set(e) == {3,4} else 1.8 for e in G.edges()]
    return {"G": G, "pos": pos, "edge_colors": edge_colors, "edge_widths": edge_widths}


def b_articulation():
    G = nx.Graph()
    G.add_nodes_from(range(8))
    G.add_edges_from([(0,1),(1,2),(2,0),(2,3),(3,4),(4,5),(5,3),(3,6),(6,7)])
    pos = circular_pos(G.nodes())
    node_colors = [NODE for _ in G.nodes()]
    node_colors[3] = NODE_HI
    return {"G": G, "pos": pos, "node_colors": node_colors}


def b_matching():
    G = nx.Graph()
    G.add_nodes_from(range(8))
    G.add_edges_from([(0,1),(2,3),(4,5),(6,7)])
    edge_colors = [NODE_ACCENT for _ in G.edges()]
    edge_widths = [3.0 for _ in G.edges()]
    return {"G": G, "pos": circular_pos(G.nodes()), "edge_colors": edge_colors, "edge_widths": edge_widths}


def b_regular():
    G = nx.cycle_graph(8)
    return {"G": G, "pos": circular_pos(G.nodes())}


def b_wheel():
    G = nx.wheel_graph(8)
    pos = shell_pos([[0], [1,2,3,4,5,6,7]])
    node_colors = [NODE_HI] + [NODE for _ in range(7)]
    return {"G": G, "pos": pos, "node_colors": node_colors}


def b_planar():
    G = nx.cubical_graph()
    pos = nx.spring_layout(G, seed=7)
    return {"G": G, "pos": pos}


def b_nonplanar():
    G = nx.complete_bipartite_graph(3, 3)
    G.add_nodes_from([6, 7])
    left = [0,1,2]
    right = [3,4,5]
    pos = {
        0:(-1,2), 1:(-1,1), 2:(-1,0),
        3:(1,2), 4:(1,1), 5:(1,0),
        6:(-2,-1.1), 7:(2,-1.1)
    }
    node_colors = [NODE for _ in left] + [NODE_ALT for _ in right] + [NODE_ACCENT, NODE_ACCENT]
    return {"G": G, "pos": pos, "node_colors": node_colors}


def d_connected_growth():
    G = nx.empty_graph(8)
    pos = circular_pos(G.nodes())
    edges = [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7)]
    frames = []
    current = []
    for i, e in enumerate(edges, start=1):
        current.append(e)
        frames.append({
            "edges": current.copy(),
            "caption": f"A graph is connected if every pair of vertices can be joined by a path. Frame {i}: connectivity is gradually being built."
        })
    return {"G": G, "pos": pos, "frames": frames}


def d_complete_growth():
    G = nx.empty_graph(8)
    pos = circular_pos(G.nodes())
    all_edges = list(nx.complete_graph(8).edges())
    chosen = all_edges[:12] + all_edges[12:20] + all_edges[20:]
    frames = []
    checkpoints = [6, 12, 18, 28]
    for k in checkpoints:
        frames.append({
            "edges": chosen[:k],
            "caption": f"A complete graph has every possible edge present. Here we are filling toward K8: {k}/28 edges."
        })
    return {"G": G, "pos": pos, "frames": frames, "node_size": 430, "font_size": 9}


def d_bipartite_growth():
    G = nx.empty_graph(8)
    left = [0,1,2,3]
    right = [4,5,6,7]
    pos = bipartite_pos(left, right)
    edges = [(0,4),(0,5),(1,5),(1,6),(2,6),(2,7),(3,4),(3,7)]
    node_colors = [NODE for _ in left] + [NODE_ALT for _ in right]
    frames = []
    current = []
    for e in edges:
        current.append(e)
        frames.append({
            "edges": current.copy(),
            "node_colors": node_colors,
            "caption": "A bipartite graph splits vertices into two parts, with edges only across the split."
        })
    return {"G": G, "pos": pos, "frames": frames, "node_size": 480}


def d_triangle_free_vs_triangle():
    G = nx.empty_graph(8)
    pos = circular_pos(G.nodes())
    seq = [
        [(0,1),(1,2)],
        [(0,1),(1,2),(2,3),(3,4)],
        [(0,1),(1,2),(2,3),(3,4),(4,5)],
        [(0,1),(1,2),(2,0),(3,4),(4,5)]
    ]
    frames = []
    for i, edges in enumerate(seq):
        caption = "Triangle-free so far." if i < 3 else "Now a triangle appears on vertices 0, 1, 2."
        edge_colors = []
        for e in edges:
            if set(e).issubset({0,1,2}) and len(edges) == 5:
                edge_colors.append(NODE_HI)
            else:
                edge_colors.append(EDGE)
        frames.append({
            "edges": edges,
            "edge_colors": edge_colors,
            "caption": caption
        })
    return {"G": G, "pos": pos, "frames": frames}


SPECS: List[VisualSpec] = [
    VisualSpec("01_empty_graph", "Empty Graph", "No edges are present.", "static", b_empty),
    VisualSpec("02_single_edge", "Single Edge", "The smallest nontrivial connection between two vertices.", "static", b_single_edge),
    VisualSpec("03_path_graph", "Path Graph", "Vertices arranged in a single chain.", "static", b_path),
    VisualSpec("04_cycle_graph", "Cycle Graph", "A closed loop in which every vertex has degree 2.", "static", b_cycle),
    VisualSpec("05_complete_graph", "Complete Graph", "Every pair of distinct vertices is connected by an edge.", "static", b_complete),
    VisualSpec("06_star_graph", "Star Graph", "One central vertex is connected to every outer vertex.", "static", b_star),
    VisualSpec("07_tree_graph", "Tree", "A connected graph with no cycles.", "static", b_tree),
    VisualSpec("08_connected_graph", "Connected Graph", "Every vertex can be reached from every other vertex.", "static", b_connected_not_complete),
    VisualSpec("09_disconnected_graph", "Disconnected Graph", "At least two components are present.", "static", b_disconnected),
    VisualSpec("10_complete_bipartite_graph", "Complete Bipartite Graph", "Vertices split into two parts, with all cross edges present.", "static", b_complete_bipartite),
    VisualSpec("11_bipartite_graph", "Bipartite Graph", "Vertices split into two parts, with no edges inside either part.", "static", b_bipartite_not_complete),
    VisualSpec("12_triangle_free_graph", "Triangle-Free Graph", "No 3-cycle appears anywhere in the graph.", "static", b_triangle_free),
    VisualSpec("13_graph_with_triangle", "Graph with a Triangle", "Vertices 0, 1, and 2 form a 3-cycle.", "static", b_has_triangle),
    VisualSpec("14_bridge_edge", "Bridge", "Removing the highlighted edge increases the number of components.", "static", b_bridge),
    VisualSpec("15_articulation_vertex", "Articulation Point", "Removing the highlighted vertex disconnects the graph.", "static", b_articulation),
    VisualSpec("16_matching", "Matching", "A set of edges with no shared endpoints.", "static", b_matching),
    VisualSpec("17_regular_graph", "Regular Graph", "Each vertex has the same degree. This one is 2-regular.", "static", b_regular),
    VisualSpec("18_wheel_graph", "Wheel Graph", "A cycle together with one hub connected to the cycle.", "static", b_wheel),
    VisualSpec("19_planar_graph", "Planar Graph", "This graph can be drawn without edge crossings.", "static", b_planar),
    VisualSpec("20_nonplanar_graph", "Non-Planar Graph", "K3,3 is a standard example of a graph that cannot be drawn planar.", "static", b_nonplanar),
    VisualSpec("21_connected_growth", "Building Connectivity", "Edges are added until the graph becomes connected.", "dynamic", d_connected_growth),
    VisualSpec("22_complete_growth", "Building a Complete Graph", "Edges are added until every pair is connected.", "dynamic", d_complete_growth),
    VisualSpec("23_bipartite_growth", "Building a Bipartite Graph", "Edges are added only across the two parts.", "dynamic", d_bipartite_growth),
    VisualSpec("24_triangle_appears", "When a Triangle Appears", "A graph can be triangle-free for a while, then suddenly acquire one.", "dynamic", d_triangle_free_vs_triangle),
]


def main():
    ensure_dir(OUTPUT_DIR)
    manifest = []

    for spec in SPECS:
        print(f"Creating {spec.slug} ...")
        payload = spec.builder()

        if spec.kind == "static":
            outputs = save_static(spec, payload)
        elif spec.kind == "dynamic":
            outputs = save_dynamic(spec, payload)
        else:
            raise ValueError(f"Unknown kind: {spec.kind}")

        manifest.append({
            "slug": spec.slug,
            "title": spec.title,
            "description": spec.description,
            "kind": spec.kind,
            "files": outputs,
        })

    manifest_path = os.path.join(OUTPUT_DIR, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Done. Files saved in: {OUTPUT_DIR}")
    print(f"Manifest saved to: {manifest_path}")


if __name__ == "__main__":
    main()
