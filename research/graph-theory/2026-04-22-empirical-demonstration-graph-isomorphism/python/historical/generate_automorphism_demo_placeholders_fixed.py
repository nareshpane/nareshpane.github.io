#!/usr/bin/env python3
from __future__ import annotations

import math
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import FancyArrowPatch

# ============================================================
# Running graph: the 12-vertex prism graph P = C6 □ K2
# ============================================================
N = 12
VERTICES = list(range(1, N + 1))
OUTER = list(range(1, 7))
INNER = list(range(7, 13))
OUTDIR = Path("automorphism_demo_pngs")

PALETTE = {
    0: "#d9d9d9",
    1: "#d55e5e",
    2: "#5e8fd5",
    3: "#5dbb76",
    4: "#d8a742",
    5: "#9b6ad6",
    6: "#4fb8b8",
    7: "#e17c43",
    8: "#c56bb0",
    9: "#7a7a2f",
    10: "#2d8f8c",
    11: "#7a5ad8",
    12: "#6d6d6d",
}

BASE_G = nx.Graph()
BASE_G.add_nodes_from(VERTICES)
for i in range(6):
    BASE_G.add_edge(OUTER[i], OUTER[(i + 1) % 6])
for i in range(6):
    BASE_G.add_edge(INNER[i], INNER[(i + 1) % 6])
for i in range(6):
    BASE_G.add_edge(OUTER[i], INNER[i])

assert BASE_G.number_of_nodes() == 12
assert BASE_G.number_of_edges() == 18

BASE_EDGES = sorted(tuple(sorted(e)) for e in BASE_G.edges())
BASE_G6 = nx.to_graph6_bytes(BASE_G, header=False).decode().strip()

EDGE_TEXT = (
    "{1,2},{2,3},{3,4},{4,5},{5,6},{6,1},"
    "{7,8},{8,9},{9,10},{10,11},{11,12},{12,7},"
    "{1,7},{2,8},{3,9},{4,10},{5,11},{6,12}"
)
ADJ_LINES = [
    "1:{2,6,7}", "2:{1,3,8}", "3:{2,4,9}", "4:{3,5,10}",
    "5:{4,6,11}", "6:{1,5,12}", "7:{1,8,12}", "8:{2,7,9}",
    "9:{3,8,10}", "10:{4,9,11}", "11:{5,10,12}", "12:{6,7,11}",
]
DEGREE_TEXT = "(3,3,3,3,3,3,3,3,3,3,3,3)"
GROUP_TEXT = "Aut(P)=<ρ,τ,σ>, |Aut(P)|=24"


def prism_positions() -> Dict[int, Tuple[float, float]]:
    pos = {}
    r_outer = 1.00
    r_inner = 0.57
    for i, v in enumerate(OUTER):
        theta = math.pi / 2 - 2 * math.pi * i / 6
        pos[v] = (r_outer * math.cos(theta), r_outer * math.sin(theta))
    for i, v in enumerate(INNER):
        theta = math.pi / 2 - 2 * math.pi * i / 6
        pos[v] = (r_inner * math.cos(theta), r_inner * math.sin(theta))
    return pos


POS = prism_positions()


# ============================================================
# Explicit automorphisms of the prism graph
# ============================================================
def rot(k: int) -> Dict[int, int]:
    k %= 6
    p = {}
    for i, v in enumerate(OUTER):
        p[v] = OUTER[(i + k) % 6]
    for i, v in enumerate(INNER):
        p[v] = INNER[(i + k) % 6]
    return p


def refl() -> Dict[int, int]:
    p = {}
    for i, v in enumerate(OUTER):
        p[v] = OUTER[(-i) % 6]
    for i, v in enumerate(INNER):
        p[v] = INNER[(-i) % 6]
    return p


def swap_rings() -> Dict[int, int]:
    p = {}
    for i in range(6):
        p[OUTER[i]] = INNER[i]
        p[INNER[i]] = OUTER[i]
    return p


def compose(p: Dict[int, int], q: Dict[int, int]) -> Dict[int, int]:
    return {v: p[q[v]] for v in VERTICES}


def perm_key(p: Dict[int, int]) -> Tuple[int, ...]:
    return tuple(p[v] for v in VERTICES)


def closure(gens: List[Dict[int, int]]) -> List[Dict[int, int]]:
    ident = {v: v for v in VERTICES}
    seen = {perm_key(ident): ident}
    stack = [ident]
    while stack:
        cur = stack.pop()
        for g in gens:
            nxt = compose(g, cur)
            key = perm_key(nxt)
            if key not in seen:
                seen[key] = nxt
                stack.append(nxt)
    return [seen[k] for k in sorted(seen)]


ALL_AUTOS = closure([rot(1), refl(), swap_rings()])
assert len(ALL_AUTOS) == 24


# ============================================================
# Colourings / partitions used in the sections
# ============================================================
def colouring_none() -> Dict[int, int]:
    return {v: 0 for v in VERTICES}


def colouring_three_blocks() -> Dict[int, int]:
    return {v: 1 if v <= 4 else 2 if v <= 8 else 3 for v in VERTICES}


def colouring_finer() -> Dict[int, int]:
    mapping = {}
    classes = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10, 11, 12)]
    for idx, cls in enumerate(classes, start=1):
        for v in cls:
            mapping[v] = idx
    return mapping


def colouring_one_singleton() -> Dict[int, int]:
    return {v: (1 if v == 1 else 0) for v in VERTICES}


def colouring_two_singletons() -> Dict[int, int]:
    return {v: (1 if v == 1 else 2 if v == 7 else 0) for v in VERTICES}


def colouring_asymmetric() -> Dict[int, int]:
    return {v: (1 if v == 1 else 2 if v == 4 else 3 if v == 8 else 0) for v in VERTICES}


def colouring_ring_split() -> Dict[int, int]:
    return {v: (1 if v in OUTER else 2) for v in VERTICES}


def colouring_discrete() -> Dict[int, int]:
    return {v: v for v in VERTICES}


def colour_sequence(c: Dict[int, int]) -> Tuple[int, ...]:
    return tuple(c[v] for v in VERTICES)


def apply_perm_to_colouring(c: Dict[int, int], p: Dict[int, int]) -> Dict[int, int]:
    # π^g(v)=π(v^g)
    return {v: c[p[v]] for v in VERTICES}


def permute_display_labels(p: Dict[int, int]) -> Dict[int, str]:
    return {v: str(p[v]) for v in VERTICES}


def colouring_name(c: Dict[int, int]) -> str:
    if c == colouring_none():
        return "uncoloured"
    if c == colouring_three_blocks():
        return "three ordered cells"
    if c == colouring_finer():
        return "finer ordered cells"
    if c == colouring_one_singleton():
        return "one singled-out vertex"
    if c == colouring_two_singletons():
        return "two singled-out vertices"
    if c == colouring_ring_split():
        return "outer/inner colouring"
    if c == colouring_discrete():
        return "discrete colouring"
    if c == colouring_asymmetric():
        return "asymmetric coloured graph"
    return "custom colouring"


def partition_notation(c: Dict[int, int]) -> str:
    cells: Dict[int, List[int]] = {}
    for v in VERTICES:
        cells.setdefault(c[v], []).append(v)
    if c == colouring_none():
        return "π = ({1,2,3,4,5,6,7,8,9,10,11,12})"
    parts = ["{" + ",".join(map(str, vals)) + "}" for _, vals in sorted(cells.items())]
    return "π = (" + ", ".join(parts) + ")"


def automorphisms_for_colouring(c: Dict[int, int]) -> List[Dict[int, int]]:
    autos = []
    for p in ALL_AUTOS:
        if all(c[v] == c[p[v]] for v in VERTICES):
            autos.append(p)
    autos.sort(key=perm_key)
    return autos


def orbit_partition(perms: List[Dict[int, int]]) -> List[List[int]]:
    remaining = set(VERTICES)
    orbits = []
    while remaining:
        v = min(remaining)
        orb = sorted({p[v] for p in perms})
        orbits.append(orb)
        remaining -= set(orb)
    return orbits


def cycle_notation(p: Dict[int, int]) -> str:
    seen = set()
    cycles = []
    for v in VERTICES:
        if v in seen or p[v] == v:
            seen.add(v)
            continue
        cur = v
        cyc = []
        while cur not in seen:
            seen.add(cur)
            cyc.append(cur)
            cur = p[cur]
        if len(cyc) > 1:
            cycles.append("(" + " ".join(map(str, cyc)) + ")")
    return " ".join(cycles) if cycles else "id"


def canonical_aut_colouring(c: Dict[int, int]) -> Tuple[Dict[int, int], Dict[int, int]]:
    best_p = None
    best_seq = None
    for p in ALL_AUTOS:
        seq = tuple(apply_perm_to_colouring(c, p)[v] for v in VERTICES)
        if best_seq is None or seq < best_seq:
            best_seq = seq
            best_p = p
    return apply_perm_to_colouring(c, best_p), best_p


# ============================================================
# Drawing helpers
# ============================================================
def node_facecolors(c: Dict[int, int]) -> List[str]:
    return [PALETTE.get(c[v], "#bbbbbb") for v in VERTICES]


def set_clean(ax, info_margin: bool = False):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    # When an algebra/info box is present, reserve blank space on the right so
    # the box does not sit on top of the graph itself.
    if info_margin:
        ax.set_xlim(-1.45, 2.35)
    else:
        ax.set_xlim(-1.32, 1.32)
    ax.set_ylim(-1.28, 1.28)
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(1.0)
        sp.set_color("#bfbfbf")
    ax.set_facecolor("#fafafa")


def graph_algebra_block(c: Dict[int, int], extra: str | None = None) -> str:
    autos = automorphisms_for_colouring(c)
    lines = [
        "P = C6 □ K2",
        "V = {1,...,12}",
        "|E| = 18, deg(v)=3 for all v",
        partition_notation(c),
        f"|Aut(P,π)| = {len(autos)}",
    ]
    if extra:
        lines.append(extra)
    return "\n".join(lines)


def draw_graph(ax, c: Dict[int, int], labels: Dict[int, str] | None = None,
               title: str = "", subtitle: str | None = None,
               highlight_nodes: List[int] | None = None,
               highlight_edges: List[Tuple[int, int]] | None = None,
               arrows: List[Tuple[int, int]] | None = None,
               algebra_text: str | None = None):
    set_clean(ax, info_margin=bool(algebra_text))
    highlight_nodes = highlight_nodes or []
    highlight_edges = highlight_edges or []
    arrows = arrows or []

    nx.draw_networkx_edges(BASE_G, POS, ax=ax, width=1.7, edge_color="#6f6f6f")
    if highlight_edges:
        nx.draw_networkx_edges(BASE_G, POS, edgelist=highlight_edges, ax=ax, width=3.2, edge_color="#d55e5e")
    sizes = [760 if v in highlight_nodes else 620 for v in VERTICES]
    borders = ["#b22222" if v in highlight_nodes else "black" for v in VERTICES]
    nx.draw_networkx_nodes(
        BASE_G, POS, ax=ax,
        node_color=node_facecolors(c), edgecolors=borders,
        linewidths=1.4, node_size=sizes,
    )
    if labels is None:
        labels = {v: str(v) for v in VERTICES}
    nx.draw_networkx_labels(BASE_G, POS, ax=ax, labels=labels, font_size=10, font_weight="bold")

    for u, v in arrows:
        x1, y1 = POS[u]
        x2, y2 = POS[v]
        arr = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="->", mutation_scale=14,
                              linewidth=2.0, color="#2b6cb0", connectionstyle="arc3,rad=0.15")
        ax.add_patch(arr)

    if title:
        ax.set_title(title, fontsize=13, pad=8)
    if subtitle:
        ax.text(0.5, 0.02, subtitle, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=9.6, color="#555555")
    if algebra_text:
        # Place the info box in the reserved right margin instead of over the graph.
        ax.text(1.38, 1.02, algebra_text, ha="left", va="top", fontsize=9.5,
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#cccccc"),
                family="monospace")


def draw_partition_legend(ax, c: Dict[int, int], title: str = "Ordered cells"):
    ax.axis("off")
    ax.set_facecolor("#fafafa")
    cells: Dict[int, List[int]] = {}
    for v in VERTICES:
        cells.setdefault(c[v], []).append(v)
    y = 0.95
    ax.text(0.02, y, title, fontsize=13, fontweight="bold", ha="left", va="top")
    y -= 0.12
    for cid, members in sorted(cells.items()):
        color = PALETTE.get(cid, "#bbbbbb")
        label = f"cell {cid}: {{{', '.join(map(str, members))}}}" if cid != 0 else f"uncoloured: {{{', '.join(map(str, members))}}}"
        ax.add_patch(plt.Rectangle((0.03, y - 0.03), 0.05, 0.05, facecolor=color, edgecolor="black"))
        ax.text(0.10, y, label, fontsize=11, ha="left", va="center")
        y -= 0.09
    ax.text(0.02, 0.06, graph_algebra_block(c), fontsize=10, ha="left", va="bottom", family="monospace")


def draw_mapping_text(ax, p: Dict[int, int], title: str = "Permutation"):
    ax.axis("off")
    lines = [title, "", cycle_notation(p), "", "mapping:"]
    line = []
    for v in VERTICES:
        line.append(f"{v}→{p[v]}")
        if len(line) == 4:
            lines.append("   ".join(line))
            line = []
    if line:
        lines.append("   ".join(line))
    lines.extend(["", GROUP_TEXT, "ρ=(1 2 3 4 5 6)(7 8 9 10 11 12)", "σ=(1 7)(2 8)(3 9)(4 10)(5 11)(6 12)"])
    ax.text(0.02, 0.95, "\n".join(lines), fontsize=11, ha="left", va="top", family="monospace")


def section_summary(fig, title: str, subtitle: str):
    fig.suptitle(title, fontsize=18, y=0.982)
    fig.text(0.5, 0.948, textwrap.fill(subtitle, width=110), ha="center", va="top", fontsize=11, color="#444444")


def add_fig_algebra(fig, c: Dict[int, int], y: float = 0.915):
    txt = (
        f"P=C_6 \\square K_2,   V={{1,...,12}},   |E|=18,   deg-seq={DEGREE_TEXT},   "
        f"{partition_notation(c)},   |Aut(P,π)|={len(automorphisms_for_colouring(c))}"
    )
    fig.text(0.5, y, textwrap.fill(txt, width=120), ha="center", va="top", fontsize=10.4, color="#333333")


# ============================================================
# Section metadata
# ============================================================
@dataclass
class Section:
    num: int
    title: str
    view_kind: str
    aut_colouring: Callable[[], Dict[int, int]]
    perm: Dict[int, int] | None = None
    perm2: Dict[int, int] | None = None


def sections() -> List[Section]:
    return [
        Section(1, "Formal definitions come first", "single_none", colouring_none),
        Section(2, "Why the search tree matters", "workflow_refine", colouring_none),
        Section(3, "A self-contained strategy section", "workflow_strategy", colouring_none),
        Section(4, "The ambient set of graphs", "single_none", colouring_none),
        Section(5, "What a colouring is", "single_three", colouring_three_blocks),
        Section(6, "The number of colours", "single_three", colouring_three_blocks),
        Section(7, "What a cell is", "cell_focus", colouring_three_blocks),
        Section(8, "Discrete colourings", "single_discrete", colouring_discrete),
        Section(9, "Discrete colouring as a permutation", "discrete_permutation", colouring_discrete, perm=rot(3)),
        Section(10, "Finer than or equal to", "coarse_vs_fine", colouring_finer),
        Section(11, "Subset intuition for finer colourings", "coarse_vs_fine", colouring_finer),
        Section(12, "Why colourings are also partitions", "single_three", colouring_three_blocks),
        Section(13, "Why colour order still matters", "same_partition_diff_order", colouring_three_blocks),
        Section(14, "What a coloured graph is", "single_three", colouring_three_blocks),
        Section(15, "The symmetric group", "perm_action", colouring_none, perm=rot(1)),
        Section(16, "Exponent notation for actions", "perm_action", colouring_none, perm=rot(2)),
        Section(17, "Image of a vertex under a permutation", "vertex_image", colouring_none, perm=rot(3)),
        Section(18, "Induced action on more complicated objects", "edge_image", colouring_none, perm=rot(3)),
        Section(19, "Action on subsets", "subset_action", colouring_none, perm=rot(3)),
        Section(20, "Action on graphs", "graph_action", colouring_none, perm=rot(3)),
        Section(21, "Discrete colouring gives G^π", "discrete_permutation", colouring_discrete, perm=rot(1)),
        Section(22, "Action on colourings", "colouring_action", colouring_three_blocks, perm=rot(1)),
        Section(23, "Action on coloured graphs", "colouring_action", colouring_three_blocks, perm=rot(2)),
        Section(24, "Isomorphism of coloured graphs", "isomorphic_pair", colouring_three_blocks, perm=rot(1)),
        Section(25, "What the isomorphism itself is", "isomorphic_pair", colouring_three_blocks, perm=rot(2)),
        Section(26, "Automorphism group", "single_none", colouring_none),
        Section(27, "Canonical form", "canonical_single", colouring_asymmetric, perm=rot(5)),
        Section(28, "Unique representative of the class", "canonical_pair", colouring_asymmetric, perm=rot(2), perm2=compose(rot(5), refl())),
        Section(29, "Why canonical forms solve comparison", "canonical_compare", colouring_asymmetric, perm=rot(1), perm2=compose(rot(4), refl())),
    ]


# ============================================================
# Section-specific graph view renderers
# ============================================================
def render_graph_view(sec: Section, path: Path):
    c = sec.aut_colouring()
    kind = sec.view_kind

    if kind in {"single_none", "single_three", "single_discrete"}:
        fig, ax = plt.subplots(figsize=(9.2, 7.4))
        extra = None
        if kind == "single_three":
            extra = graph_algebra_block(c, "k = 3 ordered colours")
        elif kind == "single_discrete":
            extra = graph_algebra_block(c, "|π| = 12 (discrete)")
        else:
            extra = graph_algebra_block(c)
        draw_graph(ax, c, title=sec.title, subtitle="same prism graph, viewed in one state", algebra_text=extra)
        fig.tight_layout(rect=[0.03, 0.03, 0.97, 0.96])

    elif kind == "workflow_refine":
        fig, axes = plt.subplots(1, 3, figsize=(14.2, 5.0))
        draw_graph(axes[0], colouring_none(), title="Start", subtitle="all vertices initially alike", algebra_text=graph_algebra_block(colouring_none()))
        draw_graph(axes[1], colouring_one_singleton(), title="Individualize", subtitle="single out vertex 1", highlight_nodes=[1], algebra_text=graph_algebra_block(colouring_one_singleton()))
        draw_graph(axes[2], colouring_discrete(), title="Refined end state", subtitle="discrete colouring", algebra_text=graph_algebra_block(colouring_discrete()))
        section_summary(fig, sec.title, "I show the search-tree idea by splitting symmetry step by step until the colouring becomes discrete.")
        fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.90])

    elif kind == "workflow_strategy":
        fig, axes = plt.subplots(1, 3, figsize=(14.2, 5.0))
        draw_graph(axes[0], colouring_none(), title="Graph G", subtitle="start with P", algebra_text=graph_algebra_block(colouring_none()))
        draw_graph(axes[1], colouring_three_blocks(), title="Colouring / partition", subtitle="organize vertices into cells", algebra_text=graph_algebra_block(colouring_three_blocks()))
        draw_graph(axes[2], colouring_discrete(), title="Rigid endpoint", subtitle="search-tree end state", algebra_text=graph_algebra_block(colouring_discrete()))
        section_summary(fig, sec.title, "I keep the workflow visible: graph → colouring → refinement/search → automorphisms or canonical form.")
        fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.90])

    elif kind == "cell_focus":
        fig = plt.figure(figsize=(13.2, 6.7))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1])
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        draw_graph(ax1, c, title=sec.title, subtitle="cell 2 is the middle block", highlight_nodes=[5, 6, 7, 8], algebra_text=graph_algebra_block(c))
        draw_partition_legend(ax2, c, title="Cells of π")
        fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.96])

    elif kind == "discrete_permutation":
        fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.4))
        draw_graph(axes[0], colouring_discrete(), title="Discrete colouring π", subtitle="every cell is a singleton", algebra_text=graph_algebra_block(colouring_discrete()))
        draw_graph(axes[1], colouring_discrete(), labels=permute_display_labels(sec.perm), title="Permutation view", subtitle=f"display labels follow {cycle_notation(sec.perm)}", algebra_text=graph_algebra_block(colouring_discrete(), f"g = {cycle_notation(sec.perm)}"))
        fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.96])

    elif kind == "coarse_vs_fine":
        fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.4))
        draw_graph(axes[0], colouring_three_blocks(), title="Coarse colouring π", subtitle="three ordered cells", algebra_text=graph_algebra_block(colouring_three_blocks()))
        draw_graph(axes[1], colouring_finer(), title="Finer colouring π'", subtitle="each π'-cell lies inside a π-cell", algebra_text=graph_algebra_block(colouring_finer()))
        fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.96])

    elif kind == "same_partition_diff_order":
        fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.4))
        c1 = colouring_three_blocks()
        c2 = {v: {1: 2, 2: 1, 3: 3}[c1[v]] for v in VERTICES}
        draw_graph(axes[0], c1, title="π with order (1,2,3)", subtitle="same partition blocks", algebra_text=graph_algebra_block(c1, partition_notation(c1)))
        draw_graph(axes[1], c2, title="σ with order (2,1,3)", subtitle="same blocks, different order", algebra_text=graph_algebra_block(c2, partition_notation(c2)))
        fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.96])

    elif kind == "perm_action":
        fig = plt.figure(figsize=(13.2, 6.3))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1])
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        draw_graph(ax1, colouring_none(), title=sec.title, subtitle="same graph, acted on by g", algebra_text=graph_algebra_block(colouring_none(), f"g = {cycle_notation(sec.perm)}"))
        draw_mapping_text(ax2, sec.perm, title="chosen g ∈ S12")
        fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.96])

    elif kind == "vertex_image":
        fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.4))
        draw_graph(axes[0], colouring_none(), title="Original vertex", subtitle="highlight v = 1", highlight_nodes=[1], algebra_text=graph_algebra_block(colouring_none(), "v = 1"))
        draw_graph(axes[1], colouring_none(), title="Image under g", subtitle=f"for g = {cycle_notation(sec.perm)}, 1^g = {sec.perm[1]}", highlight_nodes=[sec.perm[1]], arrows=[(1, sec.perm[1])], algebra_text=graph_algebra_block(colouring_none(), f"1^g = {sec.perm[1]}"))
        fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.96])

    elif kind == "edge_image":
        fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.4))
        e1 = (1, 2)
        e2 = tuple(sorted((sec.perm[1], sec.perm[2])))
        draw_graph(axes[0], colouring_none(), title="A derived object", subtitle="edge {1,2}", highlight_edges=[e1], highlight_nodes=[1, 2], algebra_text=graph_algebra_block(colouring_none(), "W = {1,2}"))
        draw_graph(axes[1], colouring_none(), title="Its image under g", subtitle=f"{{1,2}}^g = {{{sec.perm[1]},{sec.perm[2]}}}", highlight_edges=[e2], highlight_nodes=[sec.perm[1], sec.perm[2]], algebra_text=graph_algebra_block(colouring_none(), f"W^g = {{{sec.perm[1]},{sec.perm[2]}}}"))
        fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.96])

    elif kind == "subset_action":
        fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.4))
        W = [1, 2, 3]
        Wg = [sec.perm[v] for v in W]
        draw_graph(axes[0], colouring_none(), title="Subset W", subtitle="W = {1,2,3}", highlight_nodes=W, algebra_text=graph_algebra_block(colouring_none(), "W = {1,2,3}"))
        draw_graph(axes[1], colouring_none(), title="Image W^g", subtitle=f"W^g = {{{', '.join(map(str, Wg))}}}", highlight_nodes=Wg, algebra_text=graph_algebra_block(colouring_none(), f"W^g = {{{', '.join(map(str, Wg))}}}"))
        fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.96])

    elif kind == "graph_action":
        fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.4))
        draw_graph(axes[0], colouring_none(), title="G", subtitle="original labels", algebra_text=graph_algebra_block(colouring_none(), "G = P"))
        draw_graph(axes[1], colouring_none(), labels=permute_display_labels(sec.perm), title="G^g", subtitle="same adjacency after relabelling", algebra_text=graph_algebra_block(colouring_none(), f"g = {cycle_notation(sec.perm)}"))
        fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.96])

    elif kind == "colouring_action":
        fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.4))
        c2 = apply_perm_to_colouring(c, sec.perm)
        draw_graph(axes[0], c, title="(G, π)", subtitle="original colouring", algebra_text=graph_algebra_block(c))
        draw_graph(axes[1], c2, title="π^g or (G,π)^g", subtitle="colours transported by the action", algebra_text=graph_algebra_block(c2, f"g = {cycle_notation(sec.perm)}"))
        fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.96])

    elif kind == "isomorphic_pair":
        fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.4))
        c2 = apply_perm_to_colouring(c, sec.perm)
        draw_graph(axes[0], c, title="(G, π)", subtitle="first coloured graph", algebra_text=graph_algebra_block(c))
        draw_graph(axes[1], c2, title="(G', π')", subtitle="isomorphic coloured graph", algebra_text=graph_algebra_block(c2, f"g = {cycle_notation(sec.perm)}"))
        fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.96])

    elif kind == "canonical_single":
        fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.4))
        c_start = apply_perm_to_colouring(c, sec.perm)
        c_can, p = canonical_aut_colouring(c_start)
        draw_graph(axes[0], c_start, title="Input coloured graph", subtitle="a relabelled copy", algebra_text=graph_algebra_block(c_start))
        draw_graph(axes[1], c_can, title="Chosen canonical representative", subtitle="lexicographically smallest sequence", algebra_text=graph_algebra_block(c_can, f"chosen by {cycle_notation(p)}"))
        fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.96])

    elif kind == "canonical_pair":
        fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.2))
        c1 = apply_perm_to_colouring(c, sec.perm)
        c2 = apply_perm_to_colouring(c, sec.perm2)
        c1_can, _ = canonical_aut_colouring(c1)
        draw_graph(axes[0], c1, title="First copy", subtitle="same isomorphism class", algebra_text=graph_algebra_block(c1))
        draw_graph(axes[1], c2, title="Second copy", subtitle="different labelling", algebra_text=graph_algebra_block(c2))
        draw_graph(axes[2], c1_can, title="Common canonical form", subtitle="both copies reduce here", algebra_text=graph_algebra_block(c1_can))
        fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.96])

    elif kind == "canonical_compare":
        fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.2))
        c1 = apply_perm_to_colouring(c, sec.perm)
        c2 = apply_perm_to_colouring(c, sec.perm2)
        c1_can, _ = canonical_aut_colouring(c1)
        c2_can, _ = canonical_aut_colouring(c2)
        draw_graph(axes[0], c1, title="Input A", subtitle="first labelled copy", algebra_text=graph_algebra_block(c1))
        draw_graph(axes[1], c2, title="Input B", subtitle="second labelled copy", algebra_text=graph_algebra_block(c2))
        draw_graph(axes[2], c1_can, title="Canonical(A) = Canonical(B)", subtitle=str(colour_sequence(c1_can) == colour_sequence(c2_can)), algebra_text=graph_algebra_block(c1_can))
        fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.96])

    else:
        raise ValueError(f"Unknown view kind {kind}")

    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Automorphism thumbnails
# ============================================================
def render_automorphism_view(sec: Section, path: Path):
    c = sec.aut_colouring()
    autos = automorphisms_for_colouring(c)
    orbits = orbit_partition(autos)
    count = len(autos)

    ncols = min(6, max(1, count))
    nrows = math.ceil(count / ncols)
    fig_h = 2.5 + 2.25 * nrows
    fig = plt.figure(figsize=(2.55 * ncols + 1.4, fig_h))
    gs = fig.add_gridspec(nrows + 1, ncols, height_ratios=[0.9] + [1] * nrows)

    summary_ax = fig.add_subplot(gs[0, :])
    summary_ax.axis("off")
    summary_ax.text(0.01, 0.92, f"{sec.num:02d}. {sec.title}", fontsize=16, fontweight="bold", ha="left", va="top")
    summary_ax.text(0.01, 0.58, f"P = C6 □ K2,   {partition_notation(c)},   |Aut(P,π)| = {count},   Orbits = {orbits}", fontsize=11, ha="left", va="top")
    summary_ax.text(0.01, 0.24, "Each thumbnail keeps the same prism geometry and relabels the vertices by one automorphism. Algebra is repeated on purpose.", fontsize=10, ha="left", va="bottom", color="#555555")

    for idx, p in enumerate(autos):
        r = idx // ncols + 1
        cc = idx % ncols
        ax = fig.add_subplot(gs[r, cc])
        set_clean(ax)
        nx.draw_networkx_edges(BASE_G, POS, ax=ax, width=1.2, edge_color="#7a7a7a")
        nx.draw_networkx_nodes(BASE_G, POS, ax=ax, node_color=node_facecolors(c), edgecolors="black", linewidths=1.0, node_size=380)
        nx.draw_networkx_labels(BASE_G, POS, ax=ax, labels={v: str(p[v]) for v in VERTICES}, font_size=8.2, font_weight="bold")
        ax.set_title(cycle_notation(p), fontsize=8.6, pad=2)
        ax.text(0.03, 0.03, f"|π|={len(set(c.values()))}\nAut={count}", transform=ax.transAxes, fontsize=7.6,
                ha="left", va="bottom", bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#cccccc"))

    total_cells = nrows * ncols
    for idx in range(count, total_cells):
        r = idx // ncols + 1
        cc = idx % ncols
        ax = fig.add_subplot(gs[r, cc])
        ax.axis("off")

    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.98])
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Nauty-friendly export
# ============================================================
def export_nauty_friendly_data(outdir: Path):
    txt = [
        "Automorphism demo: nauty-friendly data",
        "",
        "Base graph (all sections): prism graph P = C6 □ K2",
        f"graph6: {BASE_G6}",
        "V = {1,...,12}",
        f"E = {{{EDGE_TEXT}}}",
        f"Adjacency: {'; '.join(ADJ_LINES)}",
        f"Degree sequence: {DEGREE_TEXT}",
        "Generators: ρ=(1 2 3 4 5 6)(7 8 9 10 11 12), τ=(2 6)(3 5)(8 12)(9 11), σ=(1 7)(2 8)(3 9)(4 10)(5 11)(6 12)",
        f"Total automorphisms: {len(ALL_AUTOS)}",
        "",
    ]
    for sec in sections():
        c = sec.aut_colouring()
        autos = automorphisms_for_colouring(c)
        txt.append(f"Section {sec.num:02d}: {sec.title}")
        txt.append(f"  colouring: {colouring_name(c)}")
        txt.append(f"  partition: {partition_notation(c)}")
        txt.append(f"  |Aut(P,π)| = {len(autos)}")
        txt.append("")
    (outdir / "automorphism_demo_nauty_data.txt").write_text("\n".join(txt), encoding="utf-8")


# ============================================================
# Main
# ============================================================
def main():
    outdir = Path.cwd() / OUTDIR
    outdir.mkdir(parents=True, exist_ok=True)
    for sec in sections():
        graph_path = outdir / f"{sec.num:02d}_graph_view.png"
        aut_path = outdir / f"{sec.num:02d}_automorphism_or_partition_view.png"
        render_graph_view(sec, graph_path)
        render_automorphism_view(sec, aut_path)
        print(f"saved {graph_path.name} and {aut_path.name}")
    export_nauty_friendly_data(outdir)
    print(f"\nDone. Output folder: {outdir}")


if __name__ == "__main__":
    main()
