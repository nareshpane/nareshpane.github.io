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

# ------------------------------------------------------------
# Core graph and permutations
# ------------------------------------------------------------
N = 12
VERTICES = list(range(1, N + 1))
BASE_G = nx.cycle_graph(VERTICES)

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


def circle_positions(n: int = N, radius: float = 1.0) -> Dict[int, Tuple[float, float]]:
    pos = {}
    for i in range(n):
        theta = math.pi / 2 - 2 * math.pi * i / n
        pos[i + 1] = (radius * math.cos(theta), radius * math.sin(theta))
    return pos


POS = circle_positions()


def rot(k: int) -> Dict[int, int]:
    return {v: ((v - 1 + k) % N) + 1 for v in VERTICES}


def refl() -> Dict[int, int]:
    # reflection fixing vertex 1 and reversing orientation on the cycle
    return {v: ((1 - v) % N) + 1 for v in VERTICES}


def compose(p: Dict[int, int], q: Dict[int, int]) -> Dict[int, int]:
    # p after q
    return {v: p[q[v]] for v in VERTICES}


DIHEDRAL = [rot(k) for k in range(N)] + [compose(rot(k), refl()) for k in range(N)]


# ------------------------------------------------------------
# Colourings / partitions
# ------------------------------------------------------------

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


def colouring_bipartite() -> Dict[int, int]:
    return {v: 1 if v % 2 else 2 for v in VERTICES}


def colouring_discrete() -> Dict[int, int]:
    return {v: v for v in VERTICES}


def colour_sequence(c: Dict[int, int]) -> Tuple[int, ...]:
    return tuple(c[v] for v in VERTICES)


def apply_perm_to_colouring(c: Dict[int, int], p: Dict[int, int]) -> Dict[int, int]:
    # c^g(v) = c(v^g)
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
    if c == colouring_bipartite():
        return "odd/even colouring"
    if c == colouring_discrete():
        return "discrete colouring"
    if c == colouring_asymmetric():
        return "asymmetric coloured graph"
    return "custom colouring"


# ------------------------------------------------------------
# Group / canonical helpers
# ------------------------------------------------------------

def automorphisms_for_colouring(c: Dict[int, int]) -> List[Dict[int, int]]:
    autos = []
    for p in DIHEDRAL:
        if all(c[v] == c[p[v]] for v in VERTICES):
            autos.append(p)
    autos.sort(key=lambda p: tuple(p[v] for v in VERTICES))
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
        cyc = []
        cur = v
        while cur not in seen:
            seen.add(cur)
            cyc.append(cur)
            cur = p[cur]
        if len(cyc) > 1:
            cycles.append("(" + " ".join(map(str, cyc)) + ")")
    return " ".join(cycles) if cycles else "id"


def canonical_dihedral_colouring(c: Dict[int, int]) -> Tuple[Dict[int, int], Dict[int, int]]:
    best_p = None
    best_seq = None
    for p in DIHEDRAL:
        seq = tuple(apply_perm_to_colouring(c, p)[v] for v in VERTICES)
        if best_seq is None or seq < best_seq:
            best_seq = seq
            best_p = p
    return apply_perm_to_colouring(c, best_p), best_p


# ------------------------------------------------------------
# Drawing utilities
# ------------------------------------------------------------

def node_facecolors(c: Dict[int, int]) -> List[str]:
    return [PALETTE.get(c[v], "#bbbbbb") for v in VERTICES]


def set_clean(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.25)
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(1.0)
        sp.set_color("#bfbfbf")
    ax.set_facecolor("#fafafa")


def draw_graph(ax, c: Dict[int, int], labels: Dict[int, str] | None = None,
               title: str = "", subtitle: str | None = None,
               highlight_nodes: List[int] | None = None,
               highlight_edges: List[Tuple[int, int]] | None = None,
               arrows: List[Tuple[int, int]] | None = None,
               extra_text: str | None = None):
    set_clean(ax)
    highlight_nodes = highlight_nodes or []
    highlight_edges = highlight_edges or []
    arrows = arrows or []
    nx.draw_networkx_edges(BASE_G, POS, ax=ax, width=1.6, edge_color="#6f6f6f")
    if highlight_edges:
        nx.draw_networkx_edges(
            BASE_G, POS, edgelist=highlight_edges, ax=ax, width=3.1, edge_color="#d55e5e"
        )
    sizes = [760 if v in highlight_nodes else 620 for v in VERTICES]
    borders = ["#b22222" if v in highlight_nodes else "black" for v in VERTICES]
    nx.draw_networkx_nodes(
        BASE_G, POS, ax=ax,
        node_color=node_facecolors(c),
        edgecolors=borders,
        linewidths=1.4,
        node_size=sizes,
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
                fontsize=9.8, color="#555555")
    if extra_text:
        ax.text(
            1.03,
            0.97,
            extra_text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            clip_on=False,
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc"),
        )


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


def draw_mapping_text(ax, p: Dict[int, int], title: str = "Permutation"):
    ax.axis("off")
    lines = [title, "", cycle_notation(p), "", "mapping:"]
    chunks = []
    line = []
    for v in VERTICES:
        line.append(f"{v}→{p[v]}")
        if len(line) == 4:
            chunks.append("   ".join(line))
            line = []
    if line:
        chunks.append("   ".join(line))
    lines.extend(chunks)
    ax.text(0.02, 0.95, "\n".join(lines), fontsize=11, ha="left", va="top", family="monospace")


def section_summary(fig, title: str, subtitle: str):
    fig.suptitle(title, fontsize=18, y=0.98)
    fig.text(0.5, 0.945, textwrap.fill(subtitle, width=110), ha="center", va="top", fontsize=11, color="#444444")


# ------------------------------------------------------------
# Section configurations
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# Main graph-view renderers
# ------------------------------------------------------------

def render_graph_view(sec: Section, path: Path):
    c = sec.aut_colouring()
    kind = sec.view_kind

    if kind in {"single_none", "single_three", "single_discrete"}:
        fig, ax = plt.subplots(figsize=(10.8, 7.2))
        subtitle = f"12-cycle C12 with 12 vertices and 12 edges."
        extra = None
        if kind == "single_three":
            extra = "k = 3 ordered colours"
        if kind == "single_discrete":
            extra = "|π| = 12 (discrete colouring)"
        draw_graph(ax, c, title=sec.title, subtitle=subtitle, extra_text=extra)
        if kind != "single_none":
            fig.text(0.83, 0.50, "", fontsize=1)
        fig.tight_layout(rect=[0.03, 0.03, 0.90, 0.95])

    elif kind == "workflow_refine":
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
        draw_graph(axes[0], colouring_none(), title="Start", subtitle="all vertices initially alike")
        draw_graph(axes[1], colouring_one_singleton(), title="Individualize", subtitle="single out vertex 1", highlight_nodes=[1])
        draw_graph(axes[2], colouring_discrete(), title="Refined end state", subtitle="discrete colouring")
        section_summary(fig, sec.title, "A search tree splits symmetry step by step until the graph becomes rigid enough to compare branches or extract automorphisms.")
        fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.90])

    elif kind == "workflow_strategy":
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
        draw_graph(axes[0], colouring_none(), title="Graph G", subtitle="start with the actual graph")
        draw_graph(axes[1], colouring_three_blocks(), title="Colouring / partition", subtitle="organize vertices into cells")
        draw_graph(axes[2], colouring_discrete(), title="Canonical / rigid state", subtitle="search-tree endpoint")
        section_summary(fig, sec.title, "The overall workflow is graph → colouring → refinement/search → automorphisms or canonical form.")
        fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.90])

    elif kind == "cell_focus":
        fig = plt.figure(figsize=(13, 6.6))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.4, 1])
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        draw_graph(ax1, c, title=sec.title, subtitle="cell 2 is the middle block", highlight_nodes=[5, 6, 7, 8])
        draw_partition_legend(ax2, c, title="Cells of π")
        fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.95])

    elif kind == "discrete_permutation":
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
        draw_graph(axes[0], colouring_discrete(), title="Discrete colouring π", subtitle="every cell is a singleton")
        draw_graph(axes[1], colouring_discrete(), labels=permute_display_labels(sec.perm),
                   title="Permutation view", subtitle=f"display labels follow {cycle_notation(sec.perm)}")
        fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.95])

    elif kind == "coarse_vs_fine":
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
        draw_graph(axes[0], colouring_three_blocks(), title="Coarse colouring π", subtitle="three ordered cells")
        draw_graph(axes[1], colouring_finer(), title="Finer colouring π'", subtitle="each π'-cell lies inside a π-cell")
        fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.95])

    elif kind == "same_partition_diff_order":
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
        c1 = colouring_three_blocks()
        c2 = {v: {1: 2, 2: 1, 3: 3}[c1[v]] for v in VERTICES}
        draw_graph(axes[0], c1, title="π with order (1,2,3)", subtitle="same partition blocks")
        draw_graph(axes[1], c2, title="σ with order (2,1,3)", subtitle="same blocks, different colour order")
        fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.95])

    elif kind == "perm_action":
        fig = plt.figure(figsize=(13, 6.3))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1])
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        draw_graph(ax1, colouring_none(), title=sec.title, subtitle="same 12-cycle, acted on by a permutation")
        draw_mapping_text(ax2, sec.perm, title="chosen g ∈ S12")
        fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.95])

    elif kind == "vertex_image":
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
        draw_graph(axes[0], colouring_none(), title="Original vertex", subtitle="highlight v = 1", highlight_nodes=[1])
        draw_graph(axes[1], colouring_none(), title="Image under g", subtitle=f"for g = {cycle_notation(sec.perm)}, 1^g = {sec.perm[1]}", highlight_nodes=[sec.perm[1]], arrows=[(1, sec.perm[1])])
        fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.95])

    elif kind == "edge_image":
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
        e1 = (1, 2)
        e2 = tuple(sorted((sec.perm[1], sec.perm[2])))
        draw_graph(axes[0], colouring_none(), title="A derived object", subtitle="edge {1,2}", highlight_edges=[e1], highlight_nodes=[1, 2])
        draw_graph(axes[1], colouring_none(), title="Its image under g", subtitle=f"{{1,2}}^g = {{{sec.perm[1]},{sec.perm[2]}}}", highlight_edges=[e2], highlight_nodes=[sec.perm[1], sec.perm[2]])
        fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.95])

    elif kind == "subset_action":
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
        W = [1, 2, 3]
        Wg = [sec.perm[v] for v in W]
        draw_graph(axes[0], colouring_none(), title="Subset W", subtitle="W = {1,2,3}", highlight_nodes=W)
        draw_graph(axes[1], colouring_none(), title="Image W^g", subtitle=f"W^g = {{{', '.join(map(str, Wg))}}}", highlight_nodes=Wg)
        fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.95])

    elif kind == "graph_action":
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
        draw_graph(axes[0], colouring_none(), title="G", subtitle="original labels")
        draw_graph(axes[1], colouring_none(), labels=permute_display_labels(sec.perm), title="G^g", subtitle="same adjacency pattern after relabelling")
        fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.95])

    elif kind == "colouring_action":
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
        c2 = apply_perm_to_colouring(c, sec.perm)
        draw_graph(axes[0], c, title="(G, π)", subtitle="original colouring")
        draw_graph(axes[1], c2, title="π^g or (G,π)^g", subtitle="colours transported by the action")
        fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.95])

    elif kind == "isomorphic_pair":
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
        c2 = apply_perm_to_colouring(c, sec.perm)
        draw_graph(axes[0], c, title="(G, π)", subtitle="first coloured graph")
        draw_graph(axes[1], c2, title="(G', π')", subtitle="isomorphic coloured graph")
        fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.95])

    elif kind == "canonical_single":
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
        c_start = apply_perm_to_colouring(c, sec.perm)
        c_can, _ = canonical_dihedral_colouring(c_start)
        draw_graph(axes[0], c_start, title="Input coloured graph", subtitle="a relabelled copy of the same object")
        draw_graph(axes[1], c_can, title="Chosen canonical representative", subtitle="lexicographically smallest dihedral colour sequence")
        fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.95])

    elif kind == "canonical_pair":
        fig, axes = plt.subplots(1, 3, figsize=(16, 5.0))
        c1 = apply_perm_to_colouring(c, sec.perm)
        c2 = apply_perm_to_colouring(c, sec.perm2)
        c1_can, _ = canonical_dihedral_colouring(c1)
        draw_graph(axes[0], c1, title="First copy", subtitle="same isomorphism class")
        draw_graph(axes[1], c2, title="Second copy", subtitle="different labelling, same class")
        draw_graph(axes[2], c1_can, title="Common canonical form", subtitle="both copies reduce to this representative")
        fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.95])

    elif kind == "canonical_compare":
        fig, axes = plt.subplots(1, 3, figsize=(16, 5.0))
        c1 = apply_perm_to_colouring(c, sec.perm)
        c2 = apply_perm_to_colouring(c, sec.perm2)
        c1_can, _ = canonical_dihedral_colouring(c1)
        c2_can, _ = canonical_dihedral_colouring(c2)
        draw_graph(axes[0], c1, title="Input A", subtitle="first labelled copy")
        draw_graph(axes[1], c2, title="Input B", subtitle="second labelled copy")
        draw_graph(axes[2], c1_can, title="Canonical(A) = Canonical(B)", subtitle=str(colour_sequence(c1_can) == colour_sequence(c2_can)))
        fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.95])

    else:
        raise ValueError(f"Unknown view kind {kind}")

    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------
# Automorphism view renderer
# ------------------------------------------------------------

def render_automorphism_view(sec: Section, path: Path):
    c = sec.aut_colouring()
    autos = automorphisms_for_colouring(c)
    orbits = orbit_partition(autos)
    count = len(autos)

    ncols = min(6, max(1, count))
    nrows = math.ceil(count / ncols)
    fig_h = 2.1 + 2.25 * nrows
    fig = plt.figure(figsize=(2.5 * ncols + 1.2, fig_h))
    gs = fig.add_gridspec(nrows + 1, ncols, height_ratios=[0.7] + [1] * nrows)

    summary_ax = fig.add_subplot(gs[0, :])
    summary_ax.axis("off")
    summary_ax.text(0.01, 0.90, f"{sec.num:02d}. {sec.title}", fontsize=16, fontweight="bold", ha="left", va="top")
    summary_ax.text(
        0.01,
        0.50,
        f"Graph used for group view: 12-cycle C12 with {colouring_name(c)}.   |Aut| = {count}.   Orbits: {orbits}",
        fontsize=11,
        ha="left",
        va="top",
    )
    summary_ax.text(
        0.01,
        0.12,
        "Each thumbnail keeps the same geometry and shows the relabelled vertex names under one automorphism.",
        fontsize=10,
        ha="left",
        va="bottom",
        color="#555555",
    )

    for idx, p in enumerate(autos):
        r = idx // ncols + 1
        cc = idx % ncols
        ax = fig.add_subplot(gs[r, cc])
        set_clean(ax)
        nx.draw_networkx_edges(BASE_G, POS, ax=ax, width=1.2, edge_color="#7a7a7a")
        nx.draw_networkx_nodes(BASE_G, POS, ax=ax, node_color=node_facecolors(c), edgecolors="black", linewidths=1.0, node_size=380)
        labels = {v: str(p[v]) for v in VERTICES}
        nx.draw_networkx_labels(BASE_G, POS, ax=ax, labels=labels, font_size=8.5, font_weight="bold")
        ax.set_title(cycle_notation(p), fontsize=8.8, pad=3)

    total_cells = nrows * ncols
    for idx in range(count, total_cells):
        r = idx // ncols + 1
        cc = idx % ncols
        ax = fig.add_subplot(gs[r, cc])
        ax.axis("off")

    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.98])
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------
# Nauty-friendly exports
# ------------------------------------------------------------

def export_nauty_friendly_data(outdir: Path):
    g6 = nx.to_graph6_bytes(BASE_G, header=False).decode().strip()
    txt = ["Automorphism demo: nauty-friendly data", "", f"Base graph (all sections): C12", f"graph6: {g6}", ""]
    for sec in sections():
        c = sec.aut_colouring()
        autos = automorphisms_for_colouring(c)
        cells: Dict[int, List[int]] = {}
        for v in VERTICES:
            cells.setdefault(c[v], []).append(v)
        txt.append(f"Section {sec.num:02d}: {sec.title}")
        txt.append(f"  colouring: {colouring_name(c)}")
        txt.append(f"  cells: {cells}")
        txt.append(f"  |Aut| under this colouring: {len(autos)}")
        txt.append("")
    (outdir / "automorphism_demo_nauty_data.txt").write_text("\n".join(txt), encoding="utf-8")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

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
