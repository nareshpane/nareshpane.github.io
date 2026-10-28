#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import shutil
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

DEFAULT_N = 20
DEFAULT_SEED = 314159
DEFAULT_FPS = 6
DEFAULT_HOLD_FINAL_FRAMES = 18
DEFAULT_OUTPUT = Path("./uniform_vs_permutation_20_nodes_all_edges.mp4")

NODE_FACE = "#f5f5f5"
NODE_EDGE = "#333333"
SIMPLE_EDGE = "#666666"
MULTI_EDGE = "#1565c0"
SELF_LOOP = "#c62828"
PERM_EDGE = "#222222"
PANEL_BG = "#fbfbfb"

Point = Tuple[float, float]
Edge = Tuple[int, int]


def almost_circular_layout(n: int, rng: np.random.Generator) -> Dict[int, Point]:
    """
    Place most nodes around a nearly circular ring, with a few scattered inside.
    This keeps the overall circular feel while avoiding a perfectly rigid circle.
    """
    if n < 2:
        return {0: (0.0, 0.0)}

    inner_count = max(3, min(5, n // 5))
    outer_count = max(0, n - inner_count)

    points: List[Point] = []

    if outer_count > 0:
        angle_offset = float(rng.uniform(0.0, 2.0 * math.pi))
        outer_angles = np.linspace(0.0, 2.0 * math.pi, outer_count, endpoint=False) + angle_offset
        outer_angles += rng.normal(0.0, 0.045, size=outer_count)

        for theta in outer_angles:
            r = float(np.clip(0.88 + rng.normal(0.0, 0.028), 0.80, 0.95))
            points.append((r * math.cos(theta), r * math.sin(theta)))

    attempts = 0
    while len(points) < n and attempts < 15000:
        attempts += 1
        theta = float(rng.uniform(0.0, 2.0 * math.pi))
        r = float(np.sqrt(rng.uniform(0.0, 1.0)) * 0.45)
        x = r * math.cos(theta)
        y = r * math.sin(theta)

        if all((x - px) ** 2 + (y - py) ** 2 >= 0.13**2 for px, py in points):
            points.append((x, y))

    while len(points) < n:
        theta = float(rng.uniform(0.0, 2.0 * math.pi))
        r = float(np.sqrt(rng.uniform(0.0, 1.0)) * 0.42)
        points.append((r * math.cos(theta), r * math.sin(theta)))

    return {i: points[i] for i in range(n)}


def permutation_edge_order(n: int, rng: np.random.Generator) -> List[Edge]:
    edges = list(combinations(range(n), 2))
    rng.shuffle(edges)
    return edges


def uniform_ordered_pairs(n: int, total_steps: int, rng: np.random.Generator) -> List[Edge]:
    draws: List[Edge] = []
    for _ in range(total_steps):
        u = int(rng.integers(0, n))
        v = int(rng.integers(0, n))
        draws.append((u, v))
    return draws


def grouped_uniform_state(draws: Sequence[Edge], upto: int):
    simple_counts: Counter[Tuple[int, int]] = Counter()
    loop_counts: Counter[int] = Counter()

    for u, v in draws[:upto]:
        if u == v:
            loop_counts[u] += 1
        else:
            a, b = (u, v) if u < v else (v, u)
            simple_counts[(a, b)] += 1

    repeated_pairs = sum(1 for cnt in simple_counts.values() if cnt >= 2)
    repeated_samples = sum(cnt - 1 for cnt in simple_counts.values() if cnt >= 2)
    total_loops = sum(loop_counts.values())
    distinct_simple = len(simple_counts)

    return simple_counts, loop_counts, repeated_pairs, repeated_samples, total_loops, distinct_simple


def alternating_radii(k: int, base: float = 0.10) -> List[float]:
    if k <= 1:
        return [0.0]
    radii: List[float] = []
    for i in range(k):
        step = (i // 2) + 1
        sign = -1 if i % 2 == 0 else 1
        radii.append(sign * base * step)
    radii.sort()
    return radii


def draw_curved_edge(ax, p1: Point, p2: Point, color: str, lw: float, rad: float = 0.0, alpha: float = 1.0, zorder: int = 2) -> None:
    patch = FancyArrowPatch(
        p1,
        p2,
        arrowstyle='-',
        connectionstyle=f"arc3,rad={rad}",
        linewidth=lw,
        color=color,
        alpha=alpha,
        zorder=zorder,
        shrinkA=6,
        shrinkB=6,
        capstyle='round',
        joinstyle='round',
    )
    ax.add_patch(patch)


def draw_self_loops(ax, pos: Dict[int, Point], loop_counts: Counter[int]) -> None:
    for node, count in loop_counts.items():
        x, y = pos[node]
        for i in range(count):
            radius = 0.068 + 0.022 * i
            center = (x + 0.085 + 0.016 * i, y + 0.085 + 0.016 * i)
            loop = Circle(
                center,
                radius=radius,
                fill=False,
                edgecolor=SELF_LOOP,
                linewidth=1.7,
                alpha=0.95,
                zorder=4,
            )
            ax.add_patch(loop)


def draw_uniform_panel(ax, pos: Dict[int, Point], draws: Sequence[Edge], upto: int) -> Tuple[int, int, int, int]:
    simple_counts, loop_counts, repeated_pairs, repeated_samples, total_loops, distinct_simple = grouped_uniform_state(draws, upto)

    for (u, v), count in simple_counts.items():
        p1, p2 = pos[u], pos[v]
        if count == 1:
            draw_curved_edge(ax, p1, p2, color=SIMPLE_EDGE, lw=1.00, rad=0.0, alpha=0.92, zorder=2)
        else:
            draw_curved_edge(ax, p1, p2, color=SIMPLE_EDGE, lw=0.75, rad=0.0, alpha=0.22, zorder=1)
            for rad in alternating_radii(count, base=0.09):
                draw_curved_edge(ax, p1, p2, color=MULTI_EDGE, lw=1.22, rad=rad, alpha=0.98, zorder=3)

    draw_self_loops(ax, pos, loop_counts)
    return distinct_simple, repeated_pairs, repeated_samples, total_loops


def draw_permutation_panel(ax, pos: Dict[int, Point], edges: Sequence[Edge], upto: int) -> int:
    for u, v in edges[:upto]:
        draw_curved_edge(ax, pos[u], pos[v], color=PERM_EDGE, lw=1.02, rad=0.0, alpha=0.94, zorder=2)
    return upto


def draw_nodes(ax, pos: Dict[int, Point], n: int) -> None:
    xs = [pos[i][0] for i in range(n)]
    ys = [pos[i][1] for i in range(n)]
    ax.scatter(xs, ys, s=56, facecolors=NODE_FACE, edgecolors=NODE_EDGE, linewidths=0.8, zorder=5)
    ax.set_xlim(-1.16, 1.16)
    ax.set_ylim(-1.16, 1.16)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor(PANEL_BG)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.6)
        spine.set_edgecolor('#4f4f4f')
    ax.add_patch(
        Rectangle(
            (0.004, 0.004),
            0.992,
            0.992,
            transform=ax.transAxes,
            fill=False,
            edgecolor='#4f4f4f',
            linewidth=1.4,
            zorder=50,
            clip_on=False,
        )
    )


def save_animation(anim: FuncAnimation, output_path: Path, fps: int, dpi: int = 160) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == '.mp4' and shutil.which('ffmpeg'):
        writer = FFMpegWriter(
            fps=fps,
            codec='libx264',
            bitrate=2400,
            extra_args=['-pix_fmt', 'yuv420p', '-movflags', '+faststart', '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2'],
        )
        anim.save(output_path, writer=writer, dpi=dpi)
        return output_path

    fallback = output_path.with_suffix('.gif')
    anim.save(fallback, writer=PillowWriter(fps=fps), dpi=120)
    return fallback


def make_animation(n: int, seed: int, fps: int, hold_final_frames: int, output: Path) -> Path:
    total_simple_edges = n * (n - 1) // 2
    rng_layout = np.random.default_rng(seed)
    rng_perm = np.random.default_rng(seed + 1)
    rng_uniform = np.random.default_rng(seed + 2)

    pos = almost_circular_layout(n=n, rng=rng_layout)
    perm_edges = permutation_edge_order(n=n, rng=rng_perm)
    uniform_draws = uniform_ordered_pairs(n=n, total_steps=total_simple_edges, rng=rng_uniform)

    total_frames = total_simple_edges + hold_final_frames

    fig = plt.figure(figsize=(14.8, 8.8))
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.0, 0.24],
        hspace=0.11,
        wspace=0.07,
        left=0.05,
        right=0.95,
        top=0.74,
        bottom=0.14,
    )
    ax_u = fig.add_subplot(gs[0, 0])
    ax_p = fig.add_subplot(gs[0, 1])
    info_u = fig.add_subplot(gs[1, 0])
    info_p = fig.add_subplot(gs[1, 1])
    for info_ax in (info_u, info_p):
        info_ax.axis('off')

    title_ax = fig.add_axes([0.06, 0.79, 0.88, 0.14])
    title_ax.axis('off')
    title_ax.text(
        0.5,
        0.78,
        'Uniform Model vs. Permutation Model',
        ha='center',
        va='center',
        fontsize=18,
        fontweight='bold',
    )
    title_ax.text(
        0.5,
        0.28,
        (
            f'n = {n} nodes in both panels. The permutation model reveals all {total_simple_edges} possible simple edges without replacement. '
            f'The uniform model uses the same number of steps, but samples ordered pairs with replacement, so repeated edges appear in blue and self-loops in red.'
        ),
        ha='center',
        va='center',
        fontsize=10.2,
        color='0.25',
        wrap=True,
    )

    legend_handles = [
        Line2D([0], [0], color=SIMPLE_EDGE, lw=1.5, label='Uniform simple edge'),
        Line2D([0], [0], color=MULTI_EDGE, lw=1.8, label='Uniform multi-edge'),
        Line2D([0], [0], color=SELF_LOOP, lw=1.8, label='Uniform self-loop'),
        Line2D([0], [0], color=PERM_EDGE, lw=1.6, label='Permutation-model edge'),
    ]
    fig.legend(
        handles=legend_handles,
        loc='lower center',
        ncol=4,
        bbox_to_anchor=(0.5, 0.05),
        frameon=True,
        edgecolor='#cccccc',
        fontsize=10,
    )

    def update(frame: int) -> None:
        step = min(frame + 1, total_simple_edges)

        for ax in (ax_u, ax_p):
            ax.clear()
        for info_ax in (info_u, info_p):
            info_ax.clear()
            info_ax.axis('off')

        distinct_simple, repeated_pairs, repeated_samples, total_loops = draw_uniform_panel(ax_u, pos, uniform_draws, step)
        permutation_edges_shown = draw_permutation_panel(ax_p, pos, perm_edges, step)

        draw_nodes(ax_u, pos, n)
        draw_nodes(ax_p, pos, n)

        ax_u.set_title('Uniform model\n(samples ordered pairs with replacement)', fontsize=12, pad=6)
        ax_p.set_title('Permutation model\n(reveals simple edges without replacement)', fontsize=12, pad=6)

        info_u.add_patch(
            FancyBboxPatch(
                (0.04, 0.10),
                0.92,
                0.80,
                transform=info_u.transAxes,
                boxstyle='round,pad=0.02',
                linewidth=0.95,
                edgecolor='#cfcfcf',
                facecolor='#fcfcfc',
            )
        )
        info_p.add_patch(
            FancyBboxPatch(
                (0.04, 0.10),
                0.92,
                0.80,
                transform=info_p.transAxes,
                boxstyle='round,pad=0.02',
                linewidth=0.95,
                edgecolor='#cfcfcf',
                facecolor='#fcfcfc',
            )
        )

        info_u.text(
            0.5,
            0.68,
            f'step = {step:,} / {total_simple_edges:,}',
            ha='center',
            va='center',
            fontsize=10.8,
            fontweight='bold',
            transform=info_u.transAxes,
        )
        info_u.text(
            0.5,
            0.44,
            f'distinct simple edges = {distinct_simple:,}   |   repeated edge-pairs = {repeated_pairs:,}',
            ha='center',
            va='center',
            fontsize=10.0,
            transform=info_u.transAxes,
        )
        info_u.text(
            0.5,
            0.22,
            f'extra blue edge copies = {repeated_samples:,}   |   red self-loops = {total_loops:,}',
            ha='center',
            va='center',
            fontsize=10.0,
            transform=info_u.transAxes,
        )

        info_p.text(
            0.5,
            0.68,
            f'step = {step:,} / {total_simple_edges:,}',
            ha='center',
            va='center',
            fontsize=10.8,
            fontweight='bold',
            transform=info_p.transAxes,
        )
        info_p.text(
            0.5,
            0.44,
            f'simple edges revealed = {permutation_edges_shown:,}',
            ha='center',
            va='center',
            fontsize=10.0,
            transform=info_p.transAxes,
        )
        info_p.text(
            0.5,
            0.22,
            f'edges remaining until exhaustion = {total_simple_edges - permutation_edges_shown:,}',
            ha='center',
            va='center',
            fontsize=10.0,
            transform=info_p.transAxes,
        )

    anim = FuncAnimation(fig, update, frames=total_frames, interval=1000 / max(fps, 1), repeat=False)
    saved_to = save_animation(anim, output_path=output, fps=max(fps, 1))
    plt.close(fig)
    return saved_to


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Create a two-panel animation comparing the uniform model and permutation model. '
            'Both panels use n=20 nodes by default and the animation runs until the permutation model exhausts all simple edges.'
        )
    )
    parser.add_argument('--n', type=int, default=DEFAULT_N, help='Number of nodes in each panel.')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED, help='Random seed.')
    parser.add_argument('--fps', type=int, default=DEFAULT_FPS, help='Frames per second.')
    parser.add_argument('--hold-final-frames', type=int, default=DEFAULT_HOLD_FINAL_FRAMES, help='Still frames to hold at the end.')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT, help='Output animation path.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    saved_to = make_animation(
        n=max(2, args.n),
        seed=args.seed,
        fps=max(1, args.fps),
        hold_final_frames=max(0, args.hold_final_frames),
        output=args.output,
    )
    print(f'saved animation to: {saved_to}')


if __name__ == '__main__':
    main()
