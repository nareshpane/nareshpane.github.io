"""Generate the static figures for Discretizing Balls from 2D to 7D.

Run from any directory with: python3 research/discretizing-balls-2d-to-7d/generate_ball_visuals.py
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Wedge
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401, activates 3D projection


OUT = Path(__file__).resolve().parent
INK = "#1b1b1b"
BLUE = "#1d4f91"
RUST = "#b65b3c"
GOLD = "#b58a30"
LINE = "#d8d2c7"
PAPER = "#fffdf9"


def style(ax):
    ax.set_facecolor(PAPER)
    for spine in ax.spines.values():
        spine.set_color(LINE)
    ax.tick_params(colors="#555", labelsize=9)


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)


def ladder():
    fig, axes = plt.subplots(1, 7, figsize=(14, 2.8))
    labels = ["line", "$B^2$", "$B^3$", "$B^4$", "$B^5$", "$B^6$", "$B^7$"]
    for i, ax in enumerate(axes):
        ax.set_aspect("equal")
        ax.axis("off")
        if i == 0:
            ax.plot([-0.85, 0.85], [0, 0], color=BLUE, lw=6, solid_capstyle="round")
            ax.scatter([-0.85, 0.85], [0, 0], color=INK, s=18)
        elif i == 1:
            ax.add_patch(Circle((0, 0), 0.85, facecolor="#dce8f5", edgecolor=BLUE, lw=1.8))
        else:
            ax.add_patch(Circle((0, 0), 0.85, facecolor="#dce8f5", edgecolor=BLUE, lw=1.8))
            for y in [-0.52, -0.25, 0, 0.25, 0.52]:
                half = np.sqrt(max(0, 0.85**2 - y**2))
                ax.plot([-half, half], [y, y], color=BLUE, alpha=0.42, lw=0.8)
            if i >= 3:
                for offset in np.linspace(-0.48, 0.48, i - 1):
                    ax.add_patch(Circle((offset * 0.45, 0), 0.36, fill=False,
                                        edgecolor=RUST, alpha=0.35, lw=1.1))
                ax.text(0, -1.12, "projection", ha="center", va="top", fontsize=7, color="#666")
        ax.text(0, 1.08, labels[i], ha="center", fontsize=12, color=INK, fontweight="bold")
        if i > 0:
            ax.text(0, -0.97, f"+ {i - 1} polar angle{'s' if i > 2 else ''}", ha="center", fontsize=7.5, color=RUST)
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.28, 1.3)
    fig.text(0.5, 0.01, "Each step adds one coordinate direction; from $B^3$ onward it also adds one polar angle.",
             ha="center", color="#555", fontsize=10)
    fig.subplots_adjust(wspace=0.12, bottom=0.24)
    save(fig, "dimensional_coordinate_ladder.svg")


def ball_2d():
    fig, ax = plt.subplots(figsize=(6.4, 6.1))
    ax.set_aspect("equal")
    ax.add_patch(Circle((0, 0), 1, facecolor="#eef4fb", edgecolor=BLUE, lw=2))
    angles = np.linspace(0, 2 * np.pi, 13)[:-1]
    radii = [0.25, 0.5, 0.75, 1.0]
    for r in radii:
        ax.add_patch(Circle((0, 0), r, fill=False, edgecolor="#8aaed4", lw=0.85, alpha=0.9))
    for a in angles:
        ax.plot([0, np.cos(a)], [0, np.sin(a)], color="#8aaed4", lw=0.75)
        for r in radii:
            ax.plot(r * np.cos(a), r * np.sin(a), "o", color=BLUE, ms=3.3)
    a, r = np.pi / 3, 0.75
    px, py = r * np.cos(a), r * np.sin(a)
    ax.plot([0, px], [0, py], color=RUST, lw=2.5)
    ax.plot(px, py, "o", color=RUST, ms=8, zorder=4)
    ax.add_patch(Wedge((0, 0), 0.22, 0, 60, facecolor="#f4dcbf", edgecolor=RUST, alpha=0.9))
    ax.annotate("highlighted candidate\n$(r, \\alpha)=(0.75,\\pi/3)$", xy=(px, py), xytext=(0.38, 1.1),
                arrowprops={"arrowstyle": "-", "color": RUST}, fontsize=10, color=INK)
    ax.text(0.12, 0.13, "$\\alpha$", color=RUST, fontsize=12)
    ax.text(px / 2 - 0.1, py / 2 + 0.04, "$r$", color=RUST, fontsize=12)
    ax.plot(0, 0, "o", color=INK, ms=4)
    ax.text(-0.12, -0.16, "origin", fontsize=9, color="#555")
    ax.set_xlim(-1.3, 1.45)
    ax.set_ylim(-1.25, 1.38)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title("Radial shells and angular spokes in $B^2$", loc="left", fontsize=14, color=INK, pad=12)
    style(ax)
    save(fig, "ball_2d_radial_grid.svg")


def ball_3d():
    fig = plt.figure(figsize=(7.2, 5.8))
    ax = fig.add_subplot(111, projection="3d")
    u = np.linspace(0, 2 * np.pi, 35)
    v = np.linspace(0, np.pi, 20)
    uu, vv = np.meshgrid(u, v)
    for radius, alpha in [(1.0, 0.20), (0.68, 0.13), (0.36, 0.11)]:
        x = radius * np.sin(vv) * np.cos(uu)
        y = radius * np.sin(vv) * np.sin(uu)
        z = radius * np.cos(vv)
        ax.plot_wireframe(x, y, z, rstride=3, cstride=4, color=BLUE, linewidth=0.45, alpha=alpha)
    alphas = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    betas = np.linspace(0.25, np.pi - 0.25, 7)
    rs = [0.36, 0.68, 1]
    aa, bb, rr = np.meshgrid(alphas, betas, rs, indexing="ij")
    ax.scatter((rr * np.sin(bb) * np.cos(aa)).ravel(), (rr * np.sin(bb) * np.sin(aa)).ravel(),
               (rr * np.cos(bb)).ravel(), s=8, c=RUST, alpha=0.75, depthshade=False)
    a, b, r = np.pi / 5, np.pi / 3, 1
    ax.plot([0, r*np.sin(b)*np.cos(a)], [0, r*np.sin(b)*np.sin(a)], [0, r*np.cos(b)], color=GOLD, lw=2.5)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_zlabel("$x_3$")
    ax.set_title("Projected view of three spherical shells and a finite angular grid", pad=14, fontsize=13)
    ax.view_init(elev=22, azim=35)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    save(fig, "ball_3d_coordinate_grid.png")


def ball_4d_slices():
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.25))
    vals = [-0.8, -0.4, 0, 0.6]
    for ax, x4 in zip(axes, vals):
        rho = np.sqrt(1 - x4**2)
        ax.add_patch(Circle((0, 0), rho, facecolor="#e8f0f8", edgecolor=BLUE, lw=1.5))
        for y in np.linspace(-rho * .6, rho * .6, 5):
            half = np.sqrt(rho**2 - y**2)
            ax.plot([-half, half], [y, y], color="#8aaed4", lw=0.7)
        ax.set_aspect("equal")
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.axis("off")
        ax.set_title(f"$x_4={x4:g}$\n$\\sqrt{{1-x_4^2}}={rho:.1f}$", fontsize=10, color=INK)
    fig.text(0.5, 0.01, "Each panel is a 2D projection of the 3-ball slice in $(x_1,x_2,x_3)$, not a literal 4D rendering.",
             ha="center", fontsize=9, color="#555")
    fig.subplots_adjust(bottom=0.24, wspace=0.2)
    save(fig, "ball_4d_slices.svg")


def highdim_projections():
    rng = np.random.default_rng(1960)
    d = 7
    direction = rng.normal(size=(420, d))
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    radius = rng.random(420) ** (1 / d)
    pts = direction * radius[:, None]
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2)]
    fig, axes = plt.subplots(2, 2, figsize=(8, 7))
    for ax, (i, j) in zip(axes.flat, pairs):
        ax.scatter(pts[:, i], pts[:, j], c=pts[:, 4], cmap="coolwarm", s=11, alpha=.8, edgecolor="none")
        ax.add_patch(Circle((0, 0), 1, fill=False, edgecolor=BLUE, lw=1))
        ax.axhline(0, color=LINE, lw=.7)
        ax.axvline(0, color=LINE, lw=.7)
        ax.set_aspect("equal")
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlabel(f"$x_{i+1}$")
        ax.set_ylabel(f"$x_{j+1}$")
        style(ax)
    fig.text(0.5, 0.02, "Coordinate-pair projections of a finite sample from $B^7$; colour encodes $x_5$.", ha="center", color="#555", fontsize=10)
    fig.subplots_adjust(bottom=.11, hspace=.35, wspace=.28)
    save(fig, "ball_7d_coordinate_projections.svg")


def angle_nesting():
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.1))
    for ax, d in zip(axes, [5, 6, 7]):
        beta = d - 2
        ax.set_aspect("equal")
        ax.axis("off")
        ax.plot([-1.0, 0], [0, 0], color=BLUE, lw=4, solid_capstyle="round")
        ax.plot([0, 0.78], [0, 0.58], color=BLUE, lw=4, solid_capstyle="round")
        ax.plot([0, 0], [0, 0.88], color=RUST, lw=3, solid_capstyle="round")
        theta = np.linspace(0, .64, 40)
        ax.plot(.34 * np.cos(theta), .34 * np.sin(theta), color=GOLD, lw=1.8)
        ax.text(-1.03, -.18, f"previous direction\n$v_{{{d-1}}}$", ha="left", fontsize=9, color=INK)
        ax.text(.08, .43, f"$\\beta_{beta}$", fontsize=11, color=GOLD)
        ax.text(.09, .92, f"new coordinate $x_{d}$", fontsize=9, color=RUST)
        ax.text(.23, .22, f"$\\sin\\beta_{beta}\\,v_{{{d-1}}}$", fontsize=9, color=BLUE)
        ax.set_xlim(-1.13, 1.12)
        ax.set_ylim(-.35, 1.12)
        ax.set_title(f"$\\mathbb{{R}}^{d}$", fontsize=13, color=INK)
    fig.text(.5, .01, "One more polar angle scales the old direction and introduces one new coordinate.", ha="center", color="#555", fontsize=10)
    fig.subplots_adjust(bottom=.24, wspace=.18)
    save(fig, "high_dimension_angle_nesting.svg")


def growth():
    dims = np.arange(2, 8)
    counts = 1 + 5 * 24 * 12 ** (dims - 2)
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    bars = ax.bar(dims, counts, color=["#9ab9d8", "#7fa6cd", "#608fbe", "#3f76ad", "#285f9d", "#174b85"])
    ax.set_yscale("log")
    ax.set_xticks(dims)
    ax.set_xlabel("ambient dimension $d$")
    ax.set_ylabel("nominal parameter combinations (log scale)")
    ax.set_title("Tensor-product grid growth: $N_r=5$, $N_\\alpha=24$, $N_\\beta=12$", loc="left", fontsize=13, pad=12)
    for bar, value in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, value * 1.33, f"{value:,}", ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.grid(axis="y", color=LINE, linewidth=.8)
    style(ax)
    save(fig, "tensor_grid_growth.svg")


def radial():
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5))
    ax = axes[0]
    r = np.linspace(0, 1, 400)
    for d, color in [(2, "#9ab9d8"), (4, "#5f91bd"), (7, "#1d4f91")]:
        ax.plot(r, d * r ** (d - 1), lw=2.3, color=color, label=f"$d={d}$")
    ax.set_xlabel("radius $r$")
    ax.set_ylabel("density $f_R(r)$")
    ax.set_title("Uniform-volume radial density", loc="left", fontsize=12)
    ax.legend(frameon=False)
    style(ax)
    ax = axes[1]
    equal = np.linspace(.1, 1, 10)
    uniform_volume = np.linspace(.1, 1, 10) ** (1/4)
    ax.scatter(equal, np.zeros_like(equal)+.25, color=RUST, s=42, label="equal spacing in $r$")
    ax.scatter(uniform_volume, np.zeros_like(uniform_volume)-.25, color=BLUE, s=42, label="equal-volume shells, $d=4$")
    ax.set_yticks([-.25, .25], ["equal-volume", "equal-$r$"])
    ax.set_xlabel("shell radius")
    ax.set_xlim(0, 1.04)
    ax.set_ylim(-.55, .55)
    ax.set_title("Different radial objectives", loc="left", fontsize=12)
    ax.grid(axis="x", color=LINE, linewidth=.8)
    style(ax)
    fig.tight_layout()
    save(fig, "radial_distribution.svg")


def geometry():
    dims = np.arange(2, 8)
    volumes = np.pi ** (dims / 2) / np.array([math.gamma(d / 2 + 1) for d in dims])
    outer = 1 - .9 ** dims
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    for ax, data, title, ylabel, color in [
        (axes[0], volumes, "Unit-ball volume", "$V_d$", BLUE),
        (axes[1], outer, "Volume in the outer 10% of radius", "$1-0.9^d$", RUST),
    ]:
        ax.plot(dims, data, "o-", color=color, lw=2, ms=6)
        for d, value in zip(dims, data):
            ax.text(d, value + .035, f"{value:.3f}", ha="center", fontsize=8, color=INK)
        ax.set_xticks(dims)
        ax.set_xlabel("dimension $d$")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontsize=12)
        ax.grid(axis="y", color=LINE, linewidth=.8)
        style(ax)
    axes[1].set_ylim(0, .62)
    fig.tight_layout()
    save(fig, "unit_ball_geometry.svg")


if __name__ == "__main__":
    ladder()
    ball_2d()
    ball_3d()
    ball_4d_slices()
    highdim_projections()
    angle_nesting()
    growth()
    radial()
    geometry()
    print(f"Generated figures in {OUT}")
