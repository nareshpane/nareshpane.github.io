"""
Version with centered subtitle and updated title.

Changes:
- centers the subtitle beneath the title
- renames the main title to show the tested value of c
- keeps "Not yet achieved" in blinking red
- keeps "Achieved" in green
- leaves all other animation behavior unchanged
"""

from __future__ import annotations

import argparse
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea


@dataclass(frozen=True)
class EmergenceEvent:
    edge_index: int
    label: str


@dataclass(frozen=True)
class Theorem1Checkpoint:
    n: int
    c: float
    m: int
    theory: float
    empirical: float
    abs_diff: float
    trials: int


@dataclass(frozen=True)
class GraphMilestones:
    first_cycle_at: Optional[int]
    giant_at: Optional[int]
    no_isolates_at: Optional[int]
    connected_at: Optional[int]
    events: List[EmergenceEvent]


def edge_count_n_c(n: int, c: float) -> int:
    """Return N_c = floor(0.5 * n * log n + c n), clipped to [0, n choose 2]."""
    m = math.floor(0.5 * n * math.log(n) + c * n)
    max_edges = n * (n - 1) // 2
    return max(0, min(m, max_edges))



def theorem1_limit(c: float) -> float:
    """Theorem 1 limit for connectivity in G(n, N_c)."""
    return math.exp(-math.exp(-2.0 * c))



def estimate_connectivity_probability(
    n: int,
    m: int,
    trials: int,
    rng: np.random.Generator,
) -> float:
    """Monte Carlo estimate of P[G(n,m) is connected]."""
    connected = 0
    for _ in range(trials):
        seed = int(rng.integers(0, 2**32 - 1, dtype=np.uint32))
        g = nx.gnm_random_graph(n, m, seed=seed)
        if nx.is_connected(g):
            connected += 1
    return connected / trials



def compute_theorem1_checkpoint(
    n: int,
    c: float,
    trials: int,
    seed: int,
) -> Theorem1Checkpoint:
    rng = np.random.default_rng(seed)
    m = edge_count_n_c(n, c)
    empirical = estimate_connectivity_probability(n, m, trials, rng)
    theory = theorem1_limit(c)
    return Theorem1Checkpoint(
        n=n,
        c=float(c),
        m=m,
        theory=theory,
        empirical=empirical,
        abs_diff=abs(empirical - theory),
        trials=trials,
    )



def layered_circular_layout(
    n: int,
    rng: np.random.Generator,
    outer_radius: float = 0.93,
    inner_radius: float = 0.58,
) -> Dict[int, Tuple[float, float]]:
    """Create a mostly circular layout with a clear outer shell and interior nodes."""
    if n <= 0:
        return {}

    if n <= 8:
        outer_count = max(5, n - 2)
    else:
        outer_count = max(10, int(round(0.64 * n)))
        outer_count = min(outer_count, n - 3)

    points: List[Tuple[float, float]] = []
    angle_offset = rng.uniform(0.0, 2.0 * math.pi)
    outer_angles = np.linspace(0.0, 2.0 * math.pi, outer_count, endpoint=False) + angle_offset
    outer_angles += rng.normal(0.0, 0.045, size=outer_count)

    for theta in outer_angles:
        r = outer_radius * (1.0 + rng.normal(0.0, 0.035))
        r = float(np.clip(r, 0.84, 0.99))
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        points.append((x, y))

    min_dist = max(0.04, 0.16 * math.sqrt(20.0 / max(n, 1)))
    attempts = 0
    while len(points) < n and attempts < 50_000:
        attempts += 1
        theta = rng.uniform(0.0, 2.0 * math.pi)
        r = inner_radius * math.sqrt(rng.uniform(0.0, 1.0))
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        if all((x - px) ** 2 + (y - py) ** 2 >= min_dist**2 for px, py in points):
            points.append((x, y))

    while len(points) < n:
        theta = rng.uniform(0.0, 2.0 * math.pi)
        r = inner_radius * math.sqrt(rng.uniform(0.0, 1.0))
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        points.append((x, y))

    return {i + 1: points[i] for i in range(n)}



def ordered_random_edges(nodes: List[int], rng: np.random.Generator) -> List[Tuple[int, int]]:
    edges = [(u, v) for i, u in enumerate(nodes) for v in nodes[i + 1 :]]
    rng.shuffle(edges)
    return edges



def has_cycle(g: nx.Graph) -> bool:
    return g.number_of_edges() >= g.number_of_nodes() and len(nx.cycle_basis(g)) > 0



def analyze_edge_order(nodes: List[int], edges_in_order: List[Tuple[int, int]]) -> GraphMilestones:
    g = nx.Graph()
    g.add_nodes_from(nodes)
    events: List[EmergenceEvent] = []

    first_cycle_at: Optional[int] = None
    giant_at: Optional[int] = None
    no_isolates_at: Optional[int] = None
    connected_at: Optional[int] = None
    giant_threshold = math.ceil(len(nodes) / 2)

    for idx, edge in enumerate(edges_in_order, start=1):
        g.add_edge(*edge)

        if first_cycle_at is None and has_cycle(g):
            first_cycle_at = idx
            events.append(EmergenceEvent(idx, "First cycle appears"))

        largest_component = max((len(comp) for comp in nx.connected_components(g)), default=1)
        if giant_at is None and largest_component >= giant_threshold:
            giant_at = idx
            events.append(EmergenceEvent(idx, f"Giant component emerges (size {largest_component})"))

        isolates = sum(1 for _, deg in g.degree() if deg == 0)
        if no_isolates_at is None and isolates == 0:
            no_isolates_at = idx
            events.append(EmergenceEvent(idx, "No isolated vertices remain"))

        if connected_at is None and nx.is_connected(g):
            connected_at = idx
            events.append(EmergenceEvent(idx, "Graph becomes connected"))
            break

    return GraphMilestones(
        first_cycle_at=first_cycle_at,
        giant_at=giant_at,
        no_isolates_at=no_isolates_at,
        connected_at=connected_at,
        events=events,
    )



def latest_event_text(edge_count: int, events: List[EmergenceEvent], linger_edges: int) -> str:
    candidates = [e for e in events if e.edge_index <= edge_count < e.edge_index + linger_edges]
    if not candidates:
        return ""
    return candidates[-1].label



def build_graph_state(nodes: List[int], edges_in_order: List[Tuple[int, int]], edge_count: int) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges_in_order[:edge_count])
    return g



def build_frame_schedule(
    target_edges: int,
    frames_per_edge: int,
    theorem_checkpoint_edge: int,
    theorem_pause_frames: int,
    hold_final_frames: int,
) -> List[int]:
    schedule = [0]
    for edge_count in range(1, target_edges + 1):
        schedule.extend([edge_count] * frames_per_edge)
        if edge_count == theorem_checkpoint_edge:
            schedule.extend([edge_count] * theorem_pause_frames)
    schedule.extend([target_edges] * hold_final_frames)
    return schedule



def save_animation(anim: FuncAnimation, output_path: Path, fps: int) -> Path:
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
            anim.save(output_path, writer=writer, dpi=180)
            return output_path

        gif_path = output_path.with_suffix(".gif")
        anim.save(gif_path, writer=PillowWriter(fps=fps), dpi=120)
        return gif_path

    anim.save(output_path, writer=PillowWriter(fps=fps), dpi=120)
    return output_path



def make_animation(
    n: int,
    max_edges: Optional[int],
    c: float,
    theorem_trials: int,
    seed: int,
    fps: int,
    hold_final_frames: int,
    frames_per_edge: int,
    theorem_pause_frames: int,
    output_path: Path,
) -> Tuple[Path, Theorem1Checkpoint, GraphMilestones, int]:
    rng = np.random.default_rng(seed)
    nodes = list(range(1, n + 1))
    positions = layered_circular_layout(n=n, rng=rng)
    full_edge_order = ordered_random_edges(nodes=nodes, rng=rng)

    theorem_checkpoint = compute_theorem1_checkpoint(
        n=n,
        c=c,
        trials=theorem_trials,
        seed=seed + 99173,
    )
    milestones = analyze_edge_order(nodes=nodes, edges_in_order=full_edge_order)

    if max_edges is None:
        connected_target = milestones.connected_at if milestones.connected_at is not None else theorem_checkpoint.m
        target_edges = max(theorem_checkpoint.m, connected_target)
    else:
        target_edges = max(1, min(max_edges, len(full_edge_order)))
        target_edges = max(target_edges, theorem_checkpoint.m)

    edge_order = full_edge_order[:target_edges]
    total_possible_edges = n * (n - 1) // 2
    frame_schedule = build_frame_schedule(
        target_edges=target_edges,
        frames_per_edge=frames_per_edge,
        theorem_checkpoint_edge=theorem_checkpoint.m,
        theorem_pause_frames=theorem_pause_frames,
        hold_final_frames=hold_final_frames,
    )

    node_size = max(22, 110 - int(0.6 * n))
    edge_width_old = 0.55 if n >= 100 else 1.0
    edge_width_new = 1.6 if n >= 100 else 2.2

    fig = plt.figure(figsize=(10.0, 9.2))
    graph_ax = fig.add_axes([0.05, 0.23, 0.90, 0.70])
    caption_ax = fig.add_axes([0.05, 0.04, 0.90, 0.15])
    fig.patch.set_facecolor("white")
    caption_ax.axis("off")

    info_line_1 = caption_ax.text(
        0.5,
        0.82,
        "",
        ha="center",
        va="center",
        fontsize=10,
        bbox=dict(facecolor="white", edgecolor="0.85", boxstyle="round,pad=0.25"),
    )
    info_line_2 = caption_ax.text(
        0.5,
        0.52,
        "",
        ha="center",
        va="center",
        fontsize=10,
        bbox=dict(facecolor="white", edgecolor="0.85", boxstyle="round,pad=0.25"),
    )
    event_line = caption_ax.text(
        0.5,
        0.20,
        "",
        ha="center",
        va="center",
        fontsize=10.5,
        bbox=dict(facecolor="white", edgecolor="0.70", boxstyle="round,pad=0.30"),
    )

    theorem_box = graph_ax.text(
        0.5,
        0.05,
        "",
        transform=graph_ax.transAxes,
        ha="center",
        va="center",
        fontsize=10.5,
        bbox=dict(facecolor="#fffaf0", edgecolor="black", boxstyle="round,pad=0.45"),
    )

    def update(frame_index: int) -> None:
        graph_ax.clear()
        edge_count = frame_schedule[frame_index]
        g = build_graph_state(nodes=nodes, edges_in_order=edge_order, edge_count=edge_count)

        component_sizes = sorted((len(cmp) for cmp in nx.connected_components(g)), reverse=True)
        largest_component = component_sizes[0] if component_sizes else 1
        isolates = sum(1 for _, deg in g.degree() if deg == 0)
        avg_degree = 0.0 if n == 0 else 2.0 * g.number_of_edges() / n
        density = 0.0 if n <= 1 else 2.0 * g.number_of_edges() / (n * (n - 1))
        is_connected = nx.is_connected(g) if g.number_of_edges() > 0 else False

        previous_edges = edge_order[: max(0, edge_count - 1)]
        newest_edge = edge_order[edge_count - 1 : edge_count] if edge_count > 0 else []

        nx.draw_networkx_edges(
            g,
            pos=positions,
            edgelist=previous_edges,
            ax=graph_ax,
            width=edge_width_old,
            alpha=0.38,
            edge_color="0.45",
        )
        nx.draw_networkx_edges(
            g,
            pos=positions,
            edgelist=newest_edge,
            ax=graph_ax,
            width=edge_width_new,
            alpha=0.92,
            edge_color="black",
        )
        nx.draw_networkx_nodes(
            g,
            pos=positions,
            ax=graph_ax,
            node_size=node_size,
            linewidths=0.65,
            edgecolors="black",
            node_color="white",
        )

        graph_ax.set_xlim(-1.16, 1.16)
        graph_ax.set_ylim(-1.16, 1.16)
        graph_ax.set_aspect("equal")
        graph_ax.axis("off")
        graph_ax.set_title(
            f"Probability of Graph Remaining Connected with c = {theorem_checkpoint.c:.2f}",
            fontsize=16,
            pad=30,
        )

        theorem_status = "Achieved" if edge_count >= theorem_checkpoint.m else "Not yet achieved"
        theorem_status_color = "#1a7f37" if theorem_status == "Achieved" else "#c62828"

        if theorem_status == "Not yet achieved":
            blink_on = ((frame_index // max(1, fps // 2 + 1)) % 2 == 0)
            status_alpha = 1.0 if blink_on else 0.35
        else:
            status_alpha = 1.0

        subtitle_left = TextArea(
            "Expected number of edges from Erdos and Renyi (1950) Theorem 1; Result:",
            textprops=dict(color="0.20", fontsize=10.5),
        )
        subtitle_right = TextArea(
            f" {theorem_status}.",
            textprops=dict(
                color=theorem_status_color,
                fontsize=10.5,
                fontweight="bold",
                alpha=status_alpha,
            ),
        )
        subtitle_pack = HPacker(
            children=[subtitle_left, subtitle_right],
            align="center",
            pad=0,
            sep=0,
        )
        subtitle_box = AnchoredOffsetbox(
            loc="upper center",
            child=subtitle_pack,
            pad=0.0,
            frameon=False,
            bbox_to_anchor=(0.5, 1.012),
            bbox_transform=graph_ax.transAxes,
            borderpad=0.0,
        )
        graph_ax.add_artist(subtitle_box)

        info_line_1.set_text(
            f"n = {n}   |   edges shown m = {g.number_of_edges()}/{target_edges}   |   "
            f"possible total = {total_possible_edges}   |   density = {density:.4f}"
        )
        info_line_2.set_text(
            f"avg degree = {avg_degree:.2f}   |   components = {nx.number_connected_components(g)}   |   "
            f"largest component = {largest_component}   |   isolates = {isolates}   |   connected = {'yes' if is_connected else 'no'}"
        )

        event_text = latest_event_text(edge_count, milestones.events, linger_edges=2)
        event_line.set_text(f"Emergence: {event_text}" if event_text else " ")

        if edge_count == theorem_checkpoint.m:
            theorem_box.set_text(
                "Theorem 1 checkpoint reached: "
                f"m = N_c = {theorem_checkpoint.m} for c = {theorem_checkpoint.c:.2f}.\n"
                f"Empirical P(connected) ≈ {theorem_checkpoint.empirical:.4f} from {theorem_checkpoint.trials} trials; "
                f"theory = {theorem_checkpoint.theory:.4f}; |diff| = {theorem_checkpoint.abs_diff:.4f}."
            )
            theorem_box.set_visible(True)
        else:
            theorem_box.set_visible(False)

    anim = FuncAnimation(
        fig,
        update,
        frames=len(frame_schedule),
        interval=1000 / fps,
        repeat=False,
    )
    saved_to = save_animation(anim=anim, output_path=output_path, fps=fps)
    plt.close(fig)
    return saved_to, theorem_checkpoint, milestones, target_edges



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create an MP4 animation showing the evolution of a random graph on 100 nodes, "
            "and pause at the Theorem 1 threshold checkpoint m = floor(0.5*n*log n + c*n)."
        )
    )
    parser.add_argument("--n", type=int, default=100, help="Number of nodes.")
    parser.add_argument(
        "--max-edges",
        type=int,
        default=None,
        help=(
            "How many random edges to reveal. If omitted, the video runs at least until the "
            "Theorem 1 checkpoint and continues through the first connected moment of this sample graph."
        ),
    )
    parser.add_argument(
        "--theorem-c",
        type=float,
        default=0.0,
        help="Value of c used in N_c = floor(0.5*n*log n + c*n) for the Theorem 1 checkpoint.",
    )
    parser.add_argument(
        "--theorem-trials",
        type=int,
        default=300,
        help="Monte Carlo trials used to estimate the empirical Theorem 1 connectivity probability.",
    )
    parser.add_argument("--seed", type=int, default=12345, help="Random seed.")
    parser.add_argument(
        "--fps",
        type=int,
        default=4,
        help="Frames per second. Kept at the earlier pace by default.",
    )
    parser.add_argument(
        "--frames-per-edge",
        type=int,
        default=3,
        help="How many frames each edge remains on screen before the next edge appears.",
    )
    parser.add_argument(
        "--theorem-pause-frames",
        type=int,
        default=24,
        help="Extra pause frames when the Theorem 1 checkpoint edge count is reached.",
    )
    parser.add_argument(
        "--hold-final-frames",
        type=int,
        default=18,
        help="Extra still frames at the end so the final state lingers.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./random_graph_100_node_evolution.mp4"),
        help="Output animation path. Use .mp4 or .gif.",
    )
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    max_possible_edges = args.n * (args.n - 1) // 2

    if args.max_edges is not None:
        requested_max_edges = max(1, min(args.max_edges, max_possible_edges))
    else:
        requested_max_edges = None

    saved_to, theorem_checkpoint, milestones, target_edges = make_animation(
        n=args.n,
        max_edges=requested_max_edges,
        c=args.theorem_c,
        theorem_trials=args.theorem_trials,
        seed=args.seed,
        fps=max(1, args.fps),
        hold_final_frames=max(0, args.hold_final_frames),
        frames_per_edge=max(1, args.frames_per_edge),
        theorem_pause_frames=max(0, args.theorem_pause_frames),
        output_path=args.output,
    )

    print(f"Using n = {args.n} nodes.")
    print(f"Maximum possible edges: {max_possible_edges} = {args.n}*({args.n}-1)/2")
    print(
        f"Theorem 1 checkpoint uses c = {theorem_checkpoint.c:.2f}, so "
        f"N_c = floor(0.5*n*log n + c*n) = {theorem_checkpoint.m}."
    )
    print(
        f"Empirical connectivity estimate at m = {theorem_checkpoint.m}: "
        f"{theorem_checkpoint.empirical:.6f} from {theorem_checkpoint.trials} trials"
    )
    print(f"Theorem 1 value: {theorem_checkpoint.theory:.6f}")
    print(f"Absolute difference: {theorem_checkpoint.abs_diff:.6f}")
    if milestones.connected_at is not None:
        print(f"This sampled graph first becomes connected at edge {milestones.connected_at}.")
    print(f"Animating through {target_edges} revealed edges.")
    print(f"Saved animation to: {saved_to}")


if __name__ == "__main__":
    main()
