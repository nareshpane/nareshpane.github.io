from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np

from rs_core import (
    build_codebook,
    feasible_mask_vector_for_word,
    line_word,
    make_params,
    nearest_codewords,
    shared_interpolation_distance,
)

OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)
RNG = np.random.default_rng(20260711)

# The final two cases probe an integer radius above the Johnson radius but below
# the capacity radius. They are sampled, not exhaustive.
cases = [
    {"q": 7, "n": 6, "k": 3, "t": 2, "samples": 3000},
    {"q": 11, "n": 10, "k": 3, "t": 6, "samples": 500},
    {"q": 13, "n": 12, "k": 3, "t": 8, "samples": 250},
]

all_rows = []
summary_rows = []

for case in cases:
    q, n, k, t, samples = case["q"], case["n"], case["k"], case["t"], case["samples"]
    params = make_params(q, n, k)
    _, codebook = build_codebook(params)
    case_rows = []

    for sample_id in range(samples):
        f = RNG.integers(0, q, size=n, dtype=np.int16)
        while True:
            g = RNG.integers(0, q, size=n, dtype=np.int16)
            if np.any(g):
                break

        distances = []
        for z in range(q):
            word = line_word(f, g, z, q)
            distance, _ = nearest_codewords(word, codebook)
            distances.append(distance)
        close_count = sum(distance <= t for distance in distances)

        feasible_f = feasible_mask_vector_for_word(f, codebook)
        feasible_g = feasible_mask_vector_for_word(g, codebook)
        pair_distance = shared_interpolation_distance(feasible_f, feasible_g, n)

        row = {
            "q": q,
            "n": n,
            "k": k,
            "sample_id": sample_id,
            "radius_t": t,
            "johnson_radius_real": params.johnson_radius_real,
            "johnson_radius_integer": params.johnson_radius_integer,
            "capacity_radius_integer": params.capacity_radius_integer,
            "f": " ".join(map(str, f.tolist())),
            "g": " ".join(map(str, g.tolist())),
            "line_point_distances": " ".join(map(str, distances)),
            "close_count": close_count,
            "close_fraction": close_count / q,
            "correlated_pair_distance": pair_distance,
            "pair_farther_than_t": pair_distance > t,
        }
        case_rows.append(row)
        all_rows.append(row)

    best_close = max(row["close_count"] for row in case_rows)
    best_pair_far = max((row["close_count"] for row in case_rows if row["pair_farther_than_t"]), default=0)
    summary_rows.append({
        "q": q,
        "n": n,
        "k": k,
        "radius_t": t,
        "samples": samples,
        "johnson_radius_real": params.johnson_radius_real,
        "capacity_radius_integer": params.capacity_radius_integer,
        "maximum_close_count_seen": best_close,
        "maximum_close_count_with_pair_farther_than_t": best_pair_far,
        "all_points_close_lines_seen": sum(row["close_count"] == q for row in case_rows),
    })

    frequencies = {}
    for row in case_rows:
        key = (row["correlated_pair_distance"], row["close_count"])
        frequencies[key] = frequencies.get(key, 0) + 1
    points = sorted((pair_distance, close_count, frequency) for (pair_distance, close_count), frequency in frequencies.items())

    plt.figure(figsize=(8.5, 5.4))
    plt.scatter(
        [point[0] for point in points],
        [point[1] for point in points],
        s=[18 + 8 * (point[2] ** 0.5) for point in points],
        alpha=0.65,
    )
    plt.axvline(t, linestyle="--", label="pair distance = tested radius")
    plt.xlabel("Correlated pair distance")
    plt.ylabel(f"Number of z values with distance <= {t}")
    plt.title(f"Random affine-line search: q={q}, n={n}, k={k}, samples={samples}")
    plt.xticks(range(n + 1))
    plt.yticks(range(q + 1))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / f"08_random_line_search_q{q}_n{n}_k{k}.png", dpi=180)
    plt.close()

with (OUT / "08_random_line_search_all_samples.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=all_rows[0].keys())
    writer.writeheader()
    writer.writerows(all_rows)

with (OUT / "08_random_line_search_summary.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=summary_rows[0].keys())
    writer.writeheader()
    writer.writerows(summary_rows)

# A compact table containing only the most interesting sampled lines.
top_rows = sorted(
    all_rows,
    key=lambda row: (row["pair_farther_than_t"], row["close_count"], row["correlated_pair_distance"]),
    reverse=True,
)[:75]
with (OUT / "08_random_line_search_top_candidates.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=top_rows[0].keys())
    writer.writeheader()
    writer.writerows(top_rows)

print(OUT / "08_random_line_search_summary.csv")
