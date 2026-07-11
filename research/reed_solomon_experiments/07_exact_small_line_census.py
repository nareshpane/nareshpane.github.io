from pathlib import Path
import csv
from collections import Counter

import matplotlib.pyplot as plt

from rs_core import (
    all_words,
    build_codebook,
    distance_to_code_for_words,
    feasible_masks_for_all_words,
    line_word,
    make_params,
    mod_inverse,
    shared_interpolation_distance,
    word_to_index,
)

OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)

params = make_params(q=5, n=4, k=2)
q, n = params.q, params.n
_, codebook = build_codebook(params)
words = all_words(q, n)
distance_table = distance_to_code_for_words(words, codebook)
feasible = feasible_masks_for_all_words(words, codebook)

# One canonical representative for every one-dimensional direction.
direction_set = set()
for g in words[1:]:
    first = next(int(x) for x in g if int(x) != 0)
    inverse = mod_inverse(first, q)
    direction_set.add(tuple(int((inverse * int(x)) % q) for x in g))
directions = sorted(direction_set)

line_rows = []
aggregate = {t: Counter() for t in range(n + 1)}
line_id = 0

for direction_id, direction_tuple in enumerate(directions):
    direction = list(direction_tuple)
    direction_index = word_to_index(direction, q)
    canonical_bases = set()
    for f in words:
        indices = [word_to_index(line_word(f, direction, z, q), q) for z in range(q)]
        canonical_bases.add(min(indices))

    for base_index in sorted(canonical_bases):
        base = words[base_index]
        point_indices = [word_to_index(line_word(base, direction, z, q), q) for z in range(q)]
        distances = [int(distance_table[index]) for index in point_indices]
        pair_distance = shared_interpolation_distance(feasible[base_index], feasible[direction_index], n)
        row = {
            "line_id": line_id,
            "direction_id": direction_id,
            "base_word": " ".join(map(str, base.tolist())),
            "direction_word": " ".join(map(str, direction)),
            "correlated_pair_distance": pair_distance,
            "point_distances": " ".join(map(str, distances)),
        }
        for t in range(n + 1):
            close_count = sum(distance <= t for distance in distances)
            row[f"close_count_t{t}"] = close_count
            aggregate[t][(pair_distance, close_count)] += 1
        line_rows.append(row)
        line_id += 1

with (OUT / "07_exact_unique_lines_q5_n4_k2.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=line_rows[0].keys())
    writer.writeheader()
    writer.writerows(line_rows)

summary_rows = []
aggregate_rows = []
for t in range(n + 1):
    all_close = sum(count for (pair_distance, close), count in aggregate[t].items() if close == q)
    none_close = sum(count for (pair_distance, close), count in aggregate[t].items() if close == 0)
    intermediate = len(line_rows) - all_close - none_close
    pair_far_entries = [(close, count) for (pair_distance, close), count in aggregate[t].items() if pair_distance > t]
    max_close_pair_far = max((close for close, _ in pair_far_entries), default=0)
    all_close_pair_far = sum(
        count for (pair_distance, close), count in aggregate[t].items() if close == q and pair_distance > t
    )
    summary_rows.append({
        "q": q,
        "n": n,
        "k": params.k,
        "radius_t": t,
        "relative_radius": t / n,
        "number_of_unique_affine_lines": len(line_rows),
        "all_q_points_close": all_close,
        "no_points_close": none_close,
        "intermediate_number_close": intermediate,
        "maximum_close_points_when_pair_distance_exceeds_t": max_close_pair_far,
        "all_close_but_pair_distance_exceeds_t": all_close_pair_far,
    })
    for (pair_distance, close_count), frequency in sorted(aggregate[t].items()):
        aggregate_rows.append({
            "radius_t": t,
            "correlated_pair_distance": pair_distance,
            "close_points_on_line": close_count,
            "number_of_unique_lines": frequency,
        })

with (OUT / "07_exact_line_census_summary.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=summary_rows[0].keys())
    writer.writeheader()
    writer.writerows(summary_rows)

with (OUT / "07_exact_line_census_aggregate.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=aggregate_rows[0].keys())
    writer.writeheader()
    writer.writerows(aggregate_rows)

for t in [params.unique_radius, params.johnson_radius_integer]:
    selected = [row for row in aggregate_rows if row["radius_t"] == t]
    plt.figure(figsize=(8.5, 5.4))
    plt.scatter(
        [row["correlated_pair_distance"] for row in selected],
        [row["close_points_on_line"] for row in selected],
        s=[18 + 7 * (row["number_of_unique_lines"] ** 0.5) for row in selected],
        alpha=0.65,
    )
    for row in selected:
        if row["number_of_unique_lines"] >= 100:
            plt.annotate(
                str(row["number_of_unique_lines"]),
                (row["correlated_pair_distance"], row["close_points_on_line"]),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=8,
            )
    plt.xlabel("Correlated pair distance")
    plt.ylabel(f"Number of z values with distance <= {t}")
    plt.title(f"Exact census of 19,500 affine lines in F_5^4 (radius t={t})")
    plt.xticks(range(n + 1))
    plt.yticks(range(q + 1))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / f"07_exact_line_census_t{t}.png", dpi=180)
    plt.close()

print(f"Directions: {len(directions)}")
print(f"Unique affine lines: {len(line_rows)}")
print(OUT / "07_exact_line_census_summary.csv")
