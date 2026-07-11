from pathlib import Path
import csv
import math

import matplotlib.pyplot as plt

from rs_core import make_params

OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)

cases = [
    (5, 4, 2),
    (7, 5, 2),
    (7, 6, 3),
    (11, 10, 3),
    (13, 12, 3),
    (17, 16, 4),
]
rows = []
for q, n, k in cases:
    params = make_params(q, n, k)
    codewords = q ** k
    received_words = q ** n
    word_codeword_pairs = q ** (n + k)
    coordinate_comparisons = n * word_codeword_pairs
    parameterized_affine_lines = q ** (2 * n) - q ** n  # g != 0
    line_points_to_check = q * parameterized_affine_lines
    rows.append({
        "q": q,
        "n": n,
        "k": k,
        "rate": params.rate,
        "codewords_q_to_k": codewords,
        "received_words_q_to_n": received_words,
        "word_codeword_distance_pairs_q_to_n_plus_k": word_codeword_pairs,
        "coordinate_comparisons_for_full_list_census": coordinate_comparisons,
        "distance_matrix_bytes_if_uint8": word_codeword_pairs,
        "distance_matrix_gib_if_uint8": word_codeword_pairs / (1024 ** 3),
        "parameterized_affine_lines_with_nonzero_g": parameterized_affine_lines,
        "line_points_across_all_parameterizations": line_points_to_check,
        "rough_seconds_at_100_million_coordinate_comparisons_per_second": coordinate_comparisons / 1e8,
    })

with (OUT / "09_scaling_estimates.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

labels = [f"({r['q']},{r['n']},{r['k']})" for r in rows]
plt.figure(figsize=(10, 5.8))
plt.plot(labels, [math.log10(r["codewords_q_to_k"]) for r in rows], marker="o", label="log10(codewords)")
plt.plot(labels, [math.log10(r["received_words_q_to_n"]) for r in rows], marker="o", label="log10(received words)")
plt.plot(labels, [math.log10(r["word_codeword_distance_pairs_q_to_n_plus_k"]) for r in rows], marker="o", label="log10(word-codeword pairs)")
plt.plot(labels, [math.log10(r["parameterized_affine_lines_with_nonzero_g"]) for r in rows], marker="o", label="log10(parameterized lines)")
plt.xlabel("(q,n,k)")
plt.ylabel("Base-10 logarithm of count")
plt.title("Why exhaustive proximity-gap searches become infeasible")
plt.xticks(rotation=20)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "09_scaling_growth.png", dpi=180)
plt.close()

print(OUT / "09_scaling_estimates.csv")
print(OUT / "09_scaling_growth.png")
