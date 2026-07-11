from pathlib import Path
import csv

import matplotlib.pyplot as plt

from rs_core import (
    build_codebook,
    correlated_agreement_distance,
    line_profile,
    make_params,
    polynomial_string,
)

OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)

params = make_params(q=7, n=6, k=3)
messages, codebook = build_codebook(params)

# This explicit line was built coordinate-by-coordinate from three codewords.
# It is a useful warning: every point f+zg is individually within radius 2,
# but no single pair of degree-<3 polynomials explains f and g on four common
# coordinates. Thus zero-loss correlated agreement fails, even though the
# ordinary proximity-gap alternative "all line points are close" is satisfied.
f = [1, 0, 0, 1, 1, 2]
g = [4, 4, 1, 3, 0, 5]
profile = line_profile(f, g, params.q, codebook)
pair_distance, f_code_index, g_code_index = correlated_agreement_distance(f, g, codebook)

rows = []
for item in profile:
    nearest_descriptions = []
    for index in item["nearest_indices"]:
        nearest_descriptions.append(polynomial_string(messages[index].tolist()))
    rows.append({
        "z": item["z"],
        "line_word_f_plus_zg": " ".join(map(str, item["word"].tolist())),
        "distance_to_RS": item["distance"],
        "number_of_nearest_codewords": len(item["nearest_indices"]),
        "nearest_polynomials": " | ".join(nearest_descriptions),
    })

with (OUT / "06_affine_line_profile.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

report = [
    "Experiment 6: an explicit affine line in F_7^6",
    "=" * 54,
    f"RS parameters: q={params.q}, n={params.n}, k={params.k}",
    f"Johnson radius: {params.johnson_radius_real:.6f}; integer radius {params.johnson_radius_integer}",
    f"f = {f}",
    f"g = {g}",
    "",
    "For every z in F_7, the word f+zg has distance exactly 2 from RS(7,6,3).",
    f"Correlated pair distance Delta([f,g], C^2) = {pair_distance}/6.",
    "The best pair of code polynomials is:",
    f"  P(x) = {polynomial_string(messages[f_code_index].tolist())}",
    f"  Q(x) = {polynomial_string(messages[g_code_index].tolist())}",
    "",
    "Interpretation:",
    "  * The affine line satisfies the first proximity-gap alternative: all seven points are close.",
    "  * But individual nearest polynomials change with z.",
    "  * Therefore, 'all points are t-close' does not automatically imply zero-loss correlated agreement at the same t.",
    "  * This is a finite teaching example, not a disproof of the formal asymptotic proximity-gap statements.",
]
(OUT / "06_affine_line_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")

plt.figure(figsize=(8.5, 5.2))
plt.plot([r["z"] for r in rows], [r["distance_to_RS"] for r in rows], marker="o")
plt.axhline(params.johnson_radius_real, linestyle="--", label=f"Johnson radius {params.johnson_radius_real:.3f}")
plt.xlabel("Scalar z in F_7")
plt.ylabel("Distance from f+zg to RS(7,6,3)")
plt.title("Every point on one affine line is exactly two errors from the code")
plt.xticks(range(params.q))
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "06_affine_line_distances.png", dpi=180)
plt.close()

print(OUT / "06_affine_line_profile.csv")
print(OUT / "06_affine_line_distances.png")
