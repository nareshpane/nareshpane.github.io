from pathlib import Path
from itertools import combinations
from math import comb
import csv

import matplotlib.pyplot as plt

from rs_core import evaluate_polynomial, lagrange_interpolate, make_params, polynomial_string

OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)

# The received word is r_i = alpha_i^k. It comes from a degree-k polynomial,
# while the code only permits degrees < k. Every k coordinates interpolate a
# candidate code polynomial. Because x^k - p(x) has degree k, it cannot agree
# at more than k distinct points unless it is the zero polynomial.
cases = [(7, 5, 2), (7, 6, 3), (11, 7, 3), (11, 8, 4)]
summary_rows = []
detail_rows = []

for q, n, k in cases:
    params = make_params(q, n, k)
    received = [pow(x, k, q) for x in params.evaluation_points]
    polynomials = {}

    for subset in combinations(range(n), k):
        xs = [params.evaluation_points[i] for i in subset]
        ys = [received[i] for i in subset]
        coefficients = tuple(lagrange_interpolate(xs, ys, q) + [0] * (k - len(lagrange_interpolate(xs, ys, q))))
        coefficients = coefficients[:k]
        codeword = evaluate_polynomial(coefficients, params.evaluation_points, q)
        agreement_positions = [i for i, (a, b) in enumerate(zip(received, codeword)) if int(a) == int(b)]
        polynomials[coefficients] = agreement_positions
        detail_rows.append({
            "q": q,
            "n": n,
            "k": k,
            "subset_indices": " ".join(map(str, subset)),
            "subset_x_values": " ".join(map(str, xs)),
            "interpolated_coefficients": " ".join(map(str, coefficients)),
            "interpolated_polynomial": polynomial_string(coefficients),
            "agreement_positions": " ".join(map(str, agreement_positions)),
            "number_of_agreements": len(agreement_positions),
            "distance": n - len(agreement_positions),
        })

    predicted = comb(n, k)
    observed = len(polynomials)
    agreement_counts = sorted({len(v) for v in polynomials.values()})
    summary_rows.append({
        "q": q,
        "n": n,
        "k": k,
        "received_polynomial": f"x^{k}",
        "received_word": " ".join(map(str, received)),
        "predicted_binomial_count": predicted,
        "observed_distinct_interpolants": observed,
        "agreement_counts_seen": " ".join(map(str, agreement_counts)),
        "radius_n_minus_k": n - k,
    })

with (OUT / "05_interpolation_crowding_summary.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=summary_rows[0].keys())
    writer.writeheader()
    writer.writerows(summary_rows)

with (OUT / "05_interpolation_crowding_details.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=detail_rows[0].keys())
    writer.writeheader()
    writer.writerows(detail_rows)

labels = [f"({r['q']},{r['n']},{r['k']})" for r in summary_rows]
x = range(len(labels))
width = 0.36
plt.figure(figsize=(9, 5.5))
plt.bar([i - width / 2 for i in x], [r["predicted_binomial_count"] for r in summary_rows], width=width, label="C(n,k)")
plt.bar([i + width / 2 for i in x], [r["observed_distinct_interpolants"] for r in summary_rows], width=width, label="Observed")
plt.xticks(list(x), labels)
plt.xlabel("(q,n,k)")
plt.ylabel("Number of distinct nearby degree < k polynomials")
plt.title("Interpolation crowding at radius n-k")
plt.grid(True, axis="y", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "05_interpolation_crowding.png", dpi=180)
plt.close()

print(OUT / "05_interpolation_crowding_summary.csv")
print(OUT / "05_interpolation_crowding.png")
