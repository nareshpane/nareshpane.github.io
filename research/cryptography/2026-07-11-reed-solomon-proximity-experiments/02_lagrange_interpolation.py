from pathlib import Path
import csv

from rs_core import (
    evaluate_polynomial,
    lagrange_interpolate,
    make_params,
    mod_inverse,
    poly_mul,
    poly_scale,
    polynomial_string,
)

OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)

params = make_params(q=7, n=6, k=3)
original = [3, 2, 4]
full_values = evaluate_polynomial(original, params.evaluation_points, params.q)
chosen_indices = [0, 2, 5]
xs = [params.evaluation_points[i] for i in chosen_indices]
ys = [int(full_values[i]) for i in chosen_indices]
recovered = lagrange_interpolate(xs, ys, params.q)
recovered_values = evaluate_polynomial(recovered, params.evaluation_points, params.q)

basis_rows = []
for j, (xj, yj) in enumerate(zip(xs, ys)):
    numerator = [1]
    denominator = 1
    for m, xm in enumerate(xs):
        if m == j:
            continue
        numerator = poly_mul(numerator, [(-xm) % params.q, 1], params.q)
        denominator = (denominator * (xj - xm)) % params.q
    inverse = mod_inverse(denominator, params.q)
    scaled = poly_scale(numerator, yj * inverse, params.q)
    basis_rows.append((j, xj, yj, numerator, denominator, inverse, scaled))

csv_path = OUT / "02_lagrange_basis_terms.csv"
with csv_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow(["j", "x_j", "y_j", "numerator_coefficients", "denominator_mod_7", "inverse", "contribution_coefficients"])
    writer.writerows(basis_rows)

text_path = OUT / "02_lagrange_interpolation.txt"
lines = [
    "Experiment 2: reconstruct the quadratic from three points",
    "=" * 61,
    f"Original polynomial: {polynomial_string(original)} over F_7",
    f"Chosen points: {list(zip(xs, ys))}",
    "",
    "Lagrange builds one basis polynomial for each chosen point.",
]
for j, xj, yj, numerator, denominator, inverse, scaled in basis_rows:
    lines += [
        f"j={j}: x_j={xj}, y_j={yj}",
        f"  numerator coefficients: {numerator}",
        f"  denominator mod 7: {denominator}; inverse: {inverse}",
        f"  y_j times basis contribution: {scaled}",
    ]
lines += [
    "",
    f"Recovered coefficients [a0,a1,a2]: {recovered}",
    f"Recovered polynomial: {polynomial_string(recovered)}",
    f"Values at all six points: {recovered_values.tolist()}",
    f"Exact recovery: {recovered == original}",
]
text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(text_path)
print(csv_path)
