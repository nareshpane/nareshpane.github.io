from pathlib import Path
import csv

from rs_core import evaluate_polynomial, make_params, polynomial_string

OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)

params = make_params(q=7, n=6, k=3)
coefficients = [3, 2, 4]  # p(x) = 3 + 2x + 4x^2
values = evaluate_polynomial(coefficients, params.evaluation_points, params.q)

csv_path = OUT / "01_polynomial_evaluation_q7.csv"
with csv_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow(["x", "integer_expression", "value_mod_7"])
    for x, value in zip(params.evaluation_points, values):
        expression = f"3 + 2*{x} + 4*{x}^2"
        writer.writerow([x, expression, int(value)])

text_path = OUT / "01_polynomial_walkthrough.txt"
lines = [
    "Experiment 1: evaluate a concrete polynomial over F_7",
    "=" * 58,
    f"Polynomial: p(x) = {polynomial_string(coefficients)} over F_7",
    f"Evaluation points: {list(params.evaluation_points)}",
    "",
]
for x, value in zip(params.evaluation_points, values):
    raw = 3 + 2 * x + 4 * x * x
    lines.append(f"p({x}) = 3 + 2({x}) + 4({x})^2 = {raw}, and {raw} mod 7 = {int(value)}")
lines += [
    "",
    f"Reed--Solomon word: {values.tolist()}",
    "Each coordinate is only the polynomial's value at one public evaluation point.",
]
text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(text_path)
print(csv_path)
