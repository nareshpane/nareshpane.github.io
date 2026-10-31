from pathlib import Path
import csv
from itertools import product

OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)

# GF(8) = F_2[a] / (a^3 + a + 1). Elements are stored as three-bit
# polynomials b0 + b1*a + b2*a^2. Addition is XOR.
MODULUS = 0b1011


def add(a: int, b: int) -> int:
    return a ^ b


def multiply(a: int, b: int) -> int:
    result = 0
    x, y = a, b
    while y:
        if y & 1:
            result ^= x
        y >>= 1
        x <<= 1
        if x & 0b1000:
            x ^= MODULUS
    return result & 0b111


def power(a: int, exponent: int) -> int:
    result = 1
    base = a
    while exponent:
        if exponent & 1:
            result = multiply(result, base)
        base = multiply(base, base)
        exponent >>= 1
    return result


def inverse(a: int) -> int:
    if a == 0:
        raise ZeroDivisionError("0 has no inverse in GF(8)")
    return power(a, 6)


def poly_add(p, q):
    length = max(len(p), len(q))
    out = [0] * length
    for i in range(length):
        out[i] = add(p[i] if i < len(p) else 0, q[i] if i < len(q) else 0)
    while len(out) > 1 and out[-1] == 0:
        out.pop()
    return out


def poly_multiply(p, q):
    out = [0] * (len(p) + len(q) - 1)
    for i, x in enumerate(p):
        for j, y in enumerate(q):
            out[i + j] = add(out[i + j], multiply(x, y))
    while len(out) > 1 and out[-1] == 0:
        out.pop()
    return out


def poly_scale(p, scalar):
    return [multiply(value, scalar) for value in p]


def evaluate(coefficients, x):
    value = 0
    for coefficient in reversed(coefficients):
        value = add(multiply(value, x), coefficient)
    return value


def interpolate(xs, ys):
    result = [0]
    for j, (xj, yj) in enumerate(zip(xs, ys)):
        numerator = [1]
        denominator = 1
        for m, xm in enumerate(xs):
            if m == j:
                continue
            numerator = poly_multiply(numerator, [xm, 1])  # -xm = xm in characteristic 2
            denominator = multiply(denominator, add(xj, xm))
        result = poly_add(result, poly_scale(numerator, multiply(yj, inverse(denominator))))
    return result


labels = {
    0: "0",
    1: "1",
    2: "a",
    3: "a+1",
    4: "a^2",
    5: "a^2+1",
    6: "a^2+a",
    7: "a^2+a+1",
}

# Multiplication table
with (OUT / "10_gf8_multiplication_table.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow(["x*y", *[labels[x] for x in range(8)]])
    for x in range(8):
        writer.writerow([labels[x], *[labels[multiply(x, y)] for y in range(8)]])

coefficients = [1, 2, 4]  # 1 + a*x + a^2*x^2
evaluation_points = list(range(8))
codeword = [evaluate(coefficients, x) for x in evaluation_points]
chosen_xs = [1, 2, 4]
chosen_ys = [evaluate(coefficients, x) for x in chosen_xs]
recovered = interpolate(chosen_xs, chosen_ys)

with (OUT / "10_gf8_polynomial_evaluation.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow(["x_integer", "x_symbol", "p(x)_integer", "p(x)_symbol"])
    for x, y in zip(evaluation_points, codeword):
        writer.writerow([x, labels[x], y, labels[y]])

with (OUT / "10_gf8_codebook_k3.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow(["a0", "a1", "a2", *[f"p({labels[x]})" for x in evaluation_points]])
    for message in product(range(8), repeat=3):
        writer.writerow([*message, *[evaluate(message, x) for x in evaluation_points]])

report = [
    "Experiment 10: Reed--Solomon arithmetic over GF(8)",
    "=" * 55,
    "Field definition: GF(8) = F_2[a]/(a^3+a+1)",
    "Elements are represented by integers 0,...,7, interpreted as three-bit polynomials.",
    "Addition is XOR; multiplication reduces powers using a^3 = a+1.",
    "",
    f"Polynomial coefficients [1,a,a^2] as integers: {coefficients}",
    f"Codeword over all eight field elements: {codeword}",
    f"Chosen interpolation x-values: {chosen_xs}",
    f"Chosen y-values: {chosen_ys}",
    f"Recovered coefficients: {recovered}",
    f"Exact recovery: {recovered == coefficients}",
    "",
    "There are 8^3 = 512 degree-<3 message polynomials in this codebook.",
]
(OUT / "10_gf8_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")

print(OUT / "10_gf8_report.txt")
