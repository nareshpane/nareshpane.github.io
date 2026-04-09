#!/usr/bin/env python3
"""
Generate one PNG for each of 20 standard functions, comparing the exact function
with its Taylor polynomial approximations about x = a using the first 3, 4, 5,
and 6 terms.

Revised per user request:
- Taylor plots are centered at x = 1 whenever the function is analytic there.
- The x-window is symmetric about the Taylor center.
  Example: Maclaurin window [-2, 2] becomes Taylor window [-1, 3] when a = 1.
- Folder names and output filenames are kept unchanged so the existing HTML
  continues to work.

Important exception:
Some functions are not defined at x = 1, so a Taylor expansion about x = 1 is
impossible. For these, the script keeps a safe non-singular center and states
that clearly in the plot subtitle:
    1/(1-x), 1/(1-x)^2, x/(1-x), ln(1-x), 1/sqrt(1-x)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp


x = sp.symbols("x")
ALPHA = sp.Rational(2, 3)
FIGSIZE = (10.5, 6.4)
DPI = 220


@dataclass(frozen=True)
class FunctionSpec:
    index: int
    slug: str
    title: str
    expr: sp.Expr
    mac_x_min: float
    mac_x_max: float
    a_value: sp.Expr
    note: str = ""


def symmetric_window_about(center: float, mac_x_min: float, mac_x_max: float) -> tuple[float, float]:
    half_width = max(abs(mac_x_min), abs(mac_x_max))
    return center - half_width, center + half_width


FUNCTIONS: List[FunctionSpec] = [
    FunctionSpec(1,  "exponential",             r"Exponential Function: $e^x$",                    sp.exp(x),                -2.0,  2.0,  sp.Integer(1)),
    FunctionSpec(2,  "sine",                    r"Sine Function: $\sin x$",                        sp.sin(x),                -3.0,  3.0,  sp.Integer(1)),
    FunctionSpec(3,  "cosine",                  r"Cosine Function: $\cos x$",                      sp.cos(x),                -3.0,  3.0,  sp.Integer(1)),
    FunctionSpec(4,  "sinh",                    r"Hyperbolic Sine: $\sinh x$",                     sp.sinh(x),               -2.0,  2.0,  sp.Integer(1)),
    FunctionSpec(5,  "cosh",                    r"Hyperbolic Cosine: $\cosh x$",                   sp.cosh(x),               -2.0,  2.0,  sp.Integer(1)),
    FunctionSpec(6,  "reciprocal",              r"Reciprocal Function: $1/x$",                     1 / x,                     0.35,  1.65, sp.Integer(1)),
    FunctionSpec(7,  "logarithm",               r"Natural Logarithm: $\ln x$",                     sp.log(x),                 0.35,  1.65, sp.Integer(1)),
    FunctionSpec(8,  "geometric_minus",         r"Geometric Kernel: $1/(1-x)$",                    1 / (1 - x),              -0.4,  0.75, sp.Rational(1, 2),
                 r"$a=1$ is impossible here because the function is singular at $x=1$; using $a=\frac{1}{2}$ instead"),
    FunctionSpec(9,  "geometric_plus",          r"Alternating Geometric Kernel: $1/(1+x)$",        1 / (1 + x),              -0.75, 1.2,  sp.Integer(1)),
    FunctionSpec(10, "squared_geometric_minus", r"Squared Geometric Kernel: $1/(1-x)^2$",          1 / (1 - x) ** 2,         -0.4,  0.75, sp.Rational(1, 2),
                 r"$a=1$ is impossible here because the function is singular at $x=1$; using $a=\frac{1}{2}$ instead"),
    FunctionSpec(11, "squared_geometric_plus",  r"Squared Alternating Kernel: $1/(1+x)^2$",        1 / (1 + x) ** 2,         -0.75, 1.2,  sp.Integer(1)),
    FunctionSpec(12, "x_over_1_minus_x",        r"Rational Function: $x/(1-x)$",                   x / (1 - x),              -0.4,  0.75, sp.Rational(1, 2),
                 r"$a=1$ is impossible here because the function is singular at $x=1$; using $a=\frac{1}{2}$ instead"),
    FunctionSpec(13, "x_over_1_plus_x",         r"Rational Function: $x/(1+x)$",                   x / (1 + x),              -0.75, 1.2,  sp.Integer(1)),
    FunctionSpec(14, "log_1_plus_x",            r"Logarithm of $1+x$: $\ln(1+x)$",                 sp.log(1 + x),            -0.75, 1.2,  sp.Integer(1)),
    FunctionSpec(15, "log_1_minus_x",           r"Logarithm of $1-x$: $\ln(1-x)$",                 sp.log(1 - x),            -0.4,  0.75, sp.Rational(1, 2),
                 r"$a=1$ is impossible here because the function is singular at $x=1$; using $a=\frac{1}{2}$ instead"),
    FunctionSpec(16, "binomial_alpha",          rf"General Binomial Function: $(1+x)^{{{sp.latex(ALPHA)}}}$", (1 + x) ** ALPHA, -0.75, 1.2, sp.Integer(1),
                 rf"Using $\alpha={sp.latex(ALPHA)}$ for plotting"),
    FunctionSpec(17, "sqrt_1_plus_x",           r"Square Root: $\sqrt{1+x}$",                      sp.sqrt(1 + x),           -0.75, 1.2,  sp.Integer(1)),
    FunctionSpec(18, "inv_sqrt_1_minus_x",      r"Inverse Square Root: $1/\sqrt{1-x}$",            1 / sp.sqrt(1 - x),       -0.4,  0.75, sp.Rational(1, 2),
                 r"$a=1$ is impossible here because the function is singular at $x=1$; using $a=\frac{1}{2}$ instead"),
    FunctionSpec(19, "inv_sqrt_1_plus_x",       r"Inverse Square Root: $1/\sqrt{1+x}$",            1 / sp.sqrt(1 + x),       -0.75, 1.2,  sp.Integer(1)),
    FunctionSpec(20, "arctan",                  r"Arctangent: $\arctan x$",                        sp.atan(x),               -1.5,  1.5,  sp.Integer(1)),
]


def taylor_polynomial(expr: sp.Expr, a_value: sp.Expr, n_terms: int) -> sp.Expr:
    poly = sp.Integer(0)
    for k in range(n_terms):
        coeff = sp.simplify(sp.diff(expr, x, k).subs(x, a_value) / sp.factorial(k))
        poly += coeff * (x - a_value) ** k
    return sp.expand(poly)


def safe_numeric_callable(expr: sp.Expr):
    raw = sp.lambdify(x, expr, modules=["numpy"])

    def wrapped(values: np.ndarray) -> np.ndarray:
        with np.errstate(all="ignore"):
            out = raw(values)
        out = np.asarray(out, dtype=np.complex128)
        out = np.where(np.abs(out.imag) < 1e-10, out.real, np.nan)
        out = np.asarray(out, dtype=float)
        out[~np.isfinite(out)] = np.nan
        return out

    return wrapped


def matched_y_limits(exact_y: np.ndarray) -> tuple[float, float]:
    finite = exact_y[np.isfinite(exact_y)]
    if finite.size == 0:
        return -1.0, 1.0

    y_min = float(np.min(finite))
    y_max = float(np.max(finite))

    if y_min < 0 < y_max:
        y_abs = max(abs(y_min), abs(y_max))
        pad = 0.08 * y_abs if y_abs > 0 else 1.0
        return -y_abs - pad, y_abs + pad

    span = y_max - y_min
    if span == 0:
        pad = 0.1 * max(1.0, abs(y_max))
    else:
        pad = 0.08 * span
    return y_min - pad, y_max + pad


def clip_for_display(y: np.ndarray, y_limits: tuple[float, float]) -> np.ndarray:
    low, high = y_limits
    span = high - low
    display_low = low - 0.35 * span
    display_high = high + 0.35 * span
    out = np.array(y, dtype=float)
    out[(out < display_low) | (out > display_high)] = np.nan
    return out


def make_plot(spec: FunctionSpec, output_dir: Path) -> None:
    center_float = float(sp.N(spec.a_value))
    x_min, x_max = symmetric_window_about(center_float, spec.mac_x_min, spec.mac_x_max)

    if spec.a_value == sp.Rational(1, 2) and spec.slug in {
        "geometric_minus", "squared_geometric_minus", "x_over_1_minus_x",
        "log_1_minus_x", "inv_sqrt_1_minus_x"
    }:
        x_min, x_max = 0.05, 0.95

    xs = np.linspace(x_min, x_max, 1200)
    exact_y = safe_numeric_callable(spec.expr)(xs)
    y_limits = matched_y_limits(exact_y)
    exact_y = clip_for_display(exact_y, y_limits)

    term_counts = [3, 4, 5, 6]
    polys = [taylor_polynomial(spec.expr, spec.a_value, n) for n in term_counts]
    y_polys = [clip_for_display(safe_numeric_callable(poly)(xs), y_limits) for poly in polys]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.suptitle(f"{spec.index}. {spec.title}", fontsize=14)

    subtitle = rf"Taylor approximation about $x=a$ with $a={sp.latex(spec.a_value)}$ using the first 3, 4, 5, and 6 terms"
    if spec.note:
        subtitle += f" | {spec.note}"
    ax.set_title(subtitle, fontsize=10, pad=10)

    ax.plot(xs, exact_y, linewidth=2.5, label="Exact function")
    for n, y_poly in zip(term_counts, y_polys):
        ax.plot(xs, y_poly, linewidth=1.8, linestyle="--", label=f"First {n} terms")

    ax.axvline(center_float, linestyle=":", linewidth=1.0, label=rf"Center $x={sp.latex(spec.a_value)}$")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(*y_limits)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    filename = f"{spec.index:02d}_{spec.slug}_taylor.png"
    fig.savefig(output_dir / filename, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_dir / filename}")


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    output_dir = script_dir / "taylor_about_a_pngs"
    output_dir.mkdir(parents=True, exist_ok=True)

    for spec in FUNCTIONS:
        make_plot(spec, output_dir)

    print(f"Saved {len(FUNCTIONS)} PNG files in: {output_dir}")


if __name__ == "__main__":
    main()
