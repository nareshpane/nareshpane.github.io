#!/usr/bin/env python3
"""
Generate one PNG per subsection for 20 standard examples.

Maclaurin-side plots:
- For functions that have a Maclaurin series, plot the exact function together with
  the approximations from the first 3, 4, 5, and 6 nonzero Maclaurin terms.
- For functions without a Maclaurin series at x = 0 (namely 1/x and ln x),
  generate an explanatory figure instead of a mathematically incorrect plot.

The plotting window, figure size, filename pattern, and axis limits are chosen so
that each Maclaurin figure aligns with the matching Taylor-about-a figure from the
companion script `generate_taylor_about_a_pngs_revised.py`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp


x = sp.symbols("x")
BINOMIAL_ALPHA = sp.Rational(2, 3)
FIGSIZE = (10.5, 6.4)
DPI = 220


# -----------------------------------------------------------------------------
# Shared function specifications
# -----------------------------------------------------------------------------

def build_specs() -> List[Dict]:
    return [
        {
            "index": 1,
            "slug": "exponential",
            "title": r"Exponential Function: $e^x$",
            "expr": sp.exp(x),
            "center": sp.Integer(0),
            "a_value": sp.Rational(1, 2),
            "x_range": (-2.0, 2.0),
        },
        {
            "index": 2,
            "slug": "sine",
            "title": r"Sine Function: $\sin x$",
            "expr": sp.sin(x),
            "center": sp.Integer(0),
            "a_value": sp.Rational(1, 2),
            "x_range": (-3.0, 3.0),
        },
        {
            "index": 3,
            "slug": "cosine",
            "title": r"Cosine Function: $\cos x$",
            "expr": sp.cos(x),
            "center": sp.Integer(0),
            "a_value": sp.Rational(1, 2),
            "x_range": (-3.0, 3.0),
        },
        {
            "index": 4,
            "slug": "sinh",
            "title": r"Hyperbolic Sine: $\sinh x$",
            "expr": sp.sinh(x),
            "center": sp.Integer(0),
            "a_value": sp.Rational(1, 2),
            "x_range": (-2.0, 2.0),
        },
        {
            "index": 5,
            "slug": "cosh",
            "title": r"Hyperbolic Cosine: $\cosh x$",
            "expr": sp.cosh(x),
            "center": sp.Integer(0),
            "a_value": sp.Rational(1, 2),
            "x_range": (-2.0, 2.0),
        },
        {
            "index": 6,
            "slug": "reciprocal",
            "title": r"Reciprocal Function: $1/x$",
            "expr": 1 / x,
            "center": None,
            "a_value": sp.Integer(1),
            "x_range": (0.35, 1.65),
        },
        {
            "index": 7,
            "slug": "logarithm",
            "title": r"Natural Logarithm: $\ln x$",
            "expr": sp.log(x),
            "center": None,
            "a_value": sp.Integer(1),
            "x_range": (0.35, 1.65),
        },
        {
            "index": 8,
            "slug": "geometric_minus",
            "title": r"Geometric Kernel: $1/(1-x)$",
            "expr": 1 / (1 - x),
            "center": sp.Integer(0),
            "a_value": sp.Rational(1, 2),
            "x_range": (-0.4, 0.75),
        },
        {
            "index": 9,
            "slug": "geometric_plus",
            "title": r"Alternating Geometric Kernel: $1/(1+x)$",
            "expr": 1 / (1 + x),
            "center": sp.Integer(0),
            "a_value": sp.Rational(1, 2),
            "x_range": (-0.75, 1.2),
        },
        {
            "index": 10,
            "slug": "squared_geometric_minus",
            "title": r"Squared Geometric Kernel: $1/(1-x)^2$",
            "expr": 1 / (1 - x) ** 2,
            "center": sp.Integer(0),
            "a_value": sp.Rational(1, 2),
            "x_range": (-0.4, 0.75),
        },
        {
            "index": 11,
            "slug": "squared_geometric_plus",
            "title": r"Squared Alternating Kernel: $1/(1+x)^2$",
            "expr": 1 / (1 + x) ** 2,
            "center": sp.Integer(0),
            "a_value": sp.Rational(1, 2),
            "x_range": (-0.75, 1.2),
        },
        {
            "index": 12,
            "slug": "x_over_1_minus_x",
            "title": r"Rational Function: $x/(1-x)$",
            "expr": x / (1 - x),
            "center": sp.Integer(0),
            "a_value": sp.Rational(1, 2),
            "x_range": (-0.4, 0.75),
        },
        {
            "index": 13,
            "slug": "x_over_1_plus_x",
            "title": r"Rational Function: $x/(1+x)$",
            "expr": x / (1 + x),
            "center": sp.Integer(0),
            "a_value": sp.Rational(1, 2),
            "x_range": (-0.75, 1.2),
        },
        {
            "index": 14,
            "slug": "log_1_plus_x",
            "title": r"Logarithm of $1+x$: $\ln(1+x)$",
            "expr": sp.log(1 + x),
            "center": sp.Integer(0),
            "a_value": sp.Rational(1, 2),
            "x_range": (-0.75, 1.2),
        },
        {
            "index": 15,
            "slug": "log_1_minus_x",
            "title": r"Logarithm of $1-x$: $\ln(1-x)$",
            "expr": sp.log(1 - x),
            "center": sp.Integer(0),
            "a_value": sp.Rational(1, 2),
            "x_range": (-0.4, 0.75),
        },
        {
            "index": 16,
            "slug": "binomial_alpha",
            "title": rf"General Binomial Function: $(1+x)^{{{sp.latex(BINOMIAL_ALPHA)}}}$",
            "expr": (1 + x) ** BINOMIAL_ALPHA,
            "center": sp.Integer(0),
            "a_value": sp.Rational(1, 2),
            "x_range": (-0.75, 1.2),
            "note": rf"Using $\alpha={sp.latex(BINOMIAL_ALPHA)}$ for plotting",
        },
        {
            "index": 17,
            "slug": "sqrt_1_plus_x",
            "title": r"Square Root: $\sqrt{1+x}$",
            "expr": sp.sqrt(1 + x),
            "center": sp.Integer(0),
            "a_value": sp.Rational(1, 2),
            "x_range": (-0.75, 1.2),
        },
        {
            "index": 18,
            "slug": "inv_sqrt_1_minus_x",
            "title": r"Inverse Square Root: $1/\sqrt{1-x}$",
            "expr": 1 / sp.sqrt(1 - x),
            "center": sp.Integer(0),
            "a_value": sp.Rational(1, 2),
            "x_range": (-0.4, 0.75),
        },
        {
            "index": 19,
            "slug": "inv_sqrt_1_plus_x",
            "title": r"Inverse Square Root: $1/\sqrt{1+x}$",
            "expr": 1 / sp.sqrt(1 + x),
            "center": sp.Integer(0),
            "a_value": sp.Rational(1, 2),
            "x_range": (-0.75, 1.2),
        },
        {
            "index": 20,
            "slug": "arctan",
            "title": r"Arctangent: $\arctan x$",
            "expr": sp.atan(x),
            "center": sp.Integer(0),
            "a_value": sp.Rational(1, 2),
            "x_range": (-1.5, 1.5),
        },
    ]


# -----------------------------------------------------------------------------
# Series helpers
# -----------------------------------------------------------------------------

def first_nonzero_taylor_polynomial(
    expr: sp.Expr,
    var: sp.Symbol,
    center: sp.Expr,
    nonzero_terms: int,
    max_order: int = 60,
) -> sp.Expr:
    """Return the Taylor polynomial with the first `nonzero_terms` nonzero terms."""
    pieces: List[sp.Expr] = []

    for n in range(max_order + 1):
        coeff = sp.simplify(sp.diff(expr, var, n).subs(var, center) / sp.factorial(n))
        if coeff != 0:
            pieces.append(sp.expand(coeff * (var - center) ** n))
            if len(pieces) == nonzero_terms:
                return sp.expand(sum(pieces))

    raise ValueError(
        f"Could not find {nonzero_terms} nonzero terms within derivative order {max_order}."
    )


# -----------------------------------------------------------------------------
# Numeric helpers
# -----------------------------------------------------------------------------

def safe_numeric_callable(expr: sp.Expr) -> Callable[[np.ndarray], np.ndarray]:
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


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def make_regular_plot(spec: Dict, output_dir: Path) -> None:
    expr = spec["expr"]
    center = spec["center"]
    x_min, x_max = spec["x_range"]
    note = spec.get("note", "")

    x_vals = np.linspace(x_min, x_max, 1200)
    exact_fn = safe_numeric_callable(expr)
    exact_y = exact_fn(x_vals)
    y_limits = matched_y_limits(exact_y)
    exact_y = clip_for_display(exact_y, y_limits)

    approximations = {}
    labels = {}
    for term_count in (3, 4, 5, 6):
        poly = first_nonzero_taylor_polynomial(expr, x, center, term_count)
        approximations[term_count] = clip_for_display(safe_numeric_callable(poly)(x_vals), y_limits)
        labels[term_count] = sp.sstr(poly)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.suptitle(f"{spec['index']}. {spec['title']}", fontsize=14)
    subtitle = r"Maclaurin approximation using the first 3, 4, 5, and 6 nonzero terms"
    if note:
        subtitle += f" | {note}"
    ax.set_title(subtitle, fontsize=10, pad=10)

    ax.plot(x_vals, exact_y, linewidth=2.5, label="Exact function")
    for term_count in (3, 4, 5, 6):
        ax.plot(
            x_vals,
            approximations[term_count],
            linewidth=1.8,
            linestyle="--",
            label=f"First {term_count} nonzero terms",
        )

    ax.axvline(0.0, linewidth=1.0, linestyle=":", label=r"Center $x=0$")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(*y_limits)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    filename = f"{spec['index']:02d}_{spec['slug']}_maclaurin.png"
    fig.savefig(output_dir / filename, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_dir / filename}")
    for term_count in (3, 4, 5, 6):
        print(f"    {term_count} nonzero terms: {labels[term_count]}")


def make_nonexistent_maclaurin_plot(spec: Dict, output_dir: Path) -> None:
    expr = spec["expr"]
    x_min, x_max = spec["x_range"]
    x_vals = np.linspace(x_min, x_max, 1200)
    exact_y = safe_numeric_callable(expr)(x_vals)
    y_limits = matched_y_limits(exact_y)
    exact_y = clip_for_display(exact_y, y_limits)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.suptitle(f"{spec['index']}. {spec['title']}", fontsize=14)
    ax.set_title("Maclaurin series does not exist at x = 0", fontsize=10, pad=10)

    ax.plot(x_vals, exact_y, linewidth=2.5, label="Exact function")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(*y_limits)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.25)

    message = (
        "No Maclaurin series exists here because the function\n"
        "is not analytic at x = 0. See the Taylor-about-a plot\n"
        f"for the corresponding local approximation about x = {sp.latex(spec['a_value'])}."
    )
    ax.text(
        0.03,
        0.97,
        message,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#f5efe3", "edgecolor": "#b9ac95"},
    )
    ax.legend(fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    filename = f"{spec['index']:02d}_{spec['slug']}_maclaurin.png"
    fig.savefig(output_dir / filename, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_dir / filename}")
    print("    Maclaurin series does not exist at x = 0.")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    script_dir = Path(__file__).resolve().parent
    output_dir = script_dir / "taylor_maclaurin_pngs"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving PNG files into: {output_dir}")
    print()

    for spec in build_specs():
        if spec["center"] is None:
            make_nonexistent_maclaurin_plot(spec, output_dir)
        else:
            make_regular_plot(spec, output_dir)
        print()

    print("Done.")


if __name__ == "__main__":
    main()
