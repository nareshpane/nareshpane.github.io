"""Small, transparent Reed--Solomon utilities over prime fields.

The code deliberately favors readability over asymptotic speed. Coefficients are
stored from low degree to high degree: [a0, a1, ...] means
p(x) = a0 + a1*x + ... .
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import floor, sqrt
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class RSParameters:
    q: int
    n: int
    k: int
    evaluation_points: tuple[int, ...]

    @property
    def distance(self) -> int:
        return self.n - self.k + 1

    @property
    def unique_radius(self) -> int:
        return (self.distance - 1) // 2

    @property
    def johnson_radius_real(self) -> float:
        return self.n - sqrt(self.n * (self.k - 1))

    @property
    def johnson_radius_integer(self) -> int:
        return floor(self.johnson_radius_real + 1e-12)

    @property
    def capacity_radius_integer(self) -> int:
        return self.n - self.k

    @property
    def rate(self) -> float:
        return self.k / self.n


def is_prime(q: int) -> bool:
    if q < 2:
        return False
    if q % 2 == 0:
        return q == 2
    d = 3
    while d * d <= q:
        if q % d == 0:
            return False
        d += 2
    return True


def make_params(q: int, n: int, k: int, evaluation_points: Sequence[int] | None = None) -> RSParameters:
    if not is_prime(q):
        raise ValueError(f"q={q} is not prime. These core scripts use prime fields F_q.")
    if not (1 <= k <= n <= q):
        raise ValueError("Require 1 <= k <= n <= q for these ordinary prime-field RS experiments.")
    points = tuple(range(n)) if evaluation_points is None else tuple(int(x) % q for x in evaluation_points)
    if len(points) != n or len(set(points)) != n:
        raise ValueError("Evaluation points must contain n distinct field elements.")
    return RSParameters(q=q, n=n, k=k, evaluation_points=points)


def mod_inverse(a: int, q: int) -> int:
    a %= q
    if a == 0:
        raise ZeroDivisionError("0 has no multiplicative inverse in a field.")
    return pow(a, q - 2, q)


def poly_trim(coeffs: Sequence[int], q: int) -> list[int]:
    out = [int(c) % q for c in coeffs]
    while len(out) > 1 and out[-1] == 0:
        out.pop()
    return out


def poly_add(a: Sequence[int], b: Sequence[int], q: int) -> list[int]:
    m = max(len(a), len(b))
    out = [0] * m
    for i in range(m):
        out[i] = ((a[i] if i < len(a) else 0) + (b[i] if i < len(b) else 0)) % q
    return poly_trim(out, q)


def poly_scale(a: Sequence[int], scalar: int, q: int) -> list[int]:
    return poly_trim([(scalar * x) % q for x in a], q)


def poly_mul(a: Sequence[int], b: Sequence[int], q: int) -> list[int]:
    out = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i + j] = (out[i + j] + x * y) % q
    return poly_trim(out, q)


def poly_eval(coeffs: Sequence[int], x: int, q: int) -> int:
    value = 0
    for coefficient in reversed(coeffs):
        value = (value * x + coefficient) % q
    return value


def evaluate_polynomial(coeffs: Sequence[int], evaluation_points: Sequence[int], q: int) -> np.ndarray:
    return np.array([poly_eval(coeffs, x, q) for x in evaluation_points], dtype=np.int16)


def polynomial_string(coeffs: Sequence[int], variable: str = "x") -> str:
    terms: list[str] = []
    for degree, coefficient in enumerate(coeffs):
        if coefficient == 0:
            continue
        if degree == 0:
            terms.append(str(coefficient))
        elif degree == 1:
            terms.append(f"{coefficient}{variable}")
        else:
            terms.append(f"{coefficient}{variable}^{degree}")
    return " + ".join(terms) if terms else "0"


def lagrange_interpolate(xs: Sequence[int], ys: Sequence[int], q: int) -> list[int]:
    if len(xs) != len(ys):
        raise ValueError("xs and ys must have the same length.")
    if len(set(x % q for x in xs)) != len(xs):
        raise ValueError("Interpolation x-values must be distinct in F_q.")

    result = [0]
    for j, (xj, yj) in enumerate(zip(xs, ys)):
        numerator = [1]
        denominator = 1
        for m, xm in enumerate(xs):
            if m == j:
                continue
            numerator = poly_mul(numerator, [(-xm) % q, 1], q)
            denominator = (denominator * (xj - xm)) % q
        basis_scale = (yj * mod_inverse(denominator, q)) % q
        result = poly_add(result, poly_scale(numerator, basis_scale, q), q)
    return poly_trim(result, q)


def all_messages(q: int, k: int) -> np.ndarray:
    return np.array(list(product(range(q), repeat=k)), dtype=np.int16)


def build_codebook(params: RSParameters) -> tuple[np.ndarray, np.ndarray]:
    messages = all_messages(params.q, params.k)
    points = np.array(params.evaluation_points, dtype=np.int64)
    powers = np.vstack([pow_vector(points, degree, params.q) for degree in range(params.k)]).T
    codebook = (messages.astype(np.int64) @ powers.T) % params.q
    return messages, codebook.astype(np.int16)


def pow_vector(values: np.ndarray, exponent: int, q: int) -> np.ndarray:
    if exponent == 0:
        return np.ones_like(values, dtype=np.int64)
    out = np.ones_like(values, dtype=np.int64)
    base = values.astype(np.int64) % q
    e = exponent
    while e:
        if e & 1:
            out = (out * base) % q
        base = (base * base) % q
        e >>= 1
    return out


def all_words(q: int, n: int) -> np.ndarray:
    total = q ** n
    indices = np.arange(total, dtype=np.int64)
    words = np.empty((total, n), dtype=np.int16)
    work = indices.copy()
    for position in range(n - 1, -1, -1):
        words[:, position] = work % q
        work //= q
    return words


def word_to_index(word: Sequence[int], q: int) -> int:
    index = 0
    for value in word:
        index = index * q + int(value)
    return index


def hamming_distance(a: Sequence[int], b: Sequence[int]) -> int:
    return sum(int(x) != int(y) for x, y in zip(a, b))


def nearest_codewords(word: Sequence[int], codebook: np.ndarray) -> tuple[int, np.ndarray]:
    word_array = np.asarray(word, dtype=np.int16)
    distances = np.count_nonzero(codebook != word_array, axis=1)
    minimum = int(distances.min())
    return minimum, np.flatnonzero(distances == minimum)


def distance_to_code_for_words(words: np.ndarray, codebook: np.ndarray, chunk_size: int = 4096) -> np.ndarray:
    distances = np.empty(len(words), dtype=np.int16)
    for start in range(0, len(words), chunk_size):
        chunk = words[start : start + chunk_size]
        matrix = np.count_nonzero(chunk[:, None, :] != codebook[None, :, :], axis=2)
        distances[start : start + len(chunk)] = matrix.min(axis=1)
    return distances


def list_size_summary(
    words: np.ndarray,
    codebook: np.ndarray,
    radii: Iterable[int],
    chunk_size: int = 2048,
) -> tuple[dict[int, int], dict[int, np.ndarray], dict[int, dict[int, int]]]:
    radii = sorted(set(int(t) for t in radii))
    maxima = {t: -1 for t in radii}
    witnesses: dict[int, np.ndarray] = {}
    histograms = {t: {} for t in radii}

    for start in range(0, len(words), chunk_size):
        chunk = words[start : start + chunk_size]
        matrix = np.count_nonzero(chunk[:, None, :] != codebook[None, :, :], axis=2)
        for t in radii:
            counts = np.count_nonzero(matrix <= t, axis=1)
            local_max = int(counts.max())
            if local_max > maxima[t]:
                local_index = int(np.flatnonzero(counts == local_max)[0])
                maxima[t] = local_max
                witnesses[t] = chunk[local_index].copy()
            values, frequencies = np.unique(counts, return_counts=True)
            histogram = histograms[t]
            for value, frequency in zip(values, frequencies):
                histogram[int(value)] = histogram.get(int(value), 0) + int(frequency)
    return maxima, witnesses, histograms


def match_masks(word: Sequence[int], codebook: np.ndarray) -> np.ndarray:
    word_array = np.asarray(word, dtype=np.int16)
    equal = codebook == word_array
    weights = (1 << np.arange(codebook.shape[1], dtype=np.uint64))
    return (equal.astype(np.uint64) * weights).sum(axis=1)


def correlated_agreement_distance(f: Sequence[int], g: Sequence[int], codebook: np.ndarray) -> tuple[int, int, int]:
    """Return pair distance and codeword indices attaining it.

    A coordinate agrees only when both f and g agree with their respective
    Reed--Solomon codewords at that same coordinate.
    """
    n = codebook.shape[1]
    f_masks = match_masks(f, codebook)
    g_masks = match_masks(g, codebook)
    intersections = np.bitwise_and(f_masks[:, None], g_masks[None, :])
    popcount = np.array([int(x).bit_count() for x in range(1 << n)], dtype=np.int8)
    agreements = popcount[intersections.astype(np.int64)]
    flat_index = int(np.argmax(agreements))
    i, j = np.unravel_index(flat_index, agreements.shape)
    best_agreement = int(agreements[i, j])
    return n - best_agreement, int(i), int(j)


def line_word(f: Sequence[int], g: Sequence[int], z: int, q: int) -> np.ndarray:
    return (np.asarray(f, dtype=np.int16) + z * np.asarray(g, dtype=np.int16)) % q


def line_profile(f: Sequence[int], g: Sequence[int], q: int, codebook: np.ndarray) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for z in range(q):
        word = line_word(f, g, z, q)
        distance, nearest = nearest_codewords(word, codebook)
        rows.append({
            "z": z,
            "word": word,
            "distance": distance,
            "nearest_indices": nearest,
        })
    return rows


def feasible_masks_for_all_words(words: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    """For each word and coordinate mask, record whether some codeword matches on that mask."""
    n = words.shape[1]
    total_masks = 1 << n
    feasible = np.zeros((len(words), total_masks), dtype=bool)
    feasible[:, 0] = True
    for mask in range(1, total_masks):
        coordinates = [i for i in range(n) if (mask >> i) & 1]
        # Shape: number_of_words x number_of_codewords x len(coordinates)
        agreement = words[:, None, coordinates] == codebook[None, :, coordinates]
        feasible[:, mask] = np.any(np.all(agreement, axis=2), axis=1)
    return feasible


def shared_interpolation_distance(feasible_f: np.ndarray, feasible_g: np.ndarray, n: int) -> int:
    common = np.flatnonzero(feasible_f & feasible_g)
    maximum_agreement = max(int(mask).bit_count() for mask in common)
    return n - maximum_agreement


def feasible_mask_vector_for_word(word: Sequence[int], codebook: np.ndarray) -> np.ndarray:
    """Return all coordinate subsets on which the word matches some codeword."""
    n = codebook.shape[1]
    feasible = np.zeros(1 << n, dtype=bool)
    feasible[0] = True
    for mask_value in np.unique(match_masks(word, codebook)):
        mask = int(mask_value)
        submask = mask
        while True:
            feasible[submask] = True
            if submask == 0:
                break
            submask = (submask - 1) & mask
    return feasible
