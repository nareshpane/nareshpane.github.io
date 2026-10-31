#!/usr/bin/env python3
"""
02_graph_isomorphism_zkp.py
Classic zero-knowledge proof for graph isomorphism.
Educational only; graphs are tiny so brute-force checks are possible.
"""

from __future__ import annotations
import itertools
import random
from typing import Iterable

RNG = random.Random(123)


def normalize_edge(u: int, v: int) -> tuple[int, int]:
    return (u, v) if u < v else (v, u)


def apply_perm(graph: set[tuple[int, int]], perm: list[int]) -> set[tuple[int, int]]:
    return {normalize_edge(perm[u], perm[v]) for (u, v) in graph}


def compose(p: list[int], q: list[int]) -> list[int]:
    """Return p∘q, meaning apply q first, then p."""
    return [p[q[i]] for i in range(len(p))]


def inverse(perm: list[int]) -> list[int]:
    inv = [0] * len(perm)
    for i, j in enumerate(perm):
        inv[j] = i
    return inv


def is_isomorphic(g1: set[tuple[int, int]], g2: set[tuple[int, int]], n: int) -> bool:
    for perm in itertools.permutations(range(n)):
        if apply_perm(g1, list(perm)) == g2:
            return True
    return False


def honest_round(g0, g1, sigma, rng):
    n = max(max(e) for e in g0) + 1
    tau = list(range(n))
    rng.shuffle(tau)
    H = apply_perm(g0, tau)
    b = rng.randint(0, 1)
    if b == 0:
        phi = tau
    else:
        phi = compose(tau, inverse(sigma))
    verified = (apply_perm(g0 if b == 0 else g1, phi) == H)
    return b, verified


def cheating_round(noniso_g0, noniso_g1, rng):
    """A bluffing prover can only prepare to answer one challenge."""
    n = max(max(e) for e in noniso_g0) + 1
    choice = rng.randint(0, 1)
    base = noniso_g0 if choice == 0 else noniso_g1
    tau = list(range(n))
    rng.shuffle(tau)
    H = apply_perm(base, tau)
    b = rng.randint(0, 1)
    if b == choice:
        phi = tau
        verified = (apply_perm(noniso_g0 if b == 0 else noniso_g1, phi) == H)
    else:
        verified = False
    return b, verified


def multi_round(round_fn, rounds: int, trials: int) -> float:
    wins = 0
    for _ in range(trials):
        ok = True
        for _ in range(rounds):
            _, verified = round_fn()
            if not verified:
                ok = False
                break
        wins += ok
    return wins / trials


def main() -> None:
    # Isomorphic pair
    g0 = {normalize_edge(*e) for e in [(0, 1), (1, 2), (2, 3), (3, 4), (1, 4)]}
    sigma = [2, 4, 0, 1, 3]
    g1 = apply_perm(g0, sigma)

    # Non-isomorphic pair for cheating demo
    h0 = {normalize_edge(*e) for e in [(0, 1), (1, 2), (2, 3), (3, 4)]}   # path
    h1 = {normalize_edge(*e) for e in [(0, 1), (1, 2), (2, 0), (3, 4)]}   # triangle + edge

    print('GRAPH ISOMORPHISM ZERO-KNOWLEDGE PROOF')
    print('=' * 60)
    print('We prove knowledge of a hidden permutation sigma such that G1 = sigma(G0).')
    print('Protocol equations are combinatorial rather than algebraic:')
    print('  H = tau(G0)')
    print('  if challenge b=0, reveal phi = tau')
    print('  if challenge b=1, reveal phi = tau ∘ sigma^{-1}')
    print()
    print('Sanity checks:')
    print('  G0 and G1 isomorphic?   ', is_isomorphic(g0, g1, 5))
    print('  H0 and H1 isomorphic?   ', is_isomorphic(h0, h1, 5))
    print()

    b, verified = honest_round(g0, g1, sigma, RNG)
    print(f'Example honest round challenge b = {b}, verifier accepts = {verified}')
    print()
    print(f"{'Rounds':>6}  {'Honest accept':>14}  {'Cheater accept':>15}  {'Theory cheater':>15}")
    print('-' * 60)
    for rounds in range(1, 8):
        honest = multi_round(lambda: honest_round(g0, g1, sigma, RNG), rounds, 2000)
        cheat = multi_round(lambda: cheating_round(h0, h1, RNG), rounds, 5000)
        theory = 2 ** (-rounds)
        print(f"{rounds:6d}  {honest:14.6f}  {cheat:15.6f}  {theory:15.6f}")

    print('\nInterpretation: a prover who does not know how to answer both challenges can do no better than about 1/2 per round.')


if __name__ == '__main__':
    main()
