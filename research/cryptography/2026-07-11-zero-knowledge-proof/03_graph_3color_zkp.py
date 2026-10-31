#!/usr/bin/env python3
"""
03_graph_3color_zkp.py
A toy zero-knowledge proof for 3-colourability using hash commitments.
This is a didactic simulation, not production cryptography.
"""

from __future__ import annotations
import hashlib
import os
import random

RNG = random.Random(99)
COLORS = ['red', 'green', 'blue']


def commit(value: str, salt: bytes) -> str:
    return hashlib.sha256(salt + value.encode()).hexdigest()


def permute_coloring(coloring: dict[int, str], rng: random.Random) -> dict[int, str]:
    perm = COLORS[:]
    rng.shuffle(perm)
    mapping = dict(zip(COLORS, perm))
    return {v: mapping[c] for v, c in coloring.items()}


def make_commitments(coloring: dict[int, str]) -> tuple[dict[int, str], dict[int, bytes]]:
    commitments = {}
    salts = {}
    for v, c in coloring.items():
        salt = os.urandom(16)
        salts[v] = salt
        commitments[v] = commit(c, salt)
    return commitments, salts


def verify_opening(vertex: int, claimed_color: str, salt: bytes, commitments: dict[int, str]) -> bool:
    return commit(claimed_color, salt) == commitments[vertex]


def one_round(graph_edges, coloring, rng, honest=True) -> bool:
    if honest:
        shown_coloring = permute_coloring(coloring, rng)
    else:
        shown_coloring = coloring.copy()  # fake prover uses a fixed possibly invalid coloring
    commitments, salts = make_commitments(shown_coloring)
    edge = rng.choice(graph_edges)
    u, v = edge
    cu, cv = shown_coloring[u], shown_coloring[v]
    opened_ok = (
        verify_opening(u, cu, salts[u], commitments)
        and verify_opening(v, cv, salts[v], commitments)
    )
    different = (cu != cv)
    return opened_ok and different


def experiment(graph_edges, honest_coloring, fake_coloring, rounds_max=10, trials=5000):
    print('3-COLOURING ZERO-KNOWLEDGE PROOF')
    print('=' * 60)
    print('Commitment form:')
    print('  Com(v) = H(salt_v || permuted_colour(v))')
    print('The verifier asks to open one random edge (u,v).')
    print('The prover reveals only the colours on that edge, not the full colouring.')
    print()
    print('Graph edges:', graph_edges)
    print('Honest coloring:', honest_coloring)
    print('Fake coloring:  ', fake_coloring)
    print()
    print(f"{'Rounds':>6}  {'Honest accept':>14}  {'Fake accept':>14}")
    print('-' * 40)
    for rounds in range(1, rounds_max + 1):
        honest_ok = 0
        fake_ok = 0
        for _ in range(trials):
            if all(one_round(graph_edges, honest_coloring, RNG, honest=True) for _ in range(rounds)):
                honest_ok += 1
            if all(one_round(graph_edges, fake_coloring, RNG, honest=False) for _ in range(rounds)):
                fake_ok += 1
        print(f"{rounds:6d}  {honest_ok / trials:14.6f}  {fake_ok / trials:14.6f}")


def main():
    # A 5-cycle is 3-colourable.
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
    honest_coloring = {0: 'red', 1: 'green', 2: 'blue', 3: 'red', 4: 'green'}
    # Fake colouring has one bad edge: (4,0) both red.
    fake_coloring = {0: 'red', 1: 'green', 2: 'blue', 3: 'red', 4: 'red'}
    experiment(edges, honest_coloring, fake_coloring)
    print('\nInterpretation: a bad colouring may pass one challenge if the verifier misses the bad edge, but repeated random checks drive the cheating probability down.')


if __name__ == '__main__':
    main()
