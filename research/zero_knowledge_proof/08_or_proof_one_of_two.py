#!/usr/bin/env python3
"""
08_or_proof_one_of_two.py
A sigma-protocol OR proof: the prover knows one of two secrets,
but does not reveal which one.
"""

from __future__ import annotations
import random

RNG = random.Random(2026)
p = 23
q = 11
g = 2


def inv_mod(a: int, mod: int) -> int:
    return pow(a, -1, mod)


def main():
    # Prover really knows x1 but not x2.
    x1 = RNG.randrange(1, q)
    x2 = RNG.randrange(1, q)
    y1 = pow(g, x1, p)
    y2 = pow(g, x2, p)

    # Prover prepares one real branch (for y1) and one simulated branch (for y2).
    r1 = RNG.randrange(q)
    a1 = pow(g, r1, p)

    c2 = RNG.randrange(q)
    s2 = RNG.randrange(q)
    a2 = (pow(g, s2, p) * inv_mod(pow(y2, c2, p), p)) % p

    c = RNG.randrange(q)  # verifier's global challenge
    c1 = (c - c2) % q
    s1 = (r1 + c1 * x1) % q

    check_sum = (c1 + c2) % q == c
    check1 = pow(g, s1, p) == (a1 * pow(y1, c1, p)) % p
    check2 = pow(g, s2, p) == (a2 * pow(y2, c2, p)) % p

    print('OR PROOF: KNOW ONE OF TWO DISCRETE LOGS')
    print('=' * 60)
    print('Global challenge is split as c = c1 + c2 (mod q).')
    print('One branch is real, the other branch is simulated.')
    print('The verifier cannot tell which is which.')
    print()
    print(f'y1 = {y1}, y2 = {y2}')
    print(f'Global challenge c = {c}')
    print(f'Branch 1: a1={a1}, c1={c1}, s1={s1}')
    print(f'Branch 2: a2={a2}, c2={c2}, s2={s2}')
    print()
    print('Checks:')
    print('  c1 + c2 == c (mod q)?', check_sum)
    print('  Branch 1 equation holds?', check1)
    print('  Branch 2 equation holds?', check2)
    print('Overall accept?', check_sum and check1 and check2)


if __name__ == '__main__':
    main()
