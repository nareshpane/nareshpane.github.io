#!/usr/bin/env python3
"""
07_chaum_pedersen_equality.py
Zero-knowledge proof that two public values have the same discrete logarithm.
"""

from __future__ import annotations
import random

RNG = random.Random(77)
p = 23
q = 11
g = 2
h = 3

# Ensure h is in the subgroup of order q.
assert pow(h, q, p) == 1


def main():
    x = RNG.randrange(1, q)
    y1 = pow(g, x, p)
    y2 = pow(h, x, p)
    r = RNG.randrange(q)
    a1 = pow(g, r, p)
    a2 = pow(h, r, p)
    c = RNG.randrange(q)
    s = (r + c * x) % q

    left1 = pow(g, s, p)
    right1 = (a1 * pow(y1, c, p)) % p
    left2 = pow(h, s, p)
    right2 = (a2 * pow(y2, c, p)) % p

    print('CHAUM-PEDERSEN PROOF OF EQUALITY OF DISCRETE LOGS')
    print('=' * 60)
    print('We prove that the SAME hidden x satisfies:')
    print('  y1 = g^x mod p')
    print('  y2 = h^x mod p')
    print('without revealing x.')
    print()
    print('Commitments:')
    print(f'  a1 = g^r = {a1}')
    print(f'  a2 = h^r = {a2}')
    print(f'Challenge c = {c}, response s = {s}')
    print()
    print('Verification equations:')
    print(f'  g^s == a1 * y1^c ?  {left1} == {right1} -> {left1 == right1}')
    print(f'  h^s == a2 * y2^c ?  {left2} == {right2} -> {left2 == right2}')


if __name__ == '__main__':
    main()
