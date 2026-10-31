#!/usr/bin/env python3
"""
09_and_proof_two_secrets.py
AND composition of sigma protocols: prove knowledge of TWO secrets at once.
"""

from __future__ import annotations
import random

RNG = random.Random(314)
p = 23
q = 11
g = 2


def main():
    x1 = RNG.randrange(1, q)
    x2 = RNG.randrange(1, q)
    y1 = pow(g, x1, p)
    y2 = pow(g, x2, p)

    r1 = RNG.randrange(q)
    r2 = RNG.randrange(q)
    a1 = pow(g, r1, p)
    a2 = pow(g, r2, p)
    c = RNG.randrange(1, q)
    s1 = (r1 + c * x1) % q
    s2 = (r2 + c * x2) % q

    check1 = pow(g, s1, p) == (a1 * pow(y1, c, p)) % p
    check2 = pow(g, s2, p) == (a2 * pow(y2, c, p)) % p

    print('AND PROOF: KNOW TWO SECRETS SIMULTANEOUSLY')
    print('=' * 60)
    print('The prover uses the SAME challenge c for two Schnorr proofs.')
    print('This proves knowledge of x1 and x2 together.')
    print()
    print(f'y1 = {y1}, y2 = {y2}, challenge c = {c}')
    print(f'Commitments: a1 = {a1}, a2 = {a2}')
    print(f'Responses:   s1 = {s1}, s2 = {s2}')
    print('Check 1:', check1)
    print('Check 2:', check2)
    print('Overall accept?', check1 and check2)


if __name__ == '__main__':
    main()
