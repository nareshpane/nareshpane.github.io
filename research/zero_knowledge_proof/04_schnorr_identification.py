#!/usr/bin/env python3
"""
04_schnorr_identification.py
Interactive Schnorr identification protocol.
Educational parameters only.
"""

from __future__ import annotations
import random

RNG = random.Random(7)

# Tiny safe prime example: p = 23, subgroup order q = 11, generator g = 2.
p = 23
q = 11
g = 2


def prover_commit(rng, x):
    r = rng.randrange(q)
    t = pow(g, r, p)
    return r, t


def prover_respond(r, c, x):
    return (r + c * x) % q


def verify(t, c, s, y):
    left = pow(g, s, p)
    right = (t * pow(y, c, p)) % p
    return left == right


def main():
    x = RNG.randrange(1, q)
    y = pow(g, x, p)

    print('SCHNORR IDENTIFICATION PROTOCOL')
    print('=' * 60)
    print('Public values:')
    print(f'  p = {p}, q = {q}, g = {g}')
    print(f'  y = g^x mod p = {y}')
    print('Hidden secret: x')
    print()
    print('Equations:')
    print('  commitment t = g^r mod p')
    print('  response   s = r + c x mod q')
    print('  verify  g^s = t y^c mod p')
    print()

    r, t = prover_commit(RNG, x)
    c = RNG.randrange(q)
    s = prover_respond(r, c, x)
    print(f'Example transcript: t = {t}, c = {c}, s = {s}')
    print('Verifier accepts?', verify(t, c, s, y))
    print()

    print(f"{'Round':>5}  {'Challenge c':>11}  {'Accept?':>8}")
    print('-' * 30)
    for i in range(1, 6):
        r, t = prover_commit(RNG, x)
        c = RNG.randrange(q)
        s = prover_respond(r, c, x)
        print(f"{i:5d}  {c:11d}  {str(verify(t, c, s, y)):>8}")

    print('\nThis protocol is a sigma protocol: commit, challenge, respond.')


if __name__ == '__main__':
    main()
