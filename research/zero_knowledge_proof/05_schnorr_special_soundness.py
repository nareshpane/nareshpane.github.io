#!/usr/bin/env python3
"""
05_schnorr_special_soundness.py
If the same Schnorr commitment answers two different challenges,
we can extract the secret x.
"""

from __future__ import annotations
import random

RNG = random.Random(1234)
p = 23
q = 11
g = 2


def main():
    x = RNG.randrange(1, q)
    y = pow(g, x, p)
    r = RNG.randrange(q)
    t = pow(g, r, p)

    c1, c2 = RNG.sample(range(q), 2)
    s1 = (r + c1 * x) % q
    s2 = (r + c2 * x) % q

    numerator = (s1 - s2) % q
    denominator = (c1 - c2) % q
    x_extracted = (numerator * pow(denominator, -1, q)) % q

    print('SCHNORR SPECIAL SOUNDNESS / EXTRACTOR DEMO')
    print('=' * 60)
    print('Given two accepting transcripts with the same commitment t:')
    print('  s1 = r + c1 x (mod q)')
    print('  s2 = r + c2 x (mod q)')
    print('Subtract them:')
    print('  s1 - s2 = (c1 - c2) x (mod q)')
    print('So:')
    print('  x = (s1 - s2) / (c1 - c2) (mod q)')
    print()
    print(f'p={p}, q={q}, g={g}, y={y}')
    print(f'Hidden x = {x}')
    print(f'Common commitment t = {t}')
    print(f'Transcript 1: c1={c1}, s1={s1}')
    print(f'Transcript 2: c2={c2}, s2={s2}')
    print(f'Extracted x = {x_extracted}')
    print('Extraction successful?', x_extracted == x)


if __name__ == '__main__':
    main()
