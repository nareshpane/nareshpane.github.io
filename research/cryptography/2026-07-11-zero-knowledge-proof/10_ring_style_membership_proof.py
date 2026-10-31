#!/usr/bin/env python3
"""
10_ring_style_membership_proof.py
Generalized OR proof across many public keys.
This gives a toy ring-proof / anonymous-membership flavour.
"""

from __future__ import annotations
import random

RNG = random.Random(2718)
p = 23
q = 11
g = 2


def inv_mod(a: int, mod: int) -> int:
    return pow(a, -1, mod)


def main():
    n = 5
    secrets = RNG.sample(range(1, q), n)
    public_keys = [pow(g, x, p) for x in secrets]

    known_index = 3  # prover knows only this secret
    xk = secrets[known_index]

    c_values = [None] * n
    s_values = [None] * n
    a_values = [None] * n

    # Simulate all branches except the known one.
    for i in range(n):
        if i == known_index:
            continue
        c_values[i] = RNG.randrange(q)
        s_values[i] = RNG.randrange(q)
        a_values[i] = (pow(g, s_values[i], p) * inv_mod(pow(public_keys[i], c_values[i], p), p)) % p

    # Real branch
    rk = RNG.randrange(q)
    a_values[known_index] = pow(g, rk, p)

    # Global challenge from verifier
    c_global = RNG.randrange(q)
    known_sum = sum(c for c in c_values if c is not None) % q
    c_values[known_index] = (c_global - known_sum) % q
    s_values[known_index] = (rk + c_values[known_index] * xk) % q

    # Verify
    challenge_sum_ok = sum(c_values) % q == c_global
    branch_checks = []
    for i in range(n):
        lhs = pow(g, s_values[i], p)
        rhs = (a_values[i] * pow(public_keys[i], c_values[i], p)) % p
        branch_checks.append(lhs == rhs)

    print('RING-STYLE MEMBERSHIP / MANY-WAY OR PROOF')
    print('=' * 60)
    print('Interpretation: the prover shows, "I know one secret key in this public list,')
    print('but I will not reveal which one."')
    print()
    print('Public keys:', public_keys)
    print('Global challenge:', c_global)
    print('Challenge pieces c_i:', c_values)
    print('Response pieces  s_i:', s_values)
    print('Commitments      a_i:', a_values)
    print()
    print('Do the challenge pieces sum to the global challenge?', challenge_sum_ok)
    for i, ok in enumerate(branch_checks):
        print(f'Branch {i} verifies? {ok}')
    print('Overall accept?', challenge_sum_ok and all(branch_checks))
    print(f'(The prover actually knew index {known_index}, but the transcript does not announce that.)')


if __name__ == '__main__':
    main()
