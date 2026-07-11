#!/usr/bin/env python3
"""
06_fiat_shamir_schnorr.py
Turn Schnorr's interactive protocol into a non-interactive proof
using the Fiat-Shamir transform.
"""

from __future__ import annotations
import hashlib
import random

RNG = random.Random(5)
p = 23
q = 11
g = 2


def hash_to_challenge(y: int, t: int, message: str) -> int:
    blob = f'{y}|{t}|{message}'.encode()
    return int(hashlib.sha256(blob).hexdigest(), 16) % q


def prove(x: int, y: int, message: str):
    r = RNG.randrange(q)
    t = pow(g, r, p)
    c = hash_to_challenge(y, t, message)
    s = (r + c * x) % q
    return {'t': t, 's': s, 'message': message}


def verify(y: int, proof: dict) -> bool:
    t, s, message = proof['t'], proof['s'], proof['message']
    c = hash_to_challenge(y, t, message)
    return pow(g, s, p) == (t * pow(y, c, p)) % p


def main():
    x = RNG.randrange(1, q)
    y = pow(g, x, p)
    message = 'I know the secret key for this toy example.'
    proof = prove(x, y, message)

    print('FIAT-SHAMIR NON-INTERACTIVE SCHNORR PROOF')
    print('=' * 60)
    print('Challenge is now computed, not sent by a live verifier:')
    print('  c = H(y || t || message) mod q')
    print('This removes interaction, at least in the random-oracle model.')
    print()
    print('Proof object:', proof)
    print('Verifier accepts original proof?', verify(y, proof))

    fake = dict(proof)
    fake['message'] = 'I changed the statement!'
    print('Verifier accepts tampered-message proof?', verify(y, fake))


if __name__ == '__main__':
    main()
