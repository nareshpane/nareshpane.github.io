#!/usr/bin/env python3
"""
01_ali_baba_cave.py
A first experiment on zero-knowledge proofs: the Ali Baba cave.
Educational only.
"""

from __future__ import annotations
import math
import random
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

RNG = random.Random(42)
TRIALS = 20000
MAX_ROUNDS = 12
IMAGE_PATH = Path(__file__).resolve().parents[1] / 'images' / '01_ali_baba_cave_probability.png'


def one_trial(rounds: int, honest: bool, rng: random.Random) -> bool:
    """Return True if the prover is accepted for all rounds."""
    for _ in range(rounds):
        chosen_path = rng.choice(['A', 'B'])
        verifier_request = rng.choice(['A', 'B'])
        if honest:
            success = True  # honest prover can switch paths using the secret word
        else:
            success = (chosen_path == verifier_request)
        if not success:
            return False
    return True


def estimate(rounds: int, honest: bool, trials: int, rng: random.Random) -> float:
    wins = sum(one_trial(rounds, honest, rng) for _ in range(trials))
    return wins / trials


def main() -> None:
    print('ALI BABA CAVE: EMPIRICAL SOUNDNESS TEST')
    print('=' * 60)
    print('A cheating prover has success probability (1/2)^t after t rounds.')
    print()
    print(f"{'Rounds':>6}  {'Theory':>12}  {'Empirical':>12}  {'Honest':>12}")
    print('-' * 50)

    xs, theory_vals, empirical_vals = [], [], []
    for rounds in range(1, MAX_ROUNDS + 1):
        theory = 2 ** (-rounds)
        empirical = estimate(rounds, honest=False, trials=TRIALS, rng=RNG)
        honest = estimate(rounds, honest=True, trials=2000, rng=RNG)
        xs.append(rounds)
        theory_vals.append(theory)
        empirical_vals.append(empirical)
        print(f"{rounds:6d}  {theory:12.8f}  {empirical:12.8f}  {honest:12.8f}")

    print()
    print('Interpretation:')
    print('- The honest prover always passes.')
    print('- The cheating prover quickly becomes extremely unlikely to pass many rounds.')
    print('- This illustrates soundness without revealing the secret word.')

    if plt is not None:
        plt.figure(figsize=(8, 5))
        plt.plot(xs, theory_vals, marker='o', label='Theory: $2^{-t}$')
        plt.plot(xs, empirical_vals, marker='s', label='Empirical cheating rate')
        plt.yscale('log')
        plt.xlabel('Number of rounds t')
        plt.ylabel('Cheating acceptance probability')
        plt.title('Ali Baba Cave: cheating probability falls exponentially')
        plt.grid(True, which='both', alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(IMAGE_PATH, dpi=180)
        print(f"\nSaved figure to: {IMAGE_PATH}")
    else:
        print('\nmatplotlib not available, so no PNG was created.')


if __name__ == '__main__':
    main()
