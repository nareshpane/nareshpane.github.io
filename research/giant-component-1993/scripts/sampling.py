"""Exact lazy sampling of the two Section 1 graph processes."""
import numpy as np
from numba import njit


@njit(cache=True)
def draw_edge(n, simple, used):
    # Each unordered non-loop pair has two equally likely ordered proposals.
    # Conditional on acceptance every unused pair therefore has equal probability.
    while True:
        u = np.random.randint(n)
        v = np.random.randint(n)
        if not simple:
            return u, v
        if u == v:
            continue
        if u > v:
            u, v = v, u
        key = np.int64(u) * n + v
        if key not in used:
            used.add(key)
            return u, v


@njit(cache=True)
def sample_edges(n, m, simple, seed):
    """Validation helper; not used to materialize large research histories."""
    np.random.seed(seed)
    used = {np.int64(-1)}
    out = np.empty((m, 2), np.int64)
    for i in range(m):
        out[i] = draw_edge(n, simple, used)
    return out
