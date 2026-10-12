"""Compiled independent trials; trajectory successes stop only at connectivity."""
import numpy as np
from numba import njit, prange
from .sampling import draw_edge
from .union_find import add_edge


@njit(cache=True)
def trial(n, simple, trajectory, seed):
    np.random.seed(seed)
    parent = np.arange(n)
    size = np.ones(n, np.int64)
    edges = np.zeros(n, np.int64)
    used = {np.int64(-1)}
    components, complex_count, m = n, 0, 0
    while True:
        if trajectory:
            if components == 1:
                return 1, m
        elif m == n // 2:
            for i in range(n):
                if parent[i] == i and edges[i] - size[i] + 1 > 2:
                    return 0, m
            return 1, m
        u, v = draw_edge(n, simple, used)
        complex_count, merged = add_edge(parent, size, edges, u, v, complex_count)
        components -= int(merged)
        m += 1
        if trajectory and complex_count >= 2:
            return 0, m


@njit(cache=True, parallel=True)
def batch(n, simple, trajectory, seeds):
    outcomes = np.empty(len(seeds), np.int64)
    accepted = np.empty(len(seeds), np.int64)
    for i in prange(len(seeds)):
        outcomes[i], accepted[i] = trial(n, simple, trajectory, seeds[i])
    return outcomes, accepted
