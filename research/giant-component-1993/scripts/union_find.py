"""Root vertex/edge counts and an incremental complex-component count."""
import numpy as np
from numba import njit


@njit(cache=True)
def find(parent, u):
    while parent[u] != u:
        parent[u] = parent[parent[u]]
        u = parent[u]
    return u


@njit(cache=True)
def add_edge(parent, size, edges, u, v, complex_count):
    a, b = find(parent, u), find(parent, v)
    complex_count -= int(edges[a] > size[a])
    merged = a != b
    if merged:
        complex_count -= int(edges[b] > size[b])
        if size[a] < size[b]:
            a, b = b, a
        parent[b] = a
        size[a] += size[b]
        edges[a] += edges[b] + 1
    else:
        edges[a] += 1
    complex_count += int(edges[a] > size[a])
    return complex_count, merged


@njit(cache=True)
def classify_history(n, history):
    """Return final snapshot event, persistent trajectory event, current count."""
    parent, size, edges = np.arange(n), np.ones(n, np.int64), np.zeros(n, np.int64)
    count, ever_failed = 0, False
    for u, v in history:
        count, _ = add_edge(parent, size, edges, u, v, count)
        ever_failed = ever_failed or count >= 2
    snapshot = True
    for i in range(n):
        if parent[i] == i and edges[i] - size[i] + 1 > 2:
            snapshot = False
    return snapshot, not ever_failed, count
