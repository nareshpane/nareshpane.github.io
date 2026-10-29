"""Independent, standard-library graph mathematics for the research page.

No browser code, NetworkX, nauty, or Traces is called by these calculations.
"""
from itertools import permutations, combinations
from math import factorial

def edges(n):
    return list(combinations(range(n), 2))

def matrix(n, pairs):
    a = [[0] * n for _ in range(n)]
    for u, v in pairs:
        a[u][v] = a[v][u] = 1
    return a

def from_mask(n, mask):
    return matrix(n, [e for k, e in enumerate(edges(n)) if mask >> k & 1])

def code(a, order):
    return ''.join(str(a[order[i]][order[j]]) for i, j in edges(len(a)))

def canonical_exhaustive(a):
    return max(code(a, p) for p in permutations(range(len(a))))

def graph6(line):
    n = ord(line[0]) - 63
    bits = [int(x) for c in line[1:] for x in f'{ord(c)-63:06b}']
    pairs = [(i, j) for j in range(1, n) for i in range(j)]
    return matrix(n, [e for e, bit in zip(pairs, bits) if bit])

def burnside(n):
    pair_list = edges(n)
    pair_index = {e: i for i, e in enumerate(pair_list)}
    total = 0
    for p in permutations(range(n)):
        action = [pair_index[tuple(sorted((p[u], p[v])))] for u, v in pair_list]
        visited, orbit_count = set(), 0
        for i in range(len(action)):
            if i in visited:
                continue
            orbit_count += 1
            j = i
            while j not in visited:
                visited.add(j)
                j = action[j]
        total += 1 << orbit_count
    assert total % factorial(n) == 0
    return total // factorial(n)

def automorphisms(a):
    n = len(a)
    # Independent degree-constrained permutation traversal; no IR code is shared.
    degree = [sum(row) for row in a]
    cells = {}
    for v, d in enumerate(degree):
        cells.setdefault(d, []).append(v)
    p = [-1] * n
    used = set()
    result = []
    def visit(k):
        if k == n:
            result.append(tuple(p))
            return
        for w in cells[degree[k]]:
            if w not in used and all(a[k][i] == a[w][p[i]] for i in range(k)):
                p[k] = w
                used.add(w)
                visit(k + 1)
                used.remove(w)
    visit(0)
    return result

def srg(a):
    n = len(a)
    return {'n': n, 'degrees': sorted(set(map(sum, a))),
            'adjacent_common': sorted({sum(a[u][t]*a[v][t] for t in range(n)) for u,v in edges(n) if a[u][v]}),
            'nonadjacent_common': sorted({sum(a[u][t]*a[v][t] for t in range(n)) for u,v in edges(n) if not a[u][v]}),
            'triangles': sum(all(a[u][v] for u,v in combinations(c,2)) for c in combinations(range(n),3)),
            'cliques4': sum(all(a[u][v] for u,v in combinations(c,2)) for c in combinations(range(n),4))}

def regular_examples():
    pairs = edges(16)
    rook = matrix(16, [(u,v) for u,v in pairs if u//4 == v//4 or u%4 == v%4])
    differences = {(1,0),(3,0),(0,1),(0,3),(1,1),(3,3)}
    shri = matrix(16, [(u,v) for u,v in pairs if ((v%4-u%4)%4,(v//4-u//4)%4) in differences])
    return rook, shri

if __name__ == '__main__':
    print('Run validate_examples.py to regenerate the independently checked report.')
