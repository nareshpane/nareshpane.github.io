"""Regenerate independent exact counts, family checks, and source checksums."""
from pathlib import Path
from itertools import combinations, permutations
from math import factorial
import hashlib
import json
import platform
from generate_examples import (edges, matrix, from_mask, canonical_exhaustive,
                               graph6, burnside, automorphisms, srg, regular_examples)

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = [1, 2, 4, 11, 34, 156, 1044, 12346]

def main():
    report = {'date': '2026-10-29', 'python': platform.python_version(),
              'method': 'Independent Python standard-library exhaustive permutations, Burnside edge-orbits, and direct structural checks.',
              'burnside': {}, 'exhaustive_counts': {}, 'catalogues': {}, 'checks': []}
    for n, expected in enumerate(EXPECTED, 1):
        got = burnside(n)
        assert got == expected, (n, got)
        report['burnside'][str(n)] = got
    for n in range(1, 6):
        keys = {canonical_exhaustive(from_mask(n,m)) for m in range(1 << (n*(n-1)//2))}
        assert len(keys) == EXPECTED[n-1]
        report['exhaustive_counts'][str(n)] = len(keys)
    for n in [6,7,8]:
        p = ROOT / f'data/generated-graphs/graph{n}.g6'
        lines = p.read_text().splitlines()
        assert len(lines) == EXPECTED[n-1]
        assert all(len(graph6(line)) == n for line in lines)
        item = {'count': len(lines), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
                'source': f'https://users.cecs.anu.edu.au/~bdm/data/graph{n}.g6'}
        if n == 6:
            keys = {canonical_exhaustive(graph6(line)) for line in lines}
            assert len(keys) == len(lines)
            item['independent_exhaustive_S6_unique_keys'] = len(keys)
        report['catalogues'][str(n)] = item
    pairs = {'path4': (4, [(0,1),(1,2),(2,3)], 2),
             'cycle4': (4, [(0,1),(1,2),(2,3),(3,0)], 8),
             'cycle6': (6, [(i,(i+1)%6) for i in range(6)], 12),
             'complete5': (5, edges(5), 120),
             'star6': (6, [(0,i) for i in range(1,6)], 120),
             'asymmetric7': (7, [(0,1),(0,2),(2,3),(0,4),(4,5),(5,6)], 1),
             'prism6': (6, [(i,(i+1)%3) for i in range(3)]+[(i+3,(i+1)%3+3) for i in range(3)]+[(i,i+3) for i in range(3)], 12),
             'prism12': (12, [(i,(i+1)%6) for i in range(6)]+[(i+6,(i+1)%6+6) for i in range(6)]+[(i,i+6) for i in range(6)], 24),
             'paper9': (9, [(0,1),(1,2),(2,5),(5,8),(8,7),(7,6),(6,3),(3,0),(1,3),(3,7),(7,5),(5,1),(1,4),(3,4),(5,4),(7,4)], 8)}
    report['automorphism_orders'] = {}
    for key, (n, e, expected) in pairs.items():
        a = matrix(n,e)
        aut = automorphisms(a)
        assert len(aut) == expected, (key,len(aut))
        assert all(all(a[u][v] == a[p[u]][p[v]] for u,v in edges(n)) for p in aut)
        report['automorphism_orders'][key] = len(aut)
    a = matrix(9,pairs['paper9'][1]); cells = [[0,2,6,8],[1,3,5,7],[4]]
    quotient = [[sum(a[c[0]][w] for w in d) for d in cells] for c in cells]
    assert quotient == [[0,2,0],[2,2,1],[0,4,0]]
    assert all(all(sum(a[v][w] for w in d) == quotient[i][j] for v in c) for i,c in enumerate(cells) for j,d in enumerate(cells))
    report['paper9_quotient'] = quotient
    rook, shri = regular_examples()
    report['strongly_regular'] = {'rook':srg(rook),'shrikhande':srg(shri)}
    for a in (rook,shri):
        t=srg(a); assert t['degrees']==[6] and t['adjacent_common']==[2] and t['nonadjacent_common']==[2] and t['triangles']==32
    assert srg(rook)['cliques4']==8 and srg(shri)['cliques4']==0
    cube_edges = [(v,v^(1<<b)) for v in range(32) for b in range(5) if v < (v^(1<<b))]
    seen=set()
    for coord in permutations(range(5)):
        for translation in range(32):
            p=tuple(sum(((v>>j)&1)<<coord[j] for j in range(5)) ^ translation for v in range(32))
            assert all((p[u]^p[v]).bit_count()==1 for u,v in cube_edges)
            seen.add(p)
    assert len(seen)==3840
    report['cube5']={'vertices':32,'edges':len(cube_edges),'verified_distinct_coordinate_translation_automorphisms':len(seen),'group_order':32*factorial(5)}
    lex=matrix(25,[(u,v) for u,v in edges(25) if (u//5-v//5)%5 in (1,4) or (u//5==v//5 and (u%5-v%5)%5 in (1,4))])
    assert all(sum(row)==12 for row in lex)
    for u,v in edges(25):
        if lex[u][v]:
            common=sum(lex[u][w]*lex[v][w] for w in range(25))
            assert common == (10 if u//5==v//5 else 4)
    report['lex25']={'vertices':25,'edges':sum(map(sum,lex))//2,'degree':12,'internal_edge_common_neighbours':10,'interblock_edge_common_neighbours':4,'proved_wreath_product_order':10**6}
    inventory=json.loads((ROOT/'data/validation/original-asset-inventory.json').read_text())
    for item in inventory:
        path=ROOT/item['destination']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256'],path
    report['preserved_original_assets'] = len(inventory)
    report['checks'] = ['Independent Burnside counts n=1..8', 'Independent exhaustive canonical counts n=1..5',
                        'Independent exhaustive pairwise distinction of all 156 order-six catalogue representatives',
                        'All reported small-family automorphisms preserve adjacency', 'II Figure 1 quotient checked',
                        'Both SRG(16,6,2,2) constructions and 4-clique distinction checked',
                        'All 3840 cube coordinate/translation maps verified', 'Lexicographic-product block characterization checked',
                        'Original asset hashes unchanged, including both PDFs']
    (ROOT/'data/validation/python-validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
