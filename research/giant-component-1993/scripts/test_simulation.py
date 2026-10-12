"""Independent adjacency traversal, constructed cases and sampling checks."""
import csv
import json
import math
import unittest
from pathlib import Path
import numpy as np
from .sampling import sample_edges
from .union_find import find, add_edge, classify_history
from .experiments import trial, batch
from .statistics import LIMITS, NS, wilson
from .run import seed_for


def reference(n, history):
    adjacency = [set() for _ in range(n)]
    for u, v in history:
        adjacency[u].add(v)
        adjacency[v].add(u)
    unseen = set(range(n))
    components = []
    while unseen:
        reached, stack = set(), [next(iter(unseen))]
        while stack:
            u = stack.pop()
            if u in reached:
                continue
            reached.add(u)
            stack.extend(adjacency[u] - reached)
        unseen -= reached
        # Count original edge occurrences, so loops and parallel edges are exact.
        e = sum(u in reached and v in reached for u, v in history)
        components.append((len(reached), e))
    return components


class Tests(unittest.TestCase):
    def check_history(self, n, history):
        history = np.array(history, np.int64).reshape((-1, 2))
        parent, size, edges = np.arange(n), np.ones(n, np.int64), np.zeros(n, np.int64)
        count, ever = 0, False
        for k, (u, v) in enumerate(history):
            count, _ = add_edge(parent, size, edges, u, v, count)
            expected = reference(n, history[:k+1])
            actual = sorted((int(size[i]), int(edges[i])) for i in range(n) if parent[i] == i)
            self.assertEqual(actual, sorted(expected))
            self.assertEqual(count, sum(e > v for v, e in expected))
            ever = ever or count >= 2
        expected = reference(n, history)
        self.assertEqual(classify_history(n, history),
                         (all(e-v+1 <= 2 for v, e in expected), not ever, count))

    def test_constructed(self):
        for h in [[(0,1),(1,2)], [(0,1),(1,2),(2,0)],
                  [(0,1),(1,2),(2,0),(0,3),(1,3)],
                  [(0,1),(1,2),(2,0),(0,3),(1,3),(2,3)],
                  [(0,0)], [(0,1),(0,1),(0,1)],
                  [(0,0),(0,0),(1,1),(1,1)],
                  [(0,0),(0,0),(1,1),(1,1),(0,1)]]:
            self.check_history(5,h)

    def test_failure_persists_after_merge(self):
        h=np.array([(0,0),(0,0),(1,1),(1,1),(0,1)],np.int64)
        self.assertEqual(classify_history(2,h), (False,False,1))
        # A simple graph example: two K4 minus one edge, then a bridge.
        h=[(o+u,o+v) for o in [0,4] for u,v in [(0,1),(1,2),(2,0),(0,3),(1,3)]]+[(3,4)]
        self.check_history(8,h)
        self.assertFalse(classify_history(8,np.array(h,np.int64))[1])

    def test_small_random_against_independent_traversal(self):
        for simple in [True,False]:
            for seed in range(40):
                self.check_history(9,sample_edges(9,25,simple,seed))

    def test_reproducibility_and_thread_independence(self):
        import numba
        seeds=np.array([seed_for(123,51,'simple','trajectory',j) for j in range(24)],np.uint32)
        numba.set_num_threads(1)
        x=batch(51,True,True,seeds)
        numba.set_num_threads(4)
        y=batch(51,True,True,seeds)
        for a,b in zip(x,y):
            np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(sample_edges(9,25,False,19),sample_edges(9,25,False,19))

    def test_snapshot_odd_floor_and_independent_event(self):
        for n in [9,10,21]:
            for simple in [True,False]:
                for seed in range(30):
                    h=sample_edges(n,n//2,simple,seed)
                    expected=all(e-v+1<=2 for v,e in reference(n,h))
                    outcome,m=trial(n,simple,False,seed)
                    self.assertEqual(m,n//2)
                    self.assertEqual(bool(outcome),expected)

    def test_trajectory_exact_stopping(self):
        for simple in [True,False]:
            for seed in range(30):
                outcome,m=trial(9,simple,True,seed)
                h=sample_edges(9,m,simple,seed)
                self.assertEqual(bool(outcome),classify_history(9,h)[1])
                if outcome:
                    self.assertEqual(len(reference(9,h)),1)
                else:
                    self.assertGreaterEqual(sum(e>v for v,e in reference(9,h)),2)
                if m>1:
                    self.assertGreater(len(reference(9,h[:-1])),1)
                    self.assertTrue(classify_history(9,h[:-1])[1])

    def test_model_distributions(self):
        # 50,000 ordered draws; loops 1/n, each nonloop unordered pair 2/n^2.
        n=5
        h=sample_edges(n,50000,False,371)
        for u in range(n):
            count=int(((h[:,0]==u)&(h[:,1]==u)).sum())
            expected=len(h)/n**2
            self.assertLess(abs(count-expected),6*math.sqrt(expected))
            for v in range(u+1,n):
                count=int((((h[:,0]==u)&(h[:,1]==v))|((h[:,0]==v)&(h[:,1]==u))).sum())
                expected=2*len(h)/n**2
                self.assertLess(abs(count-expected),6*math.sqrt(expected))
        # Every permutation contains each eligible pair once; first edge uniform.
        counts={}
        for seed in range(5000):
            h=sample_edges(4,6,True,seed)
            self.assertEqual(len(set(map(tuple,h))),6)
            self.assertTrue(np.all(h[:,0]<h[:,1]))
            key=tuple(h[0]); counts[key]=counts.get(key,0)+1
        self.assertEqual(len(counts),6)
        self.assertLess(max(abs(v-5000/6) for v in counts.values()),6*math.sqrt(5000/6))

    def test_constants_and_wilson(self):
        # Independent series evaluation of cosh, rather than the same function.
        from decimal import Decimal, localcontext
        with localcontext() as ctx:
            ctx.prec=50
            value=(Decimal(2)/3).sqrt()*sum((Decimal(5)/18)**r/Decimal(math.factorial(2*r)) for r in range(25))
        self.assertAlmostEqual(LIMITS['snapshot'],float(value),places=14)
        self.assertAlmostEqual(LIMITS['trajectory'],0.8726646259971648,places=14)
        for s in [0,5,10]:
            lo,hi=wilson(s,10)
            self.assertLessEqual(lo,s/10); self.assertGreaterEqual(hi,s/10)

    def test_exports_if_present(self):
        out=Path(__file__).resolve().parents[1]/'data'
        if not (out/'results.json').exists():
            self.skipTest('Run simulations before export verification')
        data=json.loads((out/'results.json').read_text())
        with (out/'results.csv').open(newline='') as f:
            csv_rows=list(csv.DictReader(f))
        self.assertEqual(len(csv_rows),len(data['rows']))
        for r,c in zip(data['rows'],csv_rows):
            for key,val in r.items():
                self.assertEqual(str(val),c[key])
            self.assertEqual(r['probability'],r['successes']/r['trials'])
            self.assertEqual((r['ci_low'],r['ci_high']),wilson(r['successes'],r['trials']))
            self.assertEqual(r['theoretical_limit'],LIMITS[r['experiment']])
        if data['metadata'].get('complete'):
            for exp in data['metadata']['configuration']['experiments']:
                for model in data['metadata']['configuration']['models']:
                    self.assertEqual(sorted(r['n'] for r in data['rows'] if r['experiment']==exp and r['model']==model),NS)


if __name__=='__main__':
    unittest.main()
