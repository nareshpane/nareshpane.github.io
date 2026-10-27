"""Independent exact audits. Run with Python 3; no third-party packages required."""
from itertools import combinations, product, permutations
from fractions import Fraction
from pathlib import Path
import json, random, hashlib

BASE = Path(__file__).resolve().parents[1]
PAPER = [(0,1),(1,0),(1,2),(2,1),(2,3),(3,2),(3,1),(1,4),(3,4),(2,4)]

def enumerate_forests(n, edges, roots):
    """Edge-subset enumeration + undirected union-find + outdegree tests."""
    result=[]
    for chosen in combinations(edges,n-len(roots)):
        degree=[0]*n; parent=list(range(n)); good=True
        def find(x):
            while parent[x]!=x: x=parent[x]
            return x
        for u,v in chosen:
            degree[u]+=1
            a,b=find(u),find(v)
            if a==b: good=False; break
            parent[a]=b
        if good and all(degree[i]==(0 if i in roots else 1) for i in range(n)):
            result.append(tuple(sorted(chosen)))
    return sorted(result)

def parity_forests(n,edges,roots):
    """Symmetric completion; reject repeated edge IDs; cancel by parity;
    orient surviving undirected forests toward roots, then remove added arcs.
    No cycle test is used during polynomial expansion.
    """
    original=set(edges); pairs=sorted({tuple(sorted(e)) for e in edges})
    choices=[[(j,p) for p in pairs if i in p for j in p if j!=i] for i in range(n) if i not in roots]
    odd=set()
    for selection in product(*choices):
        ids=tuple(sorted(p for _,p in selection))
        if len(set(ids))!=len(ids): continue
        if ids in odd: odd.remove(ids)
        else: odd.add(ids)
    result=[]
    for ids in sorted(odd):
        adjacency=[[] for _ in range(n)]
        for u,v in ids: adjacency[u].append(v); adjacency[v].append(u)
        seen=set(roots); queue=list(roots); arcs=[]
        for v in queue:
            for u in adjacency[v]:
                if u not in seen:
                    seen.add(u); queue.append(u); arcs.append((u,v))
        if len(seen)!=n: raise AssertionError('Parity left an unrooted component')
        if set(arcs)<=original: result.append(tuple(sorted(arcs)))
    return sorted(result)

def determinant(matrix):
    """Exact rational Gaussian elimination, independent of graph enumeration."""
    a=[[Fraction(x) for x in row] for row in matrix]; result=Fraction(1)
    for j in range(len(a)):
        p=next((i for i in range(j,len(a)) if a[i][j]),None)
        if p is None:return 0
        if p!=j:a[p],a[j]=a[j],a[p];result=-result
        d=a[j][j];result*=d
        for i in range(j+1,len(a)):
            t=a[i][j]/d
            for k in range(j+1,len(a)):a[i][k]-=t*a[j][k]
    return int(result)

def minor_count(n,edges,roots):
    keep=[i for i in range(n) if i not in roots]
    lap=[[0]*n for _ in range(n)]
    for u,v in edges:lap[u][u]+=1;lap[u][v]-=1
    return determinant([[lap[i][j] for j in keep] for i in keep])

def run():
    cases=0; samples=[]
    def check(n,edges,roots,label):
        nonlocal cases
        a=enumerate_forests(n,edges,roots); b=parity_forests(n,edges,roots); c=minor_count(n,edges,roots)
        assert a==b,(label,a,b)
        assert len(a)==c,(label,len(a),c)
        cases+=1
        if label.startswith('paper'):samples.append(dict(label=label,n=n,edges=edges,roots=sorted(roots),count=len(a),forests=a))
    for n in range(1,4):
        arcs=[(i,j) for i in range(n) for j in range(n) if i!=j]
        for mask in range(1<<len(arcs)):
            edges=[e for i,e in enumerate(arcs) if mask>>i&1]
            for rootmask in range(1,1<<n):check(n,edges,{i for i in range(n) if rootmask>>i&1},'exhaustive')
    rng=random.Random(1966)
    for n in range(4,7):
        for trial in range(60):
            edges=[(i,j) for i in range(n) for j in range(n) if i!=j and rng.random()<.4]
            roots=set(rng.sample(range(n),rng.randint(1,n)))
            check(n,edges,roots,'seeded')
    check(5,PAPER,{4},'paper-tree');check(5,PAPER,{2,4},'paper-2-tree')
    # Snapshot source integrity, never modifying the originals.
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((BASE/'Verbatim').glob('*'))}
    output=dict(methods=['edge-subset + union-find','symmetric completion + parity','exact Laplacian principal minor'],cases=cases,failures=0,seed=1966,domain='All loopless labelled digraphs n=1..3 and every nonempty root set; 60 seeded graphs per n=4,5,6 with one sampled nonempty root set each; two paper examples.',samples=samples,source_sha256=hashes)
    (BASE/'data').mkdir(exist_ok=True)
    (BASE/'data'/'validation-results.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in output.items() if k not in ('samples','source_sha256')},indent=2))

if __name__=='__main__':run()
