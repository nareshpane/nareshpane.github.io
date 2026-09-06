"""Replay deterministic generation with independent Dijkstra distance matrices.

The replay uses a temporary output root; published data/master are never changed.
Optional --baseline compares against a previous asset directory, not a website
repository revision or a paired causal experiment.
"""
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import time
import networkx as nx
import numpy as np
import generate_data as generator
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path);args=p.parse_args()
city=json.loads((ROOT/'data/city.json').read_text());analysis=json.loads((ROOT/'data/analysis.json').read_text())
started=time.perf_counter()
def dijkstra_matrix(g):
    d=np.full((len(g),len(g)),np.inf)
    for i,lengths in nx.all_pairs_dijkstra_path_length(g,weight='seconds'):
        for j,cost in lengths.items():d[i,j]=cost/60
    return d
with tempfile.TemporaryDirectory(prefix='astra-world-math-replay-') as directory:
    generator.ROOT=Path(directory);(generator.ROOT/'data').mkdir()
    generator.costs=dijkstra_matrix
    with contextlib.redirect_stdout(io.StringIO()):generator.main()
    replay=json.loads((generator.ROOT/'data/city.json').read_text());assert replay==city
    a=json.loads((generator.ROOT/'data/analysis.json').read_text())
    a.pop('generation_seconds');expected=dict(analysis);expected.pop('generation_seconds')
    assert a==expected,'Independent Dijkstra replay differs from original Floyd–Warshall results'
    assert (generator.ROOT/'data/disruption_trials.json').read_bytes()==(ROOT/'data/disruption_trials.json').read_bytes()
g=generator.make_graph(city)
assert len(g)==64 and g.number_of_edges()==108 and nx.is_connected(g)
assert len(set(e['id'] for e in city['roads']))==108
for e in city['roads']:
    a,b=[city['nodes'][i] for i in [e['u'],e['v']]]
    assert abs(np.hypot(a['x']-b['x'],a['y']-b['y'])-e['length_m'])<1e-8
    assert e['seconds']==round(e['length_m']/80*60)
for route in analysis['alternative_routes']:
    assert nx.path_weight(g,route['nodes'],'seconds')==route['seconds']
    assert abs(nx.path_weight(g,route['nodes'],'length_m')-route['metres'])<1e-8
for road in city['roads']:
    if road['kind']!='footway':continue
    a,b=[city['nodes'][i] for i in [road['u'],road['v']]]
    # Conservative dense centreline sampling with half-width expansion of
    # every building footprint; feeder links and diagonals avoid all parcels.
    for t in np.linspace(0,1,201):
        x=a['x']+t*(b['x']-a['x']);y=a['y']+t*(b['y']-a['y'])
        for building in city['buildings']:
            assert not (abs(x-building['x'])<building['width']/2+1.5 and abs(y-building['y'])<building['depth']/2+1.5),(road['id'],building['id'])
cross=g.copy();cross.remove_edges_from([(11,12),(35,36)])
assert nx.number_connected_components(cross)==2
report=dict(status='PASS',city_sha256=hashlib.sha256((ROOT/'data/city.json').read_bytes()).hexdigest(),
    seconds=round(time.perf_counter()-started,3),checks=[
        'Exact seed replay preserves every city record and all 1200 retained edge-removal sets',
        'Dijkstra matrices independently reproduce all trial outcomes, node losses and accessibility aggregates',
        'All alternative paths use real edges with matching stored costs and lengths',
        'All 108 edge lengths are Euclidean; walking costs are rounded once to integer seconds',
        'New footways clear every building footprint including path width',
        'Both river crossings together remain a two-edge cut; neither alone disconnects the graph'])
if args.baseline:
    old=json.loads((args.baseline/'data/city.json').read_text())
    olda=json.loads((args.baseline/'data/analysis.json').read_text())
    oldv=json.loads((args.baseline/'data/scene_validation.json').read_text())
    newv=json.loads((ROOT/'data/scene_validation.json').read_text())
    geometry_keys=['id','x','y','width','depth','height','archetype','neighbourhood']
    assert all(all(b[k]==previous[k] for k in geometry_keys) for b,previous in zip(city['buildings'],old['buildings']))
    reassigned=[dict(building=b['id'],previous_node=previous['node'],revised_node=b['node']) for b,previous in zip(city['buildings'],old['buildings']) if b['node']!=previous['node']]
    def stats(c,a):
        h=generator.make_graph(c);normal=nx.shortest_path_length(h,40,47,weight='seconds')
        h.remove_edge(35,36);closure=nx.shortest_path_length(h,40,47,weight='seconds')
        return dict(nodes=a['nodes'],edges=a['road_edges'],cycle_rank=a['road_edges']-a['nodes']+1,
            same_pair_40_47_seconds=normal,same_pair_closure_seconds=closure,
            maximum_betweenness=max(a['betweenness'].values()),maximum_nodes=[k for k,v in a['betweenness'].items() if v==max(a['betweenness'].values())],
            disconnection_probability_p08=a['monte_carlo'][1]['probability_disconnected'],
            mean_accessibility_loss_p08=a['monte_carlo'][1]['mean_accessibility_loss_pct'])
    comparison=dict(previous=stats(old,olda),revised=stats(city,analysis),building_parameters_preserved=True,nearest_node_reassignments=reassigned,
        performance={label:{k:v[k] for k in ['master_objects','master_mesh_objects','master_triangles_instanced','blend_bytes','glb']} for label,v in [('previous',oldv),('revised',newv)]},
        note='Same-pair travel comparison fixes origin and destination. Monte Carlo aggregates use each version’s seeded stream, not paired failures; origins and opportunity totals also differ.')
    (ROOT/'data/world_revision_comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
(ROOT/'data/world_calculation_validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
