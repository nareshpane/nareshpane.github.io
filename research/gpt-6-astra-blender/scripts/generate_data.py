"""Synthetic metric city and exact network analysis; no geographic source data.

Python 3.12+, NumPy, NetworkX 3.6.1. Output paths are relative to this script.
Road costs are positive integer seconds, so equal-cost paths are exact.
"""
import hashlib
import json
import math
from pathlib import Path
import random
import time
from itertools import islice
from collections import Counter
import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEED = 260905
REPS = 400
LAMBDA = .18  # inverse minutes

def write(name, data):
    (ROOT/'data'/name).write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')

def make_graph(city, transit=False, proposal=False):
    g = nx.Graph()
    g.add_nodes_from(n['id'] for n in city['nodes'])
    for edge in city['roads'] + (city['transit'] if transit else []) + (city['proposal'] if proposal else []):
        g.add_edge(edge['u'], edge['v'], **{k:v for k,v in edge.items() if k not in ['u','v']})
    return g

def costs(g):
    return np.asarray(nx.floyd_warshall_numpy(g,nodelist=range(len(g)),weight='seconds'))/60

def accessibility(dist, opportunities):
    # np.exp(-inf) = 0 implements unreachable destinations without dropping origins.
    return np.exp(-LAMBDA*dist) @ np.asarray(opportunities)

def summary(values):
    a=np.asarray(values)
    return dict(min=float(a.min()),mean=float(a.mean()),median=float(np.median(a)),
                max=float(a.max()),std_population=float(a.std()))

def main():
    start=time.perf_counter()
    rng=random.Random(SEED)
    xs=[-168,-120,-72,-24,24,72,120,168]
    ys=[-144,-96,-48,0,48,96,144]
    nodes=[dict(id=r*8+c,x=x,y=y) for r,y in enumerate(ys) for c,x in enumerate(xs)]
    roads=[]
    def edge(u,v,kind='road',seconds=None,road_class='local'):
        a,b=nodes[u],nodes[v]
        length=math.hypot(a['x']-b['x'],a['y']-b['y'])
        return dict(id=f'{min(u,v)}-{max(u,v)}',u=u,v=v,kind=kind,
                    length_m=length,seconds=round(length/80*60) if seconds is None else seconds,
                    road_class=road_class,width_m=3 if kind=='footway' else 8 if road_class=='arterial' or kind=='bridge' else 6.4)
    for r in range(7):
        for c in range(8):
            u=r*8+c
            if r<6:
                roads.append(edge(u,u+8))
            if c<7 and (c!=3 or r in [1,4]):
                roads.append(edge(u,u+1,'bridge' if c==3 else 'road',road_class='arterial' if r in [1,4] else 'local'))
    # Two genuine park diagonals; no building parcels are crossed.
    roads += [edge(40,33,'footway'),edge(13,22,'footway')]
    # Irregular outer walking corridors with designated feeders: alternative
    # loops, not a fictitious third river crossing. Node IDs 0..55 are retained.
    for points,feeders in [([(-186,-110),(-186,-24),(-183,60),(-180,124)],[8,24,32,48]),
                           ([(185,-118),(188,-30),(184,60),(181,127)],[15,31,39,55])]:
        ids=[]
        for x,y in points:
            ids.append(len(nodes));nodes.append(dict(id=len(nodes),x=x,y=y,kind='promenade_junction'))
        roads += [edge(u,v,'footway') for u,v in zip(ids,ids[1:])]
        roads += [edge(u,v,'footway') for u,v in zip(ids,feeders)]
    stations=[6,22,38,54]
    transit=[edge(a,b,'transit',54) for a,b in zip(stations,stations[1:])]
    proposal=[edge(33,38,'proposed_transit',90)]
    buildings=[]
    parks=[(0,4),(5,1)]
    blocks=[]
    for r in range(6):
        for c in [0,1,2,4,5,6]:
            x=(xs[c]+xs[c+1])/2
            y=(ys[r]+ys[r+1])/2
            park=(c,r) in parks
            hood=('Old Quay' if r<3 else 'Garden Quarter') if c<3 else ('Civic Terrace' if r<3 else 'Innovation District')
            blocks.append(dict(id=f'{c}-{r}',x=x,y=y,park=park,neighbourhood=hood))
            if park:
                continue
            for slot in range(6):
                bx=x+(-10 if slot%2==0 else 10)
                by=y+(slot//2-1)*12
                archetype=rng.choice([0,1,2] if c<3 else [2,3,4,5])
                base_height=[10,16,20,28,38,50][archetype]
                height=round(base_height*rng.uniform(.85,1.12),2)
                if hood=='Innovation District':
                    height=round(height*1.2,2)
                nearest=min(nodes,key=lambda n:(n['x']-bx)**2+(n['y']-by)**2)
                buildings.append(dict(id=f'B{len(buildings)+1:03}',x=bx,y=by,
                    width=round(rng.uniform(8,11),2),depth=round(rng.uniform(7.6,9.7),2),
                    height=height,archetype=archetype,neighbourhood=hood,node=nearest['id'],
                    nearest_node_distance_m=round(math.hypot(bx-nearest['x'],by-nearest['y']),4)))
    opportunities=[10]*len(nodes)
    families={0:[0,1,2],1:[3,4],2:[5,6],3:[7,8],4:[9,10],5:[11]}
    for i,b in enumerate(buildings):
        # Height-class draws remain unchanged; family selection uses no draws
        # from the original random stream, preserving parcel height parameters.
        choices=families[b['archetype']]
        b['family']=choices[(i//6+i)%len(choices)]
    for b in buildings:
        # Fictional destination opportunity units; no employment or monetary interpretation.
        opportunities[b['node']]+=int(b['height']/3)*(6 if b['neighbourhood']=='Innovation District' else 3)
    city=dict(seed=SEED,units='metres',crs=None,nodes=nodes,roads=roads,transit=transit,
              proposal=proposal,stations=stations,blocks=blocks,buildings=buildings,
              opportunities=opportunities,lambda_per_minute=LAMBDA,
              speed_road_m_per_minute=80,road_mode='walking',transit_speed_m_per_minute=240,
              transit_penalty_seconds_per_edge=30)
    g=make_graph(city)
    before=make_graph(city,transit=True)
    after=make_graph(city,transit=True,proposal=True)
    failed=g.copy()
    failed.remove_edge(35,36)
    source,target=40,47
    def route(graph):
        path=nx.shortest_path(graph,source,target,weight='seconds')
        return dict(nodes=path,seconds=nx.path_weight(graph,path,'seconds'),
                    metres=nx.path_weight(graph,path,'length_m'))
    degree=dict(g.degree())
    centrality=nx.betweenness_centrality(g,weight='seconds',normalized=True,endpoints=False)
    base_cost=costs(g)
    a_road=accessibility(base_cost,opportunities)
    a_before=accessibility(costs(before),opportunities)
    a_after=accessibility(costs(after),opportunities)
    alternatives=[dict(nodes=p,seconds=nx.path_weight(g,p,'seconds'),metres=nx.path_weight(g,p,'length_m'))
                  for p in islice(nx.shortest_simple_paths(g,source,target,weight='seconds'),4)]
    runs=[]
    mc=[]
    node_losses={}
    # Independent Bernoulli deletion of every road edge; transit absent in this experiment.
    for p in [.03,.08,.15]:
        loss_accum=np.zeros(len(g))
        for repetition in range(REPS):
            h=g.copy()
            removed=[(e['u'],e['v']) for e in roads if rng.random()<p]
            h.remove_edges_from(removed)
            d=costs(h)
            loss=1-accessibility(d,opportunities)/a_road
            loss_accum+=loss
            connected=nx.is_connected(h)
            pair_time=d[source,target]
            record=dict(p=p,repetition=repetition,removed=len(removed),removed_edges=[list(e) for e in removed],
                disconnected=not connected,pair_disconnected=not np.isfinite(pair_time),
                route_increase_pct=(float((pair_time/base_cost[source,target]-1)*100) if np.isfinite(pair_time) else None),
                mean_accessibility_loss_pct=float(loss.mean()*100))
            runs.append(record)
        subset=runs[-REPS:]
        count=sum(x['disconnected'] for x in subset)
        phat=count/REPS
        z=1.96
        center=(phat+z*z/(2*REPS))/(1+z*z/REPS)
        half=z*math.sqrt(phat*(1-phat)/REPS+z*z/(4*REPS*REPS))/(1+z*z/REPS)
        detours=[x['route_increase_pct'] for x in subset if x['route_increase_pct'] is not None]
        mc.append(dict(p=p,repetitions=REPS,disconnected_runs=count,probability_disconnected=phat,
            wilson_95=[center-half,center+half],
            pair_disconnected_runs=sum(x['pair_disconnected'] for x in subset),
            mean_route_increase_pct_conditional=float(np.mean(detours)),
            mean_accessibility_loss_pct=float(np.mean([x['mean_accessibility_loss_pct'] for x in subset]))))
        node_losses[str(p)]=(loss_accum/REPS*100).tolist()
    # Deterministic correctness checks independently exercise both distance implementations.
    for i in g:
        lengths=nx.single_source_dijkstra_path_length(g,i,weight='seconds')
        assert all(abs(base_cost[i,j]*60-v)<1e-8 for j,v in lengths.items())
    assert nx.is_connected(failed) and not list(nx.bridges(g))
    assert np.all(a_after>=a_before-1e-9)
    assert sum(degree.values())==2*len(roads)
    result=dict(nodes=len(nodes),road_edges=len(roads),existing_transit_edges=len(transit),
        buildings=len(buildings),blocks=len(blocks),parks=len(parks),
        degree={str(k):v for k,v in degree.items()},degree_distribution=dict(Counter(degree.values())),
        betweenness={str(k):v for k,v in centrality.items()},
        route=route(g),closure_route=route(failed),failed_edge=[35,36],source=source,target=target,
        alternative_routes=alternatives,cycle_rank=len(roads)-len(nodes)+1,
        architectural_families=len({b['family'] for b in buildings}),
        accessibility_before=a_before.tolist(),accessibility_after=a_after.tolist(),
        accessibility_gain_pct=((a_after/a_before-1)*100).tolist(),
        accessibility_mean_before=float(a_before.mean()),accessibility_mean_after=float(a_after.mean()),
        accessibility_mean_gain_pct=float((a_after.mean()/a_before.mean()-1)*100),
        road_length_total_m=sum(e['length_m'] for e in roads),height=summary([b['height'] for b in buildings]),
        road_length=summary([e['length_m'] for e in roads]),
        baseline_road_cost_minutes=summary(base_cost[np.triu_indices(len(g),1)]),
        monte_carlo=mc,node_mean_loss_pct=node_losses,simulation_repetitions_total=REPS*3,
        generation_seconds=round(time.perf_counter()-start,3),networkx_version=nx.__version__)
    write('city.json',city)
    write('analysis.json',result)
    write('disruption_trials.json',runs)
    write('calculation_validation.json',dict(status='PASS',checks=[
        f'All {len(nodes)**2:,} Floyd–Warshall costs equal independent Dijkstra costs',
        'Road graph is connected with no graph-theoretic cut edges',
        'Named crossing removal leaves a connected road graph',
        'New transit weakly increases accessibility at every origin',
        'Degree sum equals twice the road edge count'],
        city_sha256=hashlib.sha256((ROOT/'data/city.json').read_bytes()).hexdigest()))
    print(json.dumps({k:result[k] for k in ['nodes','road_edges','buildings','route','closure_route','monte_carlo','generation_seconds']},indent=2))

if __name__=='__main__':
    main()
