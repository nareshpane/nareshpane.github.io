"""Deterministic teaching constructions; no empirical model is refitted."""
from __future__ import annotations
import json
import math
import numpy as np
from common import ROOT,write_json


def calculate():
    jacobian=np.array([[.3,2.],[4.,math.cos(.3)]])
    direction=np.array([1.,-1.])
    first=np.array([.8,.15,.05]);second=np.array([.75,.1,.15])
    budget=np.array([first[0]*second[0],first[1]+first[0]*second[1],first[2]+first[0]*second[2]])
    message_b=(2+2+4)/3
    hb=math.tanh(.5+.3*message_b-.2);hd=math.tanh(.5+.3-.2)
    edge_logit=.8*hb+.4*hd+.2*.5-.6
    river=json.loads((ROOT/'outputs'/'seed-42'/'river.json').read_text(encoding='utf-8'))
    time_index=min(range(len(river['t'])),key=lambda i:abs(river['t'][i]-.8))
    concentration=river['field'][time_index][39]
    unit=concentration/.8
    row=min(range(len(river['observation_indices'])),key=lambda k:abs(river['observation_indices'][k]-time_index))
    observed=river['observations'][row][1]
    # Use the actual observation time when constructing a one-observation loss.
    obs_index=river['observation_indices'][row]
    obs_unit=river['field'][obs_index][39]/.8
    sigma=river['config']['noise_sd']
    candidates=[dict(q=q,prediction=obs_unit*q,residual=obs_unit*q-observed,
                     loss=(obs_unit*q-observed)**2/(2*sigma*sigma),
                     derivative=obs_unit*(obs_unit*q-observed)/(sigma*sigma)) for q in [.4,.8,.88]]
    demand=np.array([3,2]);capacity=np.array([3,2]);cost=np.array([[1.,4.],[2.,1.]])
    soft=np.array([[.8,.2],[.6,.4]]);flow=np.array([[3,0],[0,2]])
    load=demand@soft
    data=dict(status='validated analytical constructions; river numbers selected from saved solver output',
              ad=dict(x=[2,.3],jacobian=jacobian,jvp=jacobian@direction,vjp=direction@jacobian),
              competing=dict(first=first,second=second,budget=budget,event_bin2=.8*.1,censor_bin2=.8*.75),
              graph=dict(message_b=message_b,message_d=1.,h_b=hb,h_d=hd,edge_logit=edge_logit,edge_probability=1/(1+math.exp(-edge_logit))),
              river=dict(time=river['t'][time_index],x=.65,q=.8,concentration=concentration,
                         intensity_derivative=unit,q_changed=.88,changed_concentration=unit*.88,
                         observation_time=river['t'][obs_index],observation=observed,candidates=candidates),
              dispatch=dict(demand=demand,capacity=capacity,cost=cost,soft=soft,soft_load=load,
                            expected_cost=float(np.sum(demand[:,None]*soft*cost)/demand.sum()),
                            capacity_penalty=float(np.maximum(0,load-capacity).dot(np.maximum(0,load-capacity))/capacity.sum()),
                            integer_flow=flow,integer_cost=float(np.sum(flow*cost)/demand.sum())),
              cvar=dict(losses=[1,2,3,4],alpha=.625,eta=3,tail_mass=1.5,value=11/3),
              risk_tradeoff=dict(a_losses=[1,1,1,5],b_losses=[2.2]*4,alpha=.75,mean_a=2.,mean_b=2.2,tail_a=5.,tail_b=2.2,crossover=1/14))
    np.testing.assert_allclose(budget,[.6,.23,.17],atol=1e-15)
    np.testing.assert_allclose(load,[3.6,1.4],atol=1e-14)
    assert abs(data['dispatch']['capacity_penalty']-.072)<1e-14
    return data


if __name__=='__main__':
    write_json(ROOT/'data'/'worked-examples.json',calculate())
    print('Exported checked teaching examples; empirical outputs unchanged.')
