"""Independent, meaningful numerical checks for the complete scientific implementations."""
from __future__ import annotations
import ast
import json
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from common import ROOT, configure, environment, module, write_json


def central(function, x, h=1e-6):
    return np.array([(function(x+h*e)-function(x-h*e))/(2*h) for e in np.eye(len(x))])


def validate():
    configure(42)
    groups=[]
    for path in (ROOT/'python').glob('*.py'):
        ast.parse(path.read_text(encoding='utf-8'))
    foundation=module('00_pytorch_foundations.py')
    foundational=foundation.run()
    assert foundational['exact']['updated_loss'] == 2.53125
    groups.append('Python syntax, tensor shapes, exact scalar calculation, torch.func, weight serialization')
    survival=module('01_survival_analysis.py')
    weights=np.zeros((12,3,2))
    weights[:,0,:]=np.log(.1/.8)
    x=np.array([[1.,0.,0.],[1.,0.,0.]])
    times=np.array([3,3]); causes=np.array([1,0])
    value,_=survival.numpy_objective(weights.ravel(),x,times,causes,penalty=0)
    expected=(-2*np.log(.8)-np.log(.1)-3*np.log(.8))/2
    np.testing.assert_allclose(value,expected,atol=1e-14)
    d=survival.generate(30,101)
    probe=np.random.default_rng(5).normal(0,.3,72)
    objective=lambda w: survival.numpy_objective(w,d['x'],d['times'],d['causes'])[0]
    _,analytical=survival.numpy_objective(probe,d['x'],d['times'],d['causes'])
    np.testing.assert_allclose(central(objective,probe),analytical,atol=2e-8,rtol=1e-6)
    probability=survival.probabilities(probe.reshape(12,3,2),d['x'])
    s,f=survival.incidence(probability)
    assert np.all(np.diff(s,axis=1)<=1e-14) and np.all(np.diff(f,axis=1)>=-1e-14)
    assert np.all(probability>=0)
    np.testing.assert_allclose(probability.sum(2),1,atol=1e-14)
    np.testing.assert_allclose(survival.censor_survival([1,3,8,12]),[1,1,.75+.25*4/9,.75])
    again=survival.generate(30,101)
    assert all(np.array_equal(d[k],again[k]) for k in d)
    groups.append('Survival censor/event likelihood, independent finite gradients, probability conservation, known censor weights, reproducible generation')
    graph=module('02_financial_graph_learning.py')
    records,attr,inst=graph.generate(12,days=15,accounts=20,per_day=40)
    snapshots=graph.snapshots(records,attr,inst,15)
    poisoned=records.copy();poisoned[poisoned[:,0]>=10,3]*=1e6
    altered=graph.snapshots(poisoned,attr,inst,15)
    for left,right in zip(snapshots,altered):
        assert left['history_max_time'] < left['candidate_times'].min()
        if left['day'] < 10:
            np.testing.assert_array_equal(left['baseline'],right['baseline'])
    values=np.array([[1.,2.],[3.,4.],[8.,9.],[5.,6.]])
    src=np.array([0,0,1]);dst=np.array([1,1,2])
    expected=np.array([[0.,0.],[1.,2.],[3.,4.],[0.,0.]])
    np.testing.assert_array_equal(graph.numpy_aggregate(values,src,dst),expected)
    model=graph.GraphScorer()
    tx=torch.tensor(values)
    aggregate=torch.zeros_like(tx).index_add(0,torch.tensor(dst),tx[torch.tensor(src)])
    aggregate/=torch.bincount(torch.tensor(dst),minlength=4).clamp_min(1)[:,None]
    np.testing.assert_array_equal(aggregate.numpy(),expected)
    groups.append('Graph future-data sentinel, strict chronology, duplicate-edge/isolated-node aggregation, native operator parity')
    river=module('03_river_inverse_problem.py')
    for nx,v,D in [(31,.1,.002),(61,.35,.005),(61,.7,.015),(121,.5,.008)]:
        x,t,A,B=river.grid(nx,velocity=v,dispersion=D)
        assert A+2*B<=.75+1e-12
    x,t,field=river.numpy_simulator([.26,.8])
    np.testing.assert_allclose(field,river.torch_simulator(torch.tensor([.26,.8])).numpy(),atol=1e-13)
    assert field.min()>=-1e-14 and np.all(field[:,[0,-1]]==0)
    phi=torch.tensor([.32,.7],requires_grad=True)
    assert torch.autograd.gradcheck(lambda p: river.torch_simulator(p)[[20,40,60]][:,[27,39,51]].sum(),(phi,),eps=1e-6,atol=1e-6,rtol=1e-5)
    refinement=river.manufactured_refinement()
    assert refinement[0]['relative_l2']>refinement[1]['relative_l2']>refinement[2]['relative_l2']
    groups.append('River combined stability, boundary/positivity, forward parity, independent torch gradcheck, manufactured grid refinement')
    allocation=module('04_resource_allocation.py')
    demand=allocation.scenarios(5,25)
    probe=np.r_[(-allocation.COST).ravel(),2.5]+np.random.default_rng(11).normal(0,.1,25)
    value,gradient=allocation.numpy_objective(probe,demand)
    np.testing.assert_allclose(central(lambda x: allocation.numpy_objective(x,demand)[0],probe),gradient,atol=1e-7,rtol=1e-5)
    leaf=torch.tensor(probe,requires_grad=True)
    loss=allocation.torch_objective(leaf,torch.tensor(demand,dtype=torch.float64))
    np.testing.assert_allclose(torch.autograd.grad(loss,leaf)[0].numpy(),gradient,atol=1e-12)
    losses=np.array([1.,2.,3.,4.])
    assert abs(allocation.empirical_cvar(losses,.625)-11/3)<1e-14
    eta=3.
    assert abs(eta+np.maximum(losses-eta,0).mean()/(1-.625)-11/3)<1e-14
    for d in demand[:12]:
        greedy=allocation.feasible_dispatch(d,allocation.COST)
        exact=allocation.exact_dispatch(d)
        assert np.array_equal(greedy.sum(1),d) and np.all(greedy.sum(0)<=allocation.CAPACITY)
        assert np.sum(exact*allocation.COST)<=np.sum(greedy*allocation.COST)+1e-9
    try:
        allocation.feasible_dispatch(np.full(8,100),allocation.COST)
        raise AssertionError('Insufficient capacity was silently accepted')
    except ValueError:
        pass
    groups.append('Allocation independent gradients, fractional-tail CVaR, demand/capacity integrality, exact-cost bound, infeasible-demand rejection')
    def finite(value):
        if isinstance(value,float):
            assert np.isfinite(value)
        elif isinstance(value,dict):
            for x in value.values():finite(x)
        elif isinstance(value,list):
            for x in value:finite(x)
    for path in (ROOT/'outputs').rglob('*.json'):
        finite(json.loads(path.read_text(encoding='utf-8')))
    for path in (ROOT/'outputs'/'seed-42').glob('*-data.npz'):
        with np.load(path,allow_pickle=False) as arrays:
            assert all(np.all(np.isfinite(v)) for v in arrays.values())
    rep=json.loads((ROOT/'outputs'/'replications.json').read_text())
    assert rep['seeds']==[42,43,44] and all(not row['quick'] for row in rep['results'])
    groups.append('Finite JSON/NPZ outputs, retained substantive run statuses, independent replication records')
    result=dict(status='passed',checks=groups,count=len(groups),environment=environment())
    write_json(ROOT/'data'/'python-validation.json',result)
    return result


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        report=validate()
    print('Passed',report['count'],'numerical verification groups')
