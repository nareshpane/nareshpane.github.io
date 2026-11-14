"""Independent checks for teaching additions and preservation of scientific artifacts."""
from __future__ import annotations
import hashlib
import json
import re
import numpy as np
import torch
from common import ROOT, configure, module, write_json
from refinement_examples import calculate


def validate():
    configure(42)
    baseline=json.loads((ROOT/'data/refinement-baseline.json').read_text(encoding='utf-8'))
    counts={}
    for key in ['protected_sha256','outside_scope_sha256']:
        for relative,expected in baseline[key].items():
            path=ROOT.parent/relative
            assert hashlib.sha256(path.read_bytes()).hexdigest()==expected,relative
        counts[key]=len(baseline[key])
    html=(ROOT.parent/(ROOT.name+'.html')).read_text(encoding='utf-8')
    for sid in baseline['preserved_subsection_ids']:
        assert f'id="{sid}"' in html,sid
    assert len(re.findall(r'<details class="code-panel"',html))==26
    groups=['32 scientific artifacts and 56 outside-scope files byte-identical; 61 subsection anchors and 26 canonical source panels retained']
    d=calculate()
    x=torch.tensor([2.,.3])
    def f(v):
        return torch.stack((v[0]*v[1],v[0]**2+torch.sin(v[1])))
    v=torch.tensor([1.,-1.])
    _,jvp=torch.func.jvp(f,(x,),(v,))
    _,pullback=torch.func.vjp(f,x)
    np.testing.assert_allclose(torch.func.jacrev(f)(x),d['ad']['jacobian'],atol=1e-14)
    np.testing.assert_allclose(jvp,d['ad']['jvp'],atol=1e-14)
    np.testing.assert_allclose(pullback(v)[0],d['ad']['vjp'],atol=1e-14)
    groups.append('Analytical Jacobian, JVP, and VJP agree with executed torch.func products')
    survival=module('01_survival_analysis.py')
    probabilities=np.array([[d['competing']['first'],d['competing']['second']]])
    s,incidence=survival.incidence(probabilities)
    np.testing.assert_allclose(np.r_[s[0,-1],incidence[0,-1]],d['competing']['budget'],atol=1e-14)
    np.testing.assert_allclose(s+incidence.sum(-1),1,atol=1e-14)
    # Enumerate first/second-bin outcomes independently of the incidence reduction.
    np.testing.assert_allclose(probabilities[0,0,0]*probabilities[0,1,1],d['competing']['event_bin2'])
    np.testing.assert_allclose(np.prod(probabilities[0,:,0]),d['competing']['censor_bin2'])
    groups.append('Competing-risk budgets and event versus end-of-bin censor probabilities')
    features=torch.tensor([2.,1.,4.,1.])
    source=torch.tensor([0,0,2,1]);target=torch.tensor([1,1,1,3])
    messages=torch.zeros(4).index_add(0,target,features[source])
    degree=torch.zeros(4).index_add(0,target,torch.ones(4))
    mean=messages/degree.clamp_min(1)
    h=torch.tanh(.5*features+.3*mean-.2)
    np.testing.assert_allclose([mean[1],mean[3]],[d['graph']['message_b'],d['graph']['message_d']])
    np.testing.assert_allclose([h[1],h[3]],[d['graph']['h_b'],d['graph']['h_d']])
    groups.append('Native index_add reproduces synchronous duplicate-edge message example')
    river=module('03_river_inverse_problem.py')
    x,t,unit=river.numpy_simulator([.26,1.])
    _,_,changed=river.numpy_simulator([.26,.88])
    np.testing.assert_allclose(changed,.88*unit,atol=2e-14)
    ti=int(np.argmin(abs(t-d['river']['time'])))
    np.testing.assert_allclose(unit[ti,39],d['river']['intensity_derivative'],atol=2e-14)
    phi=torch.tensor([.26,.8],requires_grad=True)
    torch_field=river.torch_simulator(phi)
    derivative=torch.autograd.grad(torch_field[ti,39],phi)[0][1]
    np.testing.assert_allclose(derivative.detach(),unit[ti,39],atol=2e-14)
    saved=json.loads((ROOT/'outputs/seed-42/river.json').read_text())
    oi=int(np.argmin(abs(t-d['river']['observation_time'])))
    observed=d['river']['observation'];sigma=saved['config']['noise_sd']
    for row in d['river']['candidates']:
        q=torch.tensor(row['q'],requires_grad=True)
        loss=(q*unit[oi,39]-observed)**2/(2*sigma**2)
        gradient=torch.autograd.grad(loss,q)[0]
        np.testing.assert_allclose([loss.detach(),gradient.detach()],[row['loss'],row['derivative']],atol=1e-10)
    groups.append('River intensity linearity, actual-observation loss, and derivatives checked by unchanged solver/autograd')
    allocation=module('04_resource_allocation.py')
    case=d['dispatch'];soft=np.asarray(case['soft']);demand=np.asarray(case['demand']);cost=np.asarray(case['cost'])
    np.testing.assert_allclose(demand@soft,[3.6,1.4])
    np.testing.assert_allclose((soft*cost).sum(1),[1.6,1.6])
    flow=np.asarray(case['integer_flow'])
    np.testing.assert_array_equal(flow.sum(1),demand)
    np.testing.assert_array_equal(flow.sum(0),case['capacity'])
    np.testing.assert_allclose((flow*cost).sum()/demand.sum(),case['integer_cost'])
    np.testing.assert_allclose(allocation.empirical_cvar(d['cvar']['losses'],d['cvar']['alpha']),d['cvar']['value'])
    a=d['risk_tradeoff']
    np.testing.assert_allclose(allocation.empirical_cvar(a['a_losses'],a['alpha']),a['tail_a'])
    assert a['mean_a']<a['mean_b'] and a['mean_a']+.1*a['tail_a']>a['mean_b']+.1*a['tail_b']
    groups.append('Soft overload versus integer feasibility; fractional CVaR and analytical risk ranking')
    actual=json.loads((ROOT/'outputs/seed-42/allocation.json').read_text())
    golden={}
    for name,policy in actual['policies'].items():
        scores=-np.log(np.asarray(policy)+1e-15)
        flows=[allocation.feasible_dispatch(np.asarray(row),scores) for row in actual['test_scenarios']]
        losses=[float(np.sum(f*allocation.COST)/np.sum(row)) for f,row in zip(flows,actual['test_scenarios'])]
        np.testing.assert_allclose(losses,actual['evaluation'][name]['losses'],atol=1e-14)
        golden[name]=dict(flows=flows,losses=losses)
    write_json(ROOT/'data/refinement-dispatch-cases.json',dict(origin='Unchanged canonical Python feasible_dispatch, all 160 held-out demands and four stored policies',policies=golden))
    write_json(ROOT/'data/refinement-validation.json',dict(status='passed',groups=groups,preservation=counts,canonical_dispatch_cases=640))
    print(json.dumps(dict(status='passed',groups=len(groups),protected_files=sum(counts.values()),canonical_dispatch_cases=640)))


if __name__=='__main__':
    validate()
