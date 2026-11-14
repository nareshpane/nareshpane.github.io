"""Risk-aware synthetic dispatch: same relaxed objective, feasible integer realization, MILP oracle."""
from __future__ import annotations
import time
import numpy as np
from scipy.optimize import minimize, milp, LinearConstraint, Bounds
from scipy.special import softmax
import torch
from common import cli, configure, environment, plot_lines, write_json

DISTRICTS = np.array([[.1,.2],[.3,.15],[.55,.12],[.83,.22],[.15,.65],[.4,.8],[.65,.68],[.9,.8]])
STATIONS = np.array([[.18,.35],[.52,.48],[.85,.55]])
COST = 1+8*np.linalg.norm(DISTRICTS[:, None, :]-STATIONS[None, :, :], axis=2)
CAPACITY = np.array([12,10,11])


def scenarios(seed=42, n=300):
    rng = np.random.default_rng(seed)
    base = np.array([.12,.08,.10,.12,.12,.13,.15,.18])
    result = []
    for _ in range(n):
        hotspot = rng.integers(0, len(base))
        weights = base.copy()
        weights[hotspot] *= rng.uniform(1, 4)
        total = rng.integers(18, 32)
        result.append(rng.multinomial(total, weights/weights.sum()))
    return np.array(result)


def numpy_objective(vector, demand, risk=.7, fairness=.25, capacity_weight=1.5, alpha=.9):
    assignment = softmax(vector[:-1].reshape(8, 3), axis=1)
    eta = vector[-1]
    district_cost = np.sum(assignment*COST, axis=1)
    losses = demand@district_cost/demand.sum(1)
    load = demand@assignment
    excess = np.maximum(0, load-CAPACITY)
    cvar = eta+np.maximum(0, losses-eta).mean()/(1-alpha)
    disparity = np.mean((district_cost-district_cost.mean())**2)
    capacity_penalty = np.mean(np.sum(excess**2, axis=1))/CAPACITY.sum()
    value = losses.mean()+risk*cvar+fairness*disparity+capacity_weight*capacity_penalty
    weights = (1+risk*(losses > eta)/(1-alpha))/len(demand)
    gradient_a = np.einsum('s,si,ij->ij', weights/demand.sum(1), demand, COST)
    gradient_a += fairness*2/8*(district_cost-district_cost.mean())[:, None]*COST
    gradient_a += capacity_weight*2*np.einsum('si,sj->ij', demand, excess)/len(demand)/CAPACITY.sum()
    gradient_logits = assignment*(gradient_a-np.sum(gradient_a*assignment, axis=1, keepdims=True))
    gradient_eta = risk*(1-np.mean(losses > eta)/(1-alpha))
    return float(value), np.r_[gradient_logits.ravel(), gradient_eta]


def torch_objective(vector, demand, risk=.7, fairness=.25, capacity_weight=1.5, alpha=.9):
    assignment = vector[:-1].reshape(8, 3).softmax(1)
    eta = vector[-1]
    district_cost = (assignment*torch.tensor(COST)).sum(1)
    losses = demand@district_cost/demand.sum(1)
    load = demand@assignment
    cvar = eta+torch.relu(losses-eta).mean()/(1-alpha)
    disparity = ((district_cost-district_cost.mean())**2).mean()
    capacity_penalty = (torch.relu(load-torch.tensor(CAPACITY))**2).sum(1).mean()/CAPACITY.sum()
    return losses.mean()+risk*cvar+fairness*disparity+capacity_weight*capacity_penalty


def empirical_cvar(losses, alpha=.9):
    # Fractional weighting handles a noninteger number of tail observations and ties.
    ordered = np.sort(losses)[::-1]
    tail = (1-alpha)*len(ordered)
    whole = int(np.floor(tail+1e-12))
    remainder = max(0, tail-whole)
    return float((ordered[:whole].sum()+(remainder*ordered[whole] if whole < len(ordered) else 0))/tail)


def feasible_dispatch(demand, scores, capacity=CAPACITY):
    remaining = np.asarray(capacity, dtype=int).copy()
    flow = np.zeros((len(demand), len(remaining)), dtype=int)
    # Largest gap between best and second-best station gets first choice; stable tie order.
    sorted_scores = np.sort(scores, axis=1)
    priority = np.argsort(-(sorted_scores[:, 1]-sorted_scores[:, 0]), kind='stable')
    for district in priority:
        left = int(demand[district])
        for station in np.argsort(scores[district], kind='stable'):
            take = min(left, remaining[station])
            flow[district, station] += take
            remaining[station] -= take
            left -= take
        if left:
            raise ValueError('Insufficient total capacity: no fully served assignment exists')
    assert np.array_equal(flow.sum(1), demand)
    assert np.all(flow.sum(0) <= capacity)
    return flow


def exact_dispatch(demand):
    # Scenario-specific integer minimum-cost transportation; richer optimization than a static ranking.
    rows = np.zeros((8, 24))
    columns = np.zeros((3, 24))
    for i in range(8):
        rows[i, i*3:(i+1)*3] = 1
    for j in range(3):
        columns[j, j::3] = 1
    constraints = [LinearConstraint(rows, demand, demand), LinearConstraint(columns, 0, CAPACITY)]
    result = milp(COST.ravel(), integrality=np.ones(24), bounds=Bounds(0, np.inf), constraints=constraints)
    assert result.success
    flow = np.rint(result.x.reshape(8, 3)).astype(int)
    assert np.array_equal(flow.sum(1), demand) and np.all(flow.sum(0) <= CAPACITY)
    return flow


def evaluate(demand, policy, exact=False):
    losses, utilizations, disparities = [], [], []
    for d in demand:
        flow = exact_dispatch(d) if exact else feasible_dispatch(d, -np.log(policy+1e-15))
        losses.append(float(np.sum(flow*COST)/d.sum()))
        utilizations.append(flow.sum(0)/CAPACITY)
        district = (flow*COST).sum(1)/np.maximum(1, d)
        disparities.append(float(np.std(district[d > 0])))
    losses = np.array(losses)
    relaxed_load = demand@policy
    return dict(mean_cost=float(losses.mean()), p90_cost=float(np.quantile(losses, .9)), cvar90=empirical_cvar(losses),
                mean_district_cost_sd=float(np.mean(disparities)), capacity_violations_after_projection=0,
                relaxed_overload_frequency=None if exact else float(np.mean(np.any(relaxed_load > CAPACITY, axis=1))),
                utilization=np.mean(utilizations, axis=0), losses=losses)


def run(seed=42, quick=False, output=None):
    configure(seed)
    train, test = scenarios(seed, 60 if quick else 300), scenarios(seed+1, 40 if quick else 160)
    initial = np.r_[(-COST).ravel(), 2.5]
    probe = initial+np.random.default_rng(seed).normal(0, .1, len(initial))
    value, analytical = numpy_objective(probe, train)
    leaf = torch.tensor(probe, requires_grad=True)
    loss = torch_objective(leaf, torch.tensor(train, dtype=torch.float64))
    gradient = torch.autograd.grad(loss, leaf)[0].numpy()
    np.testing.assert_allclose(analytical, gradient, atol=2e-12, rtol=1e-10)
    start = time.perf_counter()
    baseline = minimize(numpy_objective, initial, args=(train,), jac=True, method='L-BFGS-B', options={'maxiter': 350, 'ftol': 1e-11})
    conventional_seconds = time.perf_counter()-start
    vector = torch.nn.Parameter(torch.tensor(initial))
    optimizer = torch.optim.Adam([vector], lr=.06)
    demand_tensor = torch.tensor(train, dtype=torch.float64)
    history, best, best_value = [], vector.detach().clone(), np.inf
    start = time.perf_counter()
    steps = 150 if quick else 1200
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        objective = torch_objective(vector, demand_tensor)
        if objective.item() < best_value:
            best, best_value = vector.detach().clone(), objective.item()
        history.append([float(objective.detach()), *vector.detach().numpy().tolist()])
        objective.backward()
        optimizer.step()
    neural_seconds = time.perf_counter()-start
    learned = softmax(best.numpy()[:-1].reshape(8, 3), axis=1)
    classical = softmax(baseline.x[:-1].reshape(8, 3), axis=1)
    nearest = np.eye(3)[COST.argmin(1)]
    # Greedy scores use cost directly; equivalent to exp(-cost) preference.
    greedy = softmax(-COST, axis=1)
    policies = dict(nearest=nearest, greedy=greedy, scipy_relaxed=classical, pytorch_relaxed=learned)
    evaluation = {name: evaluate(test, p) for name, p in policies.items()}
    # The exact per-scenario minimum cost ignores CVaR/fairness policy objective.
    evaluation['milp_cost_oracle'] = evaluate(test, greedy, exact=True)
    for name, values in evaluation.items():
        assert np.all(np.isfinite(values['losses']))
    result = dict(status='executed conventional optimization and PyTorch CPU', seed=seed, quick=quick, environment=environment(),
                  counts=dict(train=len(train), test=len(test)), config=dict(alpha=.9, risk=.7, fairness=.25, capacity_weight=1.5),
                  districts=DISTRICTS, stations=STATIONS, cost=COST, capacity=CAPACITY,
                  train_scenarios=train, test_scenarios=test, policies=policies, evaluation=evaluation,
                  equivalent_gradient_max_error=float(np.max(abs(gradient-analytical))),
                  scipy=dict(objective=float(baseline.fun), seconds=conventional_seconds, success=bool(baseline.success),
                             iterations=int(baseline.nit), message=str(baseline.message)),
                  pytorch=dict(objective=best_value, seconds=neural_seconds, steps=steps), history=history)
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
        write_json(output/'allocation.json', result)
        np.savez_compressed(output/'allocation-data.npz', train=train, test=test, cost=COST, capacity=CAPACITY)
        render_diagnostics(result, output)
    return result


def render_diagnostics(result, output):
    """Regenerate readable SVG diagnostics directly from genuine stored numerical results."""
    history=np.asarray(result['history'])
    evaluation=result['evaluation']
    count=result['counts']['test']
    labels=dict(greedy='Cost-ranked greedy',scipy_relaxed='SciPy policy',
                pytorch_relaxed='PyTorch policy',milp_cost_oracle='Exact cost comparator')
    plot_lines(output/'allocation-training.svg', np.arange(len(history)), {'Adam objective': history[:, 0]},
               'Optimization step', 'Relaxed objective (proxy units)', 'Soft policy optimization: penalties are not hard constraints')
    plot_lines(output/'allocation-test.svg', np.arange(1,count+1)/count,
               {labels[name]: np.sort(v['losses']) for name,v in evaluation.items() if name!='nearest'},
               'Empirical quantile', 'Cost / demand (synthetic proxy)', 'Held-out feasible integer dispatch, identical demand scenarios')


if __name__ == '__main__':
    args = cli(__doc__)
    result = run(args.seed, args.quick, args.output)
    print('Feasible test mean costs:', {k: v['mean_cost'] for k, v in result['evaluation'].items()})
