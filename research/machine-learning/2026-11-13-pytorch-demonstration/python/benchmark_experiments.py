"""Repeated equivalent-work CPU benchmarks. No dataset generation or file IO is timed."""
from __future__ import annotations
import time
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from common import ROOT, cli, configure, environment, module, write_json


def measure(function, repetitions=31, inner=3):
    start = time.perf_counter()
    for _ in range(3):
        function()
    warmup = time.perf_counter()-start
    times = []
    for _ in range(repetitions):
        start = time.perf_counter()
        for _ in range(inner):
            function()
        times.append((time.perf_counter()-start)*1000/inner)
    return dict(median_ms=float(np.median(times)), q25_ms=float(np.quantile(times, .25)),
                q75_ms=float(np.quantile(times, .75)), raw_ms=times, repetitions=repetitions,
                calls_per_repetition=inner, warmup_calls=3, warmup_total_seconds=warmup)


def tasks(seed=42):
    rng = configure(seed)
    survival = module('01_survival_analysis.py')
    graph = module('02_financial_graph_learning.py')
    river = module('03_river_inverse_problem.py')
    allocation = module('04_resource_allocation.py')
    data = survival.generate(400, seed)
    x, times, causes = (data[k] for k in ('x', 'times', 'causes'))
    initial = rng.normal(0, .2, 72)
    tx, tt, tc = torch.tensor(x), torch.tensor(times), torch.tensor(causes)
    def sn():
        value, grad = survival.numpy_objective(initial, x, times, causes)
        return np.r_[value, grad]
    def st():
        w = torch.tensor(initial.reshape(12, 3, 2), requires_grad=True)
        value = survival.torch_objective(w, tx, tt, tc)
        return np.r_[value.item(), torch.autograd.grad(value, w)[0].numpy().ravel()]
    features = rng.normal(size=(200, 16))
    src, dst = rng.integers(0, 200, (2, 2000))
    tf, ts, td = torch.tensor(features), torch.tensor(src), torch.tensor(dst)
    degree = torch.bincount(td, minlength=200).clamp_min(1)[:, None]
    def gn():
        return graph.numpy_aggregate(features, src, dst)
    def gt():
        return (torch.zeros_like(tf).index_add(0, td, tf[ts])/degree).numpy()
    phi = np.array([.32, .7])
    indices, sensors = np.array([20, 40, 60]), np.array([27, 39, 51])
    truth = river.numpy_simulator([.26, .8])[2][indices[:, None], sensors]
    def river_loss(p):
        return np.mean((river.numpy_simulator(p)[2][indices[:, None], sensors]-truth)**2)
    def rn():
        return river.finite_gradient(river_loss, phi)
    def rt():
        p = torch.tensor(phi, requires_grad=True)
        prediction = river.torch_simulator(p)[indices[:, None], sensors]
        loss = ((prediction-torch.tensor(truth))**2).mean()
        return torch.autograd.grad(loss, p)[0].numpy()
    demand = allocation.scenarios(seed, 300)
    vector = np.r_[(-allocation.COST).ravel(), 2.5]
    tdem = torch.tensor(demand, dtype=torch.float64)
    def an():
        value, grad = allocation.numpy_objective(vector, demand)
        return np.r_[value, grad]
    def at():
        p = torch.tensor(vector, requires_grad=True)
        loss = allocation.torch_objective(p, tdem)
        return np.r_[loss.item(), torch.autograd.grad(loss, p)[0].numpy()]
    return [dict(id='survival', question='Loss + 72-parameter gradient, N=400, H=12', conventional=sn, pytorch=st,
                 derivative='analytical versus autograd; same objective', shape='400 × 12 × 3', inner=3),
            dict(id='graph', question='Incoming-neighbor mean, 200 nodes, 2,000 edges, 16 features', conventional=gn, pytorch=gt,
                 derivative='no differentiation: identical aggregation', shape='200 × 16; 2,000 edges', inner=7),
            dict(id='river', question='Two source derivatives through 61-point stable simulator', conventional=rn, pytorch=rt,
                 derivative='central differences (4 forward solves) versus one forward/backward; derivative algorithms differ', shape='61 spatial points', inner=1),
            dict(id='allocation', question='Composite loss + 25 derivatives over 300 demand scenarios', conventional=an, pytorch=at,
                 derivative='analytical subgradient versus autograd away from kinks', shape='300 × 8 × 3', inner=5)]


def run(seed=42, repetitions=31):
    results = []
    with threadpool_limits(limits=1):
        configure(seed)
        metadata = environment()
        for task in tasks(seed):
            left, right = task['conventional'](), task['pytorch']()
            error = float(np.max(abs(left-right)))
            relative = float(np.linalg.norm(left-right)/max(1, np.linalg.norm(left)))
            assert relative < 1e-6
            conventional = measure(task['conventional'], repetitions, task['inner'])
            pytorch = measure(task['pytorch'], repetitions, task['inner'])
            result = {k: v for k, v in task.items() if k not in ('conventional', 'pytorch', 'inner')}
            result.update(conventional=conventional, pytorch=pytorch,
                          ratio_conventional_over_pytorch=conventional['median_ms']/pytorch['median_ms'],
                          max_absolute_error=error, relative_error=relative,
                          cpu_tensor_peak_bytes=None,
                          memory_note='No portable CPU tensor allocator peak API used; Python tracemalloc would omit native tensor storage.')
            results.append(result)
    return dict(status='executed repeated CPU benchmarks', seed=seed, environment=metadata,
                protocol='Sequential; float64; one torch/BLAS thread; inputs resident on CPU; no disk IO; three warmups; paired numerical checks before timing.',
                cases=results, gpu=dict(status='not executed', reason='CPU wheel; no CUDA device benchmark'),
                caveat='Microbenchmarks measure these small inputs, not all training, deployment, or large-device behavior.')


if __name__ == '__main__':
    args = cli(__doc__)
    result = run(args.seed, 7 if args.quick else 31)
    write_json(args.output/'benchmark-results.json', result)
    for case in result['cases']:
        print(case['id'], 'conventional/PyTorch median ratio:', case['ratio_conventional_over_pytorch'])
