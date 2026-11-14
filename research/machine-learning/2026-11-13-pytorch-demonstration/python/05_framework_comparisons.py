"""Eager versus torch.compile, with initialization and first-call capture reported separately."""
from __future__ import annotations
import os
import tempfile
import time
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from benchmark_experiments import measure
from common import cli, configure, environment, write_json


def run(seed=42, quick=False):
    configure(seed)
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(256, 32))
    tx = torch.tensor(x)
    def numpy_function():
        return np.sum(np.sin(x)*x, axis=1)
    def torch_function(value):
        return (value.sin()*value).sum(1)
    np.testing.assert_allclose(numpy_function(), torch_function(tx).numpy(), atol=2e-14)
    result = dict(status='executed compilation compatibility probe', environment=environment(), shape=list(x.shape),
                  numpy=measure(numpy_function), eager=measure(lambda: torch_function(tx)), compiled=[])
    with tempfile.TemporaryDirectory(prefix='pytorch-monograph-compile-', ignore_cleanup_errors=True) as cache:
        os.environ['TORCHINDUCTOR_CACHE_DIR'] = cache
        for backend in (['aot_eager'] if quick else ['inductor', 'aot_eager']):
            try:
                begin = time.perf_counter()
                compiled = torch.compile(torch_function, backend=backend, fullgraph=True)
                wrapper_seconds = time.perf_counter()-begin
                begin = time.perf_counter()
                actual = compiled(tx)
                first_seconds = time.perf_counter()-begin
                np.testing.assert_allclose(actual.numpy(), numpy_function(), atol=3e-14, rtol=1e-12)
                measured = measure(lambda: compiled(tx))
                result['compiled'].append(dict(backend=backend, status='executed', wrapper_seconds=wrapper_seconds,
                                               first_call_seconds=first_seconds, steady_state=measured,
                                               note='aot_eager exercises capture/autodiff plumbing without an optimizing code-generation backend' if backend == 'aot_eager' else 'Inductor code generation'))
            except Exception as error:
                result['compiled'].append(dict(backend=backend, status='failed compatibility probe',
                                               error_type=type(error).__name__, reason=str(error)[:900]))
    return result


if __name__ == '__main__':
    args = cli(__doc__)
    with threadpool_limits(limits=1):
        result = run(args.seed, args.quick)
    write_json(args.output/'compilation.json', result)
    print([(c['backend'], c['status']) for c in result['compiled']])
