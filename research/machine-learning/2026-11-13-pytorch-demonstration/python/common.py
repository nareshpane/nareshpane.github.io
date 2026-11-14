"""Shared experiment infrastructure. All observations in this project are synthetic."""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time
import numpy as np
import scipy

sys.dont_write_bytecode = True
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'codex-pytorch-mpl-20261113'))
ROOT = Path(__file__).resolve().parents[1]


def module(filename):
    spec = importlib.util.spec_from_file_location(filename.replace('.', '_'), ROOT / 'python' / filename)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def configure(seed=42):
    import torch
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    torch.use_deterministic_algorithms(True)
    return np.random.default_rng(seed)


def environment():
    import torch
    from threadpoolctl import threadpool_info
    return dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                pytorch=torch.__version__, platform=platform.platform(), processor=platform.processor(),
                logical_cpus=os.cpu_count(), device='cpu', dtype='float64',
                torch_threads=torch.get_num_threads(), cuda_available=torch.cuda.is_available(),
                threadpools=threadpool_info())


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    def convert(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, Path):
            return str(x)
        raise TypeError(type(x).__name__)
    path.write_text(json.dumps(obj, default=convert, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def cli(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--quick', action='store_true', help='Pipeline smoke run; not convergence evidence')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs')
    return parser.parse_args()


def plot_lines(path, x, series, xlabel, ylabel, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'svg.fonttype': 'none', 'font.family': 'DejaVu Sans', 'font.size': 10})
    fig, ax = plt.subplots(figsize=(8.8, 3.5), layout='constrained')
    for (label, y), color in zip(series.items(), ['#477DA5', '#B57A60', '#8D78A5', '#5A887B']):
        ax.plot(x, y, label=label, color=color, linewidth=2)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(alpha=.18)
    ax.legend(frameon=False)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, metadata={'Date': None})
    plt.close(fig)


def timed(function):
    start = time.perf_counter()
    result = function()
    return result, time.perf_counter() - start
