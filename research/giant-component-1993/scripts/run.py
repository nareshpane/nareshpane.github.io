"""CLI: python -m scripts.run --mode research --trials 1000 --resume."""
import argparse
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import numba
from .experiments import batch
from .statistics import NS
from .export import atomic_json, export


def seed_for(base, n, model, experiment, index):
    # An independent SeedSequence child keyed by scientific cell and trial index.
    # The compiled MT19937 accepts uint32 seeds. Record the scheme and check for
    # collisions across the entire requested run before any simulation begins.
    return int(np.random.SeedSequence([base, n, int(model == 'simple'),
                                      int(experiment == 'trajectory'), index]).generate_state(1)[0])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['quick', 'research'], default='quick')
    p.add_argument('--ns', nargs='+', type=int)
    p.add_argument('--trials', type=int)
    p.add_argument('--models', nargs='+', choices=['simple', 'multigraph'], default=['simple', 'multigraph'])
    p.add_argument('--experiments', nargs='+', choices=['snapshot', 'trajectory'], default=['snapshot', 'trajectory'])
    p.add_argument('--seed', type=int, default=20261012)
    p.add_argument('--batch-size', type=int, default=100)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--out', type=Path, default=Path(__file__).resolve().parents[1] / 'data')
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    ns = sorted(set(a.ns or (NS if a.mode == 'research' else [50, 100, 500])))
    trials = a.trials if a.trials is not None else (1000 if a.mode == 'research' else 20)
    if min(ns) < 2 or trials < 1 or a.batch_size < 1 or a.seed < 0:
        p.error('n >= 2; trials and batch size positive; seed nonnegative')
    numba.set_num_threads(a.threads)
    a.out.mkdir(parents=True, exist_ok=True)
    config = dict(ns=ns, models=a.models, experiments=a.experiments, seed=a.seed)
    checkpoint = a.out / 'checkpoint.json'
    if checkpoint.exists():
        if not a.resume:
            p.error('Checkpoint exists; use --resume or a different output directory')
        state = json.loads(checkpoint.read_text(encoding='utf8'))
        if state['metadata']['configuration'] != config:
            p.error('Resume configuration mismatch')
        if any(r['trials'] > trials for r in state['rows']):
            p.error('Cannot reduce completed trial counts')
    else:
        state = {'metadata': {'configuration': config, 'started_utc': datetime.now(timezone.utc).isoformat(),
                 'python': platform.python_version(), 'numpy': np.__version__, 'numba': numba.__version__,
                 'platform': platform.platform(), 'threads': a.threads,
                 'rng': 'Numba MT19937, SeedSequence([base,n,simple,trajectory,trial_index]); uint32 seed',
                 'stopping': 'snapshot: floor(n/2); trajectory: first failure or connectivity',
                 'interval': '95% Wilson; z=1.959963984540054'}, 'rows': []}
        for experiment in a.experiments:
            for model in a.models:
                for n in ns:
                    state['rows'].append(dict(n=n, model=model, experiment=experiment,
                         trials=0, successes=0, runtime_seconds=0., accepted_edges_total=0,
                         seed_provenance=f'SeedSequence({a.seed},{n},{int(model=="simple")},{int(experiment=="trajectory")},j)'))
    all_seeds = [seed_for(a.seed, r['n'], r['model'], r['experiment'], j)
                 for r in state['rows'] for j in range(trials)]
    if len(set(all_seeds)) != len(all_seeds):
        p.error('uint32 seed collision; choose a different base seed in a fresh output directory')
    state['metadata']['seed_count'] = len(all_seeds)
    state['metadata']['seed_sha256'] = hashlib.sha256(np.array(all_seeds, '<u4').tobytes()).hexdigest()
    state['metadata']['seed_collision_check'] = 'passed'
    state['metadata']['complete'] = False
    for r in state['rows']:
        while r['trials'] < trials:
            end = min(trials, r['trials'] + a.batch_size)
            seeds = np.array([seed_for(a.seed, r['n'], r['model'], r['experiment'], j)
                              for j in range(r['trials'], end)], np.uint32)
            start = time.perf_counter()
            successes, accepted = batch(r['n'], r['model'] == 'simple', r['experiment'] == 'trajectory', seeds)
            r['runtime_seconds'] += time.perf_counter() - start
            r['trials'] = end
            r['successes'] += int(successes.sum())
            r['accepted_edges_total'] += int(accepted.sum())
            state['metadata']['updated_utc'] = datetime.now(timezone.utc).isoformat()
            state['metadata']['target_trials'] = trials
            atomic_json(checkpoint, state)
            export(a.out, state)
        print(r['experiment'], r['model'], r['n'], r['successes'], '/', r['trials'],
              'seconds', round(r['runtime_seconds'], 2), flush=True)
    state['metadata']['complete'] = True
    atomic_json(checkpoint, state)
    export(a.out, state)


if __name__ == '__main__':
    main()
