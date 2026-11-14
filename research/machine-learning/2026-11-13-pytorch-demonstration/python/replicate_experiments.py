"""Three independent simulation seeds. Retain compact evaluations, not duplicate checkpoints."""
from common import ROOT, cli, module, write_json
from threadpoolctl import threadpool_limits
import json


def summarize(domain, result):
    keys = {'survival': ['n', 'conventional', 'neural', 'gradient_max_error', 'equivalent', 'neural_epochs'],
            'financial-network': ['counts', 'conventional', 'neural', 'new_accounts', 'aggregation_max_error', 'neural_epochs'],
            'river': ['truth', 'parameter_absolute_error', 'heldout_rmse', 'gradient', 'refinement'],
            'allocation': ['counts', 'equivalent_gradient_max_error', 'scipy', 'pytorch', 'evaluation']}[domain]
    result = {k: result[k] for k in keys}
    if domain == 'allocation':
        result['evaluation'] = {name: {k: v for k, v in metric.items() if k != 'losses'} for name, metric in result['evaluation'].items()}
    return result


if __name__ == '__main__':
    args = cli(__doc__)
    studies = [('survival', '01_survival_analysis.py'), ('financial-network', '02_financial_graph_learning.py'),
               ('river', '03_river_inverse_problem.py'), ('allocation', '04_resource_allocation.py')]
    rows = []
    with threadpool_limits(limits=1):
        for seed in [42, 43, 44]:
            record = dict(seed=seed, quick=args.quick, domains={})
            for domain, filename in studies:
                print(seed, domain, flush=True)
                existing = args.output/'seed-42'/f'{domain}.json'
                if seed == 42 and existing.exists() and not args.quick:
                    result = json.loads(existing.read_text(encoding='utf-8'))
                    # Recompute allocation to include the final null-oracle metadata correction.
                    if domain == 'allocation':
                        result = module(filename).run(seed, False, args.output/'seed-42')
                else:
                    result = module(filename).run(seed, args.quick)
                record['domains'][domain] = summarize(domain, result)
            rows.append(record)
    write_json(args.output/'replications.json', dict(status='executed independent simulation seeds',
                seeds=[42,43,44], quick=args.quick, results=rows,
                interpretation='Three descriptive replications, not a precise confidence interval or real-world validation.'))
