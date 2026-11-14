"""Sequential CPU experiment runner; use --quick for a pipeline smoke test."""
from common import ROOT, cli, module, write_json
from threadpoolctl import threadpool_limits


if __name__ == '__main__':
    args = cli(__doc__)
    with threadpool_limits(limits=1):
        for name in ['00_pytorch_foundations.py', '01_survival_analysis.py',
                     '02_financial_graph_learning.py', '03_river_inverse_problem.py', '04_resource_allocation.py']:
            print('Running', name, flush=True)
            experiment = module(name)
            if name.startswith('00'):
                result = experiment.run(args.seed)
                write_json(args.output/'foundations.json', result)
            else:
                result = experiment.run(args.seed, args.quick, args.output)
            print(name, 'completed', flush=True)
