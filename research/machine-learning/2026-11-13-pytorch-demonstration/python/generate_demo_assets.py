"""Export browser views from executed artifacts; never synthesize training metrics."""
from __future__ import annotations
import ast
import hashlib
import json
import zipfile
from common import ROOT, write_json


def read(name):
    return json.loads((ROOT/'outputs'/'seed-42'/f'{name}.json').read_text(encoding='utf-8'))


def select(result, keys):
    return {k: result[k] for k in keys}


def generate():
    from refinement_examples import calculate
    write_json(ROOT/'data'/'worked-examples.json',calculate())
    # Deterministic source bundle keeps sibling imports available to readers.
    with zipfile.ZipFile(ROOT/'python'/'python-source.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted((ROOT/'python').iterdir()):
            if path.is_file() and (path.suffix in ('.py','.ps1') or path.name=='requirements.txt'):
                info=zipfile.ZipInfo('python/'+path.name,date_time=(2026,11,13,0,0,0))
                info.compress_type=zipfile.ZIP_DEFLATED
                archive.writestr(info,path.read_bytes())
    write_json(ROOT/'data'/'foundations-demo.json', read('foundations'))
    s = read('survival')
    write_json(ROOT/'data'/'survival-demo.json', select(s, ['status','seed','n','events','gradient_max_error',
                'equivalent','conventional','neural','parameter_counts','profiles','timelines','history','neural_epochs']))
    g = read('financial-network')
    write_json(ROOT/'data'/'financial-network-demo.json', select(g, ['status','seed','counts','conventional','neural',
                'new_accounts','aggregation_max_error','parameter_counts','gallery','history','neural_epochs']))
    r = read('river')
    thin = list(range(0,len(r['t']),max(1,len(r['t'])//40)))
    if thin[-1] != len(r['t'])-1:
        thin.append(len(r['t'])-1)
    result = select(r, ['status','seed','config','truth','sensors','x','observation_indices','observations',
                        'train_observation_rows','gradient','forward_max_error','refinement','conventional',
                        'pytorch','heldout_rmse','parameter_absolute_error','profile','recovery_sensitivity'])
    result.update(t=r['t'], display_times=[r['t'][i] for i in thin], display_field=[r['fitted_field'][i] for i in thin])
    write_json(ROOT/'data'/'river-demo.json', result)
    a = read('allocation')
    result = select(a, ['status','seed','counts','config','districts','stations','cost','capacity','test_scenarios',
                         'policies','evaluation','equivalent_gradient_max_error','scipy','pytorch'])
    result['training_scenarios'] = a['train_scenarios'][:48]
    indices = list(range(0, len(a['history']), max(1,len(a['history'])//50)))
    indices.append(len(a['history'])-1)
    result['history'] = [dict(step=i, objective=a['history'][i][0], vector=a['history'][i][1:]) for i in sorted(set(indices))]
    write_json(ROOT/'data'/'allocation-demo.json', result)
    for name in ['benchmark-results','compilation']:
        path = ROOT/'outputs'/'seed-42'/f'{name}.json'
        if path.exists():
            write_json(ROOT/'data'/f'{name}.json', read(name))
    replication = ROOT/'outputs'/'replications.json'
    if replication.exists():
        write_json(ROOT/'data'/'replications.json', json.loads(replication.read_text(encoding='utf-8')))
    counts = []
    for path in sorted((ROOT/'python').glob('0[1-4]_*.py')):
        source = path.read_text(encoding='utf-8')
        tree = ast.parse(source)
        functions = {node.name: node.end_lineno-node.lineno+1 for node in tree.body if isinstance(node,(ast.FunctionDef,ast.ClassDef))}
        counts.append(dict(file=path.name, physical_lines=len(source.splitlines()),
                           nonempty_noncomment_lines=sum(bool(line.strip()) and not line.lstrip().startswith('#') for line in source.splitlines()),
                           functions=functions))
    write_json(ROOT/'data'/'source-counts.json', dict(convention='Physical lines; nonempty excludes blank and # comment-only lines, includes docstrings. Functions/classes count declaration through final source line.', files=counts))
    assets = {str(path.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(path.read_bytes()).hexdigest()
              for directory in ['python','data','images','css','js'] for path in sorted((ROOT/directory).rglob('*'))
              if path.is_file() and '__pycache__' not in path.parts and path.name not in ('generation-manifest.json','page-validation.json','browser-validation.json','refinement-validation.json','refinement-browser-validation.json','typesetting-validation.json','typesetting-browser-validation.json')}
    write_json(ROOT/'data'/'generation-manifest.json',dict(generator='python/generate_demo_assets.py',
                origin='Executed synthetic CPU experiments, seed 42; three-seed replication summary where present.',
                browser='Static stored fitted outputs plus explicitly labeled JavaScript numerical illustrations; no live Python in browser.',
                seeds=[42,43,44], asset_sha256=assets))
    print('Exported browser data and provenance hashes.')


if __name__ == '__main__':
    generate()
