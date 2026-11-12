"""Run final QA only after the approved sequential render pipeline passes."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--blender',required=True);p.add_argument('--baseline',required=True)
a=p.parse_args()
while True:
    try:
        r=json.loads((ROOT/'data/world_render_pipeline.json').read_text())
        if r['status']=='FAIL':raise SystemExit('Render pipeline failed; QA not started')
        if r['status']=='PASS':break
    except (FileNotFoundError,json.JSONDecodeError):pass
    time.sleep(10)
commands=[]
for stage in ['validate_blender','validate_animations']:
    commands.append((stage,[a.blender,'--background','--threads','4','--python-exit-code','1','--python',str(ROOT/'scripts'/f'{stage}.py')]))
for stage,args in [('package_assets',[]),('build_page',[]),('validate_article_values',['--baseline',a.baseline]),
                   ('qa_site',[]),('qa_site',['--preflight']),('package_assets',['--keep-images'])]:
    commands.append((stage,[sys.executable,str(ROOT/'scripts'/f'{stage}.py'),*args]))
report={'status':'RUNNING','stages':[]}
for i,(stage,cmd) in enumerate(commands):
    report['active_stage']=stage
    (ROOT/'data/world_final_qa.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Starting '+stage,flush=True);start=time.perf_counter()
    with (ROOT/'data'/f'world_qa_{i:02}_{stage}.log').open('w') as log:
        result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
    report['stages'].append(dict(stage=stage,seconds=round(time.perf_counter()-start,3),returncode=result.returncode))
    if result.returncode:
        report['status']='FAIL'
        (ROOT/'data/world_final_qa.json').write_text(json.dumps(report,indent=2)+'\n')
        print((ROOT/'data'/f'world_qa_{i:02}_{stage}.log').read_text()[-6000:]);raise SystemExit(result.returncode)
    print('Finished '+stage,flush=True)
report['status']='PASS';report['active_stage']=None
(ROOT/'data/world_final_qa.json').write_text(json.dumps(report,indent=2)+'\n')
