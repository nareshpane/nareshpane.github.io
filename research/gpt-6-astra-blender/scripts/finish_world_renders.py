"""Serialize approved final renders on one integrated GPU; retain stage logs.

Run only after visually inspecting world and analytical benchmark frames.
--wait-for-hero waits for render_cinematic.py's current-master PASS report.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--blender',required=True)
p.add_argument('--wait-for-hero',action='store_true');p.add_argument('--hero-frames',required=True)
p.add_argument('--resume-after-stills',action='store_true')
a=p.parse_args();digest=hashlib.sha256((ROOT/'blender/city_master.blend').read_bytes()).hexdigest()
if a.wait_for_hero:
    while True:
        try:
            r=json.loads((ROOT/'data/cinematic_runtime.json').read_text())
            if r['status']=='PASS' and r['master_sha256']==digest and len(r['frame_timings'])==576:break
        except (FileNotFoundError,json.JSONDecodeError):pass
        time.sleep(10)
commands=[('encode_cinematic',[sys.executable,str(ROOT/'scripts/encode_cinematic.py'),'--frames',a.hero_frames])]
for stage,args in [('validate_cinematic',[]),('render_world_tests',[]),('render_views',[]),('render_animations',['--','--render'])]:
    commands.append((stage,[a.blender,'--background','--threads','6','--python-exit-code','1','--python',str(ROOT/'scripts'/f'{stage}.py'),*args]))
report={'status':'RUNNING','master_sha256':digest,'stages':[]}
if a.resume_after_stills:
    previous=json.loads((ROOT/'data/world_render_pipeline.json').read_text())
    assert previous['status']=='FAIL' and previous['active_stage']=='render_views'
    # The 11 city images completed before anatomy failed. Recover their actual
    # Blender-reported times rather than rerendering them or retaining v1 times.
    log=(ROOT/'data/render_views.log').read_text()
    matches=re.findall(r"Saved: '[^']*/renders/(\w+)\.png'\nTime: (\d+):(\d+\.\d+)",log)
    times={name:int(minutes)*60+float(seconds) for name,minutes,seconds in matches}
    assert len(times)==11
    (ROOT/'data/render_runtime.json').write_text(json.dumps(times,indent=2)+'\n')
    (ROOT/'data/render_runtime_notes.json').write_text(json.dumps(dict(source='Blender core render/save timings recovered from render_views.log',
        excludes='Scene-open overhead and anatomy; original anatomy material failure resolved without repeating the eleven city renders'),indent=2)+'\n')
    commands=[('render_views_anatomy',[a.blender,'--background','--threads','6','--python-exit-code','1','--python',str(ROOT/'scripts/render_views.py'),'--','anatomy']),commands[-1]]
    report['prior_attempt']=previous
    report['resolved_error']='Anatomy material was unused by the new architectural families; illustration now creates its own local green material.'
for stage,cmd in commands:
    report['active_stage']=stage
    (ROOT/'data/world_render_pipeline.json').write_text(json.dumps(report,indent=2)+'\n')
    start=time.perf_counter();print('Starting '+stage,flush=True)
    with (ROOT/'data'/f'{stage}.log').open('w') as log:
        result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
    report['stages'].append({'stage':stage,'seconds':round(time.perf_counter()-start,3),'returncode':result.returncode})
    if result.returncode:
        report['status']='FAIL'
        (ROOT/'data/world_render_pipeline.json').write_text(json.dumps(report,indent=2)+'\n')
        print((ROOT/'data'/f'{stage}.log').read_text()[-5000:]);raise SystemExit(result.returncode)
    print('Finished '+stage,flush=True)
report['status']='PASS';report['active_stage']=None
(ROOT/'data/world_render_pipeline.json').write_text(json.dumps(report,indent=2)+'\n')
