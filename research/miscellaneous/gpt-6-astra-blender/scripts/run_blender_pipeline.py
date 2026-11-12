"""Run Blender stages with retained logs and fail on the first execution error."""
import argparse
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser()
p.add_argument('--blender',default='blender')
p.add_argument('--threads',type=int,default=6)
p.add_argument('--stages',nargs='+',default=['build_city','validate_blender','render_views','render_animations'])
a=p.parse_args()
for stage in a.stages:
    assert stage in ['build_city','validate_blender','render_views','render_animations']
    start=time.perf_counter();print('Starting '+stage,flush=True)
    with (ROOT/'data'/f'{stage}.log').open('w') as log:
        result=subprocess.run([a.blender,'--background','--threads',str(a.threads),'--python-exit-code','1',
            '--python',str(ROOT/'scripts'/f'{stage}.py')]+(['--','--render'] if stage=='render_animations' else []),stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:
        print((ROOT/'data'/f'{stage}.log').read_text()[-5000:]);raise SystemExit(result.returncode)
    print(f'Finished {stage}: {time.perf_counter()-start:.1f} s',flush=True)
