"""Small, mandatory visual gate before world-revision video renders."""
import json
from pathlib import Path
import sys
import time
import bpy
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from render_views import open_master,setup_mode
from build_city import orient
from render_cinematic import make_scene

OUT=Path('/tmp/astra-world-v2-tests');OUT.mkdir(exist_ok=True)
views=[('street','hero',(119,-125,3.4),(118,-65,5),28),
       ('pedestrian','hero',(124.4,-133,2.2),(125.9,-127,1.5),40),
       ('trees','hero',(-170,32,8),(-147,67,4),40),
       ('architecture','hero',(225,-230,180),(25,-45,14),44),
       ('overlay','overlay',None,None,None),
       ('waterfront','hero',(-7,1,24),(12,48,3),35)]
times=[]
requested=sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else []
for name,mode,pos,target,lens in views:
    if requested and name not in requested:continue
    open_master();setup_mode(mode);s=bpy.context.scene
    s.render.resolution_x,s.render.resolution_y=1280,720
    if pos:
        data=bpy.data.cameras.new('World test '+name);data.lens=lens;data.clip_end=5000
        camera=bpy.data.objects.new('World test camera',data);s.collection.objects.link(camera)
        camera.location=pos;orient(camera,target);s.camera=camera
    s.render.filepath=str(OUT/f'{name}.png');tick=time.perf_counter()
    bpy.ops.render.render(write_still=True)
    times.append(dict(view=name,seconds=round(time.perf_counter()-tick,3)))
report=ROOT/'data/world_benchmark.json'
if requested and report.exists():
    previous=json.loads(report.read_text())['frames']
    times=[next((t for t in times if t['view']==old['view']),old) for old in previous]
report.write_text(json.dumps(dict(status='RENDERED_FOR_INSPECTION',frames=times),indent=2)+'\n')
print(json.dumps(times,indent=2),flush=True)
