"""Fourteen-second Eevee analytical reels, with actual graph-derived routes.

Test representative frames before --render. Bezier camera motion within four
deliberate shots. Route progression/colour interpolation are explanatory time
compression, not a calibrated traffic or dynamic economic simulation.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import bpy
from mathutils import Vector
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from render_views import open_master, setup_mode, visible, ramp, CITY, A
from build_city import MeshBuilder, object_from, material
from render_cinematic import smooth_keys, configure_lighting
FPS=16;FRAMES=224;RESOLUTION=[960,540]
NAMES=['shortest_path','bridge_failure','transit_accessibility']

def linear(owner):
    for curve in owner.animation_data.action.fcurves:
        for key in curve.keyframe_points:key.interpolation='LINEAR'

def shown(obj,first,last=FRAMES):
    for f,value in [(1,first>1),(first,False),(last+1,True)]:
        obj.hide_render=value;obj.keyframe_insert('hide_render',frame=f)

def path_curve(name,path,col,mat,radius,first=1,last=None):
    d=bpy.data.curves.new(name,'CURVE');d.dimensions='3D';d.bevel_depth=radius;d.bevel_resolution=1
    d.resolution_u=1;d.use_fill_caps=True
    d.bevel_factor_mapping_end='SPLINE'
    spline=d.splines.new('POLY');spline.points.add(len(path)-1)
    for p,node in zip(spline.points,path):
        n=CITY['nodes'][node];p.co=(n['x'],n['y'],3.2,1)
    d.materials.append(mat);o=bpy.data.objects.new(name,d);col.objects.link(o);o['graph_path']=path
    if last:
        for f,value in [(1,0),(first,0),(last,1),(FRAMES,1)]:
            d.bevel_factor_end=value;d.keyframe_insert('bevel_factor_end',frame=f)
        linear(d)
    return o

def travel(marker,path,first,last):
    distances=[0]
    for u,v in zip(path,path[1:]):
        a,b=CITY['nodes'][u],CITY['nodes'][v]
        distances.append(distances[-1]+Vector((a['x']-b['x'],a['y']-b['y'])).length)
    for node,distance in zip(path,distances):
        n=CITY['nodes'][node];marker.location=(n['x'],n['y'],6)
        marker.keyframe_insert('location',frame=first+(last-first)*distance/distances[-1])
    linear(marker)

def cameras(s,col,shots):
    s.timeline_markers.clear()
    for i,(first,last,pos0,pos1,aim0,aim1,lens0,lens1) in enumerate(shots):
        data=bpy.data.cameras.new(f'Analytical shot {i+1}');data.clip_end=5000
        cam=bpy.data.objects.new(data.name,data);col.objects.link(cam)
        target=bpy.data.objects.new(data.name+' aim',None);col.objects.link(target)
        constraint=cam.constraints.new('TRACK_TO');constraint.target=target
        constraint.track_axis='TRACK_NEGATIVE_Z';constraint.up_axis='UP_Y'
        for f,p,a,lens in [(first,pos0,aim0,lens0),(last,pos1,aim1,lens1)]:
            cam.location=p;target.location=a;data.lens=lens
            cam.keyframe_insert('location',frame=f);target.keyframe_insert('location',frame=f);data.keyframe_insert('lens',frame=f)
        for o in [cam,target,data]:smooth_keys(o)
        s.timeline_markers.new(data.name,frame=first).camera=cam
        if i==0:s.camera=cam

WIDE=(330,-430,740)
def prepare(name):
    open_master();setup_mode('access_before' if name=='transit_accessibility' else 'overlay')
    s=bpy.context.scene;configure_lighting(s,12)
    s.frame_start=1;s.frame_end=FRAMES;s.render.fps=FPS
    s.render.resolution_x,s.render.resolution_y=RESOLUTION
    c=bpy.data.collections.new('14 Analytical camera and route animation');s.collection.children.link(c)
    gold=material('Animated route gold',(1,.58,.04),emission=.55)
    red=material('Closure detour coral',(.98,.19,.06),emission=.4)
    pale=material('Alternative route muted',(.22,.47,.52),emission=.1)
    if name!='transit_accessibility':
        for node,label in [(A['source'],'Origin'),(A['target'],'Destination')]:
            n=CITY['nodes'][node];m=MeshBuilder();m.crown((0,0,0),3.3)
            object_from(label,m.mesh(label,[gold]),c,(n['x'],n['y'],6))
        m=MeshBuilder();m.crown((0,0,0),2.7)
        marker=object_from('Journey marker',m.mesh('Marker',[gold]),c)
    if name=='shortest_path':
        for i,path in enumerate(A['alternative_routes'][1:3]):
            o=path_curve(f'Alternative {i+1}',path['nodes'],c,pale,.5);shown(o,49,176)
        path_curve('Progressive shortest path',A['route']['nodes'],c,gold,1.3,65,176)
        travel(marker,A['route']['nodes'],65,176);shown(marker,65)
        shots=[(1,48,WIDE,(285,-360,610),(0,0,0),(-15,20,0),40,41),
               (49,112,(-250,-40,275),(-170,-40,230),(-120,70,0),(-60,60,0),38,38),
               (113,176,(15,-130,320),(190,-90,340),(15,48,0),(100,65,0),38,38),
               (177,224,(285,-360,610),WIDE,(0,15,0),(0,0,0),41,40)]
        labels=[(0,3,'Walking network | origin 40 - destination 47'),(3,7,'Alternative paths | shortest path grows in gold'),
                (7,11,f'Shortest path | {A["route"]["seconds"]} seconds'),(11,14,f'{A["route"]["metres"]:.2f} m | full route')]
    elif name=='bridge_failure':
        old=path_curve('Open bridge route',A['route']['nodes'],c,gold,1.3);shown(old,1,112)
        detour=path_curve('Recalculated detour',A['closure_route']['nodes'],c,red,1.35,113,192);shown(detour,113)
        travel(marker,A['closure_route']['nodes'],113,192);shown(marker,113)
        for o in list(s.objects):
            if o.get('graph_edge')=='35-36':shown(o,1,112)
        base=bpy.data.objects['Graph edges • exact road centre lines'];shown(base,1,112)
        m=MeshBuilder()
        for e in CITY['roads']:
            if e['id']=='35-36':continue
            a,b=[CITY['nodes'][i] for i in [e['u'],e['v']]]
            m.rod((a['x'],a['y'],2),(b['x'],b['y'],2),.6)
        after=object_from('Graph after closure',m.mesh('Remaining edges',list(base.data.materials)),c);shown(after,113)
        m=MeshBuilder();m.rod((-5,43,7),(5,53,7),.8);m.rod((-5,53,7),(5,43,7),.8)
        cross=object_from('Closed crossing X',m.mesh('Closure symbol',[red]),c);shown(cross,113)
        shots=[(1,48,WIDE,(285,-360,610),(0,0,0),(0,30,0),40,41),
               (49,112,(85,-105,205),(50,-70,165),(0,48,0),(0,48,0),38,39),
               (113,176,(110,-250,470),(220,-290,540),(0,0,0),(0,-10,0),36,38),
               (177,224,(250,-330,580),WIDE,(0,0,0),(0,0,0),38,40)]
        labels=[(0,3,f'Bridge open | original route {A["route"]["seconds"]} s'),(3,7,'Northern crossing | one edge closes'),
                (7,11,'Closure | recalculate from the same origin'),(11,14,f'Detour {A["closure_route"]["seconds"]} s | southern crossing remains')]
    else:
        lo=min(A['accessibility_before']);hi=max(A['accessibility_after'])
        for i,(before,after) in enumerate(zip(A['accessibility_before'],A['accessibility_after'])):
            bsdf=bpy.data.materials[f'Accessibility_{i}'].node_tree.nodes['Principled BSDF']
            for f,value in [(1,before),(96,before),(176,after),(224,after)]:
                bsdf.inputs['Base Color'].default_value=(*ramp((value-lo)/(hi-lo)),1)
                bsdf.inputs['Base Color'].keyframe_insert('default_value',frame=f)
        visible('10',True);shown(bpy.data.objects['New transit connection • schematic corridor'],97)
        shots=[(1,48,WIDE,(285,-360,610),(0,0,0),(0,30,0),40,41),
               (49,96,(280,-100,330),(240,-30,300),(110,48,15),(75,55,12),36,37),
               (97,176,(-235,-160,390),(-210,10,400),(-70,70,5),(-40,65,5),36,38),
               (177,224,(270,-340,590),WIDE,(0,0,0),(0,0,0),38,40)]
        labels=[(0,3,'Baseline accessibility | one shared colour scale'),(3,6,'Existing transit | fixed synthetic opportunities'),
                (6,11,'New 33 - 38 transit link | colours interpolate to recomputed values'),(11,14,'After transit | illustrative accessibility, not a causal estimate')]
    cameras(s,c,shots);s.frame_set(1)
    s['analytical_stages']=json.dumps(labels)
    s['time_interpretation']='Illustrative stage timing; colour interpolation is not a dynamic economic simulation.'
    return s,labels

def encode(name,directory,labels):
    subprocess.run(['python3',str(ROOT/'scripts/validate_frames.py'),str(directory),str(FRAMES),*map(str,RESOLUTION)],check=True)
    filters=['scale=out_color_matrix=bt709:out_range=tv']
    for first,last,label in labels:
        filters.append(f"drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:text='{label}':fontcolor=white:fontsize=18:box=1:boxcolor=black@0.65:boxborderw=9:x=22:y=22:enable='gte(t,{first})*lt(t,{last})'")
    path=ROOT/'video'/f'{name}.mp4'
    subprocess.run(['ffmpeg','-y','-v','warning','-framerate',str(FPS),'-i',str(directory/'frame_%04d.png'),
        '-frames:v',str(FRAMES),'-an','-vf',','.join(filters),'-c:v','libx264','-profile:v','baseline','-level:v','3.1',
        '-crf','22','-preset','medium','-pix_fmt','yuv420p','-color_primaries','bt709','-color_trc','bt709','-colorspace','bt709',
        '-movflags','+faststart','-threads','4',str(path)],check=True)
    subprocess.run(['ffmpeg','-v','error','-i',str(path),'-f','null','-'],check=True)
    return path.stat().st_size

def main():
    p=argparse.ArgumentParser();p.add_argument('--test',action='store_true');p.add_argument('--render',action='store_true')
    p.add_argument('--name',choices=NAMES);p.add_argument('--output',type=Path,default=Path('/tmp/astra-world-v2-analytical'))
    args=p.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
    assert args.test!=args.render,'Choose --test or --render'
    path=ROOT/'data'/('animation_benchmark.json' if args.test else 'animation_runtime.json')
    report=json.loads(path.read_text()) if path.exists() else {}
    for name in [args.name] if args.name else NAMES:
        s,labels=prepare(name);directory=args.output/name;directory.mkdir(parents=True,exist_ok=True)
        if args.render:bpy.ops.wm.save_as_mainfile(filepath=str(ROOT/'blender'/f'{name}.blend'),compress=True)
        timings=[];start=time.perf_counter()
        for frame in [40,85,145,214] if args.test else range(1,FRAMES+1):
            tick=time.perf_counter();s.frame_set(frame);s.render.filepath=str(directory/f'frame_{frame:04}.png')
            bpy.ops.render.render(write_still=True);timings.append(round(time.perf_counter()-tick,3))
            print(f'ANALYTICAL_FRAME {name} {frame} {timings[-1]:.3f}s',flush=True)
        elapsed=time.perf_counter()-start
        report[name]=dict(status='PASS',seconds=round(elapsed,3),frames=FRAMES,fps=FPS,resolution=RESOLUTION,
            duration_seconds=14,average_frame_seconds=round(sum(timings)/len(timings),3),frame_timings=timings,
            samples=12,engine=s.render.engine,stages=labels,raw_directory=str(directory))
        if args.render:report[name]['bytes']=encode(name,directory,labels)
        path.write_text(json.dumps(report,indent=2)+'\n')
    print('ANIMATIONS_SUCCESS',flush=True)

if __name__=='__main__':main()
