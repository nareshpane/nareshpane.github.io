"""Inspect saved analytical keyframes against current graph values, no rendering."""
import hashlib
import json
import math
from pathlib import Path
import bpy
ROOT=Path(__file__).resolve().parents[1]
C=json.loads((ROOT/'data/city.json').read_text());A=json.loads((ROOT/'data/analysis.json').read_text())
digest=hashlib.sha256((ROOT/'data/city.json').read_bytes()).hexdigest()
reports=[]
for name in ['shortest_path','bridge_failure','transit_accessibility']:
    bpy.ops.wm.open_mainfile(filepath=str(ROOT/'blender'/f'{name}.blend'))
    s=bpy.context.scene
    assert s['city_sha256']==digest and s.render.engine=='BLENDER_EEVEE_NEXT'
    assert (s.frame_start,s.frame_end,s.render.fps)==(1,224,16)
    assert (s.render.resolution_x,s.render.resolution_y)==(960,540)
    assert len(s.timeline_markers)==4
    assert sum(o.get('role')=='pedestrian' for o in s.objects)==48
    for frame in range(1,225):
        s.frame_set(frame);p=s.camera.matrix_world.translation
        assert all(math.isfinite(v) for v in p) and p.z>100
        for b in C['buildings']:
            assert not (abs(p.x-b['x'])<b['width']/2 and abs(p.y-b['y'])<b['depth']/2 and p.z<b['height']*1.25)
    for o in s.objects:
        if o.get('graph_path'):
            path=list(o['graph_path']);points=o.data.splines[0].points
            assert len(path)==len(points)
            for node,p in zip(path,points):
                n=C['nodes'][node];assert abs(p.co.x-n['x'])<1e-5 and abs(p.co.y-n['y'])<1e-5
    if name=='bridge_failure':
        for f,closed in [(112,False),(113,True),(224,True)]:
            s.frame_set(f)
            for o in s.objects:
                if o.get('graph_edge')=='35-36':assert o.hide_render==closed
            assert s.objects['Open bridge route'].hide_render==closed
            assert s.objects['Graph edges • exact road centre lines'].hide_render==closed
            assert s.objects['Graph after closure'].hide_render!=closed
        assert list(s.objects['Recalculated detour']['graph_path'])==A['closure_route']['nodes']
    elif name=='shortest_path':
        o=s.objects['Progressive shortest path'];assert list(o['graph_path'])==A['route']['nodes']
        for f,v in [(65,0),(176,1),(224,1)]:
            s.frame_set(f);assert abs(o.data.bevel_factor_end-v)<1e-6
        assert o.data.bevel_factor_mapping_end=='SPLINE'
    else:
        lo=min(A['accessibility_before']);hi=max(A['accessibility_after'])
        def ramp(t):
            stops=[(.08,.31,.42),(.18,.62,.58),(.96,.78,.34),(.87,.28,.11)]
            t=max(0,min(1,t));i=min(2,int(t*3));f=t*3-i
            return [stops[i][j]*(1-f)+stops[i+1][j]*f for j in range(3)]
        for f,key in [(1,'accessibility_before'),(224,'accessibility_after')]:
            s.frame_set(f)
            # Materials at nodes without buildings are intentionally unused
            # and Blender purges them on save. Validate every actual facade.
            for o in s.objects:
                if o.get('role')=='building':
                    assert o.material_slots[0].material.name==f"Accessibility_{o['node_id']}"
            for i in sorted({b['node'] for b in C['buildings']}):
                value=A[key][i]
                color=bpy.data.materials[f'Accessibility_{i}'].node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value
                assert all(abs(x-y)<2e-6 for x,y in zip(color,ramp((value-lo)/(hi-lo))))
        for f,hidden in [(96,True),(97,False)]:
            s.frame_set(f);assert s.objects['New transit connection • schematic corridor'].hide_render==hidden
    reports.append(dict(name=name,status='PASS',frames=224,cameras=4,graph_sha256=digest,bytes=(ROOT/'blender'/f'{name}.blend').stat().st_size))
(ROOT/'data/animation_scene_validation.json').write_text(json.dumps(dict(status='PASS',scenes=reports),indent=2)+'\n')
print('ANALYTICAL_SCENE_VALIDATION_PASS',flush=True)
