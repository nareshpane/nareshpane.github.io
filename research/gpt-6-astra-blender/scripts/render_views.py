"""Render inspectable analytical modes from the saved city, without rebuilding it."""
import json
from pathlib import Path
import sys
import time
import bpy
from mathutils import Vector

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from build_city import MeshBuilder, material, object_from, orient, archetype
CITY=json.loads((ROOT/'data/city.json').read_text())
A=json.loads((ROOT/'data/analysis.json').read_text())

def ramp(t):
    t=max(0,min(1,t))
    stops=[(.08,.31,.42),(.18,.62,.58),(.96,.78,.34),(.87,.28,.11)]
    i=min(2,int(t*3)); f=t*3-i
    return tuple(stops[i][j]*(1-f)+stops[i+1][j]*f for j in range(3))

def visible(prefix,on):
    for c in bpy.data.collections:
        if c.name.startswith(prefix): c.hide_render=not on;c.hide_viewport=not on

def open_master():
    bpy.ops.wm.open_mainfile(filepath=str(ROOT/'blender/city_master.blend'))

def setup_mode(mode):
    s=bpy.context.scene
    s.camera=bpy.data.objects['Camera_Detail' if mode=='detail' else 'Camera_Hero' if mode=='hero' else 'Camera_Analysis']
    for prefix in ['01','02','03','04','05','06','07','11','12']:visible(prefix,True)
    for prefix in ['08','09','10']:visible(prefix,False)
    if mode in ['overlay','graph','degree','betweenness','resilience']:
        visible('08',True)
    if mode in ['graph','degree']:
        for prefix in ['01','02','03','04','05','06','07','12']:visible(prefix,False)
        # A bare graph needs more contrast than an overlay on dark streets.
        shader=bpy.data.materials['Graph cyan'].node_tree.nodes['Principled BSDF']
        shader.inputs['Base Color'].default_value=(.025,.14,.18,1)
        shader.inputs['Emission Strength'].default_value=0
    if mode in ['route','closure']:
        visible('09',True)
        bpy.data.objects['route'].hide_render=(mode=='closure')
        bpy.data.objects['closure_route'].hide_render=(mode!='closure')
        if mode=='closure':
            for o in bpy.data.objects:
                if o.get('graph_edge')=='35-36':o.hide_render=True
    if mode in ['degree','betweenness','resilience']:
        values=([A['degree'][str(i)] for i in range(len(CITY['nodes']))] if mode=='degree' else
                [A['betweenness'][str(i)] for i in range(len(CITY['nodes']))] if mode=='betweenness' else A['node_mean_loss_pct']['0.08'])
        lo,hi=(min(values),max(values)) if mode=='degree' else (0,max(values))
        for i,value in enumerate(values):
            t=(value-lo)/(hi-lo) if hi>lo else 0
            o=bpy.data.objects['Node_'+str(i)]
            o.scale=(2.3,2.3,3+22*t)
            o.material_slots[0].link='OBJECT'
            o.material_slots[0].material=material(f'{mode}_{i}',ramp(t),emission=.2)
    if mode in ['access_before','access_after']:
        values=A['accessibility_before' if mode=='access_before' else 'accessibility_after']
        lo=min(A['accessibility_before']);hi=max(A['accessibility_after'])
        mats=[material(f'Accessibility_{i}',ramp((v-lo)/(hi-lo))) for i,v in enumerate(values)]
        for o in bpy.data.objects:
            if o.get('role')=='building':o.material_slots[0].material=mats[o['node_id']]
        visible('10',mode=='access_after')

def render(mode):
    start=time.perf_counter();open_master();setup_mode(mode)
    s=bpy.context.scene;s.render.filepath=str(ROOT/'renders'/f'{mode}.png')
    bpy.ops.render.render(write_still=True)
    return round(time.perf_counter()-start,3)

def anatomy():
    start=time.perf_counter();open_master()
    master=bpy.context.scene
    # Preserve the original educational figure's softbox shadows/contrast;
    # the cinematic's unshadowed fill is meant for urban exterior animation.
    bpy.data.lights['Warm softbox'].energy=900000
    bpy.data.lights['Sky fill'].energy=700000
    for light in bpy.data.lights:light.use_shadow=True
    bpy.data.lights['Afternoon sun'].energy=2.1
    bpy.data.lights['Afternoon sun'].angle=.18
    bpy.data.objects['Afternoon sun'].rotation_euler=(.4,-.5,-.4)
    bg=master.world.node_tree.nodes['Background']
    bg.inputs[0].default_value=(.56,.67,.8,1);bg.inputs[1].default_value=.35
    s=bpy.data.scenes.new('Building anatomy • staged modelling')
    bpy.context.window.scene=s
    s.world=master.world
    c=bpy.data.collections.new('Anatomy objects');s.collection.children.link(c)
    palette=[material('Anatomy garden green',(.37,.54,.43)),bpy.data.materials['Ivory trim'],
             bpy.data.materials['Blue glazing'],bpy.data.materials['Slate roofs']]
    for x in [-24,-8,8,24]:
        m=MeshBuilder();m.box((x,0,-.5),(14,14,1),1)
        object_from('Stage plinth',m.mesh('Plinth',palette),c)
    m=MeshBuilder();m.box((-24,0,3),(6,6,6))
    object_from('01 Primitive cube',m.mesh('Cube • 8 vertices 12 edges 6 faces',palette),c)
    m=MeshBuilder();m.box((-8,0,9),(8,8,18))
    object_from('02 Scaled cube',m.mesh('Scaled cube',palette),c)
    wire=MeshBuilder()
    for z in [0,18]:
        for dx in [-4,4]:wire.rod((-8+dx,-4,z),(-8+dx,4,z),.1)
        for dy in [-4,4]:wire.rod((-12,dy,z),(-4,dy,z),.1)
    for dx in [-4,4]:
        for dy in [-4,4]:
            wire.rod((-8+dx,dy,0),(-8+dx,dy,18),.1)
            for z in [0,18]:wire.crown((-8+dx,dy,z),.28)
    object_from('Highlighted edges and vertices',wire.mesh('Topology illustration',[bpy.data.materials['Route gold']]),c)
    m=MeshBuilder();m.box((8,0,9),(8,8,18));m.box((8,0,23),(9,9,.7),1)
    for z in [3,7,11,15]:
        for x in [5.5,8,10.5]:m.box((x,-6,z),(1.5,.15,2),2)
    object_from('03 Exploded facade and roof',m.mesh('Exploded parts',palette),c)
    object_from('04 Finished archetype',archetype(5,palette),c,(24,0,0),(8,8,18))
    m=MeshBuilder();m.box((0,0,-1.2),(2000,2000,.2),1)
    object_from('Anatomy backdrop',m.mesh('Backdrop',palette),c)
    for o in master.objects:
        if o.type=='LIGHT':s.collection.objects.link(o)
    d=bpy.data.cameras.new('Anatomy camera');d.type='ORTHO';d.ortho_scale=76
    camera=bpy.data.objects.new('Camera_Anatomy',d);c.objects.link(camera);camera.location=(27,-72,50);orient(camera,(0,0,10));s.camera=camera
    s.render.engine='CYCLES';s.cycles.samples=24;s.cycles.use_denoising=True;s.cycles.seed=CITY['seed']
    s.render.resolution_x=1440;s.render.resolution_y=680;s.render.resolution_percentage=100
    s.view_settings.view_transform='AgX';s.render.filepath=str(ROOT/'renders/anatomy.png')
    bpy.ops.wm.save_as_mainfile(filepath=str(ROOT/'blender/building_anatomy.blend'),compress=True)
    bpy.ops.render.render(write_still=True)
    (ROOT/'data/anatomy_render_runtime.json').write_text(json.dumps(dict(seconds=round(time.perf_counter()-start,3),engine='CYCLES',samples=24),indent=2)+'\n')

def main():
    modes=sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [
        'hero','detail','overlay','graph','degree','betweenness','route','closure','access_before','access_after','resilience','anatomy']
    times={}
    for mode in modes:
        if mode=='anatomy':anatomy()
        else:times[mode]=render(mode)
    path=ROOT/'data/render_runtime.json'
    old=json.loads(path.read_text()) if path.exists() else {}
    old.update(times);path.write_text(json.dumps(old,indent=2)+'\n')
    print('RENDER_VIEWS_SUCCESS',flush=True)

if __name__=='__main__':main()
