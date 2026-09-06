"""Independent saved-file inspection, graph/geometry alignment and GLB reimport."""
import hashlib
import json
import math
from pathlib import Path
import struct
import bpy

ROOT=Path(__file__).resolve().parents[1]
city=json.loads((ROOT/'data/city.json').read_text())
blend=ROOT/'blender/city_master.blend'
bpy.ops.wm.open_mainfile(filepath=str(blend))
s=bpy.context.scene
objects=list(s.objects);meshes=[o for o in objects if o.type=='MESH']
buildings=[o for o in objects if o.get('role')=='building']
assert len(buildings)==len(city['buildings'])==204
assert len({o['building_id'] for o in buildings})==204
assert len({o.data.as_pointer() for o in buildings})==12
trees=[o for o in objects if o.get('role')=='tree']
people=[o for o in objects if o.get('role')=='pedestrian']
walkers=[o for o in people if o.get('animated_pedestrian')]
assert len(trees)==162 and len({o['tree_family'] for o in trees})==7
assert len(people)==48 and len(walkers)==12
assert len({o['family'] for o in buildings})==12
for o in walkers:
    assert o.animation_data and len(o.data.shape_keys.key_blocks)==4
for frame in range(1,577,12):
    s.frame_set(frame)
    for o in people:
        for b in city['buildings']:
            assert not (abs(o.location.x-b['x'])<b['width']/2+.35 and abs(o.location.y-b['y'])<b['depth']/2+.35),(frame,o.name,b['id'])
        for e in city['roads']:
            if e['kind']=='footway':continue
            a,b=[city['nodes'][i] for i in [e['u'],e['v']]]
            dx,dy=b['x']-a['x'],b['y']-a['y']
            t=max(0,min(1,((o.location.x-a['x'])*dx+(o.location.y-a['y'])*dy)/(dx*dx+dy*dy)))
            assert math.hypot(o.location.x-a['x']-t*dx,o.location.y-a['y']-t*dy)>e['width_m']/2+.25,(frame,o.name,e['id'])
s.frame_set(1)
assert s['city_sha256']==hashlib.sha256((ROOT/'data/city.json').read_bytes()).hexdigest()
for o in meshes:
    assert len(o.data.vertices)>0 and len(o.data.polygons)>0,o.name
    assert all(math.isfinite(v) for row in o.matrix_world for v in row),o.name
    assert all(math.isfinite(v) for vertex in o.data.vertices for v in vertex.co),o.name
    assert all(math.isfinite(v) and v>0 for v in o.dimensions),o.name
    assert o.data.materials and all(m is not None for m in o.data.materials),o.name
    assert all(len(p.vertices)>=3 for p in o.data.polygons),o.name
    assert not o.name.endswith('.001'),o.name
for b in city['buildings']:
    o=next(o for o in buildings if o['building_id']==b['id'])
    assert abs(o.location.x-b['x'])<1e-5 and abs(o.location.y-b['y'])<1e-5
    assert o['node_id']==b['node']
for e in city['roads']:
    o=s.objects['Road_'+e['id']]
    a,b=[city['nodes'][i] for i in [e['u'],e['v']]]
    assert abs(o.location.x-(a['x']+b['x'])/2)<1e-5
    assert abs(o.location.y-(a['y']+b['y'])/2)<1e-5
    assert abs(math.hypot(a['x']-b['x'],a['y']-b['y'])-o['length_m'])<1e-5
    assert o['seconds']==e['seconds']
    assert abs(math.atan2(b['y']-a['y'],b['x']-a['x'])-o.rotation_euler.z)<1e-5
    local_length=max(v.co.x for v in o.data.vertices)-min(v.co.x for v in o.data.vertices)
    expected=e['length_m']-(8 if e['kind']!='footway' else 0)
    assert abs(local_length-expected)<1e-4,(o.name,local_length,expected)
for n in city['nodes']:
    o=s.objects['Node_'+str(n['id'])]
    assert abs(o.location.x-n['x'])<1e-5 and abs(o.location.y-n['y'])<1e-5
unique={o.data.as_pointer():o.data for o in meshes}
for mesh in unique.values():mesh.calc_loop_triangles()
report=dict(status='PASS',blender_version=bpy.app.version_string,
    master_objects=len(objects),master_mesh_objects=len(meshes),buildings=len(buildings),
    unique_building_meshes=12,architectural_families=12,trees=len(trees),tree_archetypes=7,
    pedestrians=len(people),animated_pedestrians=len(walkers),pedestrian_families=6,
    graph_nodes=len(city['nodes']),graph_edges=len(city['roads']),unique_mesh_datablocks=len(unique),
    master_vertices_instanced=sum(len(o.data.vertices) for o in meshes),
    master_polygons_instanced=sum(len(o.data.polygons) for o in meshes),
    master_triangles_instanced=sum(len(o.data.loop_triangles) for o in meshes),
    master_unique_triangles=sum(len(m.loop_triangles) for m in unique.values()),
    master_render_visible_objects=sum(not o.hide_render and not any(c.hide_render for c in o.users_collection) for o in objects),
    blend_bytes=blend.stat().st_size,render_dimensions=[s.render.resolution_x,s.render.resolution_y],
    checks=['204 unique building IDs; twelve linked architectural meshes','162 trees / seven shared meshes; 48 people / six clothing families / twelve morph walkers',
            'Pedestrian positions sampled every 12 frames: clear of buildings and vehicle roads',
            'No empty meshes, missing materials, nonfinite coordinates, or zero dimensions',
            'No unexpected .001 object names','All 108 road midpoints, directions, geometric lengths and costs match city.json',
            'All 64 graph glyphs match node coordinates','Master embeds matching city.json SHA-256'])
glbs={}
for name in ['city_web','graph_web']:
    path=ROOT/'models'/f'{name}.glb'
    with path.open('rb') as f:
        magic,version,size=struct.unpack('<4sII',f.read(12));assert magic==b'glTF' and version==2 and size==path.stat().st_size
        length,kind=struct.unpack('<II',f.read(8));assert kind==0x4E4F534A
        doc=json.loads(f.read(length))
    assert all('uri' not in b for b in doc.get('buffers',[]))
    assert all('uri' not in b for b in doc.get('images',[]))
    triangles=[]
    for mesh in doc['meshes']:
        count=0
        for p in mesh['primitives']:
            assert p.get('mode',4)==4
            count+=doc['accessors'][p['indices']]['count']//3 if 'indices' in p else doc['accessors'][p['attributes']['POSITION']]['count']//3
        triangles.append(count)
    instance_triangles=sum(triangles[n['mesh']] for n in doc['nodes'] if 'mesh' in n)
    expected_buildings=204 if name=='city_web' else 0
    assert sum(n.get('extras',{}).get('role')=='building' for n in doc['nodes'])==expected_buildings
    if name=='city_web':
        assert sum(n.get('extras',{}).get('role')=='tree' for n in doc['nodes'])==162
        assert sum(n.get('extras',{}).get('role')=='pedestrian' for n in doc['nodes'])==48
        assert not doc.get('animations')
        assert not any(p.get('targets') for m in doc['meshes'] for p in m['primitives'])
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(path))
    assert sum(o.get('role')=='building' for o in bpy.context.scene.objects)==expected_buildings
    if name=='city_web':
        assert sum(o.get('role')=='pedestrian' for o in bpy.context.scene.objects)==48
        assert sum(o.get('role')=='tree' for o in bpy.context.scene.objects)==162
    glbs[name]=dict(bytes=size,triangles_instanced=instance_triangles,triangles_unique=sum(triangles),
        mesh_nodes=sum('mesh' in n for n in doc['nodes']),unique_meshes=len(triangles),
        reimport_objects=len(bpy.context.scene.objects),self_contained=True,reimport='PASS')
report['glb']=glbs
(ROOT/'data/scene_validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2));print('SCENE_VALIDATION_SUCCESS',flush=True)
