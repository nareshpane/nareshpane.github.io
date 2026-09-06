"""Inspect the saved cinematic scene without rendering or changing the master.

blender --background --python-exit-code 1 --python scripts/validate_cinematic.py
"""
import hashlib
import json
import math
from pathlib import Path
import sys

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from render_cinematic import SHOTS


def mesh_signature(obj):
    # Raw topology and material assignments, including every shared detail.
    mesh = obj.data
    payload = [tuple(tuple(v.co) for v in mesh.vertices),
               tuple((tuple(p.vertices),p.material_index) for p in mesh.polygons),
               tuple(slot.material.name for slot in obj.material_slots)]
    return hashlib.sha256(repr(payload).encode()).hexdigest()


master = ROOT/'blender/city_master.blend'
before = hashlib.sha256(master.read_bytes()).hexdigest()
bpy.ops.wm.open_mainfile(filepath=str(master))
original = {o.name:dict(signature=mesh_signature(o),matrix=tuple(tuple(row) for row in o.matrix_world))
            for o in bpy.context.scene.objects if o.type=='MESH'}
master_count = len(bpy.context.scene.objects)
bpy.ops.wm.open_mainfile(filepath=str(ROOT/'blender/city_cinematic.blend'))
s = bpy.context.scene
meshes = [o for o in s.objects if o.type=='MESH']
assert {o.name for o in meshes} == set(original)
for o in meshes:
    assert mesh_signature(o) == original[o.name]['signature'],o.name
    if o.name != 'Studio floor • excluded from GLB':
        assert tuple(tuple(row) for row in o.matrix_world) == original[o.name]['matrix'],o.name
assert len(s.objects) == master_count + 10 # five cameras, five empty aim targets
assert len([o for o in s.objects if o.get('role')=='building']) == 204
assert s.render.engine=='BLENDER_EEVEE_NEXT'
assert (s.frame_start,s.frame_end,s.render.fps)==(1,576,24)
assert (s.render.resolution_x,s.render.resolution_y)==(1280,720)
assert [(m.frame,m.camera.name) for m in s.timeline_markers] == [(q['start'],q['name']) for q in SHOTS]
assert len(s.timeline_markers)==5
motions = []
for shot in SHOTS:
    prior = None
    distances,angles = [],[]
    for frame in range(shot['start'],shot['end']+1):
        s.frame_set(frame)
        assert s.camera.name==shot['name']
        matrix = s.camera.matrix_world.copy()
        assert all(math.isfinite(v) for row in matrix for v in row)
        assert matrix.translation.z >= 3.39
        if prior is not None:
            distances.append((matrix.translation-prior.translation).length)
            angles.append(math.degrees(matrix.to_quaternion().rotation_difference(prior.to_quaternion()).angle))
        prior = matrix
    assert max(angles)<2.5,(shot['name'],max(angles))
    # Easing at both ends: movement per frame is much smaller than mid-shot.
    assert distances[0] < max(distances)*.10 and distances[-1] < max(distances)*.10
    motions.append(dict(shot=shot['name'],max_metres_per_frame=round(max(distances),4),
                        max_degrees_per_frame=round(max(angles),4),eased_ends=True))
poses = []
framing = []
for frame in [1,576]:
    s.frame_set(frame)
    poses.append((s.camera.matrix_world.copy(),s.camera.data.lens,s.camera.data.shift_y))
    projected=[]
    for obj in meshes:
        if obj.name.startswith('Studio floor') or obj.hide_render or any(c.hide_render for c in obj.users_collection): continue
        projected += [world_to_camera_view(s,s.camera,obj.matrix_world @ Vector(corner)) for corner in obj.bound_box]
    bounds = [min(v.x for v in projected),min(v.y for v in projected),max(v.x for v in projected),max(v.y for v in projected)]
    assert all(.01<v<.99 for v in bounds),bounds
    framing.append(dict(frame=frame,normalized_bounds=[round(v,4) for v in bounds],full_city_visible=True))
assert all(abs(a-b)<1e-5 for ra,rb in zip(poses[0][0],poses[1][0]) for a,b in zip(ra,rb))
assert poses[0][1:] == poses[1][1:]
assert hashlib.sha256(master.read_bytes()).hexdigest()==before
report = dict(status='PASS',blender_version=bpy.app.version_string,objects=len(s.objects),
              mesh_objects=len(meshes),buildings=204,added_cameras=5,added_empty_targets=5,
              original_meshes_and_material_assignments_unchanged=True,
              original_city_transforms_unchanged=True,
              presentation_only_mesh_transform='Studio floor scaled in XY to remove its nearby horizon edge',
              master_unchanged=True,master_sha256=before,loop_camera_pose_identical=True,
              camera_motion=motions,full_city_framing=framing,
              blend_bytes=(ROOT/'blender/city_cinematic.blend').stat().st_size)
(ROOT/'data/cinematic_scene_validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
