"""Five-shot Eevee reel, derived from (never overwriting) city_master.blend.

blender --background --threads 6 --python-exit-code 1 --python scripts/render_cinematic.py -- --test
Then --render after inspecting the benchmark frames. Raw frames stay in a
caller-supplied temporary directory; the editable camera scene is retained.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time

import bpy
from mathutils import Vector

ROOT = Path(__file__).resolve().parents[1]
SHOTS = [
    dict(name='01 Establishing aerial', start=1, end=120, lens=(42, 43),
         positions=[(480,-580,560), (420,-510,472), (360,-440,395)],
         targets=[(0,0,10), (5,5,12), (10,10,15)]),
    dict(name='02 Innovation District sweep', start=121, end=240, lens=(36, 39),
         positions=[(280,-65,115), (292,30,125), (270,130,135)],
         targets=[(100,67,30), (96,72,32), (93,75,34)]),
    dict(name='03 Transit street tracking', start=241, end=360, lens=(28, 30),
         positions=[(119,-126,3.4), (119,-92,4.1), (119,-58,4.8)],
         targets=[(118,-50,5), (118,-16,5.7), (118,18,6.4)]),
    dict(name='04 River and northern bridge', start=361, end=480, lens=(35, 37),
         positions=[(-14,-38,32), (-4,-14,37), (14,8,42)],
         targets=[(0,48,1), (8,48,3), (12,52,5)]),
    dict(name='05 Full-city reveal', start=481, end=576, lens=(43, 42),
         positions=[(300,-345,270), (390,-462,409), (480,-580,560)],
         targets=[(10,12,18), (5,6,14), (0,0,10)]),
]
TEST_FRAMES = [1, 180, 300, 420, 576]


def smooth_keys(id_block):
    """Auto-clamped Bezier handles ease each shot without overshoot."""
    action = id_block.animation_data.action
    for curve in action.fcurves:
        for key in curve.keyframe_points:
            key.interpolation = 'BEZIER'
            key.handle_left_type = key.handle_right_type = 'AUTO_CLAMPED'


def configure_lighting(s,samples=16):
    s.render.engine = 'BLENDER_EEVEE_NEXT'
    s.eevee.taa_render_samples = samples
    s.eevee.use_raytracing = False
    s.render.image_settings.file_format = 'PNG'
    s.render.image_settings.color_mode = 'RGB'
    s.render.image_settings.compression = 15
    s.render.use_file_extension = True
    s.render.film_transparent = False
    s.view_settings.view_transform = 'AgX'
    s.view_settings.look = 'AgX - Medium High Contrast'
    s.view_settings.exposure = .10
    # Soft warm key / cool fill. No heavy volumes, ray tracing, DOF or blur:
    # keep this small-scale model legible and the integrated-GPU render modest.
    bpy.data.lights['Afternoon sun'].energy = 2.4
    bpy.data.lights['Afternoon sun'].angle = .10
    bpy.data.objects['Afternoon sun'].rotation_euler = (.48, -.62, -.40)
    bpy.data.lights['Warm softbox'].energy = 1250000
    bpy.data.lights['Sky fill'].energy = 1100000
    for light in bpy.data.lights:
        light.use_shadow_jitter = False
        # The two enormous softboxes act as ambient fill. One sun shadow map
        # supplies contact/directional shadows without noisy area-light sampling.
        if light.type == 'AREA': light.use_shadow = False
    bg = s.world.node_tree.nodes['Background']
    bg.inputs[0].default_value = (.52, .64, .72, 1)
    bg.inputs[1].default_value = .55
    bpy.data.objects['Studio floor • excluded from GLB'].scale = (100,100,1)


def make_scene(samples):
    bpy.ops.wm.open_mainfile(filepath=str(ROOT/'blender/city_master.blend'))
    s=bpy.context.scene;s.name='Astra city • cinematic camera reel'
    configure_lighting(s,samples)
    s.render.resolution_x,s.render.resolution_y=1280,720
    s.render.resolution_percentage=100;s.render.fps=24
    s.frame_start,s.frame_end=1,576
    # The street shot looks above the studio floor: a gently coloured horizon
    # remains uncluttered, without introducing fictional surrounding buildings.
    c = bpy.data.collections.new('13 Cinematic cameras and framing targets')
    s.collection.children.link(c)
    s.timeline_markers.clear()
    for shot in SHOTS:
        data = bpy.data.cameras.new(shot['name'])
        data.type = 'PERSP'
        data.clip_start, data.clip_end = .15, 5000
        data.sensor_width = 36
        if shot['start'] in [1,481]: data.shift_y = -.055
        cam = bpy.data.objects.new(shot['name'], data)
        c.objects.link(cam)
        target = bpy.data.objects.new(shot['name']+' • aim', None)
        target.empty_display_size = 3
        c.objects.link(target)
        track = cam.constraints.new('TRACK_TO')
        track.target = target
        track.track_axis, track.up_axis = 'TRACK_NEGATIVE_Z', 'UP_Y'
        frames = [shot['start'], (shot['start']+shot['end'])/2, shot['end']]
        for frame, pos, aim in zip(frames, shot['positions'], shot['targets']):
            cam.location, target.location = pos, aim
            cam.keyframe_insert('location', frame=frame)
            target.keyframe_insert('location', frame=frame)
        for frame, lens in zip([shot['start'], shot['end']], shot['lens']):
            data.lens = lens
            data.keyframe_insert('lens', frame=frame)
        for item in [cam, target, data]: smooth_keys(item)
        marker = s.timeline_markers.new(shot['name'], frame=shot['start'])
        marker.camera = cam
    s.camera = bpy.data.objects[SHOTS[0]['name']]
    s.frame_set(1)
    s['cinematic_notes'] = 'Five perspective shots; Bezier camera/target/lens keys; deliberate cuts; final pose matches opening. Original geometry and analysis unchanged.'
    bpy.context.view_layer.update()
    # Validate all 576 camera poses against building bounds, not just samples.
    collisions = []
    buildings = [o for o in s.objects if o.get('role') == 'building']
    for frame in range(1, 577):
        s.frame_set(frame)
        p = s.camera.matrix_world.translation
        for obj in buildings:
            local = obj.matrix_world.inverted() @ p
            bounds = obj.bound_box
            if all(min(v[k] for v in bounds)-.02 <= local[k] <= max(v[k] for v in bounds)+.02 for k in range(3)):
                collisions.append(dict(frame=frame, building=obj.name))
    assert not collisions, collisions
    s.frame_set(1)
    assert len(buildings) == 204
    return s


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--test', action='store_true')
    p.add_argument('--render', action='store_true')
    p.add_argument('--samples', type=int, default=16)
    p.add_argument('--frames', help='Comma-separated benchmark frames')
    p.add_argument('--output', type=Path, help='Raw frame directory (outside repository recommended)')
    args = p.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
    assert args.test != args.render, 'Choose --test or --render'
    master = ROOT/'blender/city_master.blend'
    before = hashlib.sha256(master.read_bytes()).hexdigest()
    setup_start = time.perf_counter()
    s = make_scene(args.samples)
    setup_seconds = time.perf_counter()-setup_start
    output = args.output or Path(tempfile.mkdtemp(prefix='astra-cinematic-frames-'))
    output.mkdir(parents=True, exist_ok=True)
    frames = [int(f) for f in args.frames.split(',')] if args.frames else TEST_FRAMES if args.test else range(1,577)
    if args.render:
        s.render.filepath = str(output/'frame_')
        bpy.ops.wm.save_as_mainfile(filepath=str(ROOT/'blender/city_cinematic.blend'), compress=True)
    timings = []
    report_path = ROOT/'data'/('cinematic_benchmark.json' if args.test else 'cinematic_runtime.json')
    report = dict(status='RUNNING', blender_version=bpy.app.version_string,
                  engine=s.render.engine, samples=args.samples, resolution=[1280,720],
                  fps=24, frames=576, duration_seconds=24, shots=SHOTS,
                  master_sha256=before, camera_building_collisions=0,
                  setup_seconds=round(setup_seconds,3), raw_frame_directory=str(output),
                  frame_timings=timings)
    started = time.perf_counter()
    for frame in frames:
        tick = time.perf_counter()
        s.frame_set(frame)
        s.render.filepath = str(output/f'frame_{frame:04}.png')
        bpy.ops.render.render(write_still=True)
        elapsed = round(time.perf_counter()-tick,3)
        timings.append(dict(frame=frame, seconds=elapsed))
        print(f'CINEMATIC_FRAME {frame} {elapsed:.3f}s', flush=True)
        report['elapsed_render_seconds'] = round(time.perf_counter()-started,3)
        report_path.write_text(json.dumps(report, indent=2)+'\n')
    assert before == hashlib.sha256(master.read_bytes()).hexdigest()
    report['status'] = 'PASS'
    report['average_frame_seconds'] = round(sum(t['seconds'] for t in timings)/len(timings),3)
    report['estimated_full_render_seconds'] = round(report['average_frame_seconds']*576,1)
    # First frame includes cold shader setup; steady-state predicts long runs.
    steady = timings[1:] or timings
    report['steady_state_frame_seconds'] = round(sum(t['seconds'] for t in steady)/len(steady),3)
    report['warm_estimated_full_render_seconds'] = round(timings[0]['seconds']+575*report['steady_state_frame_seconds'],1)
    report['master_unchanged'] = True
    report_path.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['shots','frame_timings']}, indent=2), flush=True)


if __name__ == '__main__': main()
