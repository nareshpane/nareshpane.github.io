"""Package the completed Eevee frames; retain a small poster, not raw frames.

python3 scripts/encode_cinematic.py --frames /tmp/astra-cinematic-final
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

from PIL import Image, ImageChops, ImageStat

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument('--frames', type=Path, required=True)
args = p.parse_args()
runtime = json.loads((ROOT/'data/cinematic_runtime.json').read_text())
assert runtime['status'] == 'PASS' and len(runtime['frame_timings']) == 576
assert {t['frame'] for t in runtime['frame_timings']} == set(range(1,577))
assert runtime['master_sha256'] == hashlib.sha256((ROOT/'blender/city_master.blend').read_bytes()).hexdigest()
previous = first = None
frame_changes = []
for frame in range(1,577):
    with Image.open(args.frames/f'frame_{frame:04}.png') as img:
        assert img.size == (1280,720)
        # Decode every PNG, reject empty/black frames, and measure continuity.
        thumb = img.convert('L').resize((160,90))
        stats = ImageStat.Stat(thumb)
        assert 30 < stats.mean[0] < 245 and stats.stddev[0] > 10,frame
        if previous is not None:
            change = ImageStat.Stat(ImageChops.difference(thumb,previous)).mean[0]
            frame_changes.append(dict(frame=frame,mean_luma_change=round(change,4)))
        else: first=thumb.copy()
        previous=thumb
loop_change = ImageStat.Stat(ImageChops.difference(first,previous)).mean[0]
assert loop_change < 1, 'Rendered loop endpoints do not match the validated camera poses'
noncuts = [c['mean_luma_change'] for c in frame_changes if c['frame'] not in [121,241,361,481]]
assert max(noncuts) < 20, 'Unexpected visual discontinuity within a shot'
started = time.perf_counter()
video = ROOT/'video/astra_city_cinematic.mp4'
subprocess.run(['ffmpeg','-y','-v','warning','-framerate','24',
                '-i',str(args.frames/'frame_%04d.png'),'-frames:v','576',
                '-an','-c:v','libx264','-profile:v','baseline','-level:v','3.1',
                '-crf','22','-preset','medium','-pix_fmt','yuv420p',
                '-vf','scale=out_color_matrix=bt709:out_range=tv',
                '-color_primaries','bt709','-color_trc','bt709','-colorspace','bt709',
                '-movflags','+faststart','-threads','4',str(video)],check=True)
probe = json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams',
                                         '-show_format','-of','json',str(video)]))
v = probe['streams'][0]
assert len(probe['streams']) == 1, 'Silent reel must not contain an audio stream'
assert v['codec_name'] == 'h264' and v['pix_fmt'] == 'yuv420p'
assert v['color_space'] == 'bt709'
assert [v['width'],v['height']] == [1280,720]
assert int(v['nb_frames']) == 576 and v['avg_frame_rate'] == '24/1'
assert abs(float(probe['format']['duration'])-24) < .01
subprocess.run(['ffmpeg','-v','error','-i',str(video),'-f','null','-'],check=True)
# Check MP4 atom ordering: the metadata precedes media for progressive playback.
atoms = []
with video.open('rb') as stream:
    while header := stream.read(8):
        size = int.from_bytes(header[:4], 'big')
        kind = header[4:].decode('ascii')
        atoms.append(kind)
        if size == 1:
            size = int.from_bytes(stream.read(8),'big')
            stream.seek(size-16,1)
        elif size >= 8: stream.seek(size-8,1)
        else: break
assert atoms.index('moov') < atoms.index('mdat')
posters = ROOT/'renders/cinematic'
posters.mkdir(parents=True,exist_ok=True)
with Image.open(args.frames/'frame_0048.png') as img:
    img.save(posters/'poster.png',optimize=True)
    img.save(posters/'poster.webp',quality=90,method=6)
report = dict(status='PASS',codec=v['codec_name'],profile=v['profile'],
              pixel_format=v['pix_fmt'],color_space=v['color_space'],width=v['width'],height=v['height'],
              duration_seconds=float(probe['format']['duration']),frame_count=int(v['nb_frames']),
              fps=v['avg_frame_rate'],bytes=video.stat().st_size,no_audio=True,
              fast_start=True,all_frames_decoded=True,poster_frame=48,
              all_raw_frames_nonblank=True,loop_endpoint_luma_difference=round(loop_change,4),
              maximum_within_shot_luma_change=round(max(noncuts),4),
              encoding_validation_seconds=round(time.perf_counter()-started,3),
              rendering_seconds=runtime['elapsed_render_seconds'],
              average_render_seconds=runtime['average_frame_seconds'],
              raw_frames_published=False,master_unchanged=True)
(ROOT/'data/cinematic_media_validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
