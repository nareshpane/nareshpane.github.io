"""WebP derivatives, media validation and a checksummed asset inventory.

Use --keep-images when validating a media-only revision without rewriting the
existing, already-reviewed stills. Cinematic poster packaging is separate.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
charts={'city_statistics','monte_carlo','accessibility_statistics','graph_map'}
for path in sorted((ROOT/'renders').glob('*.png')):
    if '--keep-images' in sys.argv:break
    if path.stem in charts or path.stem.startswith('qa_'):continue
    with Image.open(path) as img:
        assert img.size in [(1440,1000),(1440,680)],(path,img.size)
        img.convert('RGB').save(path.with_suffix('.webp'),quality=86,method=6)
if '--images-only' in sys.argv:
    print('Web image derivatives generated');raise SystemExit(0)
videos=[]
expected={name:(960,540,224,14,'16/1') for name in ['shortest_path.mp4','bridge_failure.mp4','transit_accessibility.mp4']}
expected['astra_city_cinematic.mp4']=(1280,720,576,24,'24/1')
for path in sorted((ROOT/'video').glob('*.mp4')):
    result=subprocess.run(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(path)],capture_output=True,text=True,check=True)
    doc=json.loads(result.stdout);v=next(s for s in doc['streams'] if s['codec_type']=='video')
    width,height,frames,duration,fps=expected[path.name]
    assert v['codec_name']=='h264' and v['pix_fmt']=='yuv420p' and [v['width'],v['height']]==[width,height]
    assert int(v['nb_frames'])==frames and abs(float(doc['format']['duration'])-duration)<.1
    assert v['avg_frame_rate']==fps
    assert all(v[k]=='bt709' for k in ['color_space','color_transfer','color_primaries'])
    subprocess.run(['ffmpeg','-v','error','-i',str(path),'-f','null','-'],check=True)
    videos.append(dict(name=path.name,codec=v['codec_name'],pixel_format=v['pix_fmt'],color_space=v['color_space'],fps=v['avg_frame_rate'],frames=int(v['nb_frames']),width=v['width'],height=v['height'],duration=float(doc['format']['duration']),bytes=path.stat().st_size))
assert {v['name'] for v in videos}==set(expected),'Expected the cinematic reel and all three technical animations'
(ROOT/'data/media_validation.json').write_text(json.dumps(dict(status='PASS',videos=videos),indent=2)+'\n')
inventory=[]
for path in sorted(ROOT.rglob('*')):
    if not path.is_file() or '__pycache__' in path.parts or path.suffix=='.blend1' or path.name=='file_manifest.json':continue
    inventory.append(dict(path=str(path.relative_to(ROOT)),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
(ROOT/'data/file_manifest.json').write_text(json.dumps(dict(files=inventory,excludes=['file_manifest.json (self-referential)','__pycache__','*.blend1 (ignored local backups)']),indent=2)+'\n')
print(f'PACKAGING_PASS: {len(inventory)} files; {len(videos)} videos fully decoded')
