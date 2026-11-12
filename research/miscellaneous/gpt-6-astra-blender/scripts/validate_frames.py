"""Decode every raw frame and reject missing, wrong-size or blank images."""
from pathlib import Path
import sys
from PIL import Image,ImageStat
directory=Path(sys.argv[1]);count,width,height=map(int,sys.argv[2:5])
for frame in range(1,count+1):
    with Image.open(directory/f'frame_{frame:04}.png') as im:
        assert im.size==(width,height),(frame,im.size)
        stats=ImageStat.Stat(im.convert('L').resize((120,68)))
        assert 20<stats.mean[0]<245 and stats.stddev[0]>8,(frame,stats.mean,stats.stddev)
print(f'FRAME_VALIDATION_PASS: {count} decoded, nonblank {width}x{height} frames')
