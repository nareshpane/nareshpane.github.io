"""Persist imagegen's returned PNG bytes when its temporary save fails.

This copies bytes only: no raster processing, resizing or mathematical calculations.
The destination is fixed inside this project and cannot target other files.
"""
from pathlib import Path
import argparse,base64
DEST=Path(__file__).resolve().parents[1]/'images/algorithm-06.png'
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--init',action='store_true');parser.add_argument('--append');args=parser.parse_args()
    if args.init:DEST.write_bytes(b'')
    if args.append:
        assert len(args.append)<=20000
        data=base64.b64decode(args.append,validate=True)
        with DEST.open('ab') as f:f.write(data)
