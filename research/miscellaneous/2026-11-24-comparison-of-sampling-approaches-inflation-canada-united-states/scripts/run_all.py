"""Rebuild the data, figures and replication bundle using documented snapshots."""
import argparse
import subprocess
import sys
from zipfile import ZipFile, ZIP_DEFLATED
from acquire import ROOT

def bundle():
    archive = ROOT / 'replication.zip'
    top_page = ROOT.with_suffix('.html')
    excluded = {'replication.zip', 'worktree-initial.tmp'}
    geography_inputs = {
        'data/metadata/ne_admin1.geojson', 'data/raw/canada/18100004-eng.zip',
        'data/raw/canada/18100007-eng.zip', 'data/raw/united-states/api-2010-2019.json',
        'data/raw/united-states/api-2020-2026.json',
    }
    with ZipFile(archive, 'w', ZIP_DEFLATED) as zip_file:
        zip_file.write(top_page,top_page.name)
        for path in sorted(ROOT.rglob('*')):
            relative = path.relative_to(ROOT)
            if not path.is_file() or '__pycache__' in relative.parts or path.name in excluded:
                continue
            raw_input = relative.parts[:2] == ('data','raw') or path.suffix == '.geojson'
            if raw_input and not path.name.startswith('geo-') and relative.as_posix() not in geography_inputs:
                continue
            zip_file.write(path, ROOT.name + '/' + relative.as_posix())
    print(f'Packaged {archive.name} ({archive.stat().st_size:,} bytes)')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download',action='store_true',help='Refresh external sources before rebuilding')
    parser.add_argument('--bundle-only',action='store_true',help='Archive the current study without rebuilding data')
    args=parser.parse_args()
    if not args.bundle_only:
        files=['prepare_data.py','empirical_experiments.py','make_maps.py','prepare_geography.py','simulate.py']
        if args.download:
            files[:0]=['acquire.py','acquire_geography.py']
        for name in files:
            subprocess.run([sys.executable,str(ROOT/'scripts'/name)],check=True)
    bundle()

if __name__=='__main__':
    main()

