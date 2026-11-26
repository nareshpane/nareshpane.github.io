"""Run the complete workload from any working directory."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

SCRIPTS = Path(__file__).resolve().parent
ORDER = ('00_download_data.py', '01_prepare_geographies.py', '05_socioeconomic_geography.py',
         '02_healthcare_accessibility.py', '03_wildfire_exposure.py',
         '04_flood_infrastructure.py', '06_export_web_assets.py')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offline', action='store_true', help='Use previously downloaded inputs')
    args = parser.parse_args()
    failed = []
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    for script in ORDER:
        if script.startswith('00') and args.offline:
            continue
        print(f'\n{script}', flush=True)
        result = subprocess.run([sys.executable, str(SCRIPTS / script)], env=environment)
        if result.returncode:
            failed.append(script)
    if failed:
        print('Failed stages: ' + ', '.join(failed), file=sys.stderr)
    return bool(failed)


if __name__ == '__main__':
    sys.exit(main())

