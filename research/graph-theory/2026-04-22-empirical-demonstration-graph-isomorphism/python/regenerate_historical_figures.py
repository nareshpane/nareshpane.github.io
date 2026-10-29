"""Regenerate the preserved prism figures into a separate review output directory.

Optional historical dependencies: matplotlib, numpy, networkx. Does not overwrite
the byte-preserved PNGs or the inventory. The original generator remains unchanged.
"""
from pathlib import Path
import importlib.util
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'python/historical/generate_automorphism_demo_placeholders_fixed.py'

def main():
    if len(sys.argv) != 2:
        raise SystemExit('Usage: python regenerate_historical_figures.py OUTPUT_DIRECTORY')
    output = Path(sys.argv[1]).resolve()
    if output == ROOT or ROOT in output.parents:
        raise SystemExit('Use a review output directory outside the preserved project assets.')
    output.mkdir(parents=True,exist_ok=True)
    spec=importlib.util.spec_from_file_location('historical_prism_generator',SOURCE)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    for section in module.sections():
        module.render_graph_view(section,output/f'{section.num:02}_graph_view.png')
        module.render_automorphism_view(section,output/f'{section.num:02}_automorphism_or_partition_view.png')
    module.export_nauty_friendly_data(output)
    print(output)

if __name__=='__main__':
    main()
