"""One-time migration; only explicit assets in this project's root are moved.

Original PDFs are never moved or edited. The saved manifest records original hashes.
"""
from pathlib import Path
import hashlib
import json
ROOT = Path(__file__).resolve().parents[1]
if __name__ == '__main__':
    inventory = []
    for path in sorted(ROOT.iterdir()):
        if not path.is_file():
            continue
        folder = ('figures/historical/graph-views' if path.name.endswith('_graph_view.png') else
                  'figures/historical/partition-views' if path.name.endswith('_automorphism_or_partition_view.png') else
                  'python/historical' if path.suffix == '.py' else
                  'data/historical' if path.suffix == '.txt' else '')
        dest = ROOT / folder / path.name
        inventory.append({'original': path.name, 'bytes': path.stat().st_size,
                          'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                          'destination': dest.relative_to(ROOT).as_posix()})
        if folder:
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                raise FileExistsError(dest)
            path.rename(dest)
    target = ROOT / 'data/validation/original-asset-inventory.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.write_text(json.dumps(inventory, indent=2), encoding='utf-8')
