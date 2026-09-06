"""List actual changed/generated project files against an explicit local backup."""
import argparse
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True);a=p.parse_args()
def files(root):
    return {str(p.relative_to(root)):p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ['.blend1','.pyc'] and p.name!='world_revision_files.json'}
old=files(a.baseline/'assets');current=files(ROOT)
new=sorted(set(current)-set(old))
changed=sorted(k for k in set(current)&set(old) if hashlib.sha256(current[k].read_bytes()).digest()!=hashlib.sha256(old[k].read_bytes()).digest())
removed=sorted(set(old)-set(current))
assert not removed,removed
assert (ROOT.parent.parent/'research.html').read_bytes()==(a.baseline/'research.html').read_bytes()
report=dict(created=['research/gpt-6-astra-blender/'+k for k in new]+['research/gpt-6-astra-blender/data/world_revision_files.json'],
    modified=['research/gpt-6-astra-blender.html']+['research/gpt-6-astra-blender/'+k for k in changed],removed=[],
    research_index_unchanged_this_revision=True,local_ignored_backups_excluded=True,no_commit_or_push=True)
(ROOT/'data/world_revision_files.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:len(v) if isinstance(v,list) else v for k,v in report.items()},indent=2))
