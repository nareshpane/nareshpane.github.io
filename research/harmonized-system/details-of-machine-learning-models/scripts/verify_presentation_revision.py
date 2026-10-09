"""Protect existing computations and unrelated repository files during presentation edits."""
from pathlib import Path
import hashlib,json,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parents[2]
BASELINE=ROOT/'results/presentation-baseline.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def protected():
    paths=[ROOT/'toy_trade_dataset.csv',ROOT/'data_dictionary.csv',ROOT/'dataset_build.py',ROOT/'scripts/analyze_models.py',ROOT/'scripts/chapter_content.py',ROOT/'data_sources.md',ROOT/'model_inventory.md']
    paths+=list((ROOT/'data').glob('*'))
    paths += [p for p in (ROOT/'results').glob('*') if p.is_file() and p.suffix in ['.csv','.json'] and not p.name.startswith(('presentation-','browser_'))]
    paths+=list((ROOT/'results/figures').glob('*.svg'))
    names=subprocess.check_output(['git','ls-files'],cwd=REPO,text=True).splitlines()
    for name in names:
        p=REPO/name
        if p.is_file() and ROOT not in p.parents and p!=ROOT.parent/'details-of-machine-learning-models.html':paths.append(p)
    return sorted(set(p for p in paths if p.is_file()))
def main():
    if '--baseline' in sys.argv:
        assert not BASELINE.exists(),'Do not overwrite initial baseline'
        content={str(p.relative_to(REPO)):sha(p) for p in protected()}
        BASELINE.write_text(json.dumps(content,indent=2),encoding='utf-8')
        print('Protected baseline:',len(content),'files')
        return
    content=json.loads(BASELINE.read_text());changes=[]
    for name,h in content.items():
        p=REPO/name
        if not p.exists() or sha(p)!=h:changes.append(name)
    assert not changes,'Unexpected changes: '+repr(changes)
    report={'status':'PASS','protected_files':len(content),'canonical_data_predictions_parameters_metrics_unchanged':True,'unrelated_repository_files_unchanged':True,'models_refitted':False}
    (ROOT/'results/presentation-integrity.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
