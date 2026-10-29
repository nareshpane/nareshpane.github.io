"""Record final status and ensure outside-scope status is unchanged from this task's start."""
from pathlib import Path
import json, subprocess
ASSETS=Path(__file__).resolve().parent
status=subprocess.check_output(['git','status','--short'],cwd=ASSETS.parent).decode('utf-8')
before=(ASSETS/'qa/presentation-start-status.txt').read_text(encoding='utf-8-sig')
def outside(text):
    return sorted(line for line in text.splitlines() if line and not line[3:].startswith(ASSETS.name))
assert outside(status)==outside(before),(outside(status),outside(before))
staged=subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ASSETS.parent).decode().strip()
assert not staged,staged
subprocess.run(['git','diff','--check'],cwd=ASSETS.parent,check=True)
(ASSETS/'qa/presentation-final-status.txt').write_text(status,encoding='utf-8')
report=ASSETS/'PRESENTATION_REFINEMENT_REPORT.md'
text=report.read_text(encoding='utf-8').split('<!-- FINAL_GIT_STATUS -->')[0]
report.write_text(text+'<!-- FINAL_GIT_STATUS -->\n\nFinal `git status --short` (includes pre-existing work):\n\n```text\n'+status+'```\n',encoding='utf-8')
audit={'status':'PASS','outsideScopeStatus':'Matches task-start snapshot','stagedFiles':0,'diffWhitespace':'PASS','priorDeletedPreviews':'Preserved from task-start status','unrelatedUntrackedDirectories':'Preserved'}
(ASSETS/'qa/presentation-scope-audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
print(json.dumps(audit,indent=2));print(status)
