"""Atomic checkpoint and CSV/JSON export, including execution provenance."""
import csv
import json
from pathlib import Path
from .statistics import summarize


def atomic_json(path, data):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2) + '\n', encoding='utf8')
    temp.replace(path)


def export(out, state):
    rows = [summarize(r) for r in state['rows'] if r['trials']]
    atomic_json(out / 'results.json', {'metadata': state['metadata'], 'rows': rows})
    if rows:
        with (out / 'results.csv').open('w', newline='', encoding='utf8') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
