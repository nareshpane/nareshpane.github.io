"""Presentation-only rendering from saved artifacts; no training or downloads.

Run: python research/harmonized-system/4-hs-data-analysis/format_model_comparison.py
Also called by build_model_comparison.py so the editorial changes survive builds.
Only the target HTML is written. The numerical CSV/JSON artifacts remain read-only.
"""
import csv
import json
import math
import re
from decimal import Decimal, ROUND_HALF_UP
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PAGE = ROOT.parent / 'machine-learning-trade-sector-prediction.html'
METRIC_DEFINITIONS = ('WAPE = Weighted Absolute Percentage Error; MAE = Mean Absolute Error; '
                      'RMSLE = Root Mean Squared Logarithmic Error.')
ERROR_DEFINITIONS = ('MAE = Mean Absolute Error; WAPE = Weighted Absolute Percentage Error; '
                     'RMSLE = Root Mean Squared Logarithmic Error.')


def read_rows(root, name):
    with (root / name).open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def display_number(value, places, scale='1', suffix=''):
    """Round the underlying decimal value, never a previously formatted string."""
    number = Decimal(str(value)) * Decimal(scale)
    rounded = number.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)
    return f'{rounded:.{places}f}{suffix}'


def internal_values(row):
    return [escape(row['held_out_exporter']),
            display_number(row['cell_wape'], 1, '100', '%'),
            display_number(row['cell_mae'], 1, '0.000001'),
            display_number(row['rmsle'], 1),
            display_number(row['destination_wape'], 1, '100', '%'),
            display_number(row['rank_correlation'], 2), f"{row['top_overlap']}/3"]


def dataset_descriptions(root):
    audit = json.loads((root / 'data_audit.json').read_text(encoding='utf-8'))
    fits = read_rows(root, 'fit_records.csv')
    final = {int(r['model']): r for r in fits if r['context'].startswith('final')}
    assert set(final) == set(range(1, 7)), 'Missing final-fit records'
    grid = audit['annual_grid_cells']
    pairs, headings = audit['directed_pairs'], audit['hs4_count']
    assert grid == pairs * headings
    assert all(int(r['rows']) == grid for r in final.values())
    scans = audit['annual_scans']
    positive = int(Decimal(final[6]['positive_rows']))
    assert positive == scans['2024']['positive_cells']
    common = (f'Each cell is one exporter–destination–HS4 product observation across '
              f'{pairs:,} directed foreign-country pairs and {headings:,} headings. '
              'Domestic/self-pairs and products outside the fixed HS2022 universe are excluded; '
              'valid absent flows are retained as zeros. Missing explanatory variables are '
              'median-imputed rather than causing row deletion. Alberta 2025 outcomes are excluded.')
    outer = grid // 7
    fold = grid - outer
    assert all(int(r['rows']) == fold for r in fits
               if int(r['model']) < 6 and r['context'].startswith('holdout '))
    validation = (f'Exporter-held-out validation uses {outer:,} test cells per fold, '
                  f'separate from the {fold:,} cells supplied to each fold’s fit.')
    full2024 = (f'{grid:,} zero-filled 2024 cells enter the final fitting pipeline '
                f'({scans["2024"]["positive_cells"]:,} positive and '
                f'{scans["2024"]["zero_cells"]:,} zero-valued cells). ')
    descriptions = {
        1: (f'{grid:,} exporter–destination–HS4 transitions fit the final regression: '
            '2023 predictors and lagged trade paired with 2024 trade responses. '
            f'The responses include {scans["2024"]["positive_cells"]:,} positive and '
            f'{scans["2024"]["zero_cells"]:,} zero-valued cells. ' + common + ' ' + validation),
        2: full2024 + common + ' ' + validation,
        3: full2024 + common + ' ' + validation,
    }
    # scikit-learn 1.9.1 (recorded in quality_checks.json): train_test_split rounds
    # fractional test_size upward. MLP uses .15; automatic HGB stopping uses .10
    # for these >10,000-row fits. Preprocessing/calibration still use all input rows.
    quality = json.loads((root / 'quality_checks.json').read_text(encoding='utf-8'))
    assert quality['sklearn'] == '1.9.1', 'Reverify stopping splits for a new sklearn version'
    mlp_stop = math.ceil(grid * .15)
    boost_stop = math.ceil(grid * .10)
    positive_stop = math.ceil(positive * .10)
    descriptions[4] = (f'{grid - mlp_stop:,} 2024 cells fit the network weights; '
                       f'{mlp_stop:,} cells are reserved for training-only early stopping (15%). '
                       + full2024 + 'Preprocessing and dollar calibration use all input cells. '
                       + common + ' ' + validation)
    descriptions[5] = (f'{grid - boost_stop:,} 2024 cells fit the boosted trees; '
                       f'{boost_stop:,} cells are reserved for automatic early stopping (10%). '
                       + full2024 + 'Preprocessing and dollar calibration use all input cells. '
                       + common + ' ' + validation)
    descriptions[6] = (
        f'Classification stage: {grid:,} zero-filled 2024 cells enter the pipeline; '
        f'{grid - boost_stop:,} fit the classifier and {boost_stop:,} are reserved for '
        f'automatic early stopping. Conditional-positive-value stage: {positive:,} positive '
        f'2024 cells enter its separate pipeline; {positive - positive_stop:,} fit the '
        f'regressor and {positive_stop:,} are reserved for automatic early stopping. '
        f'The {scans["2024"]["zero_cells"]:,} zero-valued cells are excluded only from '
        'the positive-value stage. Stage-specific preprocessing uses the full input sample '
        f'for that stage; positive-dollar calibration uses all {positive:,} positive cells. '
        f'Adjustment stage: λ is estimated on {scans["2022"]["grid_cells"]:,} matched '
        '2022→2023 transitions using exporter-out-of-fold structural signals, with zeros '
        f'retained. The separate 2023→2024 temporal validation covers '
        f'{scans["2023"]["grid_cells"]:,} cells ({outer:,} per exporter) and does not refit λ. '
        + common)
    return descriptions


def diagnostic_note(model):
    first = ('2025 Alberta outcomes below are an external diagnostic, meaning that '
             'previously frozen predictions are compared with observed Alberta exports '
             'that were not used to fit or select the model. ')
    if model == 1:
        second = ('This model produces an imputation-qualified one-year prediction, meaning '
                  'that missing explanatory variables, including manufacturing shares, were '
                  'replaced with estimated values from the fitted 2023 training-sample '
                  'medians rather than directly observed data.')
    elif model == 6:
        second = ('This model produces an imputation-qualified one-year partial-adjustment '
                  'prediction, meaning that missing explanatory variables were replaced '
                  'with estimated values rather than directly observed data; the '
                  'classification and conditional-positive-value stages use separate '
                  'medians estimated from their respective training samples.')
    else:
        second = ('This model produces structural model-implied trade potential based on '
                  '2024 information, rather than an explicitly dynamic one-year forecast. '
                  'Missing explanatory variables are replaced with medians estimated '
                  'from this model’s training sample; these estimated inputs are not '
                  'directly observed values.')
    return first + second


def key_blocks(html):
    """Keep the existing representative equations/assumptions in full page builds."""
    blocks = {}
    for model in range(1, 7):
        match = re.search(
            rf'<h3 class="model-key-label" id="model-{model:02d}-key-equation">.*?'
            rf'<p class="model-key-assumption"[^>]*>.*?</p>', html, re.S)
        if match:
            blocks[model] = match.group(0)
    return blocks


def format_presentation(html, root=ROOT):
    descriptions = dataset_descriptions(root)
    valid = read_rows(root, 'validation_metrics.csv')
    summary = read_rows(root, 'model_summary.csv')
    assert len(summary) == 6
    for model in range(1, 7):
        pattern = rf'(<details class="model" id="model-{model:02d}"[^>]*>)(.*?)(</details>)'
        found = re.search(pattern, html, re.S)
        assert found, f'Missing model {model}'
        body = found.group(2)
        dataset = (f'<h3 class="model-key-label" id="model-{model:02d}-dataset-size">'
                   'Dataset Size</h3>\n'
                   f'<p class="model-dataset-size" aria-labelledby="model-{model:02d}-dataset-size">'
                   f'{descriptions[model]}</p>\n')
        body = re.sub(r'<h3[^>]*>Dataset Size\s*</h3>\s*<p[^>]*>.*?</p>\s*', '', body, flags=re.S)
        assumption = f'<h3 class="model-key-label assumption-label" id="model-{model:02d}-key-assumption">'
        assert body.count(assumption) == 1, f'Missing key block for model {model}'
        body = body.replace(assumption, dataset + assumption, 1)
        body, count = re.subn(r'<p class="notice">2025 Alberta outcomes below.*?</p>',
                              '<p class="notice">' + diagnostic_note(model) + '\n</p>',
                              body, flags=re.S)
        assert count == 1
        rows = [r for r in valid if int(r['model']) == model]
        assert len(rows) == 7
        # Restrict replacement to the table immediately below this heading.
        table_pattern = (r'(<h3>Internal validation results\s*</h3>\s*)'
                         r'(?:<p class="metric-definitions">.*?</p>\s*)?'
                         r'(<div class="table-scroll".*?<tbody>)(.*?)(</tbody>)')
        def validation_table(match):
            old_rows = re.findall(r'<tr>.*?</tr>', match[3], re.S)
            assert len(old_rows) == len(rows)
            new_rows = []
            for old, row in zip(old_rows, rows):
                old_cells = re.findall(r'<td>(.*?)</td>', old, re.S)
                assert old_cells[0] == row['held_out_exporter']
                values = internal_values(row)
                assert old_cells[-1] == values[-1]
                new_rows.append('<tr>' + ''.join(f'<td>{v}</td>' for v in values) + '\n</tr>')
            return (match[1] + f'<p class="metric-definitions">{METRIC_DEFINITIONS}</p>\n'
                    + match[2] + '\n' + '\n'.join(new_rows) + '\n' + match[4])
        body, count = re.subn(table_pattern, validation_table, body, count=1, flags=re.S)
        assert count == 1
        body, count = re.subn(
            r'(<caption>Predicted destination )(?:<span class="destination-accent">)?'
            r'(#[0-9]+ · [^<]+)(?:</span>)?(\s*</caption>)',
            r'\1<span class="destination-accent">\2</span>\3', body)
        assert count == 10, f'Expected ten predicted destination captions in model {model}'
        html = html[:found.start(2)] + body + html[found.end(2):]
    overview = re.search(r'<table class="comparison">.*?</table>', html, re.S)
    assert overview
    table = overview.group(0)
    index = 0
    def overview_row(match):
        nonlocal index
        cells = list(re.finditer(r'<td>(.*?)</td>', match[0], re.S))
        assert len(cells) == 11
        row = summary[index]
        assert f'#model-{int(row["model"]):02d}' in cells[0][1]
        index += 1
        cell = cells[8]
        return (match[0][:cell.start(1)]
                + display_number(row['destination_wape'], 1, '100', '%')
                + match[0][cell.end(1):])
    table = re.sub(r'<tr><td>.*?</tr>', overview_row, table, flags=re.S)
    assert index == 6
    html = html[:overview.start()] + table + html[overview.end():]
    html, count = re.subn(
        r'(<h3>How to read the errors\s*</h3>\s*)'
        r'(?:<p class="error-definitions">.*?</p>\s*)?',
        lambda m: m[1] + f'<p class="error-definitions">{ERROR_DEFINITIONS}</p>\n',
        html, flags=re.S)
    assert count == 1
    return html


def render_saved_page():
    original = PAGE.read_bytes()
    newline = '\r\n' if b'\r\n' in original else '\n'
    html = original.decode('utf-8').replace('\r\n', '\n')
    revised = format_presentation(html)
    assert format_presentation(revised) == revised, 'Rendering must be idempotent'
    PAGE.write_bytes(revised.replace('\n', newline).encode('utf-8'))
    print('Updated only the target HTML from saved fit/audit/metric artifacts.')


if __name__ == '__main__':
    render_saved_page()
