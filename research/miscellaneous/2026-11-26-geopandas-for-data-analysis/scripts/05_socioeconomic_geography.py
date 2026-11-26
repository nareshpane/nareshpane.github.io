"""Join 2021 Census indicators and measure global and local spatial association."""
from io import TextIOWrapper
import warnings
import zipfile

from libpysal.weights import KNN, Queen, Rook
import numpy as np
import pandas as pd

from common import ROOT, SEED, datum, json_write, read, save, summarize

INDICATORS = {
    'age_total': 'Total - Age groups of the population - 100% data',
    'age65': '65 years and over',
    'income': 'Median total income of household in 2020 ($)',
    'low_income': 'Prevalence of low income based on the Low-income measure, after tax (LIM-AT) (%)',
    'unemployment': 'Unemployment rate',
    'labour': 'In the labour force',
    'unemployed': 'Unemployed',
    'housing': 'Spending 30% or more of income on shelter costs',
    'degree': "Bachelor's degree or higher",
}
CHARACTERISTICS = {'age_total': 8, 'age65': 24, 'income': 243, 'low_income': 345,
                   'labour': 2224, 'unemployed': 2226, 'unemployment': 2230,
                   'housing': 1467, 'degree': 2008}
PERMUTATIONS = 499
WEIGHTS = ('queen', 'rook', 'knn')


def profile():
    root = datum('socioeconomic', 'raw', 'unused').parent
    files = sorted(root.glob('*.zip'))
    original = files[0] if files else None
    mirror = root / 'census_profile.parquet'
    if not original and not mirror.exists():
        return None, None
    parts = []
    if original:
        with zipfile.ZipFile(original) as archive:
            members = [m for m in archive.namelist()
                       if 'english_csv_data' in m.lower() and m.lower().endswith('.csv')]
            for member in members:
                with archive.open(member) as stream:
                    rows = pd.read_csv(TextIOWrapper(stream, encoding='latin1'), chunksize=400000,
                                       low_memory=False)
                    for chunk in rows:
                        uid = chunk.DGUID.astype(str).str.extract(r'^2021S0512(48\d{6})$')[0]
                        field = chunk['CHARACTERISTIC_ID'].map({v: k for k, v in CHARACTERISTICS.items()})
                        select = uid.notna() & field.notna()
                        parts.append(pd.DataFrame({'uid': uid[select], 'field': field[select],
                                                   'number': pd.to_numeric(chunk.loc[select, 'C1_COUNT_TOTAL'], errors='coerce')}))
        label = 'Statistics Canada Census Profile 98-401-X2021006, Prairies CSV'
    else:
        fields = {'count_total_' + str(i): name for name, i in CHARACTERISTICS.items()}
        data = pd.read_parquet(mirror, columns=['da_dguid', *fields])
        data = data.loc[data.da_dguid.str.match(r'^2021S051248\d{6}$', na=False)].rename(columns=fields)
        data['uid'] = data.da_dguid.str[-8:]
        for col in fields.values():
            data[col] = pd.to_numeric(data[col], errors='coerce').where(lambda x: x >= 0)
        return data.drop(columns='da_dguid'), 'Statistics Canada 2021 Census Profile, Data for Canada / Source Cooperative copy'
    rows = pd.concat(parts).drop_duplicates()
    # Some broad labels repeat under distinct universes. Keep only unambiguous values.
    use = rows.groupby(['uid', 'field']).number.agg(['first', 'nunique'])
    clean = use['first'].where(use['nunique'].le(1))
    return clean.unstack().reset_index(), label


def fdr(p):
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1].clip(0, 1)
    q = np.empty(len(p))
    q[order] = adjusted
    return q


def spatial_weights(geometry, mode):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        if mode == 'knn':
            w = KNN.from_dataframe(geometry.set_geometry(geometry.representative_point()), k=8)
        elif mode == 'queen':
            w = Queen.from_dataframe(geometry, use_index=False)
        else:
            w = Rook.from_dataframe(geometry, use_index=False)
    w.transform = 'r'
    return w


def permutation(values, weights, simulations=PERMUTATIONS):
    """Global random labeling and conditional local sampling, with fixed seeds."""
    rng = np.random.default_rng(SEED)
    z = (values - values.mean()) / values.std()
    lag = weights.sparse @ z
    n = len(z)
    expected = -1 / (n - 1)
    factor = n / weights.s0 / (z @ z)
    observed = factor * (z @ lag)
    exceed = 0
    for _ in range(simulations):
        p = rng.permutation(z)
        simulated = factor * (p @ (weights.sparse @ p))
        exceed += abs(simulated - expected) >= abs(observed - expected)
    # Conditional permutations keep the focal value fixed. Row-standardized binary
    # neighbours have equal weights, so the simulated lag is a sampled mean.
    local = z * lag
    prob = np.ones(n)
    eligible = np.flatnonzero(np.array([len(weights.neighbors[i]) for i in range(n)]) > 0)
    for i in eligible:
        k = len(weights.neighbors[i])
        # Random keys generate an exact uniform subset without replacement.
        # The small-k rejection sampler draws independent rows until each has k distinct labels.
        sample = rng.integers(0, n - 1, size=(simulations, k))
        duplicated = np.any(np.diff(np.sort(sample, axis=1), axis=1) == 0, axis=1)
        while duplicated.any():
            sample[duplicated] = rng.integers(0, n - 1, size=(duplicated.sum(), k))
            duplicated = np.any(np.diff(np.sort(sample, axis=1), axis=1) == 0, axis=1)
        sample += sample >= i
        sim_lag = z[sample].mean(axis=1)
        # Equal-tail two-sided Monte Carlo inference with +1 correction.
        left = (np.count_nonzero(sim_lag <= lag[i]) + 1) / (simulations + 1)
        right = (np.count_nonzero(sim_lag >= lag[i]) + 1) / (simulations + 1)
        prob[i] = min(1, 2 * min(left, right))
    q = fdr(prob)
    quadrant = np.select([(z > 0) & (lag > 0), (z < 0) & (lag < 0), (z > 0) & (lag < 0),
                          (z < 0) & (lag > 0)], ['High–high', 'Low–low', 'High–low', 'Low–high'],
                         default='No quadrant')
    return {'I': observed, 'expected': expected, 'p': (exceed + 1) / (simulations + 1),
            'islands': len(weights.islands), 'components': weights.n_components}, z, lag, local, prob, q, quadrant


def run():
    census = read('shared', 'das').sort_values('uid').reset_index(drop=True)
    input_table, source = profile()
    if input_table is not None:
        census = census.merge(input_table, on='uid', how='left', validate='1:1')
    else:
        source = 'Statistics Canada table 98-10-0015-01'
    source = str(source)
    if 'age65' in census:
        census['age_share'] = 100 * census.age65 / census.age_total.replace(0, np.nan)
    census['division'] = census.uid.str[:4]
    columns = [c for c in ['population', 'density', 'age65', 'age_share', 'income', 'low_income',
                           'unemployment', 'labour', 'unemployed', 'housing', 'degree'] if c in census]
    census[['uid', 'division', *columns]].to_csv(datum('socioeconomic', 'processed', 'census_indicators.csv'),
                                               index=False)
    selected = [c for c in ['density', 'age_share', 'income', 'low_income', 'unemployment'] if c in columns]
    # Model log-density because settlement density spans several orders of magnitude.
    plot_fields = {}
    details = {}
    model_cache = {}
    for name in selected:
        subset = census.loc[census[name].notna()].reset_index(drop=True)
        values = subset[name].to_numpy(dtype=float)
        if len(values) < 50 or np.std(values) == 0:
            continue
        x = np.log1p(values) if name == 'density' else values
        details[name] = {'n': len(values), 'missing': len(census) - len(values),
                         'mean': values.mean(), 'median': float(np.median(values)),
                         'p10': float(np.quantile(values, .10)), 'p90': float(np.quantile(values, .90)),
                         'divisions': subset.groupby('division')[name].median().to_dict(), 'weights': {}}
        for mode in WEIGHTS:
            cache_key = (tuple(subset.uid), mode)
            if cache_key not in model_cache:
                model_cache[cache_key] = spatial_weights(subset, mode)
            weights = model_cache[cache_key]
            summary, z, lag, local, p, q, quadrant = permutation(x, weights)
            rows = {uid: [round(zi, 4), round(li, 4), round(iv, 4), round(pi, 5), round(qi, 5),
                          str(qu), [subset.uid.iloc[j] for j in weights.neighbors[i]]]
                    for i, (uid, zi, li, iv, pi, qi, qu) in enumerate(zip(subset.uid, z, lag, local, p, q, quadrant))}
            summary.update(raw_significant=int((p <= .05).sum()), fdr_significant=int((q <= .05).sum()))
            details[name]['weights'][mode] = summary
            json_write(ROOT / 'maps' / 'socioeconomic' / f'{name}_{mode}.json', rows)
        plot_fields[name] = {uid: float(v) for uid, v in zip(subset.uid, values)}

    composite = [c for c in ('low_income', 'unemployment', 'income') if c in plot_fields]
    composite_sensitivity = None
    if len(composite) >= 2:
        complete = census.dropna(subset=composite)
        components = complete[composite].copy()
        components = (components - components.mean()) / components.std(ddof=0)
        if 'income' in components:
            components['income'] *= -1
        plot_fields['components'] = {uid: list(map(float, row))
                                     for uid, row in zip(complete.uid, components.to_numpy())}
        baseline = components.mean(axis=1)
        first = components.iloc[:, 0]
        composite_sensitivity = {'n': len(components), 'first': composite[0],
                                 'rank_correlation': float(baseline.corr(first, method='spearman'))}
    divisions = []
    if {'labour', 'unemployed'} <= set(census.columns):
        eligible = census.dropna(subset=['labour', 'unemployed'])
        sums = eligible.groupby('division')[['labour', 'unemployed']].sum()
        for division, row in sums.iterrows():
            if row.labour:
                divisions.append({'division': division, 'labour': row.labour,
                                  'unemployed': row.unemployed, 'rate': 100 * row.unemployed / row.labour})
    save(census[['uid', 'division', *columns, 'geometry']], 'socioeconomic', 'census')
    json_write(ROOT / 'maps' / 'socioeconomic' / 'values.json', plot_fields)
    result = {'source': source, 'profile_available': input_table is not None,
              'permutations': PERMUTATIONS, 'seed': SEED, 'indicators': details,
              'composite': composite, 'composite_sensitivity': composite_sensitivity,
              'division_unemployment': divisions}
    summarize('socioeconomic', result)
    print({name: info['weights']['queen'] for name, info in details.items()})


if __name__ == '__main__':
    run()

