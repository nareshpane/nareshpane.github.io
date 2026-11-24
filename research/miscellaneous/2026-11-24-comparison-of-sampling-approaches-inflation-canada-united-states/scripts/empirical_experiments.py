"""Public-series diagnostics and partial-basket calculations used in the study."""
import json
import math
import statistics

from prepare_data import PROCESSED, MATCH, correlation
from prepare_data import csv_write, json_write, pct, lagged

LINKS = {
    'CA': {'2023': '2024-04', '2024': '2025-04', '2025': '2026-04'},
    'US': {year: year+'-12' for year in ('2023','2024','2025')},
}
KEYS = list(MATCH)

def dot(weights, changes):
    if any(r is None for r in changes):
        return None
    return math.fsum(w*r for w,r in zip(weights,changes))

def shares_at_previous_month(data, country, year, index, price_update):
    country_index = 0 if country == 'CA' else 1
    month = data['months'].index(LINKS[country][year])
    raw = [data['weights'][country][year][MATCH[key][country_index]]/100 for key in KEYS]
    if not price_update:
        return raw
    relative = [pct(data['series'][country+'_'+key][index-1],data['series'][country+'_'+key][month])
                for key in KEYS]
    if any(r is None for r in relative):
        return raw
    updated = [w * (1+r/100) for w,r in zip(raw,relative)]
    return [w / math.fsum(updated) * math.fsum(raw) for w in updated]

def baskets(data, year, index, price_update=True):
    countries = ('CA','US')
    mapped_shares = [shares_at_previous_month(data,c,year,index,price_update) for c in countries]
    normalized = [[w/math.fsum(raw) for w in raw] for raw in mapped_shares]
    common = [(u+v)/2 for u,v in zip(*normalized)]
    relative = [[pct(data['series'][c+'_'+k][index],data['series'][c+'_'+k][index-1]) for k in KEYS]
                for c in countries]
    result = {
        'reference_month': data['months'][index], 'snapshot': year, 'price_update': price_update,
        'ca_weights_ca_prices': dot(normalized[0],relative[0]),
        'ca_weights_us_prices': dot(normalized[0],relative[1]),
        'us_weights_ca_prices': dot(normalized[1],relative[0]),
        'us_weights_us_prices': dot(normalized[1],relative[1]),
        'common_weights_ca_prices': dot(common,relative[0]),
        'common_weights_us_prices': dot(common,relative[1])
    }
    for c, shares, changes in zip(countries,mapped_shares,relative):
        headline = pct(data['series'][c+'_all'][index],data['series'][c+'_all'][index-1])
        contribution = dot(shares,changes)
        result.update({c+'_coverage': math.fsum(shares), c+'_partial_contribution_pp': contribution,
                       c+'_headline_pct': headline, c+'_residual_pp':
                       headline-contribution if headline is not None and contribution is not None else None})
    return result

def diagnostics(data):
    a,b = (lagged(data['series'][key],1) for key in ['CA_sa','US_sa'])
    complete = [(x,y) for x,y in zip(a,b) if x is not None and y is not None]
    results = {
        'paired_months': len(complete), 'pearson_correlation': correlation(a,b),
        'canada_mean_monthly_pct': statistics.fmean(x for x in a if x is not None),
        'us_mean_monthly_pct': statistics.fmean(y for y in b if y is not None),
        'canada_cumulative_pct': pct(data['series']['CA_all'][-1],data['series']['CA_all'][0]),
        'us_cumulative_pct': pct(data['series']['US_all'][-1],data['series']['US_all'][0]),
        'ca_shelter_oct_annual_pct': pct(data['series']['CA_shelter'][-1],data['series']['CA_shelter'][-13]),
    }
    lag_rows = [dict(lag=k, ca_acf=correlation(a[k:],a[:len(a)-k]),
                     us_acf=correlation(b[k:],b[:len(b)-k]),
                     ca_current_us_lagged=correlation(a[k:],b[:len(b)-k])) for k in range(37)]
    return results, lag_rows

def main():
    data=json.loads((PROCESSED/'cpi.json').read_text(encoding='utf-8'))
    rows=[baskets(data,year,i,update) for i in range(len(data['months'])-34,len(data['months']))
          for year in LINKS['CA'] for update in (False,True)]
    summary,lag_rows=diagnostics(data)
    csv_write(PROCESSED/'counterfactual-baskets.csv',rows)
    csv_write(PROCESSED/'serial-diagnostics.csv',lag_rows)
    json_write(PROCESSED/'empirical-summary.json',summary)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    main()

