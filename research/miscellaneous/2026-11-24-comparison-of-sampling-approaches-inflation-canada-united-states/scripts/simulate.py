"""Finite-population CPI design comparison with seeded, separate sample draws."""
import argparse
import json
import math
from random import Random
from statistics import fmean, stdev
from time import perf_counter

from prepare_data import ROOT, csv_write, json_write

OUT = ROOT / 'data/simulated'
DESIGNS = ['SRS', 'Stratified', 'Multistage PPS', 'Judgmental', 'Cut-off', 'Calibrated', 'Hybrid']
SCENARIOS = {
    'Baseline': 'heterogeneous outlets with observed prices',
    'A · Shock': 'food and energy shocks',
    'B · Selection': 'log sales predict inflation',
    'C · Geography': 'shock concentrated in one region',
    'D · Missingness': 'large increases more likely to be missing, carried forward',
    'E · Turnover': 'replacements are comparable with error',
    'F · Weights': 'weights lag a current shift in demand',
    'G · Feed coverage': 'feed includes large chains with different trends',
    'H · Disruption': 'collection outage and carry-forward in three regions',
}
CATEGORIES = ['Food', 'Energy', 'Apparel', 'Services', 'Durables']
REGIONS = 8
M = 30
SAMPLE = 56

def economy(seed, scenario):
    rng = Random(seed)
    products, outlets = [], []
    key = scenario[0] if scenario != 'Baseline' else ''
    regional = [rng.gauss(0,.011) for _ in range(REGIONS)]
    for region in range(REGIONS):
        for market in range(M):
            city = market < 22
            chain = rng.random() < (.53 if city else .29)
            size = math.exp(rng.gauss(1.7 if chain else .35,.7))
            pricing = rng.gauss(0,.014)
            visibility = size * (.65 + .7*rng.random())
            outlet = region * M + market
            category_weights = [size * rng.uniform(.12,.55) for _ in CATEGORIES]
            total = sum(category_weights)
            observed, current, true = 0., 0., 0.
            for c, (group, weight) in enumerate(zip(CATEGORIES,category_weights)):
                base = math.exp(1+c*.57+rng.gauss(0,.32))
                quality = rng.uniform(.55,1.6)
                innovation = rng.gauss(0,.018)
                promotion = rng.random() < .09
                changed = rng.random() < .34
                nonresponse = rng.random()
                shift = .012 + regional[region] + pricing + innovation - .025*promotion
                if key=='A' and c in (0,1):
                    shift += .13
                if key=='B':
                    shift += .034*math.log(size)
                if key=='C' and region==6:
                    shift += .17
                if key=='F':
                    shift += .024*math.log(size)
                if key=='G' and not chain:
                    shift += .07
                if key=='H' and region in (0,3,5):
                    shift += .055
                change = math.expm1(shift)*100
                quality_multiplier = math.exp(.18*changed) if key=='E' else 1.
                price = base * math.exp(shift) * quality_multiplier
                response = not ((key=='D' and change>3 and nonresponse<.72) or
                                (key=='H' and region in (0,3,5) and nonresponse<.84))
                seen = change if response else 0.
                if key=='E' and changed:
                    seen = 100*(price/base*math.exp(-.12)-1)
                observed += weight*seen
                true += weight*change
                current += weight*max(.3,1+change*.09)
                products.append(dict(region=region,outlet=outlet,urban=city,chain=chain,
                    product_id=f'{outlet:03}-{c}',category=group,quality=quality,
                    current_quality=quality*quality_multiplier, expenditure_base=weight,
                    prior_month_price=base, observed_price=price if response else None,
                    comparable_price=base*math.exp(shift), response=response,
                    promotion=promotion, change_pp=change))
            outlets.append(dict(outlet=outlet, region=region, urban=city, chain=chain,
                                weight=total, visibility=visibility, observed=observed/total,
                                true=true/total, current=current))
    w = sum(x['weight'] for x in outlets)
    target_weights = [x['current'] if key=='F' else x['weight'] for x in outlets]
    truth = sum(x['true']*t for x,t in zip(outlets,target_weights))/sum(target_weights)
    for x in outlets:
        x['share']=x['weight']/w
        x['contrib']=len(outlets)*x['share']*x['observed']
    return outlets, products, truth

def se(values):
    return stdev(values)/math.sqrt(len(values)) if len(values)>1 else 0.

def weighted(rows):
    w = sum(x['weight'] for x in rows)
    return sum(x['weight']*x['observed'] for x in rows)/w

def draw(outlets, regions, rng, design, n):
    size = len(outlets)
    if design == 'SRS':
        sample = rng.sample(outlets,n)
        v = [x['contrib'] for x in sample]
        return fmean(v), se(v)*math.sqrt(1-n/size), n
    if design == 'Stratified':
        n_h = n//len(regions)
        groups = [[x['contrib'] for x in rng.sample(r,n_h)] for r in regions]
        var = sum((len(r)/size)**2*(1-n_h/len(r))*se(v)**2 for r,v in zip(regions,groups))
        return sum((len(r)/size)*fmean(v) for r,v in zip(regions,groups)), math.sqrt(var), n_h*len(regions)
    if design == 'Multistage PPS':
        mass = [sum(x['share'] for x in r) for r in regions]
        stage1 = rng.choices(range(len(regions)),weights=mass,k=n//2)
        draws = []
        for j in stage1:
            r = regions[j]
            sample = rng.sample(r,2)
            draws.append(len(r)*fmean(x['share']*x['observed'] for x in sample)/mass[j])
        return fmean(draws),se(draws),(n//2)*2.7
    if design == 'Cut-off':
        sample = sorted(outlets,key=lambda x:x['weight'],reverse=True)[:n]
        return weighted(sample),se([x['observed'] for x in sample]),n
    if design == 'Hybrid':
        feed = [x for x in outlets if x['chain'] and x['weight'] > 11]
        feed_ids = {x['outlet'] for x in feed}
        rest = sorted([(x['visibility']*rng.uniform(.3,1.7),x['outlet'],x)
                       for x in outlets if x['outlet'] not in feed_ids], reverse=True)
        selected = [pair[2] for pair in rest[:n//2]]
        share = sum(x['share'] for x in feed)
        p = sum(x['share']*x['observed'] for x in feed)
        return p + (1-share)*weighted(selected),(1-share)*se([x['observed'] for x in selected]),9+len(selected)+.12*len(feed)
    order = sorted([(x['visibility']*rng.uniform(.3,1.7),x['outlet'],x) for x in outlets],reverse=True)
    sample = [x[2] for x in order[:n]]
    if design == 'Calibrated':
        total = variance = 0.
        for chain in (True,False):
            domain = [x for x in outlets if x['chain']==chain]
            share = sum(x['share'] for x in domain)
            domain_sample = [x for x in sample if x['chain']==chain]
            usable = domain_sample or sample
            total += share * weighted(usable)
            variance += share**2 * se([x['observed'] for x in usable])**2
        return total,math.sqrt(variance),n
    return weighted(sample),se([x['observed'] for x in sample]),n

def summarize(estimate, truth, elapsed):
    errors = [value-truth for value,_,_ in estimate]
    bias = fmean(errors)
    variance = fmean((e-bias)**2 for e in errors)
    mse = fmean(e**2 for e in errors)
    assert math.isclose(mse,variance+bias*bias,rel_tol=1e-12,abs_tol=1e-12)
    coverage = fmean(abs(v-truth)<=1.96*s for v,s,_ in estimate)
    errors.sort()
    nn=len(estimate)
    return dict(target=truth, bias=bias, variance=variance, sd=math.sqrt(variance),
        mse=mse, rmse=math.sqrt(mse), mae=fmean(map(abs,errors)), coverage=coverage,
        width=fmean(3.92*s for _,s,_ in estimate), cost=fmean(c for _,_,c in estimate),
        bias_mcse=stdev(errors)/math.sqrt(nn),
        coverage_mcse=math.sqrt(coverage*(1-coverage)/nn), q025=errors[int(.025*(nn-1))],
        q975=errors[int(.975*(nn-1))], runs=nn, compute_ms=elapsed*1000)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=int,default=1200)
    parser.add_argument('--seed',type=int,default=20261124)
    args=parser.parse_args()
    if args.runs<2:
        parser.error('At least two replications are required')
    summary=[]
    for k,scenario in enumerate(SCENARIOS):
        outlets,products,truth=economy(args.seed,scenario)
        regions=[[x for x in outlets if x['region']==i] for i in range(REGIONS)]
        if not k:
            csv_write(OUT/'population.csv',products)
        for j,design in enumerate(DESIGNS):
            rng=Random(args.seed+1000*k+j)
            start=perf_counter()
            samples=[draw(outlets,regions,rng,design,SAMPLE) for _ in range(args.runs)]
            summary.append(dict(scenario=scenario,design=design,**summarize(samples,truth,perf_counter()-start)))
    csv_write(OUT/'summary.csv',summary)
    bundle=dict(seed=args.seed,runs=args.runs,sample_size=SAMPLE,scenarios=SCENARIOS,rows=summary)
    json_write(OUT/'summary.json',bundle)
    (OUT/'simulation-data.js').write_text('globalThis.CPI_SIMULATED = '+json.dumps(bundle,ensure_ascii=False,separators=(',',':'))+';\n',encoding='utf-8')
    print(f'{len(summary)} scenario-design cells × {args.runs} replications')

if __name__=='__main__':
    main()

