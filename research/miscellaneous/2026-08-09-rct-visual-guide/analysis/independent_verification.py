"""Second implementation: CSV/standard-library arithmetic, no replication-function imports.

Keeps original records, including repeated IDs. Fisher probabilities are calculated
directly from hypergeometric margins, independently of scipy.stats.fisher_exact.
"""
from pathlib import Path
import csv, hashlib, json, math, statistics
from collections import Counter

ROOT=Path(__file__).resolve().parents[1]
PUBLISHED={'SEPSISDEATH':(206,123,.60,.48,.74,'<0.001'), 'SEPSIS':(202,117,.58,.46,.72,'<0.001'),
 'CULTPOS':(27,6,.22,.09,.53,'<0.001'),'SEPSISB':(36,19,.53,.30,.92,'0.021'),
 'SEPSISC':(139,92,.66,.51,.88,'0.002'),'DEATH':(4,6,1.50,.42,5.31,'0.526'),
 'DIARRHEA':(59,12,.20,.11,.38,'<0.001'),'LOCINF':(33,16,.48,.27,.88,'0.015'),
 'ABSCESS':(11,5,.45,.16,1.33,'0.133'),'OMPHA':(13,3,.23,.07,.81,'0.014'),
 'GRAMNEG':(16,4,.25,.08,.75,'0.007'),'GRAMPOS':(11,2,.18,.04,.82,'0.012')}

def logchoose(n,k):
    return math.lgamma(n+1)-math.lgamma(k+1)-math.lgamma(n-k+1)

def exact_fisher(a,n1,c,n0):
    k=a+c;support=range(max(0,k-n0),min(n1,k)+1)
    logp=lambda x:logchoose(n1,x)+logchoose(n0,k-x)-logchoose(n1+n0,k)
    observed=logp(a)
    return min(1.,math.fsum(math.exp(logp(x)) for x in support if logp(x)<=observed+1e-9))

def effects(a,n1,c,n0):
    r1=a/n1;r0=c/n0;rr=r1/r0;rd=r1-r0
    se=math.sqrt((1-r1)/a+(1-r0)/c);z=statistics.NormalDist().inv_cdf(.975)
    serd=math.sqrt(r1*(1-r1)/n1+r0*(1-r0)/n0)
    b=n1-a;d=n0-c;N=n1+n0
    chi=N*(a*d-b*c)**2/(n1*n0*(a+c)*(b+d))
    return dict(n1=n1,n0=n0,events1=a,events0=c,risk1=r1,risk0=r0,rr=rr,rr_ci=[math.exp(math.log(rr)-z*se),math.exp(math.log(rr)+z*se)],rd=rd,rd_ci=[rd-z*serd,rd+z*serd],pearson_p=math.erfc(math.sqrt(chi/2)),fisher_p=exact_fisher(a,n1,c,n0))

def main():
    raw=ROOT/'data/Panigrahi et al.SEPSSIF.CSV';b=raw.read_bytes()
    assert hashlib.md5(b).hexdigest()=='f57409eb20fb06db78ae14e99843623c'
    with raw.open(newline='',encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    groups={z:[r for r in rows if r['TX']==str(z)] for z in (0,1)}
    assert len(rows)==4556 and len(rows[0])==74 and len(groups[0])==len(groups[1])==2278
    current=json.loads((ROOT/'documentation/pre-refinement-summary.json').read_text(encoding='utf-8'))
    rebuilt=json.loads((ROOT/'data/derived_summary.json').read_text(encoding='utf-8'))
    checks={};results={};comparison=[]
    for key,pub in PUBLISHED.items():
        assert all(r[key] in ('0','1') for r in rows)
        a=sum(int(r[key]) for r in groups[1]);c=sum(int(r[key]) for r in groups[0]);r=effects(a,2278,c,2278)
        assert (c,a)==pub[:2]
        for field in ['rr','rd','pearson_p','fisher_p']:
            assert math.isclose(r[field],rebuilt['outcomes'][key][field],abs_tol=2e-10,rel_tol=2e-8),(key,field)
        assert all(math.isclose(x,y,abs_tol=1e-12) for x,y in zip(r['rr_ci'],rebuilt['outcomes'][key]['rr_ci']))
        results[key]=r
        comparison.append(dict(measure=key,current_rr=current['outcomes'][key]['rr'],published_rr=pub[2],recalculated_rr=r['rr'],status='Current matches independent recalculation; published point estimate matches to shown precision',published_ci=list(pub[3:5]),recalculated_ci=r['rr_ci'],published_p=pub[5],pearson_p=r['pearson_p'],fisher_p=r['fisher_p']))
    assert all(int(r['SEPSISDEATH'])==int(bool(int(r['SEPSIS']) or int(r['DEATH']))) for r in rows)
    assert all(int(r['SEPSIS'])==sum(int(r[k]) for k in ['CULTPOS','SEPSISB','SEPSISC']) for r in rows)
    baseline=[]
    for col in ['BWT','MOMAGE','MOMPAR']:
        for z in (0,1):
            vals=[float(r[col]) for r in groups[z] if r[col].strip() not in ('','.')] 
            ref=next(x for x in rebuilt['baseline'] if x['variable']==col)
            assert math.isclose(statistics.mean(vals),ref['mean'+str(z)],abs_tol=1e-12)
            assert math.isclose(statistics.stdev(vals),ref['sd'+str(z)],abs_tol=1e-10)
            baseline.append(dict(variable=col,arm=z,n=len(vals),mean=statistics.mean(vals),sd=statistics.stdev(vals)))
    sensitivity=[]
    for key,field in [('exclude_lostfu','LOSTFU'),('block_positions_1_2_3','POPBLOCK123'),('block_positions_1_2','POPBLOCK12')]:
        selected={z:[r for r in groups[z] if r[field]==('0' if field=='LOSTFU' else '1')] for z in (0,1)}
        r=effects(sum(int(x['SEPSISDEATH']) for x in selected[1]),len(selected[1]),sum(int(x['SEPSISDEATH']) for x in selected[0]),len(selected[0]))
        assert math.isclose(r['rr'],rebuilt['sensitivity'][key]['rr'],abs_tol=1e-12)
        sensitivity.append(dict(analysis=key,**r))
    checks=dict(rows=len(rows),variables=len(rows[0]),arm_counts={str(z):len(v) for z,v in groups.items()},duplicate_id_excess=len(rows)-len({r['Study_ID'] for r in rows}),exact_duplicate_rows=len(rows)-len({tuple(r.items()) for r in rows}),primary_missing=sum(r['SEPSISDEATH'].strip() in ('','.') for r in rows),time_missing=sum(r['TIME'].strip() in ('','.') for r in rows),primary_event_time_missing=sum(r['SEPSISDEATH']=='1' and r['TIME'].strip() in ('','.') for r in rows),lostfu={str(z):sum(int(r['LOSTFU']) for r in v) for z,v in groups.items()},outcome_pairs_verified=len(results),baseline_mean_sd_checks=len(baseline),block_complete_case_checks=len(sensitivity),sha256=hashlib.sha256(b).hexdigest())
    output=dict(checks=checks,outcomes=results,comparison=comparison,baseline=baseline,sensitivity=sensitivity,discrepancies=[
      'Mortality: published Table 2 gives p=0.526 and labels Fisher exact; independent two-sided Fisher is 0.7536358, while Pearson is 0.5266352. The label/value pair is not reproduced.',
      'Omphalitis: published p=0.014 versus uncorrected Pearson 0.0122659. Cause not identified from released software documentation.',
      'Published culture-positive, LRTI and abscess CI upper endpoints differ from Katz reconstruction at shown precision.',
      'Published LRTI p=0.002 versus Pearson 0.0015042 is compatible with rounding; some borderline displayed rounding conventions differ.',
      'LOSTFU versus published flow dispositions, repeated identifiers, missing maternal covariates and ID-prefix/village counts remain unresolved.' ])
    (ROOT/'data/independent_verification.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    report='''# Independent statistical recheck — November 4, 2026

A second implementation reads the original CSV using Python csv, not pandas, and does not import the existing reproduction functions. It uses standard-library means/SDs, direct chi-square arithmetic and a two-sided hypergeometric Fisher test. Source bytes and all rows are retained. Published values are transcribed from corrected Table 2; the preceding webpage is preserved separately.

| Measure | Current webpage (underlying RR) | Published RR [CI] | Independently recalculated RR [CI] | Status |
|---|---|---|---|---|
'''
    for c in comparison:
        r=results[c['measure']];pub=PUBLISHED[c['measure']]
        status='Current agrees; rounded published point agrees'
        if any(abs(a-b)>.0050001 for a,b in zip(r['rr_ci'],pub[3:5])):status+='; CI endpoint discrepancy retained'
        report+=f'| {c["measure"]} | {c["current_rr"]:.6f} | {pub[2]:.2f} [{pub[3]:.2f}, {pub[4]:.2f}] | {r["rr"]:.6f} [{r["rr_ci"][0]:.6f}, {r["rr_ci"][1]:.6f}] | {status} |\n'
    report+='\n## Hypothesis-test reconciliation\n\n| Endpoint | Published p / footnote | Independent Pearson p | Independent two-sided Fisher p |\n|---|---|---|---|\n'
    for k,r in results.items():report+=f'| {k} | {PUBLISHED[k][5]}'+(' / Fisher' if k=='DEATH' else ' / chi-square')+f' | {r["pearson_p"]:.8g} | {r["fisher_p"]:.8g} |\n'
    report+='\n## Findings\n\n'+'\n'.join('- '+x for x in output['discrepancies'])
    report+='''

The primary reconstruction remains RR 0.5970874 [0.4812610, 0.7407901], RD −0.03643547 [−0.05143019, −0.02144075], Pearson p=2.0273265e−6. This verifies an unadjusted fixed-sample reconstruction, not the unavailable original interval software or complete sequential procedure. A numerical discrepancy is not repaired by changing records or relabeling a published method.

Protocol pp.31–33 documents ITT, binary chi-square/Fisher tests, planning risks and interim monitoring; stepwise regression was proposed if imbalance was observed. No released SAS program identifies exact CI code or establishes which adjusted model was actually fitted. The corrected 88-case interpretation remains in place. Survival risk sets are not recoverable from event ages alone.

Checks: 4,556 records, 74 fields, 2,278 per arm; 12 published outcome pairs, six continuous baseline mean/SD checks, three block/complete-case checks. Two repeated-ID excess records are retained; no exactly duplicated row. Endpoint indicators have no numeric missing values, but clinical follow-up is not complete. No original CSV bytes changed.

Sources: https://www.nature.com/articles/nature23480/tables/2 ; https://doi.org/10.1038/nature25006 ; deposited README and protocol. The main article body remains subscription-restricted. Displayed method equations in the revised page are our explanatory reconstructions, not purported quotations of equations from an unavailable body text.
'''
    (ROOT/'analysis/independent-verification.md').write_text(report,encoding='utf-8')
    print(json.dumps(checks,indent=2));print('Mortality tests:',results['DEATH']['pearson_p'],results['DEATH']['fisher_p'])

if __name__=='__main__':main()
