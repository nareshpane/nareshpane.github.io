"""Aggregate replication of the CC0 Panigrahi deposit; never rewrites raw data.

Run: python reproduce_trial.py (Python 3.11+, pandas, numpy, scipy).
All intervals are unadjusted, fixed-sample approximations, not sequential intervals.
"""
from pathlib import Path
import csv
import hashlib
import json
import math
import platform
from html import escape
import numpy as np
import pandas as pd
import scipy
from scipy.stats import chi2_contingency, fisher_exact, norm
from zipfile import ZipFile
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/Panigrahi et al.SEPSSIF.CSV'
EXPECTED_MD5 = 'f57409eb20fb06db78ae14e99843623c'
Z = norm.ppf(.975)
LABELS = {'SEPSISDEATH':'Sepsis or death', 'SEPSIS':'Sepsis composite',
          'CULTPOS':'Culture-positive sepsis', 'SEPSISB':'Culture-negative sepsis B',
          'SEPSISC':'pSBI–LRTI', 'DEATH':'Death', 'DIARRHEA':'Diarrhoea',
          'LOCINF':'Local infection', 'ABSCESS':'Abscess', 'OMPHA':'Omphalitis',
          'GRAMNEG':'Gram-negative sepsis', 'GRAMPOS':'Gram-positive sepsis'}
PUBLISHED = {
    'SEPSISDEATH': [206,123,.60,.48,.74], 'SEPSIS':[202,117,.58,.46,.72],
    'CULTPOS':[27,6,.22,.09,.53], 'SEPSISB':[36,19,.53,.30,.92],
    'SEPSISC':[139,92,.66,.51,.88], 'DEATH':[4,6,1.50,.42,5.31],
    'DIARRHEA':[59,12,.20,.11,.38], 'LOCINF':[33,16,.48,.27,.88],
    'ABSCESS':[11,5,.45,.16,1.33], 'OMPHA':[13,3,.23,.07,.81],
    'GRAMNEG':[16,4,.25,.08,.75], 'GRAMPOS':[11,2,.18,.04,.82]}

def comparison(a, n1, c, n0):
    """a/c events among n1/n0 assigned synbiotic/placebo; no continuity correction."""
    p1,p0 = a/n1,c/n0
    rd=p1-p0; se=math.sqrt(p1*(1-p1)/n1+p0*(1-p0)/n0)
    rr=p1/p0; selog=math.sqrt(1/a-1/n1+1/c-1/n0)
    table=[[a,n1-a],[c,n0-c]]
    return dict(events1=int(a),n1=int(n1),events0=int(c),n0=int(n0),risk1=p1,risk0=p0,
                rd=rd,rd_ci=[rd-Z*se,rd+Z*se],rr=rr,
                rr_ci=[math.exp(math.log(rr)-Z*selog),math.exp(math.log(rr)+Z*selog)],
                se_log_rr=selog,arr=-rd,rrr=1-rr,
                odds_ratio=(a/(n1-a))/(c/(n0-c)),
                nnt=1/(-rd) if rd<0 else None,
                nnt_ci=[1/(-rd+Z*se),1/(-rd-Z*se)] if rd+Z*se<0 else None,
                pearson_p=float(chi2_contingency(table,correction=False).pvalue),
                fisher_p=float(fisher_exact(table).pvalue))

def comp_frame(df, col='SEPSISDEATH'):
    return comparison(df.loc[df.TX.eq(1),col].sum(),df.TX.eq(1).sum(),
                      df.loc[df.TX.eq(0),col].sum(),df.TX.eq(0).sum())

def svg(name,body,w=900,h=360,title=''):
    s=f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" role="img" aria-labelledby="title"><title id="title">{escape(title)}</title><rect width="100%" height="100%" fill="#fffdf9"/><g font-family="Georgia,serif" font-size="18" fill="#343a36">{body}</g></svg>'
    (ROOT/'figures'/name).write_text(s,encoding='utf-8')

def figures(results,base,time):
    p=results['SEPSISDEATH']; body=''
    for i,(name,risk,events,color) in enumerate([('Placebo',p['risk0'],p['events0'],'#a35f41'),('Synbiotic',p['risk1'],p['events1'],'#277799')]):
        y=65+i*95
        body+=f'<text x="25" y="{y+27}">{name}</text><rect x="180" y="{y}" width="600" height="40" rx="5" fill="#f0e9df"/><rect x="180" y="{y}" width="{risk/.12*600}" height="40" rx="5" fill="{color}"/><text x="190" y="{y+70}">{events} / 2,278 · {risk*100:.1f}%</text>'
    for v in [0,3,6,9,12]:body+=f'<text x="{180+v/12*600}" y="36" text-anchor="middle">{v}%</text>'
    body+=f'<text x="25" y="285">Absolute reduction: {p["arr"]*100:.1f} percentage points · RR {p["rr"]:.3f}</text><text x="25" y="320">95% log-RR interval: {p["rr_ci"][0]:.3f}–{p["rr_ci"][1]:.3f} · fixed-sample approximation</text>'
    svg('primary-risk.svg',body,title='Reproduced risks of sepsis or death: placebo 9.04%, synbiotic 5.40%')
    keys=['SEPSISDEATH','SEPSIS','CULTPOS','SEPSISB','SEPSISC','DEATH','DIARRHEA','LOCINF','OMPHA']
    x=lambda v:355+(math.log(v)-math.log(.03))/(math.log(8)-math.log(.03))*380
    body=f'<line x1="{x(1)}" y1="35" x2="{x(1)}" y2="{50+len(keys)*43}" stroke="#777" stroke-dasharray="5 5"/>'
    for i,k in enumerate(keys):
        r=results[k];y=65+i*43
        body+=f'<text x="14" y="{y+5}" font-size="16">{LABELS[k]}</text><line x1="{x(r["rr_ci"][0])}" y1="{y}" x2="{x(r["rr_ci"][1])}" y2="{y}" stroke="#277799" stroke-width="3"/><circle cx="{x(r["rr"])}" cy="{y}" r="5" fill="#277799"/><text x="755" y="{y+5}" font-size="15">{r["rr"]:.2f} [{r["rr_ci"][0]:.2f}, {r["rr_ci"][1]:.2f}]</text>'
    for v in [.05,.1,.25,.5,1,2,4,8]:body+=f'<text x="{x(v)}" y="{80+len(keys)*43}" font-size="14" text-anchor="middle">{v}</text>'
    body+=f'<text x="355" y="{110+len(keys)*43}" font-size="16">Synbiotic / placebo risk ratio · logarithmic scale</text>'
    svg('secondary-forest.svg',body,h=145+len(keys)*43,title='Reproduced unadjusted risk ratios and pointwise 95% intervals; no multiplicity correction')
    body='<line x1="520" y1="25" x2="520" y2="290" stroke="#777" stroke-dasharray="4 4"/>'
    for i,r in enumerate(base):
        y=50+i*43; xx=520+float(r['smd'])*1100
        body+=f'<text x="15" y="{y+5}" font-size="16">{escape(r["label"])}</text><line x1="300" x2="740" y1="{y}" y2="{y}" stroke="#e7dcd0"/><circle cx="{xx}" cy="{y}" r="6" fill="#277799"/><text x="770" y="{y+5}" font-size="15">{r["smd"]:.3f}</text>'
    for v in [-.2,-.1,0,.1,.2]:body+=f'<text x="{520+v*1100}" y="325" text-anchor="middle" font-size="15">{v:+.1f}</text>'
    svg('baseline-balance.svg',body,h=355,title='Baseline standardized differences: synbiotic minus placebo; descriptive balance, no pass/fail cutoff')
    body=''
    maxn=max(t['n'] for t in time)
    for i,t in enumerate(time):
        x=80+i*115;hh=t['n']/maxn*175
        body+=f'<rect x="{x}" y="{220-hh}" width="75" height="{hh}" fill="#277799" rx="3"/><text x="{x+37}" y="{210-hh}" text-anchor="middle">{t["n"]}</text><text x="{x+37}" y="250" text-anchor="middle" font-size="16">{t["label"]}</text>'
    body+='<text x="50" y="295">Age at recorded primary event, days; 2 event ages missing</text><text x="50" y="325" font-size="16">A distribution among cases, not a survival curve or a risk over time.</text>'
    svg('event-ages.svg',body,w=800,h=350,title='Age distribution for 327 recorded primary events with known age; two event ages missing')

def main():
    content=RAW.read_bytes();assert hashlib.md5(content).hexdigest()==EXPECTED_MD5
    d=pd.read_csv(RAW,na_values=['.'])  # SAS numeric missing; original bytes untouched
    assert d.shape==(4556,74)
    assert d.TX.value_counts().to_dict()=={0:2278,1:2278} or d.TX.value_counts().to_dict()=={1:2278,0:2278}
    for c in [*LABELS,'LOSTFU','POPBLOCK12','POPBLOCK123']:
        assert set(d[c].unique())<={0,1},c
    assert ((d.SEPSIS|d.DEATH)==d.SEPSISDEATH).all()
    assert (d.CULTPOS+d.SEPSISB+d.SEPSISC==d.SEPSIS).all()
    assert (d.GRAMPOS+d.GRAMNEG==d.CULTPOS).all()
    assert not (d.LOSTFU & d.SEPSISDEATH).any(), 'Sensitivity additions require flagged non-events'
    assert d.Total_Doses.between(0,7).all()
    assert d.TIME.dropna().between(0,60).all()
    results={k:comp_frame(d,k) for k in LABELS}
    # Keep every deposited row. Duplicate IDs do not establish duplicate infants.
    duplicate_groups=d.groupby('Study_ID').filter(lambda x:len(x)>1)
    duplicate_summary={'duplicate_id_excess':int(d.Study_ID.duplicated().sum()),
                       'rows_in_duplicate_id_groups':len(duplicate_groups),
                       'exact_duplicate_rows':int(d.duplicated().sum()),
                       'duplicate_groups_with_different_assignment':int(duplicate_groups.groupby('Study_ID').TX.nunique().gt(1).sum())}
    base=[]
    for label,col,kind in [('Birth weight (g)','BWT','continuous'),('Maternal age (years)','MOMAGE','continuous'),('Maternal parity','MOMPAR','continuous'),('Male infant','Sex_of_Infant','binary'),('35–36 weeks','GA','binary'),('Hospital birth','DELIV','binary')]:
        x=d[col] if kind=='continuous' else (d[col].eq({'Sex_of_Infant':'Male','GA':'35-36 weeks (preterm)','DELIV':'Hospital'}[col])).astype(float)
        a=x[d.TX.eq(1)].dropna();b=x[d.TX.eq(0)].dropna()
        smd=(a.mean()-b.mean())/math.sqrt((a.var(ddof=1)+b.var(ddof=1))/2)
        base.append(dict(label=label,variable=col,kind=kind,n1=len(a),n0=len(b),mean1=float(a.mean()),mean0=float(b.mean()),sd1=float(a.std()),sd0=float(b.std()),smd=float(smd)))
    # Dictionary uses deposited README descriptions; observed coding is listed separately.
    with ZipFile(ROOT/'sources/README_for_Panigrahi et al.SEPSSIF.DOCX') as z:
        doc=ET.fromstring(z.read('word/document.xml'))
    ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    desc={}
    for t in doc.findall('.//w:tbl',ns):
        for row in t.findall('w:tr',ns):
            c=[''.join(x.itertext()).strip() for x in row.findall('w:tc',ns)]
            if len(c)==5 and c[0].isdigit():desc[c[1]]=c[4]
    dictionary=[]
    for c in d.columns:
        vals=d[c].dropna(); unique=vals.unique()
        if len(unique)<=24: coding='; '.join(map(str,sorted(unique,key=str)))
        elif pd.api.types.is_numeric_dtype(vals):coding=f'Observed numeric range {vals.min()}–{vals.max()}; {len(unique)} distinct values'
        else:coding=f'{len(unique)} distinct text values (not exposed)'
        use='Primary endpoint' if c=='SEPSISDEATH' else 'Assigned arm' if c=='TX' else 'Baseline descriptive' if c in [b['variable'] for b in base] else 'Secondary endpoint' if c in LABELS else 'Sensitivity' if c in ['LOSTFU','POPBLOCK12','POPBLOCK123'] else 'Audit only'
        dictionary.append(dict(variable=c,original_name=c,definition=desc.get(c,'Not described in README'),coding=coding,missing=int(d[c].isna().sum()),analytical_use=use,dtype=str(d[c].dtype)))
    with (ROOT/'data/data_dictionary.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=dictionary[0].keys());w.writeheader();w.writerows(dictionary)
    dic='# Data dictionary\n\nDefinitions transcribed from deposited README; coding/ranges and missingness computed from CSV. SAS numeric `.` and blank fields parse as missing. No deduplication or recoding of endpoints. README says `BTW` where actual variable is `BWT`; use `BWT`. Subsample indicators are insufficiently defined for independent estimand identification.\n\n| Variable | Original name | Definition | Observed coding | Missing | Analytical use |\n|---|---|---|---|---:|---|\n'
    dic+='\n'.join('| '+' | '.join(str(r[k]).replace('|','/') for k in ['variable','original_name','definition','coding','missing','analytical_use'])+' |' for r in dictionary)
    (ROOT/'documentation/data-dictionary.md').write_text(dic,encoding='utf-8')
    flags={str(k):{'n':int(len(v)),'lostfu':int(v.LOSTFU.sum()),'completed_original':int(v.Completed_Study.eq('Completed study').sum()),'status':{str(x):int(y) for x,y in v.STATUS.value_counts().items()},'seven_doses':int(v.Total_Doses.eq(7).sum())} for k,v in d.groupby('TX')}
    sensitivity={'all_assigned':results['SEPSISDEATH'],'exclude_lostfu':comp_frame(d[d.LOSTFU.eq(0)]),'block_positions_1_2_3':comp_frame(d[d.POPBLOCK123.eq(1)]),'block_positions_1_2':comp_frame(d[d.POPBLOCK12.eq(1)])}
    for name,add1,add0 in [('all_flagged_events',129,82),('flagged_synbiotic_only',129,0),('flagged_placebo_only',0,82)]:
        sensitivity[name]=comparison(123+add1,2278,206+add0,2278)
    # Sex interaction on log-risk-ratio scale; disjoint groups, fixed-sample approximation.
    sub={s:comp_frame(g) for s,g in d.groupby('Sex_of_Infant')}
    logint=math.log(sub['Male']['rr'])-math.log(sub['Female']['rr'])
    seint=math.sqrt(sub['Male']['se_log_rr']**2+sub['Female']['se_log_rr']**2)
    interaction=dict(subgroups=sub,ratio_of_rr=math.exp(logint),ci=[math.exp(logint-Z*seint),math.exp(logint+Z*seint)],p=float(2*norm.sf(abs(logint/seint))),scale='Male RR / Female RR',status='Exploratory; no multiplicity adjustment')
    ev=d[d.SEPSISDEATH.eq(1)]
    time=[dict(label=f'{lo}–{hi}',n=int(ev.TIME.between(lo,hi).sum())) for lo,hi in [(1,10),(11,20),(21,30),(31,40),(41,50),(51,60)]]
    quality=dict(rows=len(d),variables=len(d.columns),**duplicate_summary,sha256=hashlib.sha256(content).hexdigest(),missing={c:int(v) for c,v in d.isna().sum().items()},arm_flags=flags,primary_time_missing=int(ev.TIME.isna().sum()),complete_censoring_times_available=False,
                 bw_range=[int(d.BWT.min()),int(d.BWT.max())],distinct_id_prefixes=int(d.Study_ID.str[:4].nunique()),dose_counts={str(x):int(y) for x,y in d.Total_Doses.value_counts().items()})
    out=dict(provenance={'dataset':'doi:10.5061/dryad.275d4','version':21098,'access_date':'2026-11-04','python':platform.python_version(),'pandas':pd.__version__,'numpy':np.__version__,'scipy':scipy.__version__},quality=quality,primary=results['SEPSISDEATH'],outcomes=results,baseline=base,sensitivity=sensitivity,interaction=interaction,event_age_bins=time,published=PUBLISHED)
    (ROOT/'data/derived_summary.json').write_text(json.dumps(out,indent=2,allow_nan=False),encoding='utf-8')
    (ROOT/'data/derived_summary.js').write_text('window.RCT_RESULTS = '+json.dumps(out,allow_nan=False)+';\n',encoding='utf-8')
    with (ROOT/'data/derived_tables.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['outcome','placebo_events','placebo_n','synbiotic_events','synbiotic_n','rr','rr_lo','rr_hi','rd','rd_lo','rd_hi','pearson_p','fisher_p'])
        for k,r in results.items():w.writerow([k,r['events0'],r['n0'],r['events1'],r['n1'],r['rr'],*r['rr_ci'],r['rd'],*r['rd_ci'],r['pearson_p'],r['fisher_p']])
    figures(results,base,time)
    report=f'''# Statistical reproduction and reconciliation

Executed deterministically on the unchanged Dryad CSV. SHA-256: `{quality['sha256']}`. Source version 21098. All 4,556 deposited rows retained, 2,278 per assignment. TX=0 placebo, TX=1 synbiotic per README. Denominators correspond to the randomized population; a complete binary indicator does not prove complete prospective follow-up.

## Methods

Crude arm risks; unpooled Wald RD interval; Katz log-RR interval; uncorrected Pearson chi-square and two-sided Fisher exact tests. These are fixed-sample, unadjusted intervals. Protocol pp.31–33 prespecifies ITT, chi-square/Fisher tests and interim monitoring. Reporting summary identifies SAS 9.4. No original SAS program or exact published CI procedure is deposited; matching rounded values is numerical replication, not proof of identical software or sequential inference.

## Published Table 2 versus released-data calculations

| Endpoint | Published placebo/synbiotic events | Deposited placebo/synbiotic events | Published RR (95% CI) | Reproduced RR (95% CI) | RR minus published |
|---|---|---|---|---|---|
'''
    for k,r in results.items():
        pub=PUBLISHED[k];report+=f'| {LABELS[k]} | {pub[0]}/{pub[1]} | {r["events0"]}/{r["events1"]} | {pub[2]:.2f} ({pub[3]:.2f}, {pub[4]:.2f}) | {r["rr"]:.5f} ({r["rr_ci"][0]:.5f}, {r["rr_ci"][1]:.5f}) | {r["rr"]-pub[2]:+.5f} |\n'
    p=results['SEPSISDEATH']
    report+=f'''\nPrimary risks: {p['risk1']:.8f} versus {p['risk0']:.8f}; RD {p['rd']:.8f} (95% CI {p['rd_ci']}); RR {p['rr']:.8f} (95% CI {p['rr_ci']}); Pearson p={p['pearson_p']:.9g}; Fisher p={p['fisher_p']:.9g}. ARR={p['arr']:.8f}; reciprocal ARR={p['nnt']:.5f} (Wald-derived interval {p['nnt_ci']}). Published NNT 27 [19,47]; the reciprocal point estimate is 27.45 and rounds up to 28 under a ceiling convention. Do not force it to 27.

## Data integrity and unresolved discrepancies

- {duplicate_summary['duplicate_id_excess']} repeated-ID excess rows, {duplicate_summary['rows_in_duplicate_id_groups']} rows in repeated-ID groups; {duplicate_summary['exact_duplicate_rows']} exactly duplicated records. Retained all records; no private IDs displayed. Distinct clinical records cannot safely be collapsed by identifier alone.
- Primary indicator equals SEPSIS OR DEATH for every row; sepsis equals CULTPOS + SEPSISB + SEPSISC. Culture-positive sepsis equals GRAMPOS + GRAMNEG. Binary coding and dose/time ranges verified.
- Dataset LOSTFU combines lost/withdrawn: placebo 82, synbiotic 129. Published Fig.1 lost/withdrawn sums are 89 and 131. Original Completed_Study values also differ from STATUS. Preserve these separate measures; do not harmonize silently.
- Published Table 1 birth weight is 2799.9 vs 2803.7 g. MOMAGE and MOMPAR have 14 and 15 missing values in the release, despite the published table's general no-missing footnote. Report variable-specific denominators.
- Corrigendum: 88 suspected cases comprise culture-positive and culture-negative infants; 94 additional cultures were negative. Dataset CULTPOS=33, SEPSISB=55 (sum 88). Do not call all 88 culture-positive; the additional 94 lack a dedicated sample/testing indicator in this deposit.
- Published LRTI interval upper endpoint is 0.88; our Katz upper endpoint is 0.85578. Culture-positive and abscess upper endpoints also differ at the published precision (0.537 vs 0.53; 1.306 vs 1.33). Do not invent a cause: original interval code is unavailable. All differences remain visible in the table.
- The release contains 95 distinct four-character Study_ID prefixes; the published setting is 149 villages. This cannot independently verify the reported number of villages and is not substituted for it.
- TIME is age of a recorded morbidity event, not an individual censoring time for everyone. Two primary-event ages missing. No defensible exact Kaplan–Meier reconstruction: loss dates and enrollment ages are not supplied. The case-age histogram is descriptive and is not a survival curve.
- PODRR is entirely missing; POP* fields are described only as subsamples. They are not used to guess primary denominators. Numeric `.` treated as missing; no other endpoint transformations.

## Sensitivity (exploratory, not recovered outcomes)

| Analysis | Synbiotic events/n | Placebo events/n | RR (95% CI) |
|---|---|---|---|
'''
    for name,r in sensitivity.items():report+=f'| {name} | {r["events1"]}/{r["n1"]} | {r["events0"]}/{r["n0"]} | {r["rr"]:.4f} ({r["rr_ci"][0]:.4f}, {r["rr_ci"][1]:.4f}) |\n'
    report+=f'''\nFlagged-outcome scenarios add assumed events among LOSTFU rows; these are deliberately hypothetical and are not observations. Excluding flagged rows is a complete-case descriptive comparison, not automatically unbiased. Block-position restrictions use supplied indicators and reduce precision; they do not recreate concealed assignment records. Published supplement restrictions were 0.59 [0.46,0.75] and 0.65 [0.48,0.86]; compare computed results above.

## Exploratory effect modification

Male RR / female RR = {interaction['ratio_of_rr']:.4f}, 95% CI {interaction['ci']}, interaction p={interaction['p']:.5g}. Wald contrast of independent log-RRs, not a difference in within-sex significance. No subgroup efficacy claim; not prespecified, no multiplicity adjustment, no sequential adjustment.

## Validation assertions and limits

Shape, checksum, arm coding/counts, binary ranges, composite consistency, dose ranges and event ages pass. Duplicate IDs, ambiguous subsamples, inconsistent follow-up summaries and missing censoring dates are reported, not suppressed. This pipeline reproduces aggregate binary estimates; it does not validate event adjudication, allocation concealment, all adverse events, protocol execution or a biological mechanism. Original publication full text was subscription-restricted; public tables, figure, abstract, corrigendum, supplements and the deposited 52-page protocol were reviewed. Publisher PDFs were held only in temporary review storage; CC0 deposited protocol and README are preserved locally.
'''
    (ROOT/'analysis/reproduction_report.md').write_text(report,encoding='utf-8')
    from refine_figures import main as refine_figures
    refine_figures()
    print(json.dumps({'quality':quality,'primary':p,'interaction':interaction,'sensitivity':sensitivity},indent=2))

if __name__=='__main__':main()
