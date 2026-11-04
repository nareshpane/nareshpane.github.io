"""Develop the archived guide into the monograph, retaining every original section.
Run after reproduce_trial.py. Build is deterministic; manuscript and styles are editable.
"""
from pathlib import Path
from html import escape
from html.parser import HTMLParser
import json, re
from refine_content import enhance_parts, enhance_originals, citation, group_parameters

ROOT=Path(__file__).resolve().parents[1]
ASSET='2026-08-09-rct-visual-guide'

def table_row(cells, header=True):
    return '<tr>'+''.join((f'<th scope="row">{c}</th>' if i==0 and header else f'<td>{c}</td>') for i,c in enumerate(cells))+'</tr>'

def main():
    old=(ROOT/'documentation/original-guide.html').read_text(encoding='utf-8')
    s=json.loads((ROOT/'data/derived_summary.json').read_text(encoding='utf-8'))
    fragment=(ROOT/'analysis/monograph-sections.html').read_text(encoding='utf-8')
    parts=dict(re.findall(r'<!-- PART:([\w-]+) -->(.*?)(?=<!-- PART:|\Z)',fragment,re.S))
    exclusions=[('Did not return after caesarean delivery in time for first dose',24),('Gestation <35 weeks',279),('Birth weight <2,000 g',344),('Infant >96 hours old',866),('Unable to tolerate oral feeds',78),('Breastfeeding not begun',40),('Did not cry immediately after birth',214),('Maternal fever within two days',179),('Foul-smelling amniotic discharge',39),('Maternal abdominal tenderness',22),('Meconium-stained fluid',9),('Infant receiving antibiotics',80),('Suspected sepsis at screening',245),('Major congenital anomalies',21),('Mother unlikely to stay 60 days',22),('Difficulty implementing study',15),('Trial material not available in time',15),('Missing/unknown reason',14)]
    assert sum(n for _,n in exclusions)==2506
    replacements={'EXCLUSION_ROWS':''.join(table_row([escape(n),str(k)]) for n,k in exclusions)}
    definitions={r['variable']:r for r in __import__('csv').DictReader((ROOT/'data/data_dictionary.csv').open(encoding='utf-8'))}
    keys=['TX','SEPSISDEATH','SEPSIS','CULTPOS','SEPSISB','SEPSISC','DEATH','LOSTFU','BWT','MOMAGE','MOMPAR','TIME','POPBLOCK123','POPBLOCK12']
    replacements['KEY_DICTIONARY']=''.join(table_row([f'<code>{k}</code>',escape(definitions[k]['definition']),definitions[k]['missing']]) for k in keys)
    def bl(r,z):
        return f'{r["mean"+z]*100:.1f}%' if r['kind']=='binary' else f'{r["mean"+z]:.1f} ({r["sd"+z]:.1f})'
    replacements['BASELINE_ROWS']=''.join(table_row([escape(r['label']),bl(r,'0'),bl(r,'1'),f'{r["n0"]:,} / {r["n1"]:,}',f'{r["smd"]:+.3f}']) for r in s['baseline'])
    p=s['primary']; ci=lambda r:f'{r["rr"]:.3f} [{r["rr_ci"][0]:.3f}, {r["rr_ci"][1]:.3f}]'
    effects=[['Synbiotic risk',f'{100*p["risk1"]:.1f}%','123 events / 2,278 assigned'],['Placebo risk',f'{100*p["risk0"]:.1f}%','206 events / 2,278 assigned'],['Risk difference',f'{100*p["rd"]:.3f} pp [{100*p["rd_ci"][0]:.3f}, {100*p["rd_ci"][1]:.3f}]','Negative means fewer adverse events'],['Risk ratio',ci(p),'Risk under synbiotic / risk under placebo'],['Absolute risk reduction',f'{100*p["arr"]:.3f} percentage points','The negative of the risk difference'],['Relative risk reduction',f'{100*p["rrr"]:.1f}%','1 − RR, relative to placebo risk'],['Odds ratio',f'{p["odds_ratio"]:.3f}','Ratio of odds; distinct from RR'],['Reciprocal ARR',f'{p["nnt"]:.4f} [{p["nnt_ci"][0]:.4f}, {p["nnt_ci"][1]:.4f}]','Participants per event prevented at this horizon; ceiling convention: 28'],['Pearson p',f'{p["pearson_p"]:.6g}','Uncorrected 1-df chi-square, two-sided'],['Fisher exact p',f'{p["fisher_p"]:.6g}','Two-sided; conditional on table margins']]
    replacements['PRIMARY_EFFECT_ROWS']=''.join(table_row(x) for x in effects)
    replacements['PRIMARY_P']=f'{p["pearson_p"]:.6g}'
    labels={'SEPSIS':'Sepsis A+B+C','CULTPOS':'Culture-positive A','SEPSISB':'Culture-negative B','SEPSISC':'pSBI–LRTI C','DEATH':'Death','GRAMNEG':'Gram-negative','GRAMPOS':'Gram-positive','DIARRHEA':'Diarrhoea','LOCINF':'Local infection','ABSCESS':'Abscess','OMPHA':'Omphalitis'}
    rows=[]
    for k,label in labels.items():
        r=s['outcomes'][k];pub=s['published'][k]
        rows.append(table_row([label,f'{pub[0]} / {pub[1]}',f'{r["events0"]} / {r["events1"]}',f'{pub[2]:.2f} [{pub[3]:.2f}, {pub[4]:.2f}]',ci(r)]))
    replacements['SECONDARY_ROWS']=''.join(rows)
    replacements['SUBGROUP_ROWS']=''.join(table_row([k,f'{r["events1"]}/{r["n1"]}',f'{r["events0"]}/{r["n0"]}',ci(r)]) for k,r in s['interaction']['subgroups'].items())
    it=s['interaction'];replacements['INTERACTION']=f'Exploratory interaction: male RR / female RR = <strong>{it["ratio_of_rr"]:.3f}</strong>, 95% CI [{it["ci"][0]:.3f}, {it["ci"][1]:.3f}]; interaction p = {it["p"]:.3f}. This does not establish sex-specific efficacy.'
    senslabels={'all_assigned':'All assigned, released coding','exclude_lostfu':'Exclude LOSTFU=1 (population changes)','block_positions_1_2_3':'Supplied block positions 1–3','block_positions_1_2':'Supplied block positions 1–2'}
    replacements['SENSITIVITY_ROWS']=''.join(table_row([senslabels[k],f'{r["events1"]}/{r["n1"]}',f'{r["events0"]}/{r["n0"]}',ci(r)]) for k,r in s['sensitivity'].items() if k in senslabels)
    for key,value in replacements.items():
        parts={k:v.replace('%%'+key+'%%',value) for k,v in parts.items()}
    assert '%%' not in ''.join(parts.values())
    parts=enhance_parts(parts)

    original=re.findall(r'<section\b.*?</section>',old,re.S)
    assert len(original)==40
    originals={}
    for i,body in enumerate(original):
        match=re.search(r'^<section[^>]*\bid="([^"]+)"',body)
        ident=match[1] if match else ['legacy-flow','legacy-cohort'][i] if i<2 else 'lab-'+re.search(r'aria-labelledby="([^"]+)"',body)[1].replace('-title','')
        if not match:body=body.replace('<section ',f'<section id="{ident}" ',1)
        # Heading hierarchy now reflects parts -> chapters -> visual subheadings.
        body=re.sub(r'<(/?)h3\b',r'<\1h4',body);body=re.sub(r'<(/?)h2\b',r'<\1h3',body)
        if i<2:body=body.replace('<h4','<h3',1).replace('</h4>','</h3>',1).replace('class="visual"','class="chapter visual"',1)
        if ident in ['d','e','h','i','k','l','m','n','o','p','v','aa'] or ident.startswith(('lab-','legacy-')):
            body=body.replace('>', '><span class="evidence synthetic-label">Synthetic educational example</span>',1)
        originals[ident]=body
    # Preserve original mathematics, correcting geometry and captions identified in the audit.
    originals['g']=originals['g'].replace('</section>','<p class="source-note">Trial connection · The actual trial used fixed blocks of four within villages. Variable blocks in general design discussions are an alternative, not a claim about this trial.</p></section>')
    originals['a']=originals['a'].replace('Adults meeting stated eligibility criteria.','Eligible infants in the Odisha trial.').replace('Assigned daily Solunex-17 tablet.','Assigned seven-day synbiotic strategy.').replace('Assigned look-alike control tablet.','Assigned maltodextrin placebo strategy.')
    originals['d']=originals['d'].replace('In this guide,','In the preserved synthetic migraine example,')
    originals['h']=originals['h'].replace('</section>','<p class="source-note">Coding reminder · “Positive” means favorable response only in the fictional migraine examples. For the infant primary endpoint, Y=1 is an adverse event, and a negative assignment effect is beneficial.</p></section>')
    originals['n']=re.sub(r'<svg viewBox="0 0 800 118".*?</svg>', '<img src="'+ASSET+'/figures/synthetic-ci.svg" width="800" height="140" alt="Synthetic risk difference 14.4 percentage points, interval 6.1 to 22.7, all positioned on one zero-to-30 percent axis.">', originals['n'],flags=re.S)
    originals['lab-extensions']=re.sub(r'<svg class="forest-svg".*?</svg>','<img src="'+ASSET+'/figures/synthetic-subgroups.svg" width="800" height="245" alt="Synthetic subgroup risk differences: lower burden 0.12, higher burden 0.08, overall 0.10; intervals shown on a common axis.">', originals['lab-extensions'],flags=re.S)
    originals['lab-extensions']=originals['lab-extensions'].replace('80 people in four clusters are not equivalent to 80 independently randomized people.','48 pictured people in four clusters are not equivalent to 48 independently randomized people.')
    originals['lab-balance-lab']=originals['lab-balance-lab'].replace('Site North, variable block','Site North, illustrative block').replace('Site South, variable block','Site South, illustrative block')
    originals['lab-sampling']=originals['lab-sampling'].replace('Each line below','This synthetic normal-estimator model uses a fixed standard error of 0.055; it is not a binomial patient simulation. Each line below')
    originals['lab-power']=originals['lab-power'].replace('not evidence about the fictional trial','not evidence about the completed synbiotic trial').replace('the control risk and absolute risk difference','control event risk and absolute risk reduction')
    originals['lab-power']=originals['lab-power'].replace('<label for="powerN">','<label for="powerBaseline">Control risk: <output id="powerBaselineOut">8%</output></label><input id="powerBaseline" type="range" min="3" max="60" step="1" value="8"><label for="powerN">').replace('min="40" max="600" step="10" value="270"','min="100" max="6000" step="100" value="2300"').replace('min="2" max="25" step="1" value="14"','min="0.5" max="7.5" step="0.1" value="3.6"').replace('Risk difference:','Risk reduction:')
    originals['lab-ri']=originals['lab-ri'].replace('The displayed two-sided Monte Carlo proportion counts simulated absolute differences at least that large.','The two-sided Monte Carlo p estimate is (extreme + 1)/(simulations + 1), including the observed assignment. A finite simulation cannot establish p=0.')
    # Trial bridges keep the existing conceptual topics connected to the central experiment.
    bridges={'b':'The trial follows this parallel-arm structure, with home surveillance and hospital referral rather than scheduled migraine assessment.', 'c':'The enrolled infant population excludes many of the most premature, smallest or already ill newborns. Randomization does not remove that eligibility boundary.', 'f':'Compare the synthetic balance lab with the released-data baseline table in Part IV. Neither is a test of successful concealment.', 'j':'For the trial, assignment is TX. SEPSISDEATH records one observed endpoint; the other potential outcome remains unobserved.', 's':'The protocol proposed regression and stepwise selection if imbalances appeared. This replication uses crude risks to keep the primary reconstruction inspectable; no post-treatment feeding or adherence covariates are adjusted for.', 't':'The trial’s multiple morbidity comparisons and early monitoring require care. The reconstructed forest intervals are pointwise, unadjusted intervals.', 'u':'The actual reporting summary identifies field personnel, hospital caregivers and investigators as blinded; the diagram below is a separate generic scenario.', 'w':'The all-assigned reconstruction uses 2,278 records per arm; it preserves assignment while sensitivity analysis addresses ascertainment.', 'x':'No CACE estimate is claimed from this deposit. Dose totals alone do not identify principal adherence strata.', 'y':'The actual sex interaction is shown in Part IV with a direct ratio-of-risk-ratios contrast.', 'z':'The synbiotic trial individually randomized infants within village blocks. The cluster trial illustration describes a different design.', 'aa':'The synbiotic-plus-FOS combination was compared with placebo. It was not a factorial test of the two ingredients.', 'ab':'The trial’s reported reduction in morbidity must be interpreted alongside eligibility, safety, measurement and missingness.', 'ac':'The retained generic pipeline is implemented here with unchanged raw data, checksums, a dictionary, deterministic calculations and generated aggregate outputs.'}
    for key,txt in bridges.items():originals[key]=originals[key].replace('</section>','<p class="source-note"><strong>Trial connection.</strong> '+txt+'</p></section>')
    originals=enhance_originals(originals)
    # JavaScript from the existing guide is retained, with auditable statistical repairs.
    scripts=re.findall(r'<script>(.*?)</script>',old,re.S);legacy=scripts[-1]
    legacy=legacy.replace('function normal(next){let s=0;for(let i=0;i<12;i+=1)s+=next();return s-6;}','function normal(next){return Math.sqrt(-2*Math.log(Math.max(next(),1e-12)))*Math.cos(2*Math.PI*next());}')
    sampling_start=legacy.index(' function drawSampling()')
    sampling_end=legacy.index(' drawSampling();document',sampling_start)
    legacy=legacy[:sampling_start]+''' function drawSampling(){const next=random(sampleSeed),truth=.12,se=.055;const intervals=Array.from({length:60},()=>{const estimate=truth+normal(next)*se;return{estimate,lo:estimate-1.96*se,hi:estimate+1.96*se};});const xmin=Math.floor(Math.min(-.15,...intervals.map(v=>v.lo))*20)/20,xmax=Math.ceil(Math.max(.4,...intervals.map(v=>v.hi))*20)/20,scale=v=>80+(v-xmin)/(xmax-xmin)*680;let covered=0;const lines=intervals.map((v,i)=>{const hit=v.lo<=truth&&v.hi>=truth;if(hit)covered++;const y=25+i*8.1,color=hit?'#287352':'#a34f4f';return `<line x1="${scale(v.lo)}" x2="${scale(v.hi)}" y1="${y}" y2="${y}" stroke="${color}" stroke-width="3"/><circle cx="${scale(v.estimate)}" cy="${y}" r="2.5" fill="${color}"/>`;});document.getElementById('samplingPlot').innerHTML=`<title>60 synthetic normal-estimator confidence intervals</title><line x1="${scale(truth)}" x2="${scale(truth)}" y1="12" y2="510" stroke="#1d6c9e" stroke-width="3"/>${lines.join('')}<text x="80" y="530" font-size="14">${(xmin*100).toFixed(0)}%</text><text x="${scale(truth)}" y="530" text-anchor="middle" font-size="14">Truth 12%</text><text x="760" y="530" text-anchor="end" font-size="14">${(xmax*100).toFixed(0)}%</text>`;document.getElementById('samplingStatus').textContent=`${covered} of 60 intervals cover the fixed 12% truth; seed ${sampleSeed}. Axis expands if necessary to show every interval.`;}
'''+legacy[sampling_end:]
    legacy=legacy.replace('extreme/fisher.length).toFixed(3)','(extreme+1)/(fisher.length+1)).toFixed(4)').replace('two-sided Monte Carlo share ${extreme}/${fisher.length} =','two-sided plus-one Monte Carlo p estimate (${extreme}+1)/(${fisher.length}+1) =')
    legacy=legacy.replace('title:"Helped: +1"','title:"Helped: +1 (favorable Y)"').replace('title:"Harmed: -1"','title:"Harmed: -1 (favorable Y)"')
    legacy=legacy.replace('<h3>${card.title}</h3>','<h4>${card.title}</h4>')
    # Fix a numeric axis which previously ran beneath the labels in the SMD plot.
    legacy=legacy.replace('x1="120" y1="${y}" x2="640"','x1="310" y1="${y}" x2="640"')
    start=legacy.index(' function power(')
    legacy=legacy[:start]+''' function power(n,d,p0){const p1=p0-d,avg=(p0+p1)/2;if(p1<=0||p1>=1)return NaN;const cutoff=1.96*Math.sqrt(2*avg*(1-avg)/n),se=Math.sqrt((p0*(1-p0)+p1*(1-p1))/n);return cdf((-cutoff+d)/se)+1-cdf((cutoff+d)/se);}
 function drawPower(){const n=+document.getElementById('powerN').value,p0=+document.getElementById('powerBaseline').value/100,e=document.getElementById('powerEffect');e.max=((p0-.005)*100).toFixed(1);if(+e.value>+e.max)e.value=e.max;const d=+e.value/100,pts=[];for(let x=100;x<=6000;x+=100)pts.push(`${55+(x-100)/5900*690},${210-power(x,d,p0)*170}`);const p=power(n,d,p0),x=55+(n-100)/5900*690;document.getElementById('powerPlot').innerHTML=`<title>Normal approximation power under chosen planning risks</title><line x1="55" y1="210" x2="745" y2="210" stroke="#777"/><line x1="55" y1="74" x2="745" y2="74" stroke="#a87524" stroke-dasharray="5 4"/><text x="58" y="67" font-size="14">80% power</text><text x="8" y="43" font-size="14">100%</text><text x="16" y="210" font-size="14">0%</text><polyline points="${pts.join(' ')}" fill="none" stroke="#1d6c9e" stroke-width="4"/><circle cx="${x}" cy="${210-p*170}" r="7" fill="#a87524"/><text x="50" y="238" font-size="14">100</text><text x="650" y="238" font-size="14">6,000 per arm</text>`;document.getElementById('powerNOut').textContent=n;document.getElementById('powerBaselineOut').textContent=(p0*100).toFixed(0)+'%';document.getElementById('powerEffectOut').textContent=(d*100).toFixed(1)+' pp';document.getElementById('powerStatus').textContent=`Synthetic planning scenario: ${n} per arm; risks ${(p0*100).toFixed(1)}% and ${((p0-d)*100).toFixed(1)}%; approximate power ${(p*100).toFixed(1)}%. Fixed-sample, two-sided 5% test.`;}
 ['powerN','powerEffect','powerBaseline'].forEach(id=>document.getElementById(id).addEventListener('input',drawPower));drawPower();
'''
    (ROOT/'legacy-labs.js').write_text(legacy,encoding='utf-8')
    css=re.search(r'<style>(.*?)</style>',old,re.S)[1]
    defs=re.search(r'<svg width="0".*?</svg>',old,re.S)[0]
    def preserved(keys):return '\n'.join(originals[k] for k in keys)
    sections=[('part-trial','Part I','The experiment','A clinical question, its documented design and the corrected outcome definitions.',parts['trial']),
      ('part-foundations','Part II','Randomization and causal inference','Retained foundations, linked to the infant trial. The fictional migraine labs show favorable response; the real trial measures adverse events.',preserved(['a','b','c'])+parts['causal-math']+preserved(['d','legacy-flow','legacy-cohort','e','f','g','lab-balance-lab','h','i','j','lab-po-lab'])),
      ('part-inference','Part III','Effects, uncertainty and design','Move from the original numerical example to derivations and reproducible inference. Every simulated result remains labeled.',preserved(['k','l'])+parts['effect-math']+preserved(['m','n','lab-sampling','o','p','lab-ri','q','r'])+parts['inference-math']+preserved(['lab-power','s','t'])),
      ('part-replication','Part IV','An empirical reconstruction','Unchanged patient-level data, transparent aggregate computations and an explicit comparison with the publication.',parts['empirical']),
      ('part-conduct','Part V','Conduct, extensions and credibility','Preserved explanations of blinding, attrition, adherence, instruments, heterogeneity, clusters and factorial designs.',preserved(['u','v','w','lab-conduct','x','y','z','aa','lab-extensions'])),
      ('part-interpretation','Part VI','Interpretation and reproducibility','What is identified, what is reproduced and what remains unresolved.',preserved(['ab','lab-validity','ac','ad'])+parts['conclusion'])]
    # All original sections remain present, even when their placement changes.
    joined=''.join(x[4] for x in sections)
    for k in originals:assert f'id="{k}"' in joined,k
    nav=[];article=[]
    for ident,num,title,description,body in sections:
        links=[]
        for match in re.finditer(r'<section\b([^>]+)>(.*?)</section>',body,re.S):
            sid=re.search(r'\bid="([^"]+)"',match[1]);heading=re.search(r'<h3[^>]*>(.*?)</h3>',match[2],re.S)
            if sid and heading:links.append(f'<a href="#{sid[1]}">{re.sub("<[^>]+>","",heading[1])}</a>')
        # Add a small number of genuine mathematical/result subsections, not every card title.
        for subid,subheading in [('observational-exchangeability','Exchangeability is not automatic in observational comparisons'),('planning-power','Power before outcomes, not a probability of truth afterward'),('primary-reconciliation','Published versus reproduced')]:
            marker='<h4>'+subheading+'</h4>'
            if marker in body:
                body=body.replace(marker,f'<h4 id="{subid}">{subheading}</h4>')
                parent={'observational-exchangeability':'identification','planning-power':'inference-design','primary-reconciliation':'primary-result'}[subid]
                for i,link in enumerate(links):
                    if f'href="#{parent}"' in link:links.insert(i+1,f'<a class="subnav" href="#{subid}">{subheading}</a>');break
        body=re.sub(r'<th\b(?![^>]*\bscope=)([^>]*)>',r'<th scope="col"\1>',body)
        body=group_parameters(body)
        nav.append(f'<details class="nav-group" open><summary>{num} · {title}</summary><a href="#{ident}">{num} overview</a>'+''.join(links)+'</details>')
        article.append(f'<section class="part" id="{ident}" aria-labelledby="{ident}-title"><header class="part-heading"><span class="part-number">{num}</span><h2 id="{ident}-title">{title}</h2><p>{description}</p></header>{body}</section>')
    header='''<header class="article-header" id="top"><div><div class="eyebrow">Visual causal inference · empirical monograph</div><h1>A Visual Guide to Randomized Controlled Trials</h1><p class="lead">From chance assignment to reproducible evidence: a close investigation of the infant synbiotic trial in rural India.</p><p class="subtitle">Panigrahi et al. · Nature (2017) · corrected interpretation · independent replication</p></div><aside class="metadata" aria-label="Page metadata"><div class="meta-row"><span class="emoji" aria-hidden="true">🗓️</span><b class="created">Created:</b><time datetime="2026-08-09">August 9, 2026</time></div><div class="meta-row"><span class="emoji" aria-hidden="true">🛠️</span><b class="modified">Modified:</b><time datetime="2026-11-04">November 4, 2026</time></div><div class="meta-row"><span class="emoji" aria-hidden="true">💻</span><b class="harness">Harness:</b><span>Codex on Windows</span></div><div class="meta-row"><span class="emoji" aria-hidden="true">🤖</span><b class="model">Model:</b><span>Not recorded</span></div><div class="meta-row"><span class="emoji" aria-hidden="true">🧠</span><b class="reasoning">Reasoning:</b><span>Not recorded</span></div></aside></header><div class="evidence-key" aria-label="Evidence categories"><span class="evidence published-label">Published study finding</span><span class="evidence reproduced-label">Independently reproduced</span><span class="evidence synthetic-label">Synthetic educational example</span><span class="evidence exploratory-label">Exploratory analysis</span></div><p id="aggregateStatus" class="small">Published evidence and reproduced calculations are distinguished throughout. The original teaching laboratories are retained.</p><noscript><p class="no-js">JavaScript is disabled. The monograph, static tables, derivations and SVG figures remain readable; simulation controls need JavaScript.</p></noscript>'''
    html='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="description" content="An audited visual RCT monograph with reproducible analysis of the Panigrahi infant synbiotic trial, corrected outcome definitions, causal inference and interactive statistical laboratories."><meta name="theme-color" content="#f5faee"><title>A Visual Guide to Randomized Controlled Trials — Panigrahi Synbiotic Replication</title><link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin><link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@600;700&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet"><style>'''+css+'</style><link rel="stylesheet" href="'+ASSET+'/monograph.css"><script>window.MathJax={tex:{inlineMath:[["$","$"],["\\\\(","\\\\)"]],displayMath:[["$$","$$"],["\\\\[","\\\\]"]]},options:{skipHtmlTags:["script","noscript","style","textarea","pre","code"]}};</script><script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script></head><body><a class="skip-link" href="#main">Skip to monograph</a><div class="page"><button id="navToggle" class="mobile-nav-toggle" type="button" aria-expanded="false" aria-controls="readerNav">☰ Contents</button><div class="layout"><nav class="reader-nav" id="readerNav" aria-label="Monograph contents"><h2>In this monograph</h2><p class="nav-subtitle">RCT design → theory → replication</p><div class="nav-tools"><button type="button" id="navExpand">Expand all</button><button type="button" id="navCollapse">Collapse all</button></div>'+''.join(nav)+'</nav><main class="sheet" id="main">'+header+''.join(article)+'<footer class="footer"><a href="#top">Back to top</a><span class="sep">|</span><a href="../../index.html">Home</a><span class="sep">|</span><a href="../../research.html">Research</a><p>Independent education and replication · source access November 4, 2026</p></footer></main></div></div>'+defs+f'<script src="{ASSET}/data/derived_summary.js"></script><script src="{ASSET}/legacy-labs.js"></script><script src="{ASSET}/monograph.js"></script></body></html>'
    html=html.replace('<h2>In this monograph</h2>','<div class="nav-header"><h2>In this monograph</h2>').replace('<button type="button" id="navCollapse">Collapse all</button></div>','<button type="button" id="navCollapse">Collapse all</button></div></div>')
    html=html.replace('<b class="model">Model:</b><span>Not recorded</span>','<b class="model">Model:</b><span>GPT-6.1 Sol</span>').replace('<b class="reasoning">Reasoning:</b><span>Not recorded</span>','<b class="reasoning">Reasoning:</b><span>High</span>')
    html=html.replace('</aside></header><div class="evidence-key"','</aside></header>'+citation()+'<div class="evidence-key"',1)
    html=html.replace(f'<link rel="stylesheet" href="{ASSET}/monograph.css">',f'<link rel="stylesheet" href="{ASSET}/monograph.css"><link rel="stylesheet" href="{ASSET}/refinement.css">')
    html=html.replace(f'<script src="{ASSET}/legacy-labs.js">',f'<script src="{ASSET}/statistical-math.js"></script><script src="{ASSET}/legacy-labs.js">').replace(f'<script src="{ASSET}/monograph.js">',f'<script src="{ASSET}/refinement-labs.js"></script><script src="{ASSET}/monograph.js">')
    (ROOT.parent/(ASSET+'.html')).write_text(html,encoding='utf-8')
    print(f'Built {len(html):,} characters; all {len(originals)} original components retained; {len(sections)} parts.')

if __name__=='__main__':main()
