"""Presentation additions to existing sections; IDs, source panels, and results are retained."""
from build_monograph import p,eq,art,table,playback_controls,ROOT
from refinement_examples import calculate


def compact_slider(id,label,minimum,maximum,value,step=1):
    return f'<label class="compact-control" for="{id}"><span class="control-head"><span>{label}</span><output id="{id}-value" for="{id}"></output></span><input id="{id}" type="range" min="{minimum}" max="{maximum}" value="{value}" step="{step}" title="{label}; range {minimum} to {maximum}"></label>'


def compact_select(id,label,options):
    return f'<label class="compact-control" for="{id}"><span class="control-head">{label}</span><select id="{id}">'+''.join(f'<option value="{value}">{text}</option>' for value,text in options)+'</select></label>'


def cell(title,body):
    return '<section class="lab-cell"><h5>'+title+'</h5>'+body+'</section>'


def river_body():
    toolbar='<div class="lab-toolbar"><div class="mode-control">'+compact_select('river-mode','Computation / playback',[('transport','Live browser transport'),('inference','Recorded PyTorch inverse fit')])+'</div>'+playback_controls()+'<span id="river-position" class="playback-position" aria-live="polite"></span></div>'
    physical='<div class="compact-controls">'+compact_slider('river-location','Position ξ',.1,.55,.4,.000001)+compact_slider('river-intensity','Intensity q',.1,1.8,.5,.000001)+compact_slider('river-velocity','Velocity v',.1,.7,.35,.01)+compact_slider('river-dispersion','Dispersion D',.002,.015,.005,.001)+'</div><p class="control-note">All variables are dimensionless; changing these controls recomputes transport, not a Python fit.</p><div id="river-scene" class="plot-wrap"></div>'
    sensors=compact_select('river-sensor','Show monitoring stations',[('all','All three stations'),('0','S1 · x=0.45'),('1','S2 · x=0.65'),('2','S3 · x=0.85')])+'<div id="river-sensor-plot" class="plot-wrap"></div><div class="sensor-key"><span style="--key:#477da5">S1 · .45</span><span style="--key:#b57a60">S2 · .65</span><span style="--key:#8d78a5">S3 · .85</span></div><p class="plot-description">Lines: prediction · dots: observations · guide: time.</p>'
    spatial=compact_slider('river-time','Displayed time t',0,1.2,1.2,.03)+'<div id="river-profile-plot" class="plot-wrap"></div><p class="plot-description">Guide: source ξ. Fit playback shows t=1.2.</p>'
    loss='<span id="river-reference-note" class="reference-note">Stored profile: original v=.35, D=.005; intensity reoptimized at every location.</span><div id="river-loss-plot" class="plot-wrap"></div><p class="plot-description">Guide: source ξ · fit dot: recorded loss, not a new profile.</p>'
    return toolbar+'<div class="lab-grid river-grid">'+cell('Source and contamination transport',physical)+cell('Sensor concentration trajectories',sensors)+cell('Spatial concentration c(x,t)',spatial)+cell('Recorded source-profile objective',loss)+'</div><div id="river-readouts" class="readouts"></div><p id="river-stage" class="stage-note" aria-live="polite"></p>'


def allocation_body():
    policy=compact_select('allocation-strategy','Strategy',[('pytorch_relaxed','PyTorch history'),('scipy_relaxed','SciPy policy'),('greedy','Cost-ranked greedy'),('nearest','Nearest + fallback'),('browser','Browser optimizer')])+compact_select('allocation-stations','Stations',[(3,'All three'),(2,'First two')])+compact_slider('allocation-capacity','Capacity ×',.5,1.5,1,.1)
    risk=compact_slider('allocation-risk','Risk λ',0,2,.7,.1)+compact_slider('allocation-fairness','Fairness μ',0,1,.25,.05)
    demand=compact_slider('allocation-concentration','Extra D8 demand',0,2,0,.125)+compact_slider('allocation-scenario','Test scenario',0,159,0)
    controls='<div class="dispatch-controls">'+''.join(f'<fieldset class="control-group {kind}"><legend>{title}</legend><div class="compact-controls">{body}</div></fieldset>' for kind,title,body in [('policy','Policy configuration',policy),('risk','Risk and fairness',risk),('demand','Demand realization',demand)])+'</div>'
    toolbar='<div class="lab-toolbar">'+playback_controls()+'<span id="allocation-position" class="playback-position" aria-live="polite"></span></div>'
    main='<div class="dispatch-main"><div class="city-view"><div id="allocation-diagram" class="plot-wrap"></div><p class="plot-description">Arrows: integer flows · red labels: shortages · station bars: load / capacity.</p></div><aside class="dispatch-statistics"><h5>Selected scenario · realized service</h5><div id="allocation-readouts" class="readouts"></div></aside></div>'
    charts='<div class="lab-grid dispatch-charts">'+cell('Optimization history and current iterate','<div id="allocation-history-plot" class="plot-wrap"></div><p id="allocation-history-note" class="plot-description"></p>')+cell('Realized cost across held-out demands','<div id="allocation-cost-plot" class="plot-wrap"></div><p id="allocation-distribution-note" class="plot-description"></p>')+'</div>'
    return controls+toolbar+main+charts+'<p id="allocation-stage" class="stage-note" aria-live="polite"></p><p class="dispatch-legend">Browser shortages are stress scenarios. Their penalty is 15 proxy units per unserved unit; the Python test scenarios have sufficient total capacity. Moving λ or μ starts the disclosed browser optimizer.</p>'


def teaching(kind,title,body,symbol='∂'):
    labels={'worked':'Worked numerical example','algorithm':'From mathematics to algorithm','concept':'Intuition and mechanism','finding':'Reading the evidence','assumption':'Assumptions and scope','limitation':'Limitations and interpretation'}
    return f'<aside class="teaching-panel {kind}"><div class="panel-label">{labels[kind]}</div><h4><span class="panel-symbol" aria-hidden="true">{symbol}</span>{title}</h4>{body}</aside>'


def sequence(items):
    return '<ol class="algorithm-sequence">'+''.join(f'<li><b>{i:02d} · {name}</b>{text}</li>' for i,(name,text) in enumerate(items,1))+'</ol>'


def illustration(name,alt,caption):
    return art(name,alt,caption,'teaching-figure')


def refine_chapters(chapters):
    d=calculate()
    extras={}
    extras['function-transformations']= (teaching('worked','A tangent goes forward; an output sensitivity goes backward',
        p('Use the same two-output function and point as the executed torch.func test. The two products use the same numerical vector (1,−1), but it lives in a different space in each calculation. First differentiate each output locally; then combine columns for an input direction or rows for an output sensitivity.')+
        eq(r'\begin{aligned}\boldsymbol v&=\begin{bmatrix}1\\-1\end{bmatrix},\qquad J_f\in\mathbb R^{2\times2},\quad\boldsymbol v\in\mathbb R^2,\\J_f\boldsymbol v&=\begin{bmatrix}0.3(1)+2(-1)\\4(1)+\cos(0.3)(-1)\end{bmatrix}=\begin{bmatrix}-1.7\\3.04466351\end{bmatrix}\in\mathbb R^2.\end{aligned}','J: 2 outputs × 2 inputs. Jv = (−1.7, 3.04466351), a vector in output space.')+
        eq(r'\begin{aligned}\boldsymbol u&=\begin{bmatrix}1\\-1\end{bmatrix}\in\mathbb R^2,\\J_f^\top\boldsymbol u&=\begin{bmatrix}0.3(1)+4(-1)\\2(1)+\cos(0.3)(-1)\end{bmatrix}=\begin{bmatrix}-3.7\\1.04466351\end{bmatrix}\in\mathbb R^2.\end{aligned}','uᵀJ = (−3.7,1.04466351), a sensitivity in input space.')+
        p('For the scalar composition ℓ=f₁−f₂, the outgoing cotangent is u=(1,−1); its VJP is exactly ∇ℓ. A JVP instead asks how both outputs change if x moves in one chosen direction. For f: ℝⁿ→ℝᵐ, J is m×n, Jv has m entries, and uᵀJ has n entries. Neither product requires storing the full Jacobian.'),'∂')+
        illustration('ad-products.svg','Two lanes distinguish a Jacobian-vector input tangent from a vector-Jacobian output sensitivity.','Analytical construction using the exact function tested by the foundation program; not a runtime benchmark.'))
    extras['discrete-hazards']= (teaching('worked','One probability budget, two different bins',
        p('Let the first bin’s conditional probabilities (no event, cause 1, cause 2) be (0.80,0.15,0.05), and the second bin’s be (0.75,0.10,0.15). Only the 0.80 still event-free can enter the second bin. This is why independent binary cumulative probabilities are not the competing-risk construction.')+
        eq(r'\begin{aligned}S(2)&=0.80(0.75)=0.60,\\F_1(2)&=0.15+0.80(0.10)=0.23,\\F_2(2)&=0.05+0.80(0.15)=0.17.\end{aligned}','At bin 2: S=.60, F₁=.23, F₂=.17; total=1.')+
        p('The conditional cause-1 probability in bin 2 is 0.10; the probability of a cause-1 event specifically in bin 2 is 0.08; cumulative cause-1 incidence through bin 2 is 0.23. These answer three different questions. A cause-1 event observed in bin 2 contributes likelihood 0.08; censoring at the end of bin 2 contributes 0.60.'),'Σ')+
        illustration('competing-risk-budget.svg','A probability tree allocates survival mass to competing causes across two bins.','Analytical probability accounting. End-of-bin censoring records survival through the complete second bin.'))
    extras['survival-likelihood']= (teaching('algorithm','Turn the observed history into one masked reduction',
        sequence([('Encode','For a bin-2 cause-1 event: labels [0,1], risk mask [1,1].'),('Normalize','log_softmax across the three categories; never across subjects or time.'),('Select','gather the observed category’s log probability in each bin.'),('Reduce','Multiply by the risk mask; sum observed bins and divide by subjects.')])+
        p('The same bin-2 censor has labels [0,0] under the end-of-bin convention. In later unobserved bins the risk mask is zero. The loss difference is an observation rule, not a different softmax model. The conventional and torch functions below implement exactly this reduction.'),'→'))
    extras['message-passing']= (teaching('worked','Identical account features can produce different embeddings',
        p('Take scalar features A=2, B=1, C=4, D=1 and historical directed edges A→B twice, C→B once, B→D once. The incoming mean at B is (2+2+4)/3=8/3, while at D it is 1. Messages in this layer use old input features simultaneously; D does not receive B’s newly updated embedding in the same layer.')+
        eq(r'm_B=\frac83,\qquad m_D=1,\qquad h_v=\tanh(0.5x_v+0.3m_v-0.2).','Controlled one-unit layer: hᵥ = tanh(.5xᵥ+.3mᵥ−.2).')+
        p(f"Therefore hB=tanh(1.1)={d['graph']['h_b']:.6f} and hD=tanh(0.6)={d['graph']['h_d']:.6f}, although B and D have the same self feature. A teaching edge head with feature e=0.5 gives logit 0.8hB+0.4hD+0.2e−0.6={d['graph']['edge_logit']:.6f}, then sigmoid score {d['graph']['edge_probability']:.6f}. These chosen coefficients are an analytical illustration, not trained model weights. The actual model below uses five inputs, twelve-dimensional embeddings, and its learned edge MLP."),'→')+
        illustration('message-aggregation.svg','Historical duplicate edges yield different neighbor means and embeddings for two equal-feature accounts.','Synchronous one-layer calculation. Repeated edges count twice, matching the native index_add operator; no synthetic score is a finding of wrongdoing.'))
    rv=d['river']
    extras['inverse-objective']= (illustration('inverse-workflow.svg','A forward model maps source parameters to sensors; inverse fitting sends residual sensitivities backward through the same solver.','A conceptual workflow for the implemented discrete inverse problem. Autograd supplies the sensitivities, not an identifiability guarantee.')+
        teaching('worked','A source-intensity change has a checkable sensor effect',
        p(f"At fixed source location 0.26, v=.35 and D=.005, the saved solver gives c={rv['concentration']:.10f} at x=.65 and t={rv['time']:.10f} for q=.8. Because the source enters linearly and the initial/boundary values are zero, c(q)=q·c(1). Thus ∂c/∂q={rv['intensity_derivative']:.10f}; changing q to .88 gives c={rv['changed_concentration']:.10f}. This is a consequence of the unchanged discrete solver, not an interpolated fitted curve.")+
        table(['q','Predicted concentration','One-observation loss','Derivative dL/dq'],[[f"{row['q']:.2f}",f"{row['prediction']:.6f}",f"{row['loss']:.6f}",f"{row['derivative']:.6f}"] for row in rv['candidates']],f"Recorded sensor observation z={rv['observation']:.6f} at t={rv['observation_time']:.6f}; σ=.025")+
        eq(r'\begin{aligned}u&=c(x_j,t_k;\xi,1),\qquad\mathcal L_{\mathrm{one}}(q)=\frac{(uq-z)^2}{2\sigma_\varepsilon^2},\\\frac{d\mathcal L_{\mathrm{one}}}{dq}&=\frac{u(uq-z)}{\sigma_\varepsilon^2}.\end{aligned}','For one observation: residual r=uq−z; loss=r²/(2σ²); gradient=u r/σ².','optimization')+
        p('This table uses one actual saved observation and a single-observation loss. The full experiment averages 36 fitting sensor/time pairs; its objective is not the number in this table. Changing location is nonlinear: ∂s/∂ξ=s(x−ξ)/σ², and the time-step recurrence propagates that source change to downstream sensors. The laboratory’s ξ/q controls change this forward calculation; replaying an existing fit does not rerun optimization.'),'∂'))
    extras['river-discretization']= (teaching('assumption','Stability, accuracy, and identifiability answer different questions',
        p('A+2B≤.75 keeps the implemented explicit step inside its monotone stability margin. Grid refinement checks discretization error. Agreement with a few sensor values checks the chosen observation loss. None of these alone establishes that source location is uniquely identified or that real transport follows the model. The four laboratory panels expose these distinct pieces together.'),'!'))
    extras['demand-and-cvar']= (teaching('worked','Fractional tail mass and the effect of risk aversion',
        p('For losses (1,2,3,4), α=.625 leaves upper-tail mass (1−α)S=1.5 observations. Give loss 4 weight 1 and loss 3 weight .5. Dividing 4+.5×3 by 1.5 gives CVaR=11/3. The auxiliary expression at η=3 gives the same value: 3+(4−3)/(4×.375).')+
        eq(r'\operatorname{CVaR}_{0.625}(1,2,3,4)=\frac{4+\tfrac12(3)}{\tfrac32}=\frac{11}{3}.','CVaR uses exactly 1.5 observations of tail mass, including fractional mass at the boundary.','optimization')+
        p('To isolate the role of λ, compare teaching loss sets A=(1,1,1,5) and B=(2.2,2.2,2.2,2.2) at α=.75, without fairness/capacity terms. Their scores are JA=2+5λ and JB=2.2+2.2λ. A wins at λ=0; at λ=.1, B wins 2.42 versus 2.50. The crossover is λ=1/14. This analytical ranking illustrates an objective tradeoff; it is not a claim about the fitted city policies.'),'Σ')+
        illustration('cvar-tail.svg','A four-loss diagram assigns one full and one half observation to the upper tail.','Analytical construction checked against the same fractional-tail convention used by the experiment.'))
    extras['feasible-realization']= (teaching('worked','A soft overload can coexist with equal district costs',
        p('For demands (3,2), capacities (3,2), costs [[1,4],[2,1]], and soft rows [[.8,.2],[.6,.4]], both district expected costs equal 1.6. The chosen fairness variance is therefore zero. Yet soft station loads are (3.6,1.4), so station 1 is over capacity. The scaled squared excess is .6²/5=.072. A fairness term does not enforce capacity.')+
        p('A feasible integer realization [[3,0],[0,2]] has station loads (3,2), serves all five units, and costs 1.0 per demand unit. The realized decision differs from the soft expectation. If capacity is insufficient, unmet units must be counted explicitly; the interactive city adds 15 proxy cost units per unmet unit, separate from the training objective’s soft capacity term.'),'→')+
        illustration('soft-vs-feasible.svg','Two demand districts have a soft overloaded allocation and a separate feasible integer flow.','A two-district analytical construction, not another empirical city fit. All original eight-district results remain intact.'))
    extras['four-kinds-of-benefit']= (teaching('finding','Read five claims independently',
        sequence([('Implementation','Which derivative expressions or state conventions must be maintained?'),('Computation','Which identical work, dtype, hardware, and timing boundary were measured?'),('Correctness','Do independent formulas, gradients, and invariants agree?'),('Quality','Did held-out statistical or realized decision performance improve?')])+
        p('A fifth question sits across all four: did the algorithm or architecture change? A larger graph model can alter prediction quality while a kernel benchmark remains unchanged. Full-training time with Adam versus L-BFGS includes optimizer differences. The panels below retain those distinct comparisons rather than combining them into a score.'),'∂'))
    result=[]
    preserved=[section[0] for chapter in chapters for section in chapter[4]]
    for id,title,short,intro,sections in chapters:
        updated=[]
        for sid,stitle,content in sections:
            if sid in extras:
                code_position=content.find('<details class="code-panel"')
                content=content[:code_position]+extras[sid]+content[code_position:] if code_position>=0 else content+extras[sid]
            updated.append((sid,stitle,content))
        result.append((id,title,short,intro,updated))
    assert preserved==[section[0] for chapter in result for section in chapter[4]]
    return result
