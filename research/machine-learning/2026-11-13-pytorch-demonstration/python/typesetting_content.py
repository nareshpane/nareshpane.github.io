"""Mathematical bridges and the concise final framework-comparison chapter."""
from __future__ import annotations
import re
from html.parser import HTMLParser
from math_typesetting import m

EQUATION_NOTES={
1:'Matrix multiplication contracts the shared feature axis. Each output entry is one row–column dot product.',
2:'The data and initial parameters specify one affine prediction, its residual, and a half-squared loss.',
3:'The chain rule multiplies the residual sensitivity by the local sensitivity of the affine predictor.',
4:'Both parameter coordinates use the old gradient. The final two lines recompute prediction and loss after the simultaneous update.',
5:'The output dimension of the inner map matches the input dimension of the outer map. Their Jacobians multiply in that order.',
6:'The activation acts coordinatewise on the three hidden preactivations. Bias and weight dimensions match the implemented network.',
7:'The parameter displacement is the negative gradient scaled by the learning rate. Both can vanish at a stationary point; they are different mathematical roles.',
8:'Input coordinates index Jacobian columns; output coordinates index its rows. Trigonometric arguments are in radians.',
9:'Forward mode maps the chosen input-space tangent into the corresponding change of both outputs.',
10:'Reverse mode maps an output-space cotangent into input-space sensitivity. All vectors in this display are columns.',
11:'An event occurring in the censoring bin is observed under the end-of-bin convention. Otherwise the observed cause is zero.',
12:'The no-event category supplies the reference logit. Normalize across the three categories at the same subject and time bin.',
13:'Only subjects who remain event-free can reach the next bin. Multiplying conditional no-event probabilities gives survival from the start.',
14:'Each new cause-specific event contribution is weighted by survival through the previous bin. Cumulative incidence starts at zero.',
15:'Only the surviving first-bin mass enters the second bin. Event-free mass and the two cumulative cause masses still sum to one.',
16:'An observed event contributes survival before its bin and the selected event probability. End-of-bin censoring contributes survival through the complete bin.',
17:'The risk mask selects observed bins. Per-subject normalization and the squared coefficient penalty match the implemented objective.',
18:'The category residual is probability minus the observed-event indicator. The last term is the derivative of the ridge penalty.',
19:'Observed events by the horizon and subjects known event-free through it receive different censor-survival weights. Earlier censored histories contribute zero.',
20:'Parallel historical transfers are retained as adjacency counts. Each account has five input features.',
21:'The baseline uses 27 engineered endpoint, neighborhood, and transaction features. Its logistic score is separate from the learned edge representation.',
22:'Incoming historical transfers determine the mean. Repeated transfers contribute repeatedly; a node without incoming edges receives a zero message.',
23:'The learned edge input has 26 coordinates: two twelve-dimensional embeddings and two observed edge features. It is distinct from the baseline’s 27-dimensional vector.',
24:'This scalar teaching layer uses chosen fixed coefficients. It demonstrates the aggregation mechanism rather than fitted graph weights.',
25:'These ratios apply when their denominators are positive. The evaluation implementation supplies its documented zero-count convention.',
26:'The first line expresses flux conservation. The second uses spatially constant velocity and dispersion, matching this experiment.',
27:'Position and sustained injection intensity parameterize a smooth Gaussian source with a fixed width. The source is constant over the simulated interval.',
28:'The interior scheme uses explicit forward stepping, positive-velocity upwinding, and centered diffusion. The source is evaluated on the same spatial grid.',
29:'Collecting the neighbor and center terms exposes the stencil coefficients. Their nonnegativity is checked by the restriction below.',
30:'The sensor index runs over three fixed stations; the time index runs only over fitting rows. There are 36 fitting sensor/time pairs.',
31:'The unit-intensity response is evaluated at the same location, sensor, and time. Linearity of the unchanged discrete solver gives the displayed derivative.',
32:'Central perturbations estimate one parameter sensitivity using two objective evaluations. Step size balances truncation and rounding error.',
33:'The fixed interior stencil propagates both state and source sensitivities. Its matrix symbol is distinct from the scalar advection coefficient.',
34:'Each soft row is a distribution over stations. Scenario cost weights district expected costs by realized demand; these are not integer flows.',
35:'Expected station loads include demand. The capacity term penalizes positive overload; the fairness term gives each district equal weight.',
36:'Minimizing over the auxiliary threshold yields the empirical upper-tail average, including fractional observation mass at a boundary.',
37:'Four equally probable losses leave one and a half observations of tail mass at this confidence level.',
38:'This cost-only transportation problem serves every district and enforces station capacities with integer flows.',
39:'The threshold remains explicit in the joint optimization. The bracket is the auxiliary CVaR expression before minimizing over that threshold.'}


def training_flow():
    steps=[('Input data',r'(X,\boldsymbol y)','blue'),('Forward pass',r'\widehat{\boldsymbol y}=f_{\theta_t}(X)','lavender'),
           ('Loss',r'\mathcal L(\theta_t)','peach'),('Reverse differentiation',r'\boldsymbol g_t=\nabla_\theta\mathcal L(\theta_t)','blue'),
           ('Optimizer',r'\Delta\theta_t=-\eta\boldsymbol g_t','peach'),('Updated parameters',r'\theta_{t+1}=\theta_t+\Delta\theta_t','spring')]
    return ('<figure class="training-flow"><ol class="training-steps">'+''.join(f'<li class="{tone}"><b>{i:02d} · {title}</b>{m(tex,True)}</li>' for i,(title,tex,tone) in enumerate(steps,1))+
            '</ol><div class="training-feedback">'+m(r'\theta_{t+1}')+' returns to the next forward pass.'+
            '<svg viewBox="0 0 1000 38" aria-hidden="true"><defs><marker id="training-feedback-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0L10 5L0 10" fill="#799b89"/></marker></defs><path d="M915 2V29H250V2" fill="none" stroke="#799b89" stroke-width="1.5" marker-end="url(#training-feedback-arrow)"/></svg></div>'+
            '<figcaption>Plain gradient descent is shown. Gradients measure sensitivity; the explicit optimizer update changes the parameters. Other optimizers maintain additional state.</figcaption></figure>')


def comparison_chapter():
    from build_monograph import p,eq,table,code,panel,ROOT
    import json
    verified=json.loads((ROOT/'data/framework-comparison-validation.json').read_text())
    torch_version=verified['frameworks']['pytorch']['version'];tf_version=verified['frameworks']['tensorflow']['version']
    paired=lambda tf,pt,title:'<div class="framework-pair" aria-label="'+title+'">'+code(tf,title='TensorFlow',opened=True,compact=True,framework='tensorflow')+code(pt,title='PyTorch',opened=True,compact=True,framework='pytorch')+'</div>'
    sections=[]
    sections.append(('frameworks-shared-mathematics','Two frameworks, one mathematical foundation',
        p('PyTorch and TensorFlow are general-purpose differentiable tensor-computation frameworks used to construct and train neural networks and other mathematical models. Both support eager computation and mechanisms for graph capture or compilation. Their APIs and compilation boundaries differ; the old static-versus-dynamic slogan does not describe the modern choice.')+
        p('For input '+m(r'\boldsymbol x')+', parameters '+m(r'\theta')+', and a scalar supervised response, choose the same model and mean squared-error objective in either framework:')+
        eq(r'\begin{aligned}\widehat y_i&=f_\theta(\boldsymbol x_i),\\\mathcal L(\theta)&=\frac1n\sum_{i=1}^{n}\bigl(y_i-f_\theta(\boldsymbol x_i)\bigr)^2,\\\boldsymbol g_t&=\nabla_\theta\mathcal L(\theta_t),\\\theta_{t+1}&=\theta_t-\eta\boldsymbol g_t.\end{aligned}',
           'The model, loss, gradient, and update rule specify the mathematical method. A framework implements these operations; it does not change this objective merely by expressing it through another API.','optimization')+
        training_flow()+
        p('TensorFlow’s <a href="https://www.tensorflow.org/guide/function">function guide</a> describes traced, cached graphs and retracing. <code>tf.function</code> does not automatically mean XLA: <code>jit_compile=True</code> requests that compiler where supported. <a href="https://docs.pytorch.org/docs/stable/generated/torch.compile.html"><code>torch.compile</code></a> can compile captured regions while graph breaks leave other code eager; complete capture has stricter requirements. Both paths have startup costs and workload-specific limitations.')))
    rows=[['Core tensor','<code>tf.Tensor</code>','<code>torch.Tensor</code>'],
          ['Reverse-mode AD','<code>tf.GradientTape</code>','<code>torch.autograd</code>'],
          ['Neural-network API','Keras / <code>tf.keras</code>','<code>torch.nn</code>'],
          ['Core execution','Eager','Eager'],
          ['Graph / compiler entry','<code>tf.function</code>; optional XLA','<code>torch.compile</code>'],
          ['Training interface','Keras <code>fit()</code> and custom loops','Native loops; optional high-level APIs'],
          ['Accelerators','Supported platform configurations','Supported platform configurations'],
          ['Practical fit','Target runtime, existing systems, expertise','Target runtime, existing systems, expertise']]
    sections.append(('frameworks-at-a-glance','PyTorch and TensorFlow at a glance',
        '<div class="framework-table">'+table(['Feature','TensorFlow','PyTorch'],rows,'Current API distinctions; capabilities depend on the installed release and platform')+'</div>'+
        p('These are overlapping capabilities rather than exclusive territories. Both frameworks support custom differentiable calculations, GPU computation on compatible configurations, research, and production workflows. Keras’s <code>model.compile(optimizer=…, loss=…)</code> configures training; it is a different operation from the compiler transformation performed by <code>torch.compile(model)</code>. The four empirical studies above compare PyTorch with classical implementations; they do not establish a TensorFlow-versus-PyTorch speed or accuracy ranking.')))
    sections.append(('frameworks-same-derivative','The same derivative in two implementations',
        p('A polynomial makes the mathematical agreement directly visible. Both snippets use reverse-mode automatic differentiation at the same float64 input:')+
        eq(r'\begin{aligned}f(w)&=w^2+3w+2,\\\frac{df}{dw}&=2w+3,\\\left.\frac{df}{dw}\right|_{w=2}&=7.\end{aligned}',
           'The exact derivative is seven. Recording operations and differentiating them does not itself perform a parameter update.')+
        paired('comparison_tensorflow_gradient.py','comparison_pytorch_gradient.py','Two implementations of the same polynomial derivative')+
        p('TensorFlow records the relevant operations inside <code>GradientTape</code> and returns the requested derivative from <code>tape.gradient()</code>. PyTorch records supported operations when gradient recording is enabled; <code>backward()</code> accumulates derivatives into leaf gradients such as <code>w.grad</code>. The tape scope and gradient storage differ, while both examples apply the reverse chain rule. See the official <a href="https://www.tensorflow.org/guide/autodiff">TensorFlow autodiff</a> and <a href="https://docs.pytorch.org/docs/stable/autograd.html">PyTorch autograd</a> guides.')+
        panel('Executed example',f'Both canonical files were executed: TensorFlow {tf_version} and PyTorch {torch_version} each printed <code>7.0</code>. These are correctness checks, not timing measurements.','sage')))
    sections.append(('frameworks-neural-models','Neural-network construction and training',
        p('The corresponding model definitions represent the same affine–ReLU–affine architecture:')+
        eq(r'\begin{aligned}\mathbb R^{10}&\longrightarrow\mathbb R^{32}\longrightarrow\mathbb R,\\\boldsymbol h&=\operatorname{ReLU}(W_1\boldsymbol x+\boldsymbol b_1),\\\widehat y&=W_2\boldsymbol h+b_2.\end{aligned}',
           'Ten features enter 32 hidden units with ReLU; a linear scalar output follows. Including biases, each model has '+m(r'32(10+1)+(32+1)=385')+' trainable parameters.')+
        paired('comparison_tensorflow_model.py','comparison_pytorch_model.py','Matching neural-network architectures')+
        p('The definitions do not initialize identical numerical parameters. Output equivalence requires matching inputs, weights, biases, precision, and kernel conventions. Both actual models passed a '+m(r'2\times10')+' input batch to a '+m(r'2\times1')+' finite output. A separate controlled-weight check transposed Keras kernels into PyTorch’s weight convention and found matching float32 outputs on two fixed inputs; it does not make the independently initialized snippets equal.')+
        p('Keras supplies the high-level <code>fit()</code> workflow; native PyTorch examples commonly spell out the training loop. Both support custom training and higher-level abstractions. A training step still needs the same scientific choices: loss reduction, gradient clearing or accumulation, parameter update, and a separate evaluation procedure.')))
    sections.append(('frameworks-keras-ecosystem','Keras 3 and the modern ecosystem',
        panel('Model interface and numerical backend are separate choices',
              'Modern standalone Keras can use backends including TensorFlow, PyTorch, and JAX, with compatible installations and supported functionality. Select the backend before importing Keras.','sage')+
        '<div class="keras-example">'+code('comparison_keras_backend.py',title='Keras · choose a backend before import',opened=True,compact=True,framework='keras')+'</div>'+
        p('This exact backend-selection file also executed successfully with the PyTorch backend. Portability depends on the operations used: Keras layers and <code>keras.ops</code> support backend-agnostic code, while backend-specific custom training may require separate implementations. See <a href="https://keras.io/getting_started/">Keras getting started</a> and the <a href="https://keras.io/guides/migrating_to_keras_3/">Keras 3 migration guide</a>. A high-level model interface and its computational backend are related architectural decisions, rather than one indivisible framework choice.')))
    sections.append(('frameworks-practical-choice','Choosing a framework in practice',
        p('Choose against the work you need to maintain: custom mathematics and derivative rules, numerical experiments, model construction, debugging, accelerator compatibility, deployment targets, and existing data or serving infrastructure. Team expertise and reproducible environments can matter more than a generic preference. Both ecosystems support flexible research and production use; neither is inherently faster or more statistically accurate for every model.')+
        panel('Windows and accelerator compatibility',
              'PyTorch’s <a href="https://pytorch.org/get-started/locally/">official installer</a> offers native Windows CPU and compatible NVIDIA CUDA configurations. Match the supported wheel, Python, GPU, and driver environment. TensorFlow 2.10 was the last release with the standard native-Windows NVIDIA CUDA path; the <a href="https://www.tensorflow.org/install/pip">current TensorFlow CUDA guidance</a> uses Linux or WSL2. Native Windows CPU remains an option. CUDA, ROCm, DirectML, and other backends are not interchangeable installation targets. The examples here were checked on CPU.','lavender')+
        '<aside class="framework-conclusion"><div class="panel-label">Method and implementation</div><p><strong>The mathematical method determines what is being computed; the framework influences how that computation is implemented, differentiated, accelerated, and deployed.</strong></p></aside>'))
    return ('framework-comparison','PyTorch versus TensorFlow','PyTorch versus TensorFlow','One mathematical foundation, two modern implementation ecosystems. Compare the same derivative and architecture before considering workflow, hardware, and deployment.',sections)


def enhance_chapters(chapters):
    from build_monograph import p,eq
    additions={
        'tensor-shapes':p('A broadcast offset is a vector, not a new matrix of independently chosen values. For an '+m(r'n\times p')+' feature matrix and a '+m(r'p')+'-vector, the indexed rule is:')+
            eq(r'B_{ij}=X_{ij}+b_j,\qquad\boldsymbol b\in\mathbb R^p,\quad B\in\mathbb R^{n\times p}.','Broadcasting shares the same feature offset across rows; the workbench also exposes incompatible axes.'),
        'discrete-hazards':p('The no-event mass remaining at each step is the only mass available for later causes. Substituting the two recurrences gives the conservation law:')+
            eq(r'\begin{aligned}S_i(t)+\sum_{k=1}^{2}F_{ik}(t)&=S_i(t-1)\left(p_{i0t}+\sum_{k=1}^{2}p_{ikt}\right)+\sum_{k=1}^{2}F_{ik}(t-1)\\&=S_i(t-1)+\sum_{k=1}^{2}F_{ik}(t-1)=1.\end{aligned}','The three conditional probabilities sum to one; starting from survival one and incidence zero preserves the total probability budget.'),
        'transport-equation':p('The implemented initial and boundary conditions are homogeneous. The domain and simulated time are dimensionless:')+
            eq(r'0\le x\le1,\quad0\le t\le1.2,\qquad c(x,0)=0,\qquad c(0,t)=c(1,t)=0.','The solver inserts zero endpoint values after every explicit step; no inflow concentration or nonzero boundary forcing is added.'),
        'river-discretization':p('Here '+m(r's_j=s(x_j;\boldsymbol\phi)')+' is constant in time. On the 61-point grid, the interior update uses the dimensionless advection and diffusion coefficients:')+
            eq(r'\Delta x=\frac1{60},\qquad A=\frac{v\Delta t}{\Delta x},\qquad B=\frac{D\Delta t}{(\Delta x)^2}.','The upwind term uses the left neighbor because the documented velocity is nonnegative. The centered diffusion term uses both neighbors.')+
            eq(r'A\ge0,\qquad B\ge0,\qquad A+2B\le1,\qquad A+2B\le0.75\ \text{(implemented margin)}.','Nonnegative stencil coefficients give the monotonicity restriction. The code chooses a stricter 0.75 margin by increasing the number of time steps.'),
        'differentiate-simulator':p('In this recurrence, '+m(r'P\in\mathbb R^{59\times59}')+' is the fixed interior stencil matrix, distinct from the scalar Courant coefficient '+m(r'A')+'. The state '+m(r'\boldsymbol c^n')+' contains the 59 interior concentrations. Velocity and dispersion stay fixed while the two source parameters are differentiated; zero boundary sensitivities follow from the fixed zero endpoints.'),
        'feasible-realization':p('For a capacity shortage, introduce nonnegative integer unmet units explicitly. Continuous row weights '+m(r'a_{ij}')+' and realized integer flows '+m(r'f_{ij}')+' are different variables:')+
            eq(r'\begin{aligned}f_{ij},u_i&\in\mathbb Z_{\ge0},\\\sum_{j=1}^{3}f_{ij}+u_i&=d_i,\qquad\sum_{i=1}^{8}f_{ij}\le K_j,\\L_{\mathrm{realized}}&=\frac{\sum_{i,j}c_{ij}f_{ij}+15\sum_i u_i}{\sum_i d_i}.\end{aligned}','The browser charges for all unmet demand while respecting every station capacity. The original held-out Python cases have enough total capacity and therefore use zero unmet units.','optimization'),
        'differentiable-allocation':p('The explicit threshold '+m(r'\eta')+' is the final coordinate of the code’s parameter vector. The displayed objective is optimized jointly over logits and that threshold; replacing its bracketed term by an already-minimized CVaR would hide this dependency. The row probabilities are:')+
            eq(r'a_{ij}(\theta)=\frac{\exp(\theta_{ij})}{\sum_{r=1}^{3}\exp(\theta_{ir})},\qquad\theta\in\mathbb R^{8\times3},\quad\eta\in\mathbb R.','There are 24 logits and one auxiliary threshold. This parameterization enforces row normalization, while the soft capacity penalty remains distinct from hard integer feasibility.','optimization'),
        'demand-and-cvar':eq(r'\frac{\partial}{\partial\eta}\left[\eta+\frac{\sum_s(L_s-\eta)_+}{(1-\alpha)S}\right]=1-\frac{\#\{s:L_s>\eta\}}{(1-\alpha)S}.','Away from ties, the threshold derivative compares the empirical tail count with the desired tail mass. At ties the objective uses a subgradient.','optimization'),
        'benchmark-protocol':p('Use a fixed ratio direction and compare like-shaped numerical outputs. The displayed dashboard follows these definitions:')+
            eq(r'R=\frac{\operatorname{median}(T_{\mathrm{classical}})}{\operatorname{median}(T_{\mathrm{torch}})},\qquad e_\infty=\max_r|a_r-b_r|.','A ratio above one favors PyTorch for that measured operation. The absolute maximum discrepancy compares identical output or gradient entries; it does not measure statistical prediction quality.')
    }
    result=[]
    for identifier,title,short,intro,sections in chapters:
        updated=[]
        for sid,stitle,body in sections:
            if sid in additions:
                # Keep the mathematical bridge near the original formula and before long source.
                if sid=='river-discretization':
                    extra=additions[sid];second=extra.find('<div class="equation',extra.find('<div class="equation')+1)
                    start=body.rfind('<div class="equation',0,body.index('data-audit-old-number="28"'))
                    body=body[:start]+extra[:second]+body[start:]
                    match=re.search(r'<div class="equation[^>]*data-audit-old-number="29".*?</div><p class="equation-explanation">.*?</p>',body,re.S)
                    assert match
                    body=body[:match.end()]+extra[second:]+body[match.end():]
                else:
                    position=(body.find('<div class="lab') if sid=='tensor-shapes' else
                              body.find('<aside class="teaching-panel worked"') if sid in {'discrete-hazards','demand-and-cvar'} else body.find('<details class="code-panel"'))
                    body=body[:position]+additions[sid]+body[position:] if position>=0 else body+additions[sid]
            if sid=='function-transformations':
                body=body.replace('A VJP with the same output cotangent gives vᵀJ=(−3.7,2−cos(0.3)).',
                    'With the output seed '+m(r'\boldsymbol u=(1,-1)^\top')+', the column VJP is '+m(r'J_f^\top\boldsymbol u=(-3.7,\,2-\cos(0.3))^\top')+'. The row convention is its transpose, '+m(r'\boldsymbol u^\top J_f=(J_f^\top\boldsymbol u)^\top')+'.')
            if sid=='censoring-and-risks':
                body+=p('Here '+m(r'T_i')+' is the latent first-event bin, '+m(r'K_i')+' its cause, and '+m(r'C_i')+' the bin at whose end observation stops. The equality case observes the event, matching the generator and likelihood.')
            updated.append((sid,stitle,body))
        result.append((identifier,title,short,intro,updated))
    result.append(comparison_chapter())
    return result


def polish_inline_notation(markup):
    """Typeset recurring prose notation without touching source, labs, or existing math."""
    tokens={'∂c/∂q':r'\partial c/\partial q','∂s/∂ξ=s(x−ξ)/σ²':r'\partial s/\partial\xi=s(x-\xi)/\sigma^2',
            '[f(x+h)−f(x)]/h':r'\frac{f(x+h)-f(x)}{h}','[f(x+h)−f(x−h)]/(2h)':r'\frac{f(x+h)-f(x-h)}{2h}',
            'vᵀJ':r'\boldsymbol u^\top J','r′=(1−5η)r':r'r^\prime=(1-5\eta)r',
            'v=(1,−1)':r'\boldsymbol v=(1,-1)^\top','u=(1,−1)':r'\boldsymbol u=(1,-1)^\top',
            'Jv=(−1.7,4−cos(0.3))':r'J\boldsymbol v=(-1.7,\,4-\cos(0.3))^\top',
            '∇ℓ':r'\nabla\ell','ℓ=f₁−f₂':r'\ell=f_1-f_2','uᵀJ':r'\boldsymbol u^\top J',
            'ℝⁿ→ℝᵐ':r'\mathbb R^n\to\mathbb R^m','Jv':r'J\boldsymbol v',
            'A+2B≤.75':r'A+2B\le0.75','A+2B≤0.75':r'A+2B\le0.75',
            'λ':r'\lambda','μ':r'\mu','η':r'\eta','θ':r'\theta','β':r'\beta','ξ':r'\xi','φ':r'\phi','σ':r'\sigma','σ²':r'\sigma^2'}
    # Recurring prose references use the same subscripts, operators and spaces as displays.
    tokens.update({'xᵢ=(1,ageᵢ,markerᵢ)':r'\boldsymbol x_i=(1,\operatorname{age}_i,\operatorname{marker}_i)^\top',
        'Sᵢ(t)+Fᵢ₁(t)+Fᵢ₂(t)=1':r'S_i(t)+F_{i1}(t)+F_{i2}(t)=1','CVaR₀.₉':r'\operatorname{CVaR}_{0.9}',
        'ℝ²→ℝ³':r'\mathbb R^2\to\mathbb R^3','ℝ³→ℝ':r'\mathbb R^3\to\mathbb R',
        'Sᵢ(t−1)':r'S_i(t-1)','Sᵢ(t)':r'S_i(t)','Sᵢ(0)':r'S_i(0)',
        'Fᵢ₁(t)':r'F_{i1}(t)','Fᵢ₂(t)':r'F_{i2}(t)','F₁(t)':r'F_1(t)','F₂(t)':r'F_2(t)',
        'F₁(2)':r'F_1(2)','F₂(2)':r'F_2(2)','F₁':r'F_1','F₂':r'F_2',
        'pᵢₖₜ':r'p_{ikt}','pₖₜ':r'p_{kt}','p₀':r'p_0','p₁':r'p_1',
        'Aᵤᵥ':r'A_{uv}','Eₜ':r'E_t','Gₜ':r'G_t','Xₜ':r'X_t','Hᵢⱼ':r'H_{ij}',
        'Kⱼ':r'K_j','L²':r'L^2','Lₛ':r'L_s','Rᵢₜ':r'R_{it}','Yᵢₜ':r'Y_{it}',
        'W₁':r'W_1','W₂':r'W_2','Wₛ':r'W_s','Wₖⱼ':r'W_{kj}','Xᵢₖ':r'X_{ik}',
        'aᵢⱼ':r'a_{ij}','b₁':r'b_1','b₂':r'b_2','cᵢⱼ':r'c_{ij}','dᵢₛ':r'd_{is}',
        'fᵢⱼₛ':r'f_{ijs}','fᵢⱼ':r'f_{ij}','f₁':r'f_1','f₂':r'f_2',
        'hᵥ':r'\boldsymbol h_v','mᵥ':r'\boldsymbol m_v','qᵢ':r'q_i','r²':r'r^2','h²':r'h^2',
        'uⱼₖ²':r'u_{jk}^2','uⱼₖ':r'u_{jk}','zⱼₖ':r'z_{jk}',
        'xᵢᵀ':r'\boldsymbol x_i^\top','xᵢ':r'\boldsymbol x_i','xᵤ':r'\boldsymbol x_u','xᵥ':r'\boldsymbol x_v',
        'x₁²':r'x_1^2','x₁':r'x_1','x₂':r'x_2','xⱼ':r'x_j','βₜₖ':r'\boldsymbol\beta_{tk}',
        'ℝ²':r'\mathbb R^2','ℝ³':r'\mathbb R^3','ℝᵐ':r'\mathbb R^m','ℝⁿ':r'\mathbb R^n'})
    pattern=re.compile('|'.join(re.escape(k) for k in sorted(tokens,key=len,reverse=True)))
    class Inline(HTMLParser):
        def __init__(self):super().__init__(convert_charrefs=False);self.parts=[];self.stack=[]
        def handle_starttag(self,tag,attrs):
            cls=dict(attrs).get('class','').split();parent=self.stack[-1][1] if self.stack else False
            skip=parent or tag in {'code','pre','script','style','svg','math','label','button','select','output'} or bool(set(cls)&{'math-fragment','code-panel','lab'})
            self.parts.append(self.get_starttag_text())
            if tag not in {'img','br','hr','input','meta','link','source','wbr'}:self.stack.append((tag,skip))
        def handle_startendtag(self,tag,attrs):self.parts.append(self.get_starttag_text())
        def handle_endtag(self,tag):
            self.parts.append('</'+tag+'>')
            if self.stack and self.stack[-1][0]==tag:self.stack.pop()
        def handle_data(self,data):self.parts.append(data if self.stack and self.stack[-1][1] else pattern.sub(lambda hit:m(tokens[hit.group()]),data))
        def handle_entityref(self,name):self.parts.append('&'+name+';')
        def handle_charref(self,name):self.parts.append('&#'+name+';')
        def handle_comment(self,data):self.parts.append('<!--'+data+'-->')
    parser=Inline();parser.feed(markup);return ''.join(parser.parts)
