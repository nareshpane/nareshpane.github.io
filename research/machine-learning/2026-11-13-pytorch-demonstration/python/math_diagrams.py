"""Semantic HTML teaching figures using the monograph's single math renderer.

Only mathematical fragments go through ``m``; labels and interpretation remain
ordinary HTML.  Imports are deferred until ``diagram`` so the build helpers can
load this module without creating a circular import.
"""
from __future__ import annotations

from html import escape


def _card(title, body, tone="blue"):
    return (f'<section class="math-card math-node {tone}">'
            f'<h5>{escape(title)}</h5>{body}</section>')


def _arrow(direction="right"):
    symbol = "&#8592;" if direction == "left" else "&#8594;"
    return f'<span class="math-arrow {direction}" aria-hidden="true">{symbol}</span>'


def _figure(stem, title, subtitle, body, caption):
    identifier = f"math-diagram-{stem}"
    return (f'<figure class="math-diagram {identifier}" aria-labelledby="{identifier}-title">'
            f'<h4 class="math-figure-heading" id="{identifier}-title">{escape(title)}</h4>'
            f'<p class="math-figure-subtitle">{escape(subtitle)}</p>{body}'
            f'<figcaption>{caption}</figcaption></figure>')


def diagram(name):
    """Return one of the six existing teaching figures as accessible HTML.

    ``name`` may be the existing SVG filename or its stem.  Values are obtained
    from the checked analytical examples; no empirical model is fitted here.
    """
    from math_typesetting import m
    from refinement_examples import calculate

    stem = name.removesuffix(".svg")
    supported = {"ad-products", "competing-risk-budget", "message-aggregation",
                 "inverse-workflow", "cvar-tail", "soft-vs-feasible"}
    if stem not in supported:
        raise ValueError(f"Unknown mathematical teaching diagram: {name}")
    data = calculate()

    def formula(tex):
        return m(tex, display=True, scrollable=True)

    def paragraph(text):
        return f"<p>{text}</p>"

    if stem == "ad-products":
        ad = data["ad"]
        forward = ad["jvp"]
        reverse = ad["vjp"]
        definition = (
            '<div class="math-grid math-definition-grid">'
            + _card("A two-output function", formula(
                r"f(\boldsymbol{x})=\begin{bmatrix}x_1x_2\\x_1^2+\sin x_2\end{bmatrix},"
                r"\qquad\boldsymbol{x}=\begin{bmatrix}2\\0.3\end{bmatrix}"), "lavender")
            + _card("Rows are outputs; columns are inputs", formula(
                r"J_f(\boldsymbol{x})=\begin{bmatrix}x_2&x_1\\2x_1&\cos x_2\end{bmatrix}"
                r"=\begin{bmatrix}0.3&2\\4&0.95533649\end{bmatrix}"), "spring")
            + '</div>')
        forward_lane = (
            '<div class="math-lane-label">Forward mode: an input tangent propagates to output space.</div>'
            '<div class="math-lane forward">'
            + _card("Input tangent", formula(r"\boldsymbol{v}=\begin{bmatrix}1\\-1\end{bmatrix}")
                    + formula(r"\boldsymbol{v}\in\mathbb R^2"))
            + _arrow()
            + _card("Combine the Jacobian columns", formula(
                r"J_f\boldsymbol{v}=\begin{bmatrix}0.3(1)+2(-1)\\4(1)+\cos(0.3)(-1)\end{bmatrix}"))
            + _arrow()
            + _card("Output tangent", formula(
                rf"J_f\boldsymbol{{v}}=\begin{{bmatrix}}{forward[0]:.1f}\\{forward[1]:.8f}\end{{bmatrix}}")
                    + formula(r"J_f\boldsymbol{v}\in\mathbb R^2"))
            + '</div>')
        reverse_lane = (
            '<div class="math-lane-label">Reverse mode: an output cotangent propagates to input space.</div>'
            '<div class="math-lane reverse">'
            + _card("Input-space sensitivity", formula(
                rf"J_f^\top\boldsymbol{{u}}=\begin{{bmatrix}}{reverse[0]:.1f}\\{reverse[1]:.8f}\end{{bmatrix}}")
                    + formula(r"J_f^\top\boldsymbol{u}\in\mathbb R^2"), "peach")
            + _arrow("left")
            + _card("Combine the Jacobian rows", formula(
                r"J_f^\top\boldsymbol{u}=\begin{bmatrix}0.3(1)+4(-1)\\2(1)+\cos(0.3)(-1)\end{bmatrix}"), "peach")
            + _arrow("left")
            + _card("Output cotangent seed", formula(r"\boldsymbol{u}=\begin{bmatrix}1\\-1\end{bmatrix}")
                    + formula(r"\boldsymbol{u}\in\mathbb R^2"), "peach")
            + '</div>')
        caption = (
            "Analytical construction using the exact function tested by the foundation program. "
            "All vectors shown are columns; " + m(r"J_f\in\mathbb R^{2\times2}") + ". The equivalent row convention is "
            + m(r"\boldsymbol{u}^\top J_f=(J_f^\top\boldsymbol{u})^\top")
            + ". Rounded values are for display; the checked calculations retain full precision.")
        return _figure(stem, "The same Jacobian; two directions of sensitivity",
                       "An analytical example, with an input direction and an output seed in distinct spaces.",
                       definition + forward_lane + reverse_lane, caption)

    if stem == "competing-risk-budget":
        survival, first, second = data["competing"]["budget"]
        conditional = ('<div class="math-grid">'
            + _card("Bin 1: conditional probabilities", formula(
                r"(p_{0,1},p_{1,1},p_{2,1})=(0.80,0.15,0.05)")
                + paragraph("The full population enters the first bin event-free."))
            + _card("Bin 2: conditional probabilities", formula(
                r"(p_{0,2},p_{1,2},p_{2,2})=(0.75,0.10,0.15)")
                + paragraph("Only the surviving probability mass of 0.80 enters the second bin."))
            + '</div>')
        cumulative = ('<div class="math-grid math-grid-three">'
            + _card("Event-free through both bins", formula(
                rf"S(2)=0.80(0.75)={survival:.2f}"), "spring")
            + _card("Cause 1 by the end of bin 2", formula(
                rf"F_1(2)=0.15+0.80(0.10)={first:.2f}"), "peach")
            + _card("Cause 2 by the end of bin 2", formula(
                rf"F_2(2)=0.05+0.80(0.15)={second:.2f}"), "lavender")
            + '</div>')
        observed = ('<div class="math-grid">'
            + _card("Cause 1 observed in bin 2", formula(
                rf"\mathscr L_{{\mathrm{{event}}}}=p_{{0,1}}p_{{1,2}}={data['competing']['event_bin2']:.2f}")
                + paragraph("No event in bin 1; cause 1 in bin 2."), "peach")
            + _card("Censored at the end of bin 2", formula(
                rf"\mathscr L_{{\mathrm{{censor}}}}=p_{{0,1}}p_{{0,2}}={data['competing']['censor_bin2']:.2f}")
                + paragraph("Event-free through the entirety of both observed bins."), "spring")
            + '</div>')
        body = conditional + '<div class="math-lane-label">Allocate the surviving mass to obtain cumulative probabilities.</div>'
        body += cumulative + formula(r"S(2)+F_1(2)+F_2(2)=0.60+0.23+0.17=1") + observed
        return _figure(stem, "A probability budget across two competing-risk bins",
                       "Conditional probabilities apply only to subjects still event-free; cumulative probabilities refer to the initial population.",
                       body, "Analytical probability accounting under the implemented end-of-bin censoring convention. An event in the censoring bin is observed.")

    if stem == "message-aggregation":
        graph = data["graph"]
        features = '<div class="math-grid math-feature-grid">' + ''.join(
            _card(node, formula(rf"x_{{{node}}}={value}"), "lavender")
            for node, value in (("A", 2), ("B", 1), ("C", 4), ("D", 1))) + '</div>'
        edges = _card("Historical directed edges", formula(
            r"A\xrightarrow{\,2\,}B,\qquad C\xrightarrow{\,1\,}B,\qquad B\xrightarrow{\,1\,}D")
            + paragraph("Arrow labels give edge multiplicity. Repeated transfers contribute repeated messages."), "spring")
        calculation = ('<div class="math-grid">'
            + _card("Account B: mean incoming message", formula(r"m_B=\frac{2+2+4}{3}=\frac83")
                + formula(r"z_B=0.5(1)+0.3\left(\frac83\right)-0.2=1.1")
                + formula(rf"h_B=\tanh(1.1)={graph['h_b']:.6f}"))
            + _card("Account D: mean incoming message", formula(r"m_D=\frac11=1")
                + formula(r"z_D=0.5(1)+0.3(1)-0.2=0.6")
                + formula(rf"h_D=\tanh(0.6)={graph['h_d']:.6f}"), "peach")
            + '</div>')
        body = features + edges + formula(r"h_v=\tanh(0.5x_v+0.3m_v-0.2)") + calculation
        body += paragraph("B and D have the same self-feature and use the same transformation, yet their neighborhoods yield different embeddings. The layer updates synchronously: D receives B's original feature, not its newly computed embedding.")
        return _figure(stem, "Messages reveal what self-features alone cannot distinguish",
                       "A scalar-feature example with fixed teaching weights; these are not fitted transaction scores.",
                       body, "The fitted experiment extends this mechanism to five input features, twelve embedding coordinates, and a learned edge head.")

    if stem == "inverse-workflow":
        forward = ('<div class="math-lane-label">Forward prediction</div><div class="math-lane forward">'
            + _card("Source parameters", formula(r"\boldsymbol{\phi}=\begin{bmatrix}\xi\\q\end{bmatrix}")
                + formula(r"s(x;\boldsymbol{\phi})") + paragraph("Position and sustained injection intensity."), "lavender")
            + _arrow()
            + _card("Unchanged stable time-step solver", formula(r"A+2B\le0.75")
                + paragraph("Upwind advection and centered diffusion with zero Dirichlet endpoints."))
            + _arrow()
            + _card("Sample three sensors", formula(r"\widehat z_{jk}=c(x_j,t_k;\boldsymbol{\phi})")
                + formula(r"(x_1,x_2,x_3)=(0.45,0.65,0.85)"), "spring")
            + '</div>')
        backward = ('<div class="math-lane-label">Inverse fitting: residual sensitivities guide a new source proposal.</div>'
            '<div class="math-lane reverse">'
            + _card("Bound-aware parameter update", formula(r"0.1\le\xi\le0.55,\qquad0.1\le q\le2")
                + paragraph("L-BFGS-B uses the loss and its gradient to propose the next physical parameters."), "spring")
            + _arrow("left")
            + _card("Reverse sensitivities", formula(
                r"\nabla_{\boldsymbol{\phi}}\mathcal L=\begin{bmatrix}\partial\mathcal L/\partial\xi\\\partial\mathcal L/\partial q\end{bmatrix}")
                + paragraph("Differentiate through the same discrete time steps."), "peach")
            + _arrow("left")
            + _card("Compare with observations", formula(r"r_{jk}=\widehat z_{jk}-z_{jk}")
                + formula(r"\mathcal L=\frac{1}{2M\sigma_\varepsilon^2}\sum_{(j,k)\in\mathcal O_{\rm fit}}r_{jk}^2")
                + paragraph("Use the 36 fitting sensor/time pairs; keep held-out observations separate."), "peach")
            + '</div>')
        return _figure(stem, "Forward simulation predicts; inverse inference adjusts the source",
                       "The fitting direction uses derivatives of the implemented discretization, with the physical model held fixed.",
                       forward + backward,
                       "Sensor predictions and recorded observations enter the residual comparison. A lower loss shows agreement with this model; source identifiability and statistical validity require separate evidence.")

    if stem == "cvar-tail":
        cvar = data["cvar"]
        losses = '<div class="math-grid math-feature-grid">' + ''.join(
            _card(f"Scenario {i}", formula(rf"L_{i}={loss}") + formula(rf"w_{i}={weight}"), tone)
            for i, loss, weight, tone in ((1, 1, "0", "blue"), (2, 2, "0", "blue"),
                                          (3, 3, r"\tfrac12", "peach"), (4, 4, "1", "peach"))) + '</div>'
        tail = _card("Exact upper-tail average", formula(r"\alpha=0.625,\qquad S=4,\qquad(1-\alpha)S=1.5")
            + formula(r"\operatorname{CVaR}_{0.625}=\frac{4+\frac12(3)}{1.5}=\frac{11}{3}")
            + paragraph("Retain all of the highest loss and half the observation mass at the next loss."), "spring")
        auxiliary = _card("Auxiliary-threshold check", formula(rf"\eta={cvar['eta']}")
            + formula(r"\sum_{s=1}^{4}(L_s-3)_+=0+0+0+1=1")
            + formula(r"\eta+\frac{\sum_s(L_s-\eta)_+}{(1-\alpha)S}=3+\frac{1}{1.5}=\frac{11}{3}"), "lavender")
        return _figure(stem, "CVaR averages tail mass, including part of an observation",
                       "An analytical example with four equally probable losses; no risk distribution is fitted.",
                       losses + '<div class="math-grid">' + tail + auxiliary + '</div>',
                       "The weights describe retained observation mass. Half the weight at loss 3 leaves its loss value unchanged. Changing the confidence level changes the fraction of probability mass averaged.")

    dispatch = data["dispatch"]
    setup = _card("Two districts and two stations", formula(
        r"\boldsymbol d=\begin{bmatrix}3\\2\end{bmatrix},\quad"
        r"\boldsymbol K=\begin{bmatrix}3\\2\end{bmatrix},\quad "
        r"C=\begin{bmatrix}1&4\\2&1\end{bmatrix}")
        + paragraph("Rows are districts; columns are stations. Every district can reach either station."), "lavender")
    soft = _card("Continuous assignment weights", formula(r"a=\begin{bmatrix}0.8&0.2\\0.6&0.4\end{bmatrix}")
        + formula(r"a^\top\boldsymbol d=\begin{bmatrix}3.6\\1.4\end{bmatrix},\qquad q_1=q_2=1.6")
        + formula(rf"L_{{\rm soft}}=\frac{{3(1.6)+2(1.6)}}5={dispatch['expected_cost']:.1f}")
        + formula(rf"\mathcal C=\frac{{(3.6-3)^2}}5={dispatch['capacity_penalty']:.3f}")
        + paragraph("Both district expected costs are equal, but station 1 is overloaded. Fairness alone does not enforce capacity."))
    feasible = _card("One feasible integer realization", formula(r"f=\begin{bmatrix}3&0\\0&2\end{bmatrix}")
        + formula(r"\sum_i f_{ij}=K_j,\qquad\sum_jf_{ij}=d_i")
        + formula(rf"L_{{\rm integer}}=\frac{{3(1)+2(1)}}5={dispatch['integer_cost']:.1f}")
        + paragraph("All five demand units are served, with station loads 3 and 2. The integer decision differs from the soft expectation."), "spring")
    return _figure(stem, "A relaxed policy and a realized dispatch answer different questions",
                   "A two-district analytical construction with synthetic proxy costs; this is not another empirical city fit.",
                   setup + '<div class="math-grid">' + soft + feasible + '</div>',
                   "The relaxed policy supports gradient-based optimization. Integer flows require a separate feasibility calculation. This example does not claim that every realization improves cost.")
