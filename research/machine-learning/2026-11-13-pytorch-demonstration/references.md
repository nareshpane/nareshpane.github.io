# Checked source register

Checked November 13, 2026. References support established methods/interfaces; no externally reported empirical result is presented as this project's result. Prose, diagrams, generators, experiments, and laboratory implementations are original. No long passages are reproduced. A small upstream MathJax extension is distributed under its included Apache 2.0 license.

## Current framework comparison and mathematical renderer

Official sources checked again for the final comparison chapter:

- [PyTorch autograd](https://docs.pytorch.org/docs/stable/autograd.html) and [autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html): reverse-mode recording, gradient accumulation, and separation from optimizer updates.
- [PyTorch torch.compile](https://docs.pytorch.org/docs/stable/generated/torch.compile.html): optional compiled regions, graph breaks, fullgraph capture requirements, backend compatibility and overhead. Stable documentation resolved to 2.15 during this check.
- [TensorFlow automatic differentiation](https://www.tensorflow.org/guide/autodiff): GradientTape recording and returned derivatives.
- [TensorFlow function guide](https://www.tensorflow.org/guide/function) and [tf.function API](https://www.tensorflow.org/api_docs/python/tf/function): eager execution, tracing/caching/retracing, and explicit XLA requests with `jit_compile=True`. Graph tracing and compiler optimization are distinct.
- [Keras getting started](https://keras.io/getting_started/), [Keras 3 migration](https://keras.io/guides/migrating_to_keras_3/), and [Keras for engineers](https://keras.io/getting_started/intro_to_keras_for_engineers/): backend selection before importing standalone Keras, compatible TensorFlow/PyTorch/JAX backends, `keras.ops`, high-level training, and qualifications for backend-specific custom code.
- [PyTorch installation selector](https://pytorch.org/get-started/locally/) and [TensorFlow pip installation](https://www.tensorflow.org/install/pip): native Windows CPU/CUDA configurations, TensorFlow 2.10 as the last standard native-Windows NVIDIA GPU release, and current TensorFlow CUDA guidance through Linux/WSL2. Do not treat accelerator backends or release compatibility as interchangeable.
- [MathJax TeX configuration](https://docs.mathjax.org/en/v3.2/options/input/tex.html) and [boldsymbol extension](https://docs.mathjax.org/en/v3.2/input/tex/extensions/boldsymbol.html): the existing TeX/SVG engine remains in use. The matching 3.2.2 extension is in `js/mathjax-extensions/boldsymbol.js`; its license is adjacent. The temporary build-time MathJax installation is not part of the shipped runtime.

The derivative/model/Keras examples are original canonical files, not copied documentation examples. Actual execution records identify TensorFlow 2.21.0, PyTorch 2.15.0+cpu, and Keras 3.15.1. The controlled-weight equality check concerns two tested float32 inputs, not a performance or universal numerical-equivalence claim.

The published [TensorFlow reference monograph](https://nareshpane.github.io/research/machine-learning/2026-11-12-tensorflow-demonstration.html) was read and verified HTTP 200, 460363 bytes, SHA256 `b2486b2951edf8ecfb76dcd2c218cb5c400904a0b808a5217e263b7c584ce401`, byte-identical to the local reference. Neither reference HTML nor its assets were modified. Comparison content and code layouts were developed for this PyTorch page.

## PyTorch primary documentation

- https://docs.pytorch.org/docs/stable/ — broad public API reference. Stable pages redirected to version 2.15 during authoring.
- https://docs.pytorch.org/docs/2.15/autograd.html — automatic-differentiation interface; https://docs.pytorch.org/docs/stable/autograd.html is the requested moving entry point.
- https://docs.pytorch.org/docs/2.15/notes/autograd.html — graph semantics, gradient accumulation, no-grad and inference modes.
- https://docs.pytorch.org/docs/2.15/func.api.html — grad, vmap, jacrev/jacfwd, JVP/VJP, Hessians, functional_call; requested stable entry https://docs.pytorch.org/docs/stable/func.api.html .
- https://docs.pytorch.org/docs/2.15/generated/torch.compile.html — backend compilation interface. Requested stable entry https://docs.pytorch.org/docs/stable/generated/torch.compile.html .
- https://docs.pytorch.org/docs/2.15/profiler.html — profiler reference; the requested `/stable/profiler` entry is normalized to the `.html` documentation page. Profiling is discussed, not falsely reported as a performed profiler study.
- https://docs.pytorch.org/docs/2.15/benchmark_utils.html — repeat measurement and warm-up context; requested stable entry https://docs.pytorch.org/docs/stable/benchmark_utils.html . The project uses an explicit perf_counter CPU protocol with raw samples.
- https://docs.pytorch.org/docs/2.15/notes/randomness.html — reproducibility limitations across devices/releases and deterministic algorithm choices.
- https://pytorch.org/get-started/locally/ — official platform/device selector. Its page contained mixed older stable-version text, so it was not treated as an authoritative version number. Official CPU package-index compatibility was separately verified for CPython 3.13 Windows AMD64; torch 2.15.0+cpu was installed in an isolated temporary environment and actually executed.
- https://download.pytorch.org/whl/cpu — official CPU wheel index used for installation; not an arbitrary CUDA recommendation.

These links explain interfaces; mathematical examples and numerical derivations in the page are independently developed. Runtime behavior and exact availability are supported by the local recorded execution, not merely a documentation link.

## Survival / competing risks

Aalen, Odd O., and Søren Johansen (1978). *An Empirical Transition Matrix for Non-Homogeneous Markov Chains Based on Censored Observations*. Scandinavian Journal of Statistics **5**, 141–150.

Authors' institutional 1977 preprint record/PDF:
https://www.math.ku.dk/bibliotek/arkivet/preprints-fra-ims/1977/preprint_1977_-_no_6_aalen__odd__johansen__s_ren_-_an_empirical_transition_matrix_for_non-homogeneous---.pdf

Historical competing-risk state-probability context. Search verified the institutional title/authors; the archive's octet-stream PDF could not be parsed by the web reader. Final journal details were cross-checked against the scholarly Wiley entry https://onlinelibrary.wiley.com/doi/10.1002/9781118445112.stat06001 . Do not conflate the 1977 preprint with the 1978 publication. This project **does not implement or reproduce the Aalen–Johansen estimator**; its discrete-time softmax likelihood and censoring-aware tests are independently derived. Conditional independent censoring and positivity are explicitly stated. No hypothetical cause-elimination causal estimand is identified.

## Graph learning

Gilmer, Justin; Samuel S. Schoenholz; Patrick F. Riley; Oriol Vinyals; and George E. Dahl (2017). *Neural Message Passing for Quantum Chemistry*. Proceedings of the 34th International Conference on Machine Learning, PMLR **70**, 1263–1272.

Primary proceedings page and verified bibliographic metadata:
https://proceedings.mlr.press/v70/gilmer17a.html

The paper establishes a general message-passing framework. Our directed incoming-mean transaction model is a compact original adaptation, not a reproduction of molecular experiments or their performance. Synthetic noisy latent-regime labels do not establish financial wrongdoing.

## Numerical transport / inverse inference

LeVeque, Randall J. (2007). *Finite Difference Methods for Ordinary and Partial Differential Equations: Steady-State and Time-Dependent Problems*. Society for Industrial and Applied Mathematics. ISBN 978-0-89871-629-0; eISBN 978-0-89871-783-9. DOI: 10.1137/1.9780898717839.

Primary publisher page:
https://epubs.siam.org/doi/book/10.1137/1.9780898717839

Discretization/stability context. This page's explicit upwind-advection/centered-diffusion combined restriction is derived directly by nonnegative coefficients. Source inference, manufactured forcing, observation conventions, and gradient checks are original. No river field dataset or real transport parameter claim is borrowed. Same-grid source recovery is an optimistic control and is labeled accordingly.

## Risk optimization

Rockafellar, R. Tyrrell, and Stanislav Uryasev (2000). *Optimization of Conditional Value-at-Risk*. Journal of Risk **2**(3), 21–41. DOI: 10.21314/jor.2000.038.

https://doi.org/10.21314/jor.2000.038

Bibliographic details independently checked in the author's primary publication register (item 85): https://uryasev.github.io/publications/ . The paper supplies the auxiliary-threshold empirical CVaR formulation. Synthetic city/demand generation, composite fairness/capacity objective, differentiable implementation, greedy integer realization, and transportation comparison are original. The cost-only exact solver is not an optimizer for the whole soft multiobjective criterion.

## Rendering and licenses

An existing repository MathJax TeX-to-SVG renderer is referenced from `research/miscellaneous/harmonized-system/details-of-machine-learning-models/js/vendor/tex-svg.js`; its adjacent `MATHJAX-LICENSE.txt` supplies Apache 2.0 terms. SHA256 is documented in README.md. No TensorFlow internal dependency is used and no additional renderer copy is included. Original SVGs are editable text/vector assets; no external photograph or generic stock ML image appears. Python dependencies are installed externally and retain their own upstream notices/licenses; no upstream library source is redistributed by this asset folder.
