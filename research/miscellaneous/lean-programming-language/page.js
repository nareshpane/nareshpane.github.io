'use strict';

// Every widget enhances existing HTML independently. No Lean runs in this page.
const $ = id => document.getElementById(id);
const reducedMotion = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches;
function signalChange(element) {
  if (!reducedMotion() && element.animate) {
    element.animate([{ opacity: 0.45, transform: 'translateY(3px)' }, { opacity: 1, transform: 'translateY(0)' }], { duration: 180 });
  }
}
function revealTarget(target) {
  for (let parent = target.parentElement; parent; parent = parent.parentElement) {
    if (parent.matches('details')) parent.open = true;
    if (parent.classList.contains('tactic-panel')) {
      $('tactic-select').value = parent.id.replace('tactic-', '');
      $('tactic-select').dispatchEvent(new Event('change'));
    }
    if (parent.hasAttribute('data-view')) parent.closest('.compare').querySelector('[data-mode="both"]').click();
  }
  target.scrollIntoView({ behavior: reducedMotion() ? 'instant' : 'smooth', block: 'start' });
  target.setAttribute('tabindex', '-1');
  target.focus({ preventScroll: true });
}

function setupCode() {
  const tokens = /(--[^\n]*|\b(?:theorem|example|by|fun|with|have|Prop|Type|exact|intro|constructor|rw|simp|simpa|only|using|ring|rfl|norm_num|linarith|nlinarith|induction|rcases|use|zero|succ)\b|[ℕℤℚℝ]|\b\d+\b)/g;
  const keywords = new Set(['theorem', 'example', 'by', 'fun', 'with', 'have']);
  const types = new Set(['Prop', 'Type', 'ℕ', 'ℤ', 'ℚ', 'ℝ']);
  const explanations = {
    arithmetic: 'norm_num constructs evidence for exact numerical arithmetic. rfl proves that x and x are definitionally equal.',
    algebra: 'ring normalizes both sides into the same polynomial using the algebraic laws of ℝ.',
    logic: 'intro names the assumed conjunction. constructor opens two goals. h.2 and h.1 supply their proofs. The second declaration writes the term directly.',
    rewrite: 'rw [h] replaces x by y. The resulting equality is reflexive, so the tactic closes it.',
    odd_sum: 'induction supplies a base case and a successor case with ih. The two rewrites isolate the last summand and use ih; ring handles the remaining identity.',
    even_sum: 'rcases extracts the witnesses a and b. use supplies a+b as the new witness. Rewriting exposes a polynomial equality.',
    linear: 'The type of T includes linearity. map_add, map_zero, and map_smul expose its consequences without coordinate calculations.',
    matrix: 'Matrix.mulVec_add is a general distributivity theorem. The underscore lets elaboration infer the displayed matrix.',
    square_deriv: 'hasDerivAt_pow specializes the power rule to exponent 2. simpa simplifies its derivative expression to 2*x.',
    product_deriv: 'The local square_derivative theorem and Mathlib’s exponential derivative are combined by the proved product rule.',
    affine: 'A.hasFDerivAt says that the continuous linear map is its own derivative. Adding the constant b leaves that derivative unchanged.',
    ode: 'h establishes the inner derivative. h.exp applies the exponential chain rule, const_mul scales it, and commutative multiplication puts it in ODE form.',
    chain: 'hg.comp x hf applies Mathlib’s chain rule to the supplied differentiability hypotheses. B.comp A applies A first and B second.',
    tactic_simp: 'simp uses the established addition-by-zero simplification rule.',
    tactic_linarith: 'The hypothesis implies 2*x ≤ 4, hence x ≤ 2. linarith constructs a linear arithmetic proof.',
    tactic_nlinarith: 'sq_nonneg x supplies 0 ≤ x². nlinarith combines that bound with the positive constant 1.'
  };
  for (const code of document.querySelectorAll('.language-lean')) {
    const source = code.textContent;
    const fragment = document.createDocumentFragment();
    let end = 0;
    for (const match of source.matchAll(tokens)) {
      fragment.append(document.createTextNode(source.slice(end, match.index)));
      const span = document.createElement('span');
      const token = match[0];
      span.className = token.startsWith('--') ? 'comment' : keywords.has(token) ? 'kw' : types.has(token) ? 'type' : /^\d+$/.test(token) ? 'num' : 'tactic';
      span.textContent = token;
      fragment.append(span);
      end = match.index + token.length;
    }
    fragment.append(document.createTextNode(source.slice(end)));
    code.replaceChildren(fragment);
    const head = code.closest('.code-block').querySelector('.code-head');
    const tools = document.createElement('span');
    tools.className = 'code-tools';
    const copy = document.createElement('button');
    copy.type = 'button'; copy.className = 'copy'; copy.textContent = 'Copy';
    copy.setAttribute('aria-label', `Copy Lean example: ${code.dataset.example.replaceAll('_', ' ')}`);
    copy.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(source);
        copy.textContent = 'Copied';
      } catch {
        const range = document.createRange(); range.selectNodeContents(code);
        const selection = window.getSelection(); selection.removeAllRanges(); selection.addRange(range);
        copy.textContent = 'Selected: copy manually';
      }
      setTimeout(() => { copy.textContent = 'Copy'; }, 2500);
    });
    const explain = document.createElement('button');
    explain.type = 'button'; explain.className = 'explain'; explain.textContent = '💡 Explain';
    const note = document.createElement('p');
    note.id = `explain-${code.dataset.example}`; note.className = 'code-explanation'; note.hidden = true;
    note.textContent = explanations[code.dataset.example];
    explain.setAttribute('aria-controls', note.id); explain.setAttribute('aria-expanded', 'false');
    explain.setAttribute('aria-label', `Explain Lean example: ${code.dataset.example.replaceAll('_', ' ')}`);
    explain.addEventListener('click', () => { note.hidden = !note.hidden; explain.setAttribute('aria-expanded', String(!note.hidden)); });
    tools.append(copy, explain); head.append(tools); code.closest('.code-block').append(note);
  }
}

function setupDifficulty() {
  const levels = Array.from(document.querySelectorAll('[data-level]'));
  function select(level) {
    levels.forEach(link => { link.classList.toggle('active', link === level); if (link === level) link.setAttribute('aria-current', 'step'); else link.removeAttribute('aria-current'); });
    $('level-title').textContent = level.textContent;
    $('level-description').textContent = level.dataset.description;
    $('level-prerequisite').textContent = level.dataset.prerequisite;
    $('level-jump').href = level.getAttribute('href');
    signalChange($('difficulty-preview'));
  }
  levels.forEach(level => level.addEventListener('click', event => { if (!event.ctrlKey && !event.metaKey && !event.shiftKey && !event.altKey) { event.preventDefault(); select(level); } }));
  $('difficulty-preview').hidden = false;
  $('difficulty-preview').setAttribute('aria-live', 'polite');
  select(levels[0]);
  const menu = $('section-select'); menu.parentElement.hidden = false;
  menu.addEventListener('change', () => { if (menu.value) { location.hash = menu.value; revealTarget($(menu.value)); } });
}

function setupConcepts() {
  const list = document.querySelector('#concept-explorer [role="tablist"]');
  const tabs = Array.from(list.querySelectorAll('button'));
  function activate(tab) {
    tabs.forEach(button => {
      const active = button === tab;
      button.setAttribute('aria-selected', String(active)); button.tabIndex = active ? 0 : -1;
      const panel = $(button.getAttribute('aria-controls'));
      panel.setAttribute('role', 'tabpanel'); panel.tabIndex = 0; panel.hidden = !active;
    });
    signalChange($(tab.getAttribute('aria-controls')));
  }
  tabs.forEach((tab, index) => {
    tab.addEventListener('click', () => activate(tab));
    tab.addEventListener('keydown', event => {
      let next;
      if (event.key === 'ArrowRight') next = (index + 1) % tabs.length;
      if (event.key === 'ArrowLeft') next = (index + tabs.length - 1) % tabs.length;
      if (event.key === 'Home') next = 0;
      if (event.key === 'End') next = tabs.length - 1;
      if (next !== undefined) { event.preventDefault(); activate(tabs[next]); tabs[next].focus(); }
    });
  });
  list.hidden = false; activate(tabs[0]);
}

function setupComparisons() {
  document.querySelectorAll('.compare').forEach((pair, index) => {
    const controls = document.createElement('div'); controls.className = 'segmented comparison-controls';
    controls.setAttribute('role', 'group'); controls.setAttribute('aria-label', `${pair.dataset.compare}: representation`);
    const panels = Array.from(pair.querySelectorAll(':scope > [data-view]'));
    panels.forEach(panel => { panel.id = `comparison-${index}-${panel.dataset.view}`; });
    for (const [mode, label] of [['math', '📐 Mathematics'], ['lean', '💻 Lean'], ['both', '↔ Both']]) {
      const button = document.createElement('button'); button.type = 'button'; button.dataset.mode = mode; button.textContent = label;
      button.setAttribute('aria-pressed', String(mode === 'both'));
      button.setAttribute('aria-controls', panels.map(panel => panel.id).join(' '));
      button.addEventListener('click', () => {
        controls.querySelectorAll('button').forEach(b => b.setAttribute('aria-pressed', String(b === button)));
        pair.classList.toggle('single-view', mode !== 'both');
        panels.forEach(panel => { panel.hidden = mode !== 'both' && panel.dataset.view !== mode; });
        signalChange(pair);
      });
      controls.append(button);
    }
    pair.prepend(controls);
  });
}

function setupProof() {
  const states = Array.from(document.querySelectorAll('#proof-states > .state'));
  const commands = ['Start with the implication', 'intro h', 'constructor', 'exact h.2', 'exact h.1'];
  const previous = $('proof-prev'), next = $('proof-next'), showAll = $('proof-all');
  let step = 0, allVisible = false;
  function render() {
    $('proof-states').classList.toggle('enhanced', !allVisible);
    $('proof-states').parentElement.classList.toggle('paired-state', !allVisible && step > 0);
    states.forEach((state, i) => { state.hidden = !allVisible && i !== step; });
    $('proof-before').hidden = allVisible || step === 0;
    if (step > 0) {
      const before = states[step - 1].cloneNode(true); before.hidden = false;
      const label = document.createElement('div'); label.className = 'label'; label.textContent = 'Before this command';
      $('proof-before').replaceChildren(label, before);
    }
    previous.disabled = allVisible || step === 0; next.disabled = allVisible || step === states.length - 1;
    showAll.textContent = allVisible ? '▶ Step through proof' : 'Show all steps';
    $('proof-position').textContent = allVisible ? 'All five states' : `Step ${step + 1} of ${states.length}`;
    $('proof-step').value = String(step); $('proof-step').disabled = allVisible;
    $('proof-command').hidden = allVisible;
    $('proof-command').textContent = step === 0 ? '🎯 Start: what must we prove?' : `🧩 Command applied: ${commands[step]}`;
    signalChange(states[step]);
  }
  previous.addEventListener('click', () => { step = Math.max(0, step - 1); render(); });
  next.addEventListener('click', () => { step = Math.min(states.length - 1, step + 1); render(); });
  showAll.addEventListener('click', () => { allVisible = !allVisible; render(); });
  $('proof-step').addEventListener('change', event => { step = Number(event.target.value); render(); });
  $('proof-controls').hidden = false; render();
}

function setupTactics() {
  const menu = $('tactic-select'); const panels = document.querySelectorAll('.tactic-panel');
  function render() { panels.forEach(panel => { panel.hidden = panel.id !== `tactic-${menu.value}`; }); signalChange($(`tactic-${menu.value}`)); }
  menu.addEventListener('change', render); $('tactic-controls').hidden = false; render();
}

function setupTangent() {
  const slider = $('tangent-x');
  const plotX = t => 350 + 150 * t, plotY = y => 255 - 55 * y;
  function render() {
    const x = Number(slider.value), line = t => x * x + 2 * x * (t - x);
    $('tangent-line').setAttribute('d', `M${plotX(-2)} ${plotY(line(-2))}L${plotX(2)} ${plotY(line(2))}`);
    $('tangent-point').setAttribute('cx', plotX(x)); $('tangent-point').setAttribute('cy', plotY(x * x));
    $('tangent-value').textContent = `x = ${x.toFixed(2)}; slope = ${(2 * x).toFixed(2)}`;
    $('tangent-title').textContent = `The parabola and its tangent at x equals ${x.toFixed(2)}`;
  }
  slider.addEventListener('input', render); $('tangent-controls').hidden = false; render();
}

function setupFrechet() {
  const controls = document.querySelector('.frechet-controls');
  const descriptions = { nonlinear: 'Nonlinear map: horizontal displacements bend into the parabola (s, s²).', linear: 'Linear approximation: the identity sends (s, 0) to (s, 0).', overlay: 'Overlay: the vertical gap is the second-order error s².' };
  controls.querySelectorAll('button').forEach(button => button.addEventListener('click', () => {
    const mode = button.dataset.frechet;
    controls.querySelectorAll('button').forEach(b => b.setAttribute('aria-pressed', String(b === button)));
    document.querySelectorAll('[data-layer]').forEach(layer => { layer.style.display = mode === 'overlay' || layer.dataset.layer === mode ? '' : 'none'; });
    $('frechet-mode').textContent = descriptions[mode];
  }));
  controls.hidden = false;
}

function setupODE() {
  const slider = $('ode-a');
  function render() {
    const a = Number(slider.value), behavior = a > 0 ? 'growth' : a < 0 ? 'decay' : 'constant';
    const path = Array.from({ length: 101 }, (_, i) => { const t = i / 50; return `${i ? 'L' : 'M'}${(55 + 295 * t).toFixed(2)} ${(235 - 27.5 * Math.exp(a * t)).toFixed(2)}`; }).join(' ');
    $('ode-curve').setAttribute('d', path);
    $('ode-value').textContent = `a = ${a.toFixed(2)} · ${behavior}`;
    $('ode-title').textContent = `Exponential ${behavior} with a equal to ${a.toFixed(2)} and initial value 1`;
    $('ode-insight').textContent = a === 0 ? 'At a = 0, the derivative is zero and the solution stays at its initial value.' : `For this positive solution, ${a > 0 ? 'positive' : 'negative'} a means ${a > 0 ? 'positive' : 'negative'} slope at every time. At t = 2: y ≈ ${Math.exp(2 * a).toFixed(3)}, y′ = a y ≈ ${(a * Math.exp(2 * a)).toFixed(3)}.`;
  }
  slider.addEventListener('input', render); $('ode-controls').hidden = false; render();
}

for (const setup of [setupCode, setupDifficulty, setupConcepts, setupComparisons, setupProof, setupTactics, setupTangent, setupFrechet, setupODE]) {
  try { setup(); } catch (error) { console.error(`${setup.name}:`, error); }
}
// Links still reach examples tucked inside interactive views or disclosures.
function followHash() { const target = $(decodeURIComponent(location.hash.slice(1))); if (target) revealTarget(target); }
window.addEventListener('hashchange', followHash);
if (location.hash) followHash();
