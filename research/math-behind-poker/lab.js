/* Curated laboratory UI. No automatic simulation or permanent animation loop. */
'use strict';
(() => {
  const element = document.getElementById('poker-lab-data');
  if (!element) return;
  const data = JSON.parse(element.textContent);
  const math = window.PokerMath;
  const $ = id => document.getElementById(id);
  const suits = {s:['♠','spades'],h:['♥','hearts'],d:['♦','diamonds'],c:['♣','clubs']};
  const scenario = id => data.scenarios.find(item => item.id === id);
  const percent = value => `${(100*value).toFixed(3)}%`;
  const money = value => `$${Math.abs(value).toFixed(2)}`;
  const signedMoney = value => `${value < -1e-9 ? '−' : '+'}${money(Math.abs(value) < 1e-9 ? 0 : value)}`;
  function hand(codes, label) {
    const group = document.createElement('div');
    group.className = 'hand lab-hand-row';
    group.dataset.hand = codes.join(' ');
    group.setAttribute('role','group');
    group.setAttribute('aria-label',label);
    for (const code of codes) {
      const r = code.slice(0,-1), s = code.slice(-1), [glyph,suit] = suits[s];
      const face = document.createElement('span');
      face.className = `playing-card ${'hd'.includes(s) ? 'red' : 'black'}`;
      face.dataset.card = code;
      face.setAttribute('role','img');
      face.setAttribute('aria-label',`${({A:'Ace',K:'King',Q:'Queen',J:'Jack'})[r] || r} of ${suit}`);
      face.innerHTML = `<span class="corner" aria-hidden="true">${r}<small>${glyph}</small></span><span class="pip" aria-hidden="true">${glyph}</span><span class="corner bottom" aria-hidden="true">${r}<small>${glyph}</small></span>`;
      group.append(face);
    }
    return group;
  }
  function equation(id, lines) {
    const box = document.createElement('div');
    box.className = 'math-equation';
    for (const text of lines) {const span = document.createElement('span'); span.textContent = text; box.append(span);}
    $(id).replaceChildren(box);
  }
  function probabilityMarkup(p) {return `<strong>${p.fraction}</strong> ≈ ${p.value.toFixed(6)} ≈ <strong>${percent(p.value)}</strong>`;}
  function updateExact() {
    const s = scenario($('exact-scenario').value);
    const atTurn = $('exact-street').value === 'turn';
    const U = atTurn ? 46 : 47, o = s.out_count;
    $('exact-hole').replaceChildren(hand(s.hole,'Private hole cards'));
    $('exact-flop').replaceChildren(hand(s.flop,'Community flop'));
    $('exact-turn-wrap').hidden = !atTurn;
    $('exact-turn').replaceChildren(hand([s.miss_turn],'Observed non-out on turn'));
    $('exact-outs').replaceChildren(hand(s.outs,'Distinct current outs'));
    $('exact-target').textContent = s.target;
    $('exact-caution').textContent = s.caution;
    $('exact-known').textContent = `${atTurn ? 6 : 5} known cards → U = ${U} unseen cards; o = ${o} outs.`;
    $('next-label').textContent = `Next card · ${atTurn ? 'river' : 'turn'}`;
    $('river-label').textContent = atTurn ? 'By river · the same one remaining draw' : 'At least one out by river';
    const next = math.drawProbability(o,U,1), byRiver = math.drawProbability(o,U,atTurn ? 1 : 2);
    $('exact-next').innerHTML = probabilityMarkup(next);
    $('exact-river').innerHTML = probabilityMarkup(byRiver);
    const lines = [`P(next hit) = o/U = ${o}/${U} = ${next.fraction}`];
    if (!atTurn) lines.push('P(≥1 hit in two cards) = 1 − [(U − o)(U − o − 1)]/[U(U − 1)]', `= 1 − (${U-o} × ${U-o-1})/(${U} × ${U-1}) = ${byRiver.fraction}`);
    else lines.push(`Only the river remains: P(by river) = ${next.fraction}. The observed miss was removed from the denominator.`);
    equation('exact-equation',lines);
    $('exact-status').textContent = `${s.label}, ${atTurn ? 'after the shown turn miss' : 'after flop'}. Next-card target probability ${percent(next.value)}; by river ${percent(byRiver.value)}. These are target-hit probabilities, not equity.`;
  }
  $('exact-controls').hidden = false;
  $('exact-scenario').addEventListener('change',updateExact);
  $('exact-street').addEventListener('change',updateExact);

  function updatePot() {
    const potInput = $('pot-before'), betInput = $('pot-bet');
    const valid = [potInput,betInput].every(input => input.value !== '' && input.validity.valid && Number.isFinite(input.valueAsNumber));
    $('pot-error').hidden = valid;
    $('pot-results').hidden = !valid;
    if (!valid) { $('pot-error').textContent = 'Enter whole-dollar amounts from 0 to 10,000 for the pot and bet. Results wait for valid inputs.'; return; }
    const pot = potInput.valueAsNumber, bet = betInput.valueAsNumber, p = Number($('pot-probability').value)/100;
    const result = math.potModel(pot,bet,p);
    $('pot-probability-label').textContent = `${(100*p).toFixed(1)}%`;
    $('pot-risk').textContent = money(result.call);
    $('pot-reward').textContent = money(result.reward);
    $('pot-final').textContent = money(result.finalPot);
    $('pot-ratio').textContent = result.rewardRisk === null ? 'No paid call' : `${result.rewardRisk.toFixed(3)} : 1`;
    $('pot-threshold').textContent = result.breakEven === null ? 'Undefined (0/0)' : percent(result.breakEven);
    $('pot-ev').textContent = signedMoney(result.ev);
    $('pot-reading').textContent = result.finalPot === 0 ? 'The pot and call are both zero: EV is zero and the break-even ratio 0/0 is undefined.' : `${bet === 0 ? 'There is no call cost in this setting. ' : ''}At ${(100*p).toFixed(1)}% win probability, modeled EV is ${signedMoney(result.ev)} relative to folding. This is an assumed average net payoff, not a guaranteed result.`;
    [['pot-base-bar',pot],['pot-bet-bar',bet],['pot-call-bar',bet]].forEach(([id,value]) => {
      $(id).style.flex = String(value);
      $(id).hidden = value === 0;
      $(id).textContent = result.finalPot && value/result.finalPot >= .12 ? { 'pot-base-bar':'Before bet','pot-bet-bar':'Bet','pot-call-bar':'Call' }[id] : '';
    });
    $('pot-composition').setAttribute('aria-label',`Final pot ${result.finalPot} dollars: ${pot} previously in pot, ${bet} opponent bet, ${bet} call.`);
    equation('pot-worked',[
      `R = P₀ + B = ${pot} + ${bet} = ${result.reward}; C = B = ${bet}; F = R + C = ${result.finalPot}`,
      'EV = pR − (1 − p)C = pF − C',
      `= ${p.toFixed(3)} × ${result.reward} − ${(1-p).toFixed(3)} × ${bet} = ${signedMoney(result.ev)}`,
      result.finalPot ? `EV = 0 ⇒ p* = C/F = ${bet}/${result.finalPot} = ${math.fraction(bet,result.finalPot)} ≈ ${percent(result.breakEven)}` : 'F = 0: the break-even fraction C/F = 0/0 is undefined.'
    ]);
  }
  function preset(p) { $('pot-before').value = '100'; $('pot-bet').value = '25'; $('pot-probability').value = String(p); updatePot(); }
  $('pot-controls').hidden = false;
  ['pot-before','pot-bet','pot-probability'].forEach(id => $(id).addEventListener('input',updatePot));
  $('pot-positive').addEventListener('click',() => preset(20));
  $('pot-negative').addEventListener('click',() => preset(10));
  $('pot-reset').addEventListener('click',() => preset(20));

  // Each run holds one finite timer chain, cancelled on stop/change/page exit.
  let timer = null, generation = 0, active = false;
  let current = scenario('flush');
  let n = 1000, hits = data.reference.hits, categories = [...data.reference.categories];
  let points = data.reference_points.map(point => ({...point}));
  let last = null;
  function cancel() {
    generation++;
    if (timer !== null) clearTimeout(timer);
    timer = null;
    active = false;
    $('sim-run').disabled = false;
    $('sim-stop').disabled = true;
    $('simulation-chart').setAttribute('aria-busy','false');
  }
  function renderSimulation() {
    const estimate = n ? hits/n : null;
    $('sim-n').textContent = n.toLocaleString('en-US');
    $('sim-hits').textContent = String(hits);
    $('sim-estimate').textContent = n ? percent(estimate) : '—';
    $('sim-exact').textContent = percent(current.by_river_flop.decimal);
    $('sim-se').textContent = n ? `${(100*math.standardError(estimate,n)).toFixed(3)} pp` : '—';
    $('sim-progress').value = n;
    $('sim-progress').max = Number($('sim-size').value);
    const y = 235-current.by_river_flop.decimal*210;
    $('sim-exact-line').setAttribute('d',`M58 ${y}H598`);
    const displayed = points.filter(point => point.n >= 10);
    if (n >= 10 && !displayed.some(point => point.n === n)) displayed.push({n,estimate});
    const path = displayed.map((point,index) => `${index ? 'L' : 'M'}${58+(Math.log10(point.n)-1)*180},${235-point.estimate*210}`).join(' ');
    $('sim-estimate-line').setAttribute('d',path);
    $('sim-checkpoints').innerHTML = points.map(point => `<tr><th scope="row">${point.n.toLocaleString('en-US')}</th><td>${point.hits}</td><td>${percent(point.estimate)}</td><td>${(100*point.se).toFixed(3)}</td></tr>`).join('');
    $('sim-categories').innerHTML = [...math.categories].map((_,i) => 8-i).map(i => `<tr><th scope="row">${math.categories[i]}</th><td>${categories[i]}</td><td>${n ? percent(categories[i]/n) : '—'}</td><td>${percent(current.exact_category_counts[i]/1081)}</td></tr>`).join('');
    $('sim-last').hidden = !last;
    if (last) {
      $('sim-runout').replaceChildren(hand(last.runout,'Sampled turn and river'));
      $('sim-best').replaceChildren(hand(last.best.cards,'Strongest five-card subset of the last trial'));
      $('sim-last-label').textContent = `${last.best.category}. ${last.hit ? 'The fixed-out target was hit.' : 'The fixed-out target was not hit.'} A best-hand category and the target event are separate measurements.`;
    }
  }
  function resetSimulation() {
    cancel();
    current = scenario($('sim-scenario').value);
    n = 0; hits = 0; categories = Array(9).fill(0); points = []; last = null;
    $('sim-hole').replaceChildren(hand(current.hole,'Fixed private cards'));
    $('sim-flop').replaceChildren(hand(current.flop,'Fixed flop'));
    $('sim-event').textContent = `Event: hit at least one of the ${current.out_count} current fixed outs by the river. ${current.target} ${current.caution}`;
    $('sim-status').textContent = 'No trials yet. Choose Run simulation to start this seed from its beginning.';
    renderSimulation();
  }
  function runSimulation() {
    resetSimulation();
    const target = Number($('sim-size').value), seed = Number($('sim-seed').value);
    const sample = math.makeTrialSource(current,seed), token = generation;
    active = true;
    $('sim-run').disabled = true;
    $('sim-stop').disabled = false;
    $('simulation-chart').setAttribute('aria-busy','true');
    $('sim-status').textContent = `Running ${target.toLocaleString('en-US')} trials with seed ${seed}…`;
    let lastPaint = 0;
    function batch() {
      timer = null;
      if (token !== generation) return;
      const deadline = performance.now()+8;
      do {
        last = sample(); n++; hits += Number(last.hit); categories[last.best.rank[0]]++;
        if (data.checkpoints.includes(n)) points.push({n,hits,estimate:hits/n,se:math.standardError(hits/n,n)});
      } while (n < target && performance.now() < deadline);
      if (performance.now()-lastPaint > 120 || n === target) { renderSimulation(); lastPaint = performance.now(); }
      if (n < target) timer = setTimeout(batch,0);
      else {
        cancel();
        $('sim-status').textContent = `Complete: ${n.toLocaleString('en-US')} trials, ${hits} target hits, seed ${seed}. Estimate ${percent(hits/n)}; exact ${percent(current.by_river_flop.decimal)}. No simulation work remains scheduled.`;
      }
    }
    timer = setTimeout(batch,0);
  }
  $('simulation-controls').hidden = false;
  ['sim-scenario','sim-seed','sim-size'].forEach(id => $(id).addEventListener('change',resetSimulation));
  $('sim-run').addEventListener('click',runSimulation);
  $('sim-reset').addEventListener('click',resetSimulation);
  $('sim-stop').addEventListener('click',() => {
    if (!active) return;
    cancel(); renderSimulation();
    $('sim-status').textContent = `Stopped after ${n.toLocaleString('en-US')} completed trials. No further work is scheduled. Run restarts the selected seed.`;
  });
  window.addEventListener('pagehide',() => {
    if (!active) return;
    cancel(); renderSimulation();
    $('sim-status').textContent = `Stopped after ${n.toLocaleString('en-US')} trials when leaving the page. Run restarts the selected seed.`;
  });
})();
