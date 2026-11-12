/* One virtual 55-second timeline, scaled by a live playback rate. All movement,
   fades, reveals and holds use the same elapsed time. No independent timers. */
'use strict';
(() => {
  const $ = id => document.getElementById(id);
  const stage = $('intro-stage');
  if (!stage) return;
  const film = $('poker-introduction'), summary = $('intro-summary');
  const play = $('intro-play'), pause = $('intro-pause'), position = $('intro-position');
  const speed = $('intro-speed'), time = $('intro-time'), caption = $('intro-caption'), mode = $('intro-mode');
  const motion = matchMedia('(prefers-reduced-motion: reduce)');
  const duration = 55000, transition = 1100;
  const suits = {s: ['♠', 'spades'], h: ['♥', 'hearts'], d: ['♦', 'diamonds'], c: ['♣', 'clubs']};
  const names = {A: 'Ace', K: 'King', Q: 'Queen', J: 'Jack'};
  function card(code, start = 0) {
    const rank = code.slice(0, -1), suit = code.slice(-1);
    return `<span class="playing-card ${'hd'.includes(suit) ? 'red' : 'black'}" role="img" aria-label="${names[rank] || rank} of ${suits[suit][1]}" data-card="${code}" data-enter="${start}"><span class="corner" aria-hidden="true">${rank}<small>${suits[suit][0]}</small></span><span class="pip" aria-hidden="true">${suits[suit][0]}</span><span class="corner bottom" aria-hidden="true">${rank}<small>${suits[suit][0]}</small></span></span>`;
  }
  const hand = (codes, extra = '', start = 0) => `<div class="hand ${extra}">${codes.split(' ').map((c, i) => card(c, start + i * 130)).join('')}</div>`;
  const ranks = 'A 2 3 4 5 6 7 8 9 10 J Q K'.split(' ');
  const deck = `<div class="intro-full-deck">${Object.entries(suits).map(([s, [glyph, name]]) => `<div class="intro-suit-row" role="group" aria-label="All thirteen ${name}"><span class="intro-suit-label ${'hd'.includes(s) ? 'red-text' : ''}">${glyph} ${name}</span><div class="intro-ranks">${ranks.map(r => card(r + s)).join('')}</div></div>`).join('')}</div>`;
  // Three legible milestones instead of six one-second flashes. The full ladder
  // remains in the guide; these examples also share identities across scenes.
  const previews = [
    ['One pair', 'Ks Kh Qs Js 9s'], ['Full house', 'Ks Kh Kd 9s 9h'],
    ['Straight flush', '9s 10s Js Qs Ks']
  ];
  const rankPanels = `<div class="intro-rank-panels">${previews.map(([name, cards]) => `<div class="intro-rank-panel">${hand(cards)}<div class="intro-equation">${name}</div></div>`).join('')}</div>`;
  // Durations at 1×: 7 + 5 + 9 + 7 + 9 + 6 + 6 + 6 = 55 seconds.
  const scenes = [
    {at: 0, title: 'One deck. A finite universe.', html: `${deck}<div class="intro-equation">4 suits × 13 ranks = 52 cards</div>`, caption: 'Each suit contains A, 2–10, J, Q and K. Together the four rows show all 52 distinct cards.'},
    {at: 7000, title: 'Two cards belong to you.', html: `<span class="small-label">From the deck → your hand</span>${hand('As Qs')}<div class="intro-equation">2 private cards</div><p>Your hole cards are known to you. Other players’ cards remain hidden.</p>`, caption: 'The ace and queen of spades leave the deck and become your private hole cards.'},
    {at: 12000, title: 'The board unfolds.', html: `<div class="intro-board"><div>${hand('Js 10s Kd', '', 1400)}<small>Flop · three cards</small></div><div>${hand('9s', '', 4100)}<small>Turn · one card</small></div><div>${hand('2c', '', 6500)}<small>River · one card</small></div></div><div class="intro-hole">${hand('As Qs')}<small>Your two private cards</small></div><p>Once the river arrives: <strong>2 private + 5 community cards = 7 available.</strong></p>`, caption: 'Your two cards stay in view. The flop, turn and river add five shared community cards, one stage at a time.'},
    {at: 21000, title: 'Choose the best 5 of 7.', html: `${hand('As Qs Js 10s Kd 9s 2c', 'intro-seven')}<div class="intro-equation">C(7, 5) = 21 possible subsets</div><p id="intro-subset-label">✓ marks a candidate five-card hand.</p>`, caption: 'Compare five-card subsets. The same seven cards contain an ace-high straight, a king-high straight and a stronger ace-high flush.'},
    {at: 28000, title: 'Learn the language of patterns.', html: `${rankPanels}<p>One pair → full house → straight flush</p>`, caption: 'Three increasingly strong patterns. Explore the complete hand ladder below.'},
    {at: 37000, title: 'Count hands, not card orders.', html: `${hand('9s 10s Js Qs Ks')}<div class="intro-equation">C(52, 5) = 2,598,960</div><p>Five distinct cards form one unordered hand.<br>Rearranging them does not create a new hand.</p>`, caption: 'Keep the same five cards, change their order: still one hand. Combinations count all 2,598,960 unordered five-card hands.'},
    {at: 43000, title: 'Information changes the odds.', html: `<p>A new example: A♥ Q♥ in hand; 8♥ 3♥ K♣ on the flop.</p>${hand('2h 4h 5h 6h 7h 9h 10h Jh Kh', 'intro-outs', 450)}<div class="intro-equation">P(next heart) = <span class="intro-nowrap">9 / 47</span></div><p>9 heart outs among 47 unseen cards.<br>Completing a flush is not a guarantee of winning.</p>`, caption: 'Known cards leave the unseen deck. Nine heart outs among 47 unseen cards give the probability of a heart on the next card.'}
  ];
  scenes.forEach((scene, index) => {
    const node = document.createElement('div');
    node.className = 'intro-scene'; node.setAttribute('aria-hidden', 'true');
    node.innerHTML = `<span class="stage-label">${String(index + 1).padStart(2, '0')} · Cards to mathematics</span><h3>${scene.title}</h3>${scene.html}`;
    stage.append(node); scene.node = node;
  });
  scenes.push({at: 49000, node: summary, caption: 'Combinatorics → Probability → Expected Value → Simulation. Continue below to explore each step.'});
  const flights = document.createElement('div');
  flights.className = 'intro-flights'; flights.setAttribute('aria-hidden', 'true'); stage.append(flights);
  let elapsed = 0, anchorTime = 0, anchorElapsed = 0, rate = 1;
  let frame = null, running = false, active = -1, bridge = null;
  let autoplayPending = !motion.matches, visible = false;
  const ease = v => { const x = Math.max(0, Math.min(1, v)); return x * x * (3 - 2 * x); };
  const stamp = ms => `${Math.floor(ms / 60000)}:${String(Math.floor(ms / 1000) % 60).padStart(2, '0')}`;
  const textSelector = 'h3, .stage-label, .small-label, .intro-equation, p, .intro-suit-label, .intro-board > div > small, .intro-hole > small';
  function paintScene(index, local) {
    const scene = scenes[index];
    scene.node.querySelectorAll(textSelector).forEach(text => { text.style.opacity = '1'; });
    scene.node.querySelectorAll('[data-enter]').forEach((face, i) => {
      const reveal = index === 0 || index === 4 ? 1 : ease((local - Number(face.dataset.enter)) / 1000);
      face.style.opacity = String(reveal);
      face.style.transform = `translateY(${(1 - reveal) * 18}px)`;
      if (index === 0) {
        // Highlight one complete rank row at a time, leaving all 52 cards visible.
        const row = Math.min(3, Math.floor(local / 1600));
        face.classList.toggle('intro-deck-focus', Math.floor(i / 13) === row);
      }
      if (index === 3) {
        const step = local < 2400 ? 0 : local < 4500 ? 1 : 2;
        const keep = [[0, 1, 2, 3, 4], [1, 2, 3, 4, 5], [0, 1, 2, 3, 5]][step].includes(i);
        face.classList.toggle('intro-selected', keep);
        face.style.opacity = String(reveal * (keep ? 1 : .48));
      }
    });
    if (index === 3) $('intro-subset-label').textContent = local < 2400 ? '✓ Candidate 1: an ace-high straight.' : local < 4500 ? '✓ Candidate 2: a king-high straight.' : '✓ Best five: an ace-high spade flush. Leave K♦ and 2♣.';
    if (index === 4) {
      const current = Math.min(2, Math.floor(local / 3000));
      const blend = ease((local % 3000) / 650);
      scene.node.querySelectorAll('.intro-rank-panel').forEach((panel, i) => {
        panel.dataset.active = String(i === current);
        panel.setAttribute('aria-hidden', String(i !== current));
        panel.style.visibility = i === current || (i === current - 1 && blend < 1) ? 'visible' : 'hidden';
        panel.style.opacity = String(i === current ? (current ? blend : 1) : i === current - 1 ? 1 - blend : 0);
      });
    }
    if (index === 5) {
      // The five persistent card objects exchange places, then return. Geometry
      // follows the responsive layout, so this also works on wrapped mobile rows.
      const cards = [...scene.node.querySelectorAll('[data-card]')];
      const boxes = cards.map(face => baseRect(face));
      const amount = ease((local - 1500) / 1400) * (1 - ease((local - 4000) / 1400));
      cards.forEach((face, i) => { const to = boxes[cards.length - 1 - i], from = boxes[i]; face.style.transform = `translate(${(to.left - from.left) * amount}px, ${(to.top - from.top) * amount}px)`; });
    }
  }
  function baseRect(face) {
    const saved = face.style.transform; face.style.transform = 'none';
    const rect = face.getBoundingClientRect(); face.style.transform = saved; return rect;
  }
  function clearBridge() { flights.replaceChildren(); bridge = null; }
  function sceneCards(scene) {
    return [...scene.node.querySelectorAll('[data-card]')].filter(face => !face.closest('.intro-rank-panel') || face.closest('.intro-rank-panel').dataset.active === 'true');
  }
  function buildBridge(index) {
    clearBridge();
    const box = stage.getBoundingClientRect(), from = sceneCards(scenes[index - 1]);
    bridge = {index, cards: []};
    sceneCards(scenes[index]).forEach(target => {
      const source = from.find(face => face.dataset.card === target.dataset.card);
      if (!source) return;
      const a = baseRect(source), b = baseRect(target);
      const clone = target.cloneNode(true); clone.classList.remove('intro-selected');
      clone.style.cssText = `--card-width:${b.width}px;left:${b.left - box.left}px;top:${b.top - box.top}px;`;
      flights.append(clone); bridge.cards.push({source, target, clone, a, b});
    });
  }
  function render() {
    const index = scenes.findLastIndex(scene => elapsed >= scene.at);
    const local = elapsed - scenes[index].at;
    const blending = index > 0 && local < transition && !motion.matches;
    if (index !== active) {
      clearBridge(); active = index; caption.textContent = scenes[index].caption;
    }
    scenes.forEach((scene, i) => {
      scene.node.setAttribute('aria-hidden', String(i !== index));
      scene.node.classList.toggle('intro-transition-out', blending && i === index - 1);
      scene.node.style.opacity = i === index ? '1' : '0';
    });
    paintScene(index, local);
    if (blending) {
      paintScene(index - 1, scenes[index].at - scenes[index - 1].at - 1);
      const blend = ease(local / transition);
      scenes[index - 1].node.style.opacity = String(1 - blend);
      scenes[index].node.style.opacity = String(blend);
      // Clear the outgoing labels before introducing new ones; shared cards
      // continue travelling throughout this short, readable text handoff.
      scenes[index - 1].node.querySelectorAll(textSelector).forEach(text => { text.style.opacity = String(1 - ease(local / 450)); });
      scenes[index].node.querySelectorAll(textSelector).forEach(text => { text.style.opacity = String(ease((local - 450) / 650)); });
      if (!bridge || bridge.index !== index) buildBridge(index);
      bridge.cards.forEach(({source, target, clone, a, b}) => {
        source.style.opacity = '0'; target.style.opacity = '0';
        clone.style.transform = `translate(${(a.left - b.left) * (1 - blend)}px, ${(a.top - b.top) * (1 - blend)}px) scale(${a.width / b.width + (1 - a.width / b.width) * blend}, ${a.height / b.height + (1 - a.height / b.height) * blend})`;
      });
    } else if (bridge) clearBridge();
    position.value = String(elapsed / 1000);
    position.setAttribute('aria-valuetext', `${(elapsed / 1000).toFixed(1)} seconds of 55; ${index === 7 ? 'mathematical summary' : scenes[index].title}`);
    time.textContent = `${stamp(elapsed)} / 0:55`;
    film.dataset.state = running ? 'playing' : elapsed === duration ? 'complete' : elapsed ? 'paused' : 'ready';
  }
  function sample(now) {
    // The first frame timestamp can precede the input event that anchored it.
    elapsed = Math.min(duration, anchorElapsed + Math.max(0, now - anchorTime) * rate);
  }
  function stop() {
    if (running) sample(performance.now());
    running = false;
    if (frame !== null) cancelAnimationFrame(frame);
    frame = null;
  }
  function tick(now) {
    frame = null;
    if (!running) return;
    sample(now);
    if (elapsed === duration) {
      running = false; pause.disabled = true; pause.textContent = 'Pause';
      mode.textContent = 'Introduction complete. Replay, or continue into the guide below.';
    }
    render();
    if (running) frame = requestAnimationFrame(tick);
  }
  function start() {
    if (motion.matches || running || elapsed >= duration) return;
    autoplayPending = false; anchorTime = performance.now(); anchorElapsed = elapsed; running = true;
    pause.disabled = false; pause.textContent = 'Pause'; play.textContent = 'Replay';
    mode.textContent = 'Playing · 55 seconds at 1×. Speed changes take effect immediately.';
    render(); frame = requestAnimationFrame(tick);
  }
  play.addEventListener('click', () => { autoplayPending = false; stop(); elapsed = 0; start(); });
  pause.addEventListener('click', () => {
    autoplayPending = false;
    if (running) { stop(); render(); pause.textContent = 'Resume'; mode.textContent = 'Paused. Resume from here, or replay from the beginning.'; }
    else start();
  });
  position.addEventListener('input', () => {
    autoplayPending = false; stop(); elapsed = Number(position.value) * 1000; render();
    pause.disabled = elapsed === duration; pause.textContent = 'Resume'; play.textContent = 'Replay';
    mode.textContent = 'Timeline paused at your chosen position.';
  });
  speed.addEventListener('input', () => {
    const now = performance.now();
    if (running) sample(now); // Preserve the exact virtual position under the OLD rate.
    rate = Number(speed.value); anchorElapsed = elapsed; anchorTime = now;
    $('intro-rate').textContent = `${rate.toFixed(2).replace(/0$/, '')}×`;
    speed.setAttribute('aria-valuetext', `${rate} times normal speed`);
    render(); // Keep the existing frame chain; never schedule a second one.
  });
  $('intro-skip').addEventListener('click', () => {
    autoplayPending = false; stop(); elapsed = duration; render(); pause.disabled = true; pause.textContent = 'Pause';
    mode.textContent = 'Introduction skipped. The complete guide starts below.';
    const heading = $('game-heading'); heading.setAttribute('tabindex', '-1'); heading.focus({preventScroll: true});
    heading.scrollIntoView({behavior: 'instant', block: 'start'});
  });
  function suspend() {
    if (running) { stop(); render(); pause.textContent = 'Resume'; mode.textContent = 'Paused while out of view. Resume whenever you are ready.'; }
  }
  function tryAutoplay() { if (autoplayPending && visible && !document.hidden && !motion.matches) start(); }
  document.addEventListener('visibilitychange', () => { if (document.hidden) suspend(); else tryAutoplay(); });
  window.addEventListener('pagehide', suspend);
  new IntersectionObserver(entries => {
    const entry = entries[entries.length - 1];
    visible = entry.intersectionRatio >= .25;
    if (!entry.isIntersecting) suspend();
    else tryAutoplay();
  }, {threshold: [0, .25]}).observe(stage);
  new ResizeObserver(() => { clearBridge(); render(); }).observe(stage);
  function setMotion() {
    stop(); elapsed = motion.matches ? duration : 0;
    if (motion.matches) autoplayPending = false;
    play.disabled = motion.matches; pause.disabled = true; pause.textContent = 'Pause';
    position.disabled = motion.matches; speed.disabled = motion.matches;
    render();
    mode.textContent = motion.matches ? 'Reduced motion is enabled: a static summary replaces the animation.' : autoplayPending ? 'Plays once when in view. Pause or adjust the speed at any time.' : 'Use Play or Replay to start. Pause or adjust the speed at any time.';
  }
  motion.addEventListener('change', () => { setMotion(); tryAutoplay(); });
  $('intro-controls').hidden = false;
  setMotion();
})();
