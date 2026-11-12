/* A 32-second, dependency-free illustration. No analytical data are read. */
(() => {
  'use strict';
  const root = document.getElementById('trade-intro');
  if (!root) return;
  root.classList.add('ta-enhanced');
  const svg = root.querySelector('.ta-stage');
  const find = id => root.querySelector(`#${id}`);
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  const narrow = matchMedia('(max-width: 650px)');
  const duration = 32;
  const layers = Object.fromEntries(['canada', 'economy', 'logistics', 'world', 'data', 'prediction']
    .map(name => [name, find(`ta-${name}`)]));
  const headline = root.querySelector('.ta-headline');
  const heading = root.querySelector('.ta-heading');
  const kicker = root.querySelector('.ta-kicker');
  const subline = root.querySelector('.ta-subline');
  const pauseButton = root.querySelector('[data-ta-pause]');
  const replayButton = root.querySelector('[data-ta-replay]');
  const controls = root.querySelector('.ta-controls');
  const staticNote = root.querySelector('.ta-static-note');
  const progress = root.querySelector('.ta-progress-fill');
  const partners = [...root.querySelectorAll('.ta-partners span')];
  const routes = [...root.querySelectorAll('[data-ta-route]')].map(path => ({
    path, dot: find(`${path.id}-dot`), start: Number(path.dataset.start),
    origin: path.dataset.origin.split(',').map(Number),
    end: path.dataset.end.split(',').map(Number),
    control: path.dataset.control.split(',').map(Number)
  }));
  const tokens = [...root.querySelectorAll('[data-ta-token]')];
  const bars = [...root.querySelectorAll('[data-ta-bar]')];
  const thoughts = [...root.querySelectorAll('[data-ta-thought]')];
  const canadaFocus = find('ta-canada-focus').dataset.focus.split(',').map(Number);
  let elapsed = 0, last = 0, lastPaint = 0, frame = 0;
  let visible = true, userPaused = false, phase = '';
  const clamp = x => Math.max(0, Math.min(1, x));
  const ease = x => { x = clamp(x); return x * x * (3 - 2 * x); };
  const ramp = (t, start, end) => ease((t - start) / (end - start));
  const windowOpacity = (t, start, end, fade = .6) =>
    ramp(t, start - fade, start + fade) * (1 - ramp(t, end - fade, end + fade));
  const opacity = (element, value) => { element.style.opacity = value.toFixed(3); };
  const translate = (element, x, y, scale = 1) =>
    element.setAttribute('transform', `translate(${x.toFixed(2)} ${y.toFixed(2)}) scale(${scale})`);
  function setCopy(name, title, subtitle, label) {
    if (phase === `${name}:${title}`) return;
    phase = `${name}:${title}`;
    root.dataset.phase = name;
    headline.textContent = title;
    subline.textContent = subtitle;
    kicker.textContent = label;
  }
  function layout() {
    svg.setAttribute('viewBox', narrow.matches ? (reduced.matches ? '0 0 560 460' : '0 0 560 630') : '0 0 1120 490');
    render(reduced.matches ? 30 : elapsed % duration);
  }
  function render(t) {
    const mobile = narrow.matches;
    const closing = 1 - ramp(t, 31.3, 32);
    const opening = ramp(t, 0, .5);
    const copyFade = Math.min(...[2.7, 5, 10, 15, 21, 23.5, 26, 29].map(change => ease(Math.abs(t - change) / .3)));
    opacity(heading, Math.min(opening, closing, copyFade));
    root.dataset.animationTime = t.toFixed(2);
    opacity(layers.canada, (1 - ramp(t, 4.4, 5.5)) * opening);
    opacity(layers.economy, windowOpacity(t, 5, 14.7));
    opacity(layers.logistics, windowOpacity(t, 10, 15.6));
    opacity(layers.world, Math.max(windowOpacity(t, 15, 22.8), ramp(t, 28.9, 29.6) * closing));
    opacity(layers.data, windowOpacity(t, 21.4, 26.3));
    opacity(layers.prediction, windowOpacity(t, 26, 29.2));
    const focus = 1 + .07 * ramp(t, .7, 4.4);
    const [focusX, focusY] = canadaFocus;
    find('ta-canada-focus').setAttribute('transform', `translate(${focusX * (1 - focus)} ${focusY * (1 - focus)}) scale(${focus})`);
    const pulse = 1 + .045 * Math.sin(t * 2.2);
    find('ta-alberta-glow').setAttribute('transform', `translate(${focusX * (1 - pulse)} ${focusY * (1 - pulse)}) scale(${pulse})`);
    const retreat = ramp(t, 9.5, 11.2);
    find('ta-production-layout').setAttribute('transform', mobile
      ? `translate(${60 * retreat} ${-50 * retreat}) scale(${1 - .24 * retreat})`
      : `translate(${125 * retreat} ${-52 * retreat}) scale(${1 - .24 * retreat})`);
    translate(find('ta-grain'), 74 + 15 * Math.sin(t * 1.4), 130);
    translate(find('ta-conveyor-box'), 65 + ((t * 15) % 58), 133);
    translate(find('ta-worker-motion'), 160 + Math.sin(t * 1.2) * 4, 151);
    translate(find('ta-mine-truck'), 83 + Math.sin(t * .9) * 14, 107);
    const travel = ramp(t, 10.1, 14.8);
    const railY = mobile ? 556 : 425;
    find('ta-rail-line').setAttribute('d', `M0 ${railY} H${mobile ? 560 : 1120}`);
    find('ta-rail-sleepers').setAttribute('y', railY - 2);
    translate(find('ta-train'), (mobile ? -175 : -250) + (mobile ? 590 : 1090) * travel, railY - 44, mobile ? .9 : 1);
    translate(find('ta-road-truck'), (mobile ? 10 : 210) + (mobile ? 275 : 520) * ramp(t, 10.8, 14.6), mobile ? 507 : 365);
    translate(find('ta-road-semi'), (mobile ? -190 : -100) + (mobile ? 315 : 690) * ramp(t, 10.15, 14.9), mobile ? 510 : 366, mobile ? .8 : 1);
    translate(find('ta-road-container'), -140 + 550 * ramp(t, 11.1, 15.2), 366);
    find('ta-road-line').setAttribute('d', mobile ? 'M0 522 H410 Q500 522 510 455' : 'M70 380 H790 Q935 380 985 330');
    for (const [i, route] of routes.entries()) {
      const drawn = ramp(t, route.start, route.start + 1.15);
      const settled = t > 27.9 ? 1 : drawn;
      route.path.style.strokeDashoffset = (1 - settled).toFixed(3);
      const u = ((Math.max(0, t - route.start) * .38) % 1);
      const [cx, cy] = route.control, [ex, ey] = route.end, [sx, sy] = route.origin;
      const active = t > route.start && t < 21.8;
      const x = (1-u)**2 * sx + 2*(1-u)*u*cx + u*u*ex;
      const y = (1-u)**2 * sy + 2*(1-u)*u*cy + u*u*ey;
      translate(route.dot, active ? x : sx, active ? y : sy);
      opacity(route.dot, active ? drawn : 0);
      partners[i].classList.toggle('is-connected', settled > .85);
    }
    const voyage = ramp(t, 15.4, 20.7);
    opacity(find('ta-ship'), windowOpacity(t, 15.4, 21.6));
    translate(find('ta-ship'), mobile ? 350 - 130 * voyage : 860 - 270 * voyage,
      (mobile ? 440 : 351) + Math.sin(t * 1.7) * 2, mobile ? .88 : 1);
    const dataLayout = mobile ? [[50,155],[310,155],[50,220],[310,220],[50,285],[310,285],[180,350]]
      : [[135,115],[405,91],[730,106],[913,161],[151,301],[709,333],[912,298]];
    // Reserve a separate right-hand (desktop) / lower (mobile) analyst space.
    const thinking = ramp(t, 22.8, 23.7);
    layers.data.setAttribute('transform', mobile ? 'translate(0 0)' : `translate(${20 * thinking} ${20 * thinking}) scale(${1 - .3 * thinking})`);
    layers.prediction.setAttribute('transform', mobile ? 'translate(0 0)' : 'translate(20 20) scale(.7)');
    tokens.forEach((token, i) => {
      const [x,y] = dataLayout[i];
      translate(token, x, y - (mobile ? 75 * thinking : 0) + 3 * Math.sin(t * .65 + i), mobile ? 1.15 : 1);
      opacity(token, ramp(t, 21.15 + i * .16, 21.9 + i * .16));
    });
    translate(find('ta-observation-row'), mobile ? 35 : 275, mobile ? 442 - 92 * thinking : 390, mobile ? .87 : 1);
    translate(find('ta-data-nodes'), mobile ? 280 : 560, mobile ? 105 - 69 * thinking : 230, mobile ? .6 : 1);
    translate(find('ta-observed-cluster'), mobile ? 38 : 235, mobile ? 120 : 160);
    translate(find('ta-model-node'), mobile ? 285 : 560, mobile ? 190 : 232);
    translate(find('ta-estimated-chart'), mobile ? 375 : 775, mobile ? 115 : 146);
    find('ta-model-links').setAttribute('d', mobile ? 'M183 190H230 M333 190H370' : 'M386 232 H510 M612 232 H752');
    bars.forEach((bar, i) => bar.setAttribute('width', (48 + i * 32 + ramp(t, 26.7, 28.6) * [20,-13,27][i]).toFixed(2)));
    // The place marker belongs to Canada, not to the cross-scene timeline.
    // Containment and an explicit phase gate prevent leakage on every loop.
    const seedActive = t < 5;
    translate(find('ta-story-seed'), focusX + (seedActive ? 3 * Math.sin(t * .8) : 0), focusY);
    opacity(find('ta-story-seed'), seedActive ? .8 * opening * (1 - ramp(t, 4.4, 5)) : 0);
    opacity(find('ta-analyst'), windowOpacity(t, 23.7, 28.7, .4));
    thoughts.forEach((word, i) => opacity(word, ramp(t, 23.6 + i * .12, 24.1 + i * .12)));
    opacity(find('ta-analyst-screen'), ramp(t, 23.3, 24));
    opacity(find('ta-analyst-question'), windowOpacity(t, 24.2, 27.8, .4));
    opacity(find('ta-analyst-estimate'), ramp(t, 27.2, 28));
    find('ta-analyst-screen').querySelector('text').style.opacity = (1 - ramp(t, 27.2, 28)).toFixed(3);
    progress.style.transform = `scaleX(${t / duration})`;
    if (t < 5) setCopy('canada', t < 2.7 ? 'Alberta within Canada' : 'Goods begin here.', 'A provincial economy, connected to the world.', '01 / 06 · Place');
    else if (t < 10) setCopy('production', 'An economy that makes things.', 'Agriculture · Energy & mining · Manufacturing · Business', '02 / 06 · Production');
    else if (t < 15) setCopy('logistics', 'Goods move.', 'Rail · Road · Port', '03 / 06 · Inland logistics');
    else if (t < 21) setCopy('global', 'Real goods. Real destinations.', mobile ? 'Alberta, Canada → global partners' : 'Physical merchandise connects Alberta to global trading partners.', '04 / 06 · Global trade');
    else if (t < 26) setCopy('data', t < 23.5 ? 'We observe 2024.' : 'What might 2025 look like?', 'Trade, economic size, people, geography and product markets.', '05 / 06 · From flows to information');
    else if (t < 29) setCopy('prediction', 'Six mathematical approaches.', '2024 observed information → 2025 estimated trade', '06 / 06 · The prediction question');
    else setCopy('final', 'From trade flows to next-year prediction', 'Six models. Different assumptions.', 'Alberta → World → Next year');
  }
  function staticFrame() {
    render(30);
    opacity(layers.world, 1);
    routes.forEach(route => { route.path.style.strokeDashoffset = '0'; opacity(route.dot, 0); });
    progress.style.transform = 'scaleX(1)';
    controls.hidden = true;
    staticNote.hidden = false;
  }
  function tick(now) {
    if (!last) last = now;
    elapsed += Math.min((now - last) / 1000, .12);
    last = now;
    if (now - lastPaint > 30) { render(elapsed % duration); lastPaint = now; }
    frame = requestAnimationFrame(tick);
  }
  function reconcile() {
    const run = !reduced.matches && !userPaused && visible && !document.hidden;
    if (run && !frame) { last = 0; frame = requestAnimationFrame(tick); }
    else if (!run && frame) { cancelAnimationFrame(frame); frame = 0; last = 0; }
    pauseButton.textContent = userPaused ? 'Play' : 'Pause';
    pauseButton.setAttribute('aria-label', `${userPaused ? 'Play' : 'Pause'} animated trade introduction`);
  }
  pauseButton.addEventListener('click', () => { userPaused = !userPaused; reconcile(); });
  replayButton.addEventListener('click', () => { elapsed = 0; last = 0; userPaused = false; render(0); reconcile(); });
  document.addEventListener('visibilitychange', reconcile);
  if ('IntersectionObserver' in window) new IntersectionObserver(entries => {
    visible = entries[0].isIntersecting;
    reconcile();
  }, { threshold: .08 }).observe(root);
  narrow.addEventListener('change', layout);
  reduced.addEventListener('change', () => {
    layout();
    if (reduced.matches) staticFrame();
    else { controls.hidden = false; staticNote.hidden = true; elapsed = 0; render(0); }
    reconcile();
  });
  layout();
  if (reduced.matches) staticFrame();
  else { controls.hidden = false; staticNote.hidden = true; render(0); }
  reconcile();
})();
