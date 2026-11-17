// A deferred classic script also runs when this standalone page opens via file://.
// Keep section tracking independent of the economic model and animation clock.
(() => {
const nav = document.querySelector('nav.nav');
if (nav) {
  const links = [...nav.querySelectorAll('a[href^="#"]')].filter(link => document.getElementById(link.hash.slice(1)));
  const sections = links.map(link => document.getElementById(link.hash.slice(1)));
  let tops = [], offset = 76, active, frame, measure = true;

  function activate(link) {
    if (active === link) return;
    active?.removeAttribute('aria-current');
    active = link;
    active.setAttribute('aria-current', 'location');
  }

  function update() {
    frame = undefined;
    if (measure) {
      offset = nav.getBoundingClientRect().height + 16;
      document.documentElement.style.setProperty('--research-nav-offset', `${offset}px`);
      tops = sections.map(section => section.getBoundingClientRect().top + window.scrollY);
      measure = false;
    }
    // Cached positions avoid section layout reads during ordinary scrolling.
    const line = window.scrollY + offset + 1;
    let index = 0;
    for (let i = 1; i < tops.length && tops[i] <= line; i++) index = i;
    activate(links[index]);
  }

  function schedule(remeasure = false) {
    measure ||= remeasure;
    if (!frame) frame = requestAnimationFrame(update);
  }

  window.addEventListener('scroll', () => schedule(), {passive:true});
  window.addEventListener('resize', () => schedule(true));
  window.addEventListener('load', () => schedule(true));
  window.addEventListener('hashchange', () => schedule(true));
  window.addEventListener('pageshow', () => schedule(true));
  // Charts, disclosures and responsive text can change section heights.
  const observer = new ResizeObserver(() => schedule(true));
  document.querySelectorAll('main, section.card, nav.nav').forEach(el => observer.observe(el));
  document.fonts?.ready.then(() => schedule(true));
  nav.addEventListener('click', event => {
    const link = event.target.closest('a');
    if (links.includes(link) && !event.ctrlKey && !event.metaKey && !event.shiftKey && !event.altKey) activate(link);
  });
  schedule(true);
}

})();
