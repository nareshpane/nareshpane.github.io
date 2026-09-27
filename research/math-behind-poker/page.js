/* Progressive enhancement for one fixed, auditable five-card comparison. */
'use strict';
(() => {
  const lab = document.querySelector('.comparison-lab');
  if (!lab) return;
  const next = document.getElementById('compare-next');
  const reset = document.getElementById('compare-reset');
  const status = document.getElementById('compare-status');
  const hands = [...lab.querySelectorAll('.lab-hand')];
  const trace = [...lab.querySelectorAll('[data-step]')];
  const steps = [
    'First card: ace equals ace. Both flushes are ace-high; compare the next card.',
    'Second card: jack equals jack. The first two ranks match; compare the third card.',
    'Third card: 9 beats 8. Hand A wins. Stop here: A’s remaining 6 and 3, and B’s remaining 7 and 4, cannot change the result.'
  ];
  const labels = ['Compare highest cards', 'Compare second cards', 'Compare third cards', 'Comparison complete'];
  let position = -1;

  function clear() {
    position = -1;
    hands.forEach(hand => hand.querySelectorAll('.playing-card').forEach(card => {
      card.classList.remove('compared', 'deciding');
    }));
    trace.forEach(item => { item.hidden = true; item.classList.remove('current'); });
    status.textContent = 'Start with the highest card in each flush. Compare ranks from left to right until the first difference.';
    next.disabled = false;
    next.textContent = labels[0];
    reset.disabled = true;
  }

  next.addEventListener('click', () => {
    if (position >= steps.length - 1) return;
    position += 1;
    hands.forEach(hand => {
      const cards = hand.querySelectorAll('.playing-card');
      cards.forEach(card => card.classList.remove('deciding'));
      cards[position].classList.add('compared');
      if (position === 2) cards[position].classList.add('deciding');
    });
    trace.forEach((item, index) => {
      item.hidden = index > position;
      item.classList.toggle('current', index === position);
    });
    status.textContent = steps[position];
    next.textContent = labels[position + 1];
    next.disabled = position === steps.length - 1;
    reset.disabled = false;
  });
  reset.addEventListener('click', () => {
    clear();
    next.focus();
  });
  lab.querySelector('.lab-controls').hidden = false;
  clear();
})();
