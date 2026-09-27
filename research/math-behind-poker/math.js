/* Deterministic Phase 2 enhancements; explanations and fallback examples are static. */
'use strict';
(() => {
  const source = document.getElementById('poker-math-data');
  if (!source) return;
  const data = JSON.parse(source.textContent);
  const suits = {s: ['♠', 'spades'], h: ['♥', 'hearts'], d: ['♦', 'diamonds'], c: ['♣', 'clubs']};
  const cardName = code => {
    const rank = code.slice(0, -1);
    return `${({A: 'Ace', K: 'King', Q: 'Queen', J: 'Jack'})[rank] || rank} of ${suits[code.slice(-1)][1]}`;
  };
  const glyph = code => code.slice(0, -1) + suits[code.slice(-1)][0];
  function card(code) {
    const rank = code.slice(0, -1);
    const suit = code.slice(-1);
    const face = document.createElement('span');
    face.className = `playing-card ${'hd'.includes(suit) ? 'red' : 'black'}`;
    face.setAttribute('role', 'img');
    face.setAttribute('aria-label', cardName(code));
    face.dataset.card = code;
    // Only the verified local finite deck supplies rank and suit values.
    face.innerHTML = `<span class="corner" aria-hidden="true">${rank}<small>${suits[suit][0]}</small></span><span class="pip" aria-hidden="true">${suits[suit][0]}</span><span class="corner bottom" aria-hidden="true">${rank}<small>${suits[suit][0]}</small></span>`;
    return face;
  }
  function hand(codes, label) {
    const group = document.createElement('div');
    group.className = 'hand math-hand';
    group.setAttribute('role', 'group');
    group.setAttribute('aria-label', label);
    group.dataset.hand = codes.join(' ');
    codes.forEach(code => group.append(card(code)));
    return group;
  }

  const reorder = document.getElementById('reorder-cards');
  reorder.closest('.math-controls').hidden = false;
  reorder.addEventListener('click', () => {
    const row = document.querySelector('.order-rearranged');
    const faces = [...row.children].reverse();
    row.replaceChildren(...faces);
    row.dataset.hand = faces.map(face => face.dataset.card).join(' ');
    row.setAttribute('aria-label', 'The same royal flush, rearranged');
    document.getElementById('order-status').textContent = `${faces.map(face => glyph(face.dataset.card)).join(' ')}: the order changed, but the set is still one royal flush. Each five-card set has 120 orders.`;
  });

  const subsetChoice = document.getElementById('subset-choice');
  const previous = document.getElementById('subset-prev');
  const next = document.getElementById('subset-next');
  const subsetSource = [...document.querySelectorAll('#subset-source .choice-card')];
  let subsetIndex = 0;
  function showSubset(index) {
    subsetIndex = index;
    const subset = data.subsets[index];
    subsetChoice.value = String(index);
    subsetSource.forEach((item, position) => {
      const included = subset.indices.includes(position);
      const face = item.querySelector('.playing-card');
      face.classList.toggle('chosen', included);
      face.classList.toggle('not-chosen', !included);
      item.querySelector('.subset-mark').textContent = included ? '✓ keep' : '− leave';
    });
    document.getElementById('subset-result').replaceChildren(hand(subset.cards, 'Current selected five-card subset'));
    const isBest = JSON.stringify(subset.rank) === JSON.stringify(data.subsets[data.best_subset_index].rank);
    document.getElementById('subset-status').textContent = `Subset ${index + 1} of 21: ${subset.cards.map(glyph).join(' ')}. ${subset.category}. Rank (${subset.rank[0]}; ${subset.rank.slice(1).join(', ')}). ${isBest ? 'This is the best five-card value available: the ace-high flush.' : 'The ace-high spade flush is stronger. Choose “Show best hand” to locate it.'}`;
    previous.disabled = index === 0;
    next.disabled = index === data.subsets.length - 1;
  }
  document.getElementById('subset-controls').hidden = false;
  subsetChoice.addEventListener('change', () => showSubset(Number(subsetChoice.value)));
  previous.addEventListener('click', () => showSubset(subsetIndex - 1));
  next.addEventListener('click', () => showSubset(subsetIndex + 1));
  document.getElementById('subset-best').addEventListener('click', () => showSubset(data.best_subset_index));

  const drawChoice = document.getElementById('draw-choice');
  const cells = [...document.querySelectorAll('#outs-deck .deck-cell')];
  function showDraw(id) {
    const scenario = data.scenarios.find(s => s.id === id);
    if (!scenario) return;
    const known = new Set([...scenario.hole, ...scenario.flop]);
    const outs = new Set(scenario.outs);
    document.getElementById('draw-hole').replaceChildren(hand(scenario.hole, 'Your two hole cards'));
    document.getElementById('draw-flop').replaceChildren(hand(scenario.flop, 'The three flop cards'));
    document.getElementById('draw-target').textContent = scenario.target;
    document.getElementById('draw-caution').textContent = scenario.caution;
    cells.forEach(cell => {
      const code = cell.dataset.deckCard;
      const state = known.has(code) ? 'known' : outs.has(code) ? 'out' : 'other';
      cell.classList.remove('known', 'out', 'other');
      cell.classList.add(state);
      cell.querySelector('.deck-badge').textContent = {known: '×', out: '+', other: '·'}[state];
      cell.setAttribute('aria-label', {known: 'known, removed', out: 'out, unseen useful card', other: 'unseen, not an out'}[state]);
    });
    document.getElementById('draw-status').textContent = `5 known · 47 unseen · ${scenario.out_count} outs. Next card: ${scenario.out_count}/47 ≈ ${scenario.turn.percent.toFixed(3)}%. At least one of these outs by the river: ${scenario.by_river.fraction} ≈ ${scenario.by_river.percent.toFixed(3)}%. This is a target-hit probability, not a winning probability.`;
  }
  document.getElementById('draw-controls').hidden = false;
  drawChoice.addEventListener('change', () => showDraw(drawChoice.value));
})();
