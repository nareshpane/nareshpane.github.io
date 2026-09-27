/* Pure, shared mathematics. Browser and Node; no DOM, network, or timers. */
'use strict';
(function (root) {
  const suitNames = ['s', 'h', 'd', 'c'];
  const ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A'];
  const deck = ranks.flatMap(rank => suitNames.map(suit => rank + suit));
  const categories = ['High card', 'One pair', 'Two pair', 'Three of a kind', 'Straight', 'Flush', 'Full house', 'Four of a kind', 'Straight flush'];
  function compare(a, b) {
    for (let i = 0; i < Math.max(a.length, b.length); i++) {
      const difference = (a[i] || 0) - (b[i] || 0);
      if (difference) return difference;
    }
    return 0;
  }
  function rankFive(cards) {
    const values = cards.map(card => ranks.indexOf(card.slice(0, -1)) + 2).sort((a, b) => b - a);
    const counts = new Map();
    values.forEach(value => counts.set(value, (counts.get(value) || 0) + 1));
    const groups = [...counts].map(([rank, count]) => [count, rank]).sort((a, b) => b[0] - a[0] || b[1] - a[1]);
    const flush = cards.every(card => card.slice(-1) === cards[0].slice(-1));
    const wheel = values.join(',') === '14,5,4,3,2';
    const straight = wheel ? 5 : counts.size === 5 && values[0] - values[4] === 4 ? values[0] : 0;
    if (straight && flush) return [8, straight];
    if (groups[0][0] === 4) return [7, groups[0][1], groups[1][1]];
    if (groups[0][0] === 3 && groups[1][0] === 2) return [6, groups[0][1], groups[1][1]];
    if (flush) return [5, ...values];
    if (straight) return [4, straight];
    if (groups[0][0] === 3) return [3, ...groups.map(group => group[1])];
    if (groups[0][0] === 2 && groups[1][0] === 2) return [2, ...groups.map(group => group[1])];
    if (groups[0][0] === 2) return [1, ...groups.map(group => group[1])];
    return [0, ...values];
  }
  function bestOfSeven(cards) {
    let best = null;
    let selected = [];
    // Omitting each pair visits exactly C(7,2) = C(7,5) = 21 subsets.
    for (let a = 0; a < 6; a++) for (let b = a + 1; b < 7; b++) {
      const subset = cards.filter((_, index) => index !== a && index !== b);
      const score = rankFive(subset);
      if (!best || compare(score, best) > 0) { best = score; selected = subset; }
    }
    return {rank: best, cards: selected, category: categories[best[0]]};
  }
  function gcd(a, b) { return b ? gcd(b, a % b) : a; }
  function fraction(numerator, denominator) {
    const divisor = gcd(numerator, denominator);
    return `${numerator / divisor}/${denominator / divisor}`;
  }
  function drawProbability(outs, unseen, cardsToCome) {
    const numerator = cardsToCome === 1 ? outs : unseen * (unseen - 1) - (unseen - outs) * (unseen - outs - 1);
    const denominator = cardsToCome === 1 ? unseen : unseen * (unseen - 1);
    return {fraction: fraction(numerator, denominator), value: numerator / denominator};
  }
  function potModel(pot, bet, probability) {
    const call = bet;
    const reward = pot + bet;
    const finalPot = reward + call;
    return {call, reward, finalPot, breakEven: finalPot ? call / finalPot : null,
      rewardRisk: call ? reward / call : null, ev: probability * reward - (1 - probability) * call};
  }
  function standardError(probability, n) { return Math.sqrt(probability * (1 - probability) / n); }
  function generator(seed) {
    let state = seed >>> 0;
    // Mulberry32 with explicit unsigned integer arithmetic, independently ported to Python.
    return function () {
      state = (state + 0x6D2B79F5) >>> 0;
      let word = Math.imul(state ^ (state >>> 15), state | 1);
      word ^= word + Math.imul(word ^ (word >>> 7), word | 61);
      return (word ^ (word >>> 14)) >>> 0;
    };
  }
  function uniformIndex(random, size) {
    const limit = Math.floor(4294967296 / size) * size;
    let word;
    do { word = random(); } while (word >= limit);
    return word % size;
  }
  function makeTrialSource(scenario, seed) {
    const known = scenario.hole.concat(scenario.flop);
    const remaining = deck.filter(card => !known.includes(card));
    const outs = new Set(scenario.outs);
    const random = generator(seed);
    return function () {
      const first = uniformIndex(random, remaining.length);
      let second = uniformIndex(random, remaining.length - 1);
      if (second >= first) second++;
      const runout = [remaining[first], remaining[second]];
      const seven = known.concat(runout);
      return {hit: runout.some(card => outs.has(card)), runout, seven, best: bestOfSeven(seven)};
    };
  }
  const api = {deck, categories, compare, rankFive, bestOfSeven, fraction, drawProbability, potModel, standardError, generator, makeTrialSource};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.PokerMath = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
