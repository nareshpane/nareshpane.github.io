/* Cross-check the browser engine against independent Python reference results. */
'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const engine = require('./poker-engine.js');
const data = JSON.parse(fs.readFileSync(path.join(__dirname, 'verified_lab.json'), 'utf8'));
for (const scenario of data.scenarios) {
  const known = scenario.hole.concat(scenario.flop);
  const remaining = engine.deck.filter(card => !known.includes(card));
  const hash = crypto.createHash('sha256');
  const counts = Array(9).fill(0);
  for (let a = 0; a < 46; a++) for (let b = a + 1; b < 47; b++) {
    const rank = engine.bestOfSeven(known.concat(remaining[a], remaining[b])).rank;
    counts[rank[0]]++;
    hash.update(`${a},${b}:${rank.join(',')}\n`);
  }
  assert.deepEqual(counts, scenario.exact_category_counts);
  assert.equal(hash.digest('hex'), scenario.exact_rank_digest);
  for (const seed of data.seeds) {
    const trial = engine.makeTrialSource(scenario, seed);
    let hits = 0;
    const categories = Array(9).fill(0);
    for (let n = 1; n <= 10000; n++) {
      const outcome = trial();
      assert.equal(new Set(outcome.seven).size, 7);
      hits += Number(outcome.hit);
      categories[outcome.best.rank[0]]++;
      const expected = scenario.replays[String(seed)].find(point => point.n === n);
      if (expected) {
        assert.equal(hits, expected.hits);
        assert.deepEqual(categories, expected.categories);
        assert.deepEqual(outcome.runout, expected.runout);
        assert.deepEqual(outcome.best.rank, expected.rank);
        assert(Math.abs(engine.standardError(hits/n,n)-expected.se) < 1e-14);
      }
    }
  }
  for (const [unseen, cards, reference] of [[47,1,scenario.next_flop],[47,2,scenario.by_river_flop],[46,1,scenario.next_turn]]) {
    const p = engine.drawProbability(scenario.out_count,unseen,cards);
    assert.equal(p.fraction, reference.fraction);
    assert(Math.abs(p.value-reference.decimal)<1e-14);
  }
  console.log('Passed exact ranks, seeded checkpoints, and draw fractions:',scenario.id);
}
assert.deepEqual(engine.rankFive(['As','2d','3c','4h','5s']),[4,5]);
assert.deepEqual(engine.rankFive(['Qs','Kd','Ac','2h','3s']),[0,14,13,12,3,2]);
assert.deepEqual(engine.rankFive(['10h','Jh','Qh','Kh','Ah']),[8,14]);
assert.equal(engine.potModel(100,25,0.2).ev,5);
assert.equal(engine.potModel(100,25,0.1).ev,-10);
assert(Math.abs(engine.potModel(100,25,1/6).ev)<1e-12);
assert.equal(engine.potModel(0,0,0.5).breakEven,null);
assert.equal(engine.potModel(100,0,0.5).ev,50);
for (const point of data.uncertainty) assert(Math.abs(point.se-engine.standardError(378/1081,point.n))<1e-14);
console.log('All 5,405 complete-board ranks, 150,000 simulated trials, and 150 checkpoints match Python.');
