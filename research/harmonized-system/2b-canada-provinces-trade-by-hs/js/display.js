/* Shared display-only formatting; values and ranks retain source precision. */
window.TradeDisplay = Object.freeze({
  percent: n => n === null || n === undefined ? '—' : n.toFixed(1) + '%',
  shade(rank,count) {
    const first = [[174,64,27],[230,126,83],[242,173,136]];
    if (rank <= 3) return 'rgb(' + first[rank-1].join(',') + ')';
    const u = count <= 4 ? 0 : (rank-4)/(count-4);
    const start=[246,192,157], end=[250,216,180];
    return 'rgb(' + start.map((v,i)=>Math.round(v+(end[i]-v)*u)).join(',') + ')';
  }
});
