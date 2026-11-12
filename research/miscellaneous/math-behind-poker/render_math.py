#!/usr/bin/env python3
"""Regenerate only the marked Phase 2 HTML from verified_math.json.

Optional authoring tool, not a hosting requirement. Run verify_math.py first,
then this script. Cards, the logarithmic SVG, and numerical tables are emitted
as static HTML so all explanations survive without JavaScript.
"""
from pathlib import Path
from html import escape
from math import log10
import json
import re

HERE = Path(__file__).resolve().parent
PAGE = HERE.parent / 'math-behind-poker.html'
DATA = json.loads((HERE / 'verified_math.json').read_text())
SUITS = {'s': ('♠', 'spades'), 'h': ('♥', 'hearts'), 'd': ('♦', 'diamonds'), 'c': ('♣', 'clubs')}
RANKS = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']


def card(code, cls=''):
    r, s = code[:-1], code[-1]
    glyph, suit = SUITS[s]
    rank = {'A': 'Ace', 'K': 'King', 'Q': 'Queen', 'J': 'Jack'}.get(r, r)
    return f'<span class="playing-card {"red" if s in "hd" else "black"} {cls}" role="img" aria-label="{rank} of {suit}" data-card="{code}"><span class="corner" aria-hidden="true">{r}<small>{glyph}</small></span><span class="pip" aria-hidden="true">{glyph}</span><span class="corner bottom" aria-hidden="true">{r}<small>{glyph}</small></span></span>'


def hand(codes, label='Example cards', cls=''):
    if isinstance(codes, str):
        codes = codes.split()
    return f'<div class="hand math-hand {cls}" role="group" aria-label="{label}" data-hand="{" ".join(codes)}">'+''.join(card(c) for c in codes)+'</div>'


def equation(content, label=None):
    return f'<div class="math-equation"'+(f' role="math" aria-label="{escape(label)}"' if label else '')+f'>{content}</div>'


def factor(value, explanation, cards=''):
    return f'<div class="count-factor">{hand(cards) if cards else ""}<strong>{value}</strong><span>{explanation}</span></div>'


def deck_grid(outs=False):
    rows=[]
    scenario=DATA['scenarios'][0]
    known=set(scenario['hole']+scenario['flop'])
    hits=set(scenario['outs'])
    for s,(glyph,name) in SUITS.items():
        cells=[]
        for rank in RANKS:
            code=rank+s
            state=('known' if code in known else 'out' if code in hits else 'other') if outs else ''
            badge={'known':'×','out':'+','other':'·'}.get(state,'')
            label={'known':'known, removed','out':'out, unseen heart','other':'unseen, not an out'}.get(state,'')
            cells.append(f'<span class="deck-cell {state}" data-deck-card="{code}"'+(f' role="group" aria-label="{label}"' if outs else '')+'>'+card(code)+(f'<span class="deck-badge" aria-hidden="true">{badge}</span>' if outs else '')+'</span>')
        rows.append(f'<div class="suit-row"><strong class="suit-name">{glyph} {name.capitalize()}</strong><div class="suit-cards">'+''.join(cells)+'</div></div>')
    return '<div class="deck-grid"'+(' id="outs-deck"' if outs else '')+'>'+''.join(rows)+'</div>'


def rarity_svg():
    categories=list(reversed(DATA['categories']))
    # Axis x = 230 + 82 log10(count), so each decade has equal width.
    body=['<svg class="rarity-chart" viewBox="0 0 900 440" role="img" aria-labelledby="rarity-title rarity-desc"><title id="rarity-title">Five-card hand counts on a base-ten logarithmic scale</title><desc id="rarity-desc">Each equally spaced tick multiplies the count by ten. Counts range from 40 straight flushes to 1,302,540 high-card hands. Exact values are also in the table above.</desc>']
    for power in range(7):
        x=230+82*power
        body.append(f'<path d="M{x} 40V395" class="rarity-grid"/><text x="{x}" y="424" text-anchor="middle">{10**power:,}</text>')
    for i,c in enumerate(categories):
        y=59+38*i;x=230+82*log10(c['count'])
        body.append(f'<text x="210" y="{y+5}" text-anchor="end">{c["label"]}</text><path d="M230 {y}H{x:.3f}" class="rarity-stem"/><circle cx="{x:.3f}" cy="{y}" r="5" class="rarity-dot"/><text x="{x+12:.3f}" y="{y+5}" class="rarity-value">{c["count"]:,}</text>')
    body.append('</svg><div class="rarity-mobile" role="group" aria-label="Hand counts on a logarithmic axis, one to ten million">')
    for c in categories:
        position = 100 * log10(c['count']) / 7
        body.append(f'<div class="rarity-mobile-row"><span>{c["label"]}<b>{c["count"]:,}</b></span><div class="rarity-track" style="--position:{position:.5f}%" aria-hidden="true"></div></div>')
    body.append('<div class="rarity-mobile-axis" aria-hidden="true">')
    for power, label in enumerate(['1', '10', '100', '1k', '10k', '100k', '1m', '10m']):
        body.append(f'<span style="left:{100*power/7:.5f}%">{label}</span>')
    body.append('</div><p class="small">k = thousand; m = million. Each tick is ×10.</p></div>')
    return ''.join(body)


sections=[]
sections.append('''<div class="math-part" id="mathematics">
<div class="part-heading"><span class="stage-label">Part II · Counting and uncertainty</span><h2>From recognizing a hand<br>to measuring its rarity.</h2><p>First count equally likely outcomes. Then ask how seeing cards changes the possibilities.</p></div>
<nav class="toc math-toc" aria-label="Mathematics contents"><a href="#deck-math">08 · Deck &amp; combinations</a><a href="#total-hands">09 · All five-card hands</a><a href="#hand-counts">10 · Count each pattern</a><a href="#rarity">11 · Rarity scale</a><a href="#seven-card-math">12 · Best of seven</a><a href="#draw-probabilities">13 · Draws &amp; outs</a><a href="#conditional-math">14 · Conditional probability</a><a href="#without-replacement">15 · Probability tree</a><a href="#bayes-math">16 · Bayes’ rule</a><a href="#expected-value">17 · Expected value</a></nav>
<section class="chapter" id="deck-math"><span class="stage-label">08 · The deck as a mathematical object</span><h2>Four suits. Thirteen ranks. Fifty-two distinct cards.</h2>
<p>A <strong>rank</strong> is the value, such as queen. A <strong>suit</strong> is the symbol, such as hearts. A card is one rank–suit pair: Q♥ and Q♣ share a rank, but are different cards.</p>'''+deck_grid()+equation('4 suits × 13 ranks = <strong>52 cards</strong>')+'''
<div class="rank-suit-key"><div><span class="small-label">Hold rank fixed</span>'''+hand('Qs Qh Qd Qc','The four queens')+'''<p>One rank, four possible suits.</p></div><div><span class="small-label">Hold suit fixed</span>'''+hand('Ah 4h 7h Jh','Four different heart ranks')+'''<p>One suit, different ranks.</p></div></div>
<h3>“Choose” means select a set, not arrange a line.</h3>
<p>From these seven cards, keep the five marked ✓. Excluding the other two identifies exactly the same choice. Rearranging the five kept cards creates no new subset.</p>
<div class="choose-seven" role="group" aria-label="Seven available cards, first five selected">'''+''.join('<span class="choice-card">'+card(c,'chosen' if i<5 else 'not-chosen')+f'<small>{"✓ keep" if i<5 else "− leave"}</small></span>' for i,c in enumerate(DATA['seven_cards']))+'''</div>
'''+equation('<span>C(n, k) = <span class="fraction"><span>n!</span><span>k! (n − k)!</span></span></span>','C of n comma k equals n factorial divided by k factorial times n minus k factorial')+'''
<p class="formula-key"><strong>n</strong> = number of distinct available objects; <strong>k</strong> = number selected, with 0 ≤ k ≤ n. The factorial <strong>n!</strong> means n × (n − 1) × … × 1, and 0! = 1. C(n, k), also written “n choose k,” counts unordered selections without replacement.</p>
<div class="factor-flow">'''+factor('7 × 6 × 5 × 4 × 3','Count choices for five successive positions.')+factor('÷ 5! = ÷ 120','Each selected set appears in 120 different orders.')+factor('= C(7, 5) = 21','Exactly 21 distinct five-card subsets.')+'''</div>
<p class="small">Why the factorial formula? The ordered count is n! / (n − k)!; dividing by the k! arrangements of each selected set removes duplicate orderings. For seven cards: C(7, 5) = 7! / (5! 2!) = 21.</p>
</section>''')

sections.append('''<section class="chapter" id="total-hands"><span class="stage-label">09 · Define the sample space</span><h2>How many five-card hands are possible?</h2>
<p>Assume a thoroughly shuffled standard deck: every set of five distinct cards is equally likely. The <strong>sample space Ω</strong> is the collection of these sets; |Ω| is its size.</p>
<div class="order-demo"><span class="small-label">One royal flush, two written orders</span>'''+hand('As Ks Qs Js 10s','Royal flush in descending order', 'order-original')+'''<div class="order-equals">= <span>same five physical cards</span></div>'''+hand('10s Js Qs Ks As','The same royal flush in ascending order','order-rearranged')+'''<div class="math-controls" hidden><button type="button" id="reorder-cards">Reverse the second row</button></div><p id="order-status" role="status">Both rows represent one hand, not two.</p></div>
'''+equation('<span>|Ω| = C(52, 5)</span><span>= <span class="fraction"><span>52 × 51 × 50 × 49 × 48</span><span>5 × 4 × 3 × 2 × 1</span></span></span><span>= <span class="fraction"><span>311,875,200</span><span>120</span></span> = <strong>2,598,960</strong></span>')+'''
<p>The numerator counts ordered deals. For every particular five-card set, there are 5! = 120 orders, so divide once by 120. A shuffled deck has no repeated physical cards, even when two cards share a rank.</p>
'''+equation('<span>P(category) = <span class="fraction"><span>number of hands in that category</span><span>2,598,960 equally likely hands</span></span></span>')+'''<p class="formula-key"><strong>P</strong> denotes probability. For example, the four royal flushes have probability 4 / 2,598,960 = 1 / 649,740 ≈ <strong>0.000154%</strong>. This concerns an ordinary five-card deal.</p>
</section>''')

count_details=[
('four-of-a-kind','Four of a Kind','Four suits are forced once the rank is chosen.',
 [('13','Choose one of the 13 ranks.','9s 9h 9d 9c'),('1','Use all four suits of that rank.',''),('48','Choose the kicker: 52 − 4 cards.','Kd')],
 'C(13, 1) × 1 × 48 = 13 × 48 = <strong>624</strong>',
 'The kicker cannot be a fifth nine: the deck contains only four. Every four-of-a-kind hand has one unique repeated rank and one unique kicker, so no hand is counted twice.'),
('full-house','Full House','A triplet and a pair play different roles.',
 [('13','Choose the triplet’s rank.','Ks Kd Kc'),('C(4, 3) = 4','Choose three suits for that rank.',''),('12','Choose a different rank for the pair.','7h 7c'),('C(4, 2) = 6','Choose two suits for the pair.','')],
 '13 × C(4, 3) × 12 × C(4, 2) = 13 × 4 × 12 × 6 = <strong>3,744</strong>',
 'Do not divide the two rank choices by two: kings full of sevens and sevens full of kings are different hands. The rank chosen for the triplet has a specific job.'),
('straight-flush','Straight Flush','Ten legal rank sequences, four possible suits.',
 [('10','Choose a sequence: A–2–3–4–5 through 10–J–Q–K–A.','5h 6h 7h 8h 9h'),('4','Choose the one common suit.','')],
 '10 × 4 = <strong>40</strong>',
 'The ace-low wheel counts once; no wraparound sequence is allowed. Four of these 40 hands are royal flushes, leaving 36 non-royal straight flushes. The summary counts all 40 together.'),
('flush','Flush, excluding straight flushes','First count all single-suit hands. Then remove the overlap.',
 [('4','Choose the suit.','Ah Jh 8h 5h 2h'),('C(13, 5) = 1,287','Choose five distinct ranks within that suit.',''),('− 40','Remove the straight flushes already classified above.','')],
 '4 × C(13, 5) − 40 = 5,148 − 40 = <strong>5,108</strong>',
 'The initial 5,148 includes every same-suit five-card set, including the 40 that also form a sequence. Subtract them to make the categories mutually exclusive.'),
('straight','Straight, excluding straight flushes','Fix the ranks, then choose a suit independently for each rank.',
 [('10','Choose a legal five-rank sequence.','5s 6d 7c 8h 9s'),('4⁵ = 1,024','For each of five distinct ranks, choose one of four suits.',''),('− 40','Remove the same-suit sequences.','')],
 '10 × 4<sup>5</sup> − 40 = 10,240 − 40 = <strong>10,200</strong>',
 'For each sequence, four of the 1,024 suit assignments use a single suit. The subtraction is therefore also 10 × (4⁵ − 4). No extra division by 5! is needed: the distinct ranks already identify the five choices.'),
('three-of-a-kind','Three of a Kind','One triplet; two kickers of different ranks.',
 [('13 × C(4, 3)','Choose the triplet rank and its three suits.','Qs Qh Qd'),('C(12, 2) = 66','Choose two distinct other ranks, without ordering them.',''),('4² = 16','Choose one suit for each kicker.','8c 3s')],
 '13 × C(4, 3) × C(12, 2) × 4<sup>2</sup> = <strong>54,912</strong>',
 'Distinct kicker ranks prevent a full house. Excluding the triplet rank prevents four of a kind. Repeated ranks already rule out a straight or flush.'),
('two-pair','Two Pair','Choose the two pair ranks as an unordered set.',
 [('C(13, 2) = 78','Choose two distinct pair ranks.','Js Jd 4h 4c'),('C(4, 2)² = 36','Choose two suits separately for each rank.',''),('11 × 4 = 44','Choose a kicker from one of the remaining 11 ranks.','As')],
 'C(13, 2) × C(4, 2)<sup>2</sup> × 11 × 4 = <strong>123,552</strong>',
 'Using 13 × 12 for the pair ranks would double-count each hand: jacks-and-fours and fours-and-jacks are the same two pairs. C(13, 2) fixes that.'),
('one-pair','One Pair','One pair; three kickers of different ranks.',
 [('13 × C(4, 2)','Choose the pair rank and its two suits.','10s 10h'),('C(12, 3) = 220','Choose three distinct other ranks.',''),('4³ = 64','Choose one suit for each kicker.','Kc 7d 3c')],
 '13 × C(4, 2) × C(12, 3) × 4<sup>3</sup> = <strong>1,098,240</strong>',
 'Choosing distinct ranks for the kickers prevents a second pair or a triplet. None can reuse the pair’s rank.'),
('high-card','High Card','Five different ranks, with neither a sequence nor a single suit.',
 [('C(13, 5) − 10 = 1,277','Choose a rank set that is not one of the ten sequences.','As Jd 8c 5h 2s'),('4⁵ − 4 = 1,020','Assign suits, excluding the four all-one-suit assignments.','')],
 '[C(13, 5) − 10] × (4<sup>5</sup> − 4) = <strong>1,302,540</strong>',
 'Choosing distinct ranks excludes all repeated-rank categories. Removing sequences and uniform suits leaves exactly high card. This also equals the total minus the other eight categories.')]

sections.append('''<section class="chapter" id="hand-counts"><span class="stage-label">10 · Build the counts, factor by factor</span><h2>Every factor answers a choice.</h2><p>These are <strong>exclusive five-card categories</strong>: each hand belongs to its highest category exactly once. Multiplication combines successive choices; subtraction removes hands assigned to a stronger category.</p>''')
for slug,title,intro,factors,formula,explanation in count_details:
    c=next(c for c in DATA['categories'] if c['id']==slug)
    suit_choices = ''
    if slug == 'full-house':
        suit_choices = '<div class="rank-suit-key"><div><span class="small-label">Triplet: keep three of four suits</span><div class="hand math-hand">'+''.join(card(code, 'chosen' if code != 'Kh' else 'not-chosen') for code in ['Ks','Kh','Kd','Kc'])+'</div><p>Example: keep ♠ ♦ ♣; leave ♥. Four possible omitted suits.</p></div><div><span class="small-label">Pair: keep two of four suits</span><div class="hand math-hand">'+''.join(card(code, 'chosen' if code in ['7h','7c'] else 'not-chosen') for code in ['7s','7h','7d','7c'])+'</div><p>Example: keep ♥ ♣. The six choices are ♠♥, ♠♦, ♠♣, ♥♦, ♥♣, ♦♣.</p></div></div>'
    sections.append(f'<article class="derivation" id="count-{slug}"><h3>{title}</h3><p>{intro}</p><div class="factor-flow">'+''.join(factor(*f) for f in factors)+'</div>'+suit_choices+equation(formula)+f'<p>{explanation}</p><p class="probability-line"><strong>P = {c["count"]:,} / 2,598,960 ≈ {c["percent"]:.6f}%</strong><span>Exact fraction first; rounded percentage second.</span></p></article>')
sections.append('''<h3 id="frequency-table">The complete five-card distribution</h3><p class="small">Every row is disjoint. Percentages are rounded to six decimal places; “1 in N” uses N = 2,598,960 / count and is rounded to two decimals. It describes a long-run average, not a schedule or guarantee.</p><div class="math-table-wrap"><table class="frequency-table"><caption>Uniformly chosen five-card hands from a standard 52-card deck</caption><thead><tr><th scope="col">Category</th><th scope="col">Hands</th><th scope="col">Probability</th><th scope="col">About 1 in N</th></tr></thead><tbody>''')
for c in DATA['categories']:
    label=c['label']+(' (incl. royal)' if c['id']=='straight-flush' else '')
    sections.append(f'<tr data-count="{c["count"]}"><th scope="row">{label}</th><td>{c["count"]:,}</td><td>{c["percent"]:.6f}%</td><td>{c["one_in"]:,.2f}</td></tr>')
sections.append('''</tbody><tfoot><tr><th scope="row">Total</th><td>2,598,960</td><td>100% exactly</td><td>1</td></tr></tfoot></table></div><aside class="note"><strong>Royal flush detail, not an additional row.</strong> The 4 royal flushes are already inside the 40 straight flushes. Their probability is 1 / 649,740 ≈ 0.000154%; counting them again would make the total wrong.</aside>
<p class="verification-note">Checked by <a href="math-behind-poker/verify_math.py">exhaustive Python enumeration</a> of all 2,598,960 hands, independently against the formulas above. <a href="math-behind-poker/verified_math.json">Download the exact counts and verification results</a>.</p></section>''')

sections.append('''<section class="chapter" id="rarity"><span class="stage-label">11 · Seeing orders of magnitude</span><h2>Rare hands vanish on an ordinary bar chart.</h2><p>There are 1,098,240 one-pair hands, 624 four-of-a-kind hands, and just 40 straight flushes. On a linear chart where one pair fills 500 pixels, four of a kind gets only <strong>0.284 pixels</strong>, and straight flush gets <strong>0.018 pixels</strong>.</p><figure class="rarity-figure">'''+rarity_svg()+'''<figcaption>A logarithmic count axis. One equal step right means <strong>×10 as many hands</strong>, not “add ten.” The chart includes royal flushes in straight flush.</figcaption></figure>
'''+equation('<span>log<sub>10</sub>(10) = 1</span><span>log<sub>10</sub>(100) = 2</span><span>log<sub>10</sub>(1,000) = 3</span>')+'''<p class="formula-key"><strong>log<sub>10</sub>(x)</strong> is the power to which 10 must be raised to get the positive count x. Equal horizontal distances represent equal <em>ratios</em>. Zero has no finite position on this axis. Dot positions, not apparent line-length ratios, encode counts.</p><p>The ordering of these exclusive five-card categories follows their rarity. This explains the hierarchy’s combinatorial structure; it does not mean hand rankings change with a particular board.</p></section>''')

first=DATA['subsets'][0]
sections.append('''<section class="chapter" id="seven-card-math"><span class="stage-label">12 · Hold’em changes the experiment</span><h2>Seven cards, twenty-one candidates, one best value.</h2><p>The five-card table describes a random set of five. At a Hold’em river you have seven available cards, and keep the strongest of their C(7, 5) = <strong>21</strong> five-card subsets. These subsets share cards; they are <strong>not independent trials</strong>.</p>
<div class="subset-lab"><div class="subset-source" id="subset-source">'''+''.join('<span class="choice-card">'+card(c,'chosen' if i<5 else 'not-chosen')+f'<small>{"Hole" if i<2 else "Flop" if i<5 else "Turn" if i==5 else "River"}<br><span class="subset-mark">{"✓ keep" if i<5 else "− leave"}</span></small></span>' for i,c in enumerate(DATA['seven_cards']))+'''</div><div class="math-controls" id="subset-controls" hidden><label for="subset-choice">Five-card subset</label><select id="subset-choice">'''+''.join(f'<option value="{i}">{i+1:02} of 21 · leave out {" &amp; ".join(DATA["seven_cards"][j][:-1]+SUITS[DATA["seven_cards"][j][-1]][0] for j in range(7) if j not in s["indices"])}</option>' for i,s in enumerate(DATA['subsets']))+'''</select><button type="button" id="subset-prev" disabled>Previous</button><button type="button" id="subset-next">Next</button><button type="button" id="subset-best">Show best hand</button></div>
<div id="subset-result">'''+hand(first['cards'],'Current selected five-card subset')+'''</div><p class="comparison-status" id="subset-status" role="status">Subset 1 of 21: A♠ Q♠ J♠ 10♠ K♦ is an ace-high straight. Another subset is stronger: A♠ Q♠ J♠ 10♠ 9♠ makes an ace-high flush.</p></div>
'''+equation('<span>best_hand(S) = max<sub>T ⊆ S, |T| = 5</sub> rank(T)</span>')+'''<p class="formula-key"><strong>S</strong> is the set of seven available cards. <strong>T ⊆ S</strong> means T uses only cards from S; <strong>|T| = 5</strong> requires exactly five. <strong>rank(T)</strong> is its ordered hand value: category first, then the tie-breaking ranks. <strong>max</strong> selects the greatest value under that comparison rule. Multiple subsets can share the same maximum.</p>
<p class="small">For a numerical encoding, assign categories 0 through 8 from high card through straight flush; ace = 14, king = 13, queen = 12, jack = 11. Here the flush has rank (5; 14, 12, 11, 10, 9), beating the straight’s (4; 14) at the category entry. These tuples are compared lexicographically, never added.</p><aside class="note"><strong>Do not multiply a five-card probability by 21.</strong> The candidate subsets overlap heavily. Computing seven-card frequencies requires counting seven-card sets and classifying each by its best subset.</aside></section>''')

scenario=DATA['scenarios'][0]
sections.append('''<section class="chapter" id="draw-probabilities"><span class="stage-label">13 · Known cards, unseen cards, outs</span><h2>Probability starts by removing what you know.</h2><p>At the flop, your two hole cards and the three community cards give <strong>5 known cards</strong>. With no other card information, <strong>47 are unseen</strong>. Opponents’ hidden cards and unseen burn cards are included in that uncertainty; do not subtract unidentified cards from the denominator.</p><aside class="note"><strong>Model for these examples.</strong> Condition only on the shown cards; all unseen identities are equally likely at the next board position. Assume both turn and river will be dealt when calculating a two-card probability. Betting actions or exposed opponent cards can supply additional information.</aside>
<div class="outs-lab"><div class="math-controls" id="draw-controls" hidden><label for="draw-choice">Draw example</label><select id="draw-choice">'''+''.join(f'<option value="{s["id"]}">{s["label"]}</option>' for s in DATA['scenarios'])+'''</select></div><div class="draw-cards"><div><span class="small-label">Your hole cards</span><div id="draw-hole">'''+hand(scenario['hole'])+'''</div></div><div><span class="small-label">Flop · shared cards</span><div id="draw-flop">'''+hand(scenario['flop'])+'''</div></div></div><p id="draw-target">'''+scenario['target']+'''</p><div class="deck-legend"><span>× Known / removed</span><span>+ Current out / useful unseen card</span><span>· Other unseen card</span></div>'''+deck_grid(outs=True)+'''<p id="draw-status" class="comparison-status" role="status">5 known · 47 unseen · 9 outs. Next card: 9/47 ≈ 19.149%. At least one of these outs by the river: 378/1081 ≈ 34.968%.</p><p id="draw-caution" class="small">'''+scenario['caution']+'''</p></div>
<h3>Work the flush draw exactly.</h3><div class="factor-flow">'''+factor('13 − 4 = 9','Hearts left: the four visible hearts are already removed.')+factor('52 − 5 = 47','Unseen cards eligible for the next board position.')+factor('9 / 47','One-card flush-completion probability.')+'''</div>
'''+equation('<span>P(heart on turn) = 9 / 47 ≈ <strong>19.149%</strong></span>')+'''
<p>An <strong>out</strong> here is an unseen card that completes the stated target on the next card. Four hearts are visible; any of the other nine completes a heart flush. That event is different from winning the pot.</p>
'''+equation('<span>P(flush by river) = 1 − P(no heart on either card)</span><span>= 1 − (38 / 47)(37 / 46)</span><span>= 1 − C(38, 2) / C(47, 2)</span><span>= 378 / 1,081 ≈ <strong>34.968%</strong></span>')+'''
<p>The 38 non-hearts can produce C(38, 2) = 703 miss–miss pairs out of C(47, 2) = 1,081 possible two-card runouts. Subtract those from the total. The ordered calculation reaches the same answer because there are two orders for every pair.</p>
<h3>Four targets, four different out sets</h3><p class="small">All examples below have five known cards and 47 unseen. “By river” means hitting <strong>at least one of the current fixed outs</strong> in two cards; it is not a general winning probability or a count of every runner–runner improvement.</p><div class="draw-reference">''')
for s in DATA['scenarios']:
    o=s['out_count']
    sections.append(f'<article class="draw-reference-row"><div><h4>{s["label"]}</h4><span class="small-label">Hole cards</span>{hand(s["hole"])}<span class="small-label">Flop</span>{hand(s["flop"])}</div><div><p>{s["target"]}</p><p class="out-list"><strong>{o} outs:</strong> '+', '.join(c[:-1]+SUITS[c[-1]][0] for c in s['outs'])+f'</p><p>Turn: {o}/47 ≈ <strong>{s["turn"]["percent"]:.3f}%</strong><br>By river: {s["by_river"]["fraction"]} ≈ <strong>{s["by_river"]["percent"]:.3f}%</strong></p><p class="small">{s["caution"]}</p></div></article>')
sections.append('''</div>'''+equation('<span>P(at least one fixed out in two cards)</span><span>= 1 − <span class="fraction"><span>(47 − o)(46 − o)</span><span>47 × 46</span></span></span>')+'''<p class="formula-key"><strong>o</strong> is the number of distinct current outs, all among the 47 unseen cards. For o = 8, the result is 1 − (39 × 38)/(47 × 46) = 340/1,081 ≈ 31.452%. The fixed-out condition defines the event being counted.</p>
<aside class="note"><strong>Outs need a target and context.</strong> A nominal out can improve your hand while giving an opponent a stronger one: it is then not a clean winning out. If one card helps two draws, count that physical card once, not twice. A new turn can create or remove useful river cards. The gutshot’s “hit a six” calculation deliberately excludes alternative runner–runner straights.</aside>
<h3>Only now: the rule of 2 and 4</h3><p>A quick approximation multiplies the out count by <strong>2%</strong> for one card and <strong>4%</strong> for two cards. Here we compare it with the exact fixed-out calculation from the flop. With nine outs, it gives 18% and 36%.</p><div class="math-table-wrap"><table class="approx-table"><caption>Exact probabilities versus the approximation; all displayed percentages rounded</caption><thead><tr><th scope="col">Outs</th><th scope="col">Turn exact</th><th scope="col">×2 estimate</th><th scope="col">By river exact</th><th scope="col">×4 estimate</th></tr></thead><tbody>''')
for s in DATA['scenarios']:
    sections.append(f'<tr><th scope="row">{s["out_count"]}</th><td>{s["turn"]["percent"]:.3f}%</td><td>{s["rule_two_percent"]}%</td><td>{s["by_river"]["percent"]:.3f}%</td><td>{s["rule_four_percent"]}%</td></tr>')
sections.append('''</tbody></table></div><p class="small">For nine outs, ×2 understates the turn chance by 1.149 <em>percentage points</em>; ×4 overstates the two-card chance by 1.032 points. On the turn, nine outs among 46 unseen cards give 9/46 ≈ 19.565%, so a one-card “×2” estimate also depends on the street. The shortcut becomes less reliable for large out counts; use the exact formula when the precision matters.</p></section>''')

sections.append('''<section class="chapter" id="conditional-math"><span class="stage-label">14 · Conditioning changes the denominator</span><h2>“Given what I know” is part of the probability.</h2>'''+equation('<span>P(A | B) = <span class="fraction"><span>P(A ∩ B)</span><span>P(B)</span></span>, with P(B) &gt; 0</span>')+'''<p class="formula-key"><strong>A</strong> is the event of interest; <strong>B</strong> is the given information expressed as an event. The vertical bar means “given”; <strong>A ∩ B</strong> means both events happen. Conditioning restricts the sample space to B and rescales its probability to 1.</p>
<div class="conditioning-flow"><div><strong>52 candidates</strong><span>A: candidate card is a heart.</span><b>13 hearts</b></div><span class="flow-arrow" aria-hidden="true">→</span><div><strong>Remove 5 known</strong><span>A♥ Q♥ 8♥ 3♥ K♣</span><b>4 hearts + 1 non-heart</b></div><span class="flow-arrow" aria-hidden="true">→</span><div><strong>47 candidates remain</strong><span>B: candidate is not a known card.</span><b>9 hearts</b></div></div>
<p>To model card removal, start with a uniformly selected candidate from all 52 cards and condition on it being outside the <em>fixed</em> known set. Before that restriction, P(A ∩ B) = 9/52 and P(B) = 47/52.</p>'''+equation('<span>P(A | B) = (9/52) / (47/52) = <strong>9/47 ≈ 19.149%</strong></span>')+'''<p>The original 13/52 = 25% heart fraction is no longer the right denominator-and-numerator pair. Four hearts are already visible, so the next unseen card is a heart with probability 9/47. The deck’s physical size stays 52; your set of possible next cards shrinks.</p></section>''')

sections.append('''<section class="chapter" id="without-replacement"><span class="stage-label">15 · Without replacement</span><h2>The second draw remembers the first.</h2><p>Return to the nine-heart flush draw. A turn heart leaves 8 hearts among 46 unseen cards; a non-heart leaves 9 among 46. Unlike independent coin flips, the river’s distribution depends on what the turn removed.</p>
<div class="probability-tree" role="group" aria-label="Two-stage probability tree for heart hits and misses"><div class="tree-root">Start · 9 hearts / 47 unseen</div><div class="tree-branches"><section class="tree-branch"><h3>Turn hits a heart <span>9/47</span></h3><div class="tree-leaf"><strong>River hits · 8/46</strong><span>(9/47)(8/46) = 36/1,081</span><b>Two hearts · flush completed</b></div><div class="tree-leaf"><strong>River misses · 38/46</strong><span>(9/47)(38/46) = 171/1,081</span><b>Turn heart alone · flush completed</b></div></section><section class="tree-branch"><h3>Turn misses <span>38/47</span></h3><div class="tree-leaf"><strong>River hits · 9/46</strong><span>(38/47)(9/46) = 171/1,081</span><b>River heart alone · flush completed</b></div><div class="tree-leaf tree-miss"><strong>River misses · 37/46</strong><span>(38/47)(37/46) = 703/1,081</span><b>No heart · flush not completed</b></div></section></div></div>
'''+equation('<span>P(T ∩ R) = P(T) × P(R | T)</span>')+'''<p class="formula-key"><strong>T</strong> is a specified turn outcome (hit or miss); <strong>R</strong> is a specified river outcome. Multiply conditional probabilities along a path. For the hit–hit path: (9/47)(8/46) = 36/1,081 ≈ 3.330%.</p>
'''+equation('<span>P(at least one heart)</span><span>= (36 + 171 + 171) / 1,081</span><span>= 378 / 1,081 ≈ <strong>34.968%</strong></span>')+'''<p>Add the three <strong>mutually exclusive</strong> success paths. All four leaves sum to 1,081/1,081 = 1. Alternatively, subtract the only failure leaf: 1 − 703/1,081.</p><aside class="note"><strong>After a miss, update.</strong> Once a non-heart turn is observed, the relevant remaining chance is 9/46 ≈ 19.565%, not the original two-card 34.968%. The increase from 9/47 comes from removing one non-heart, not from a belief that you are “due” a heart.</aside></section>''')

sections.append('''<section class="chapter" id="bayes-math"><span class="stage-label">16 · Inference from an observation</span><h2>Bayes’ rule: weight possibilities by their predictions.</h2><p>Use a deliberately simplified, <strong>hypothetical</strong> opponent model. At one fixed decision point, every possible holding belongs to exactly one of two classes: <strong>H</strong> (“strong”) or <strong>L</strong> (“other”). These are model labels, not specific poker categories.</p>
<div class="bayes-cohorts"><div><span class="small-label">Prior model · before the action</span><h3>1,000 hypothetical cases</h3><div class="cohort-bar"><span class="cohort-h">H · 200</span><span class="cohort-l">L · 800</span></div><p>P(H) = 20%; P(L) = 80%.</p></div><div><span class="small-label">Assumed action policy</span><h3>Observe a bet, event B</h3><p>H bets in 75% of cases: <strong>150 of 200</strong>.<br>L bets in 25% of cases: <strong>200 of 800</strong>.</p></div><div><span class="small-label">Posterior model · after seeing B</span><h3>350 betting cases remain</h3><div class="cohort-bar posterior"><span class="cohort-h">H · 150</span><span class="cohort-l">L · 200</span></div><p>H accounts for 150/350 = 3/7 ≈ <strong>42.857%</strong>.</p></div></div>
'''+equation('<span>P(H | B) = <span class="fraction"><span>P(B | H) P(H)</span><span>P(B | H) P(H) + P(B | L) P(L)</span></span></span><span>= <span class="fraction"><span>0.75 × 0.20</span><span>0.75 × 0.20 + 0.25 × 0.80</span></span></span><span>= 0.15 / 0.35 = <strong>3/7 ≈ 42.857%</strong></span>')+'''<p class="formula-key"><strong>P(H)</strong> and <strong>P(L)</strong> are prior probabilities, summing to 1. <strong>P(B | H)</strong> and <strong>P(B | L)</strong> are the assumed probabilities of the observed action in each class. The denominator is <strong>P(B)</strong>, the total probability of a bet; <strong>P(H | B)</strong> is the updated probability of H.</p>
<p>In words: <strong>posterior ∝ likelihood × prior</strong>. The symbol ∝ means “proportional to”; divide both class weights by their sum to make probabilities add to one. H becomes more plausible after a bet, but L still accounts for more betting cases because it was much more common initially.</p><aside class="note"><strong>Assumptions in, conclusions out.</strong> The class proportions and action policy above are invented teaching assumptions, not measured player behavior or a strategy recommendation. Different priors or policies give different posteriors. This example demonstrates inference, not mind-reading.</aside></section>''')

sections.append('''<section class="chapter" id="expected-value"><span class="stage-label">17 · Bridge to decision mathematics</span><h2>A likely outcome is not the same as an average payoff.</h2><p>Probability describes possible outcomes. <strong>Expected value</strong> weights the net payoff of each outcome by its probability.</p>'''+equation('<span>EV = Σ<sub>i</sub> p<sub>i</sub> x<sub>i</sub></span>','Expected value equals the sum over outcomes i of probability p i times net payoff x i')+'''<p class="formula-key"><strong>i</strong> indexes mutually exclusive, exhaustive outcomes. <strong>p<sub>i</sub></strong> is the probability of outcome i, with Σ<sub>i</sub> p<sub>i</sub> = 1; <strong>x<sub>i</sub></strong> is its signed <em>net</em> payoff measured from the same starting point. Σ means add all outcomes’ contributions.</p>
<div class="payoff-example"><div><span class="small-label">Win · probability 3/10</span><strong>+20 units net</strong><span>Contribution: (3/10)(20) = +6</span></div><div><span class="small-label">Lose · probability 7/10</span><strong>−5 units net</strong><span>Contribution: (7/10)(−5) = −3.5</span></div><div><span class="small-label">Expected net payoff</span><strong>+2.5 units</strong><span>EV = 6 − 3.5 = 2.5</span></div></div>
<p>In this hypothetical two-outcome example there are no ties or additional costs. A single play returns either +20 or −5, never +2.5. EV is the probability-weighted mean; positive EV still comes with a 70% chance of losing on one play.</p>
<aside class="note"><strong>Keep the meanings separate.</strong> A draw-completion probability is not automatically a win probability. To evaluate an actual decision, the outcomes, net payoffs, ties, future actions, and uncertainty about other players must be modeled consistently.</aside>
<p class="next-phase"><a href="#laboratory">Continue · Connect probabilities to decisions and simulation ↓</a></p>
</section>
<aside class="math-reproducibility"><h3>Reproduce the mathematics</h3><p>The verification uses only Python’s standard library. It enumerates every five-card hand and all 1,081 turn–river pairs for each fixed-out example; no random sampling is used.</p><pre><code>python3 research/math-behind-poker/verify_math.py
python3 research/math-behind-poker/render_math.py</code></pre><p class="small">The first script writes <a href="math-behind-poker/verified_math.json">verified_math.json</a>. The optional <a href="math-behind-poker/render_math.py">renderer</a> updates only this marked mathematics section and its static figures. The served page requires no Python, package install, or build step.</p></aside>
</div>''')
# Shared finite examples: no fetch needed, including when opened as a local file.
interactive_data = {k:DATA[k] for k in ('scenarios','seven_cards','subsets','best_subset_index')}
sections.append('<script type="application/json" id="poker-math-data">'+json.dumps(interactive_data,ensure_ascii=False).replace('</','<\\/')+'</script>')
block='\n<!-- BEGIN GENERATED PHASE 2: render_math.py -->\n'+'\n'.join(sections)+'\n<!-- END GENERATED PHASE 2 -->\n'
text=PAGE.read_text()
if '<!-- BEGIN GENERATED PHASE 2:' in text:
    text=re.sub(r'\n<!-- BEGIN GENERATED PHASE 2:.*?<!-- END GENERATED PHASE 2 -->\n',lambda _:block,text,flags=re.S)
else:
    text=text.replace('<aside class="sources"',block+'<aside class="sources"',1)
phase_three_marker = '<!-- Phase 3 insertion point: interactive decision and simulation laboratory. -->'
if phase_three_marker not in text:
    text=text.replace('<!-- END GENERATED PHASE 2 -->', '<!-- END GENERATED PHASE 2 -->\n'+phase_three_marker, 1)
if 'href="math-behind-poker/math.css"' not in text:
    text=text.replace('<script defer src="math-behind-poker/page.js"></script>', '<link rel="stylesheet" href="math-behind-poker/math.css">\n<script defer src="math-behind-poker/page.js"></script>\n<script defer src="math-behind-poker/math.js"></script>')
text=text.replace('A visual mathematical guide · Phase 1 draft','A visual mathematical guide · Phases 1–2 draft')
text=text.replace('<!-- Phase 2 insertion point: five-card counting, then seven-card and conditional models. -->','<!-- The mathematical core continues below. -->')
text=text.replace('Next · From visual patterns to combinatorics and probability.','<a href="#deck-math">Continue · From visual patterns to combinatorics and probability ↓</a>')
text=text.replace('<a href="#mathematical-bridge">07 · Why this order?</a></nav>','<a href="#mathematical-bridge">07 · Why this order?</a><a href="#mathematics">Part II · The mathematics ↓</a></nav>')
text=text.replace('All cards, examples, and comparison results are visible without JavaScript. JavaScript adds the optional step-through controls.', 'All explanations, counts, and worked examples are visible without JavaScript. JavaScript adds optional ordering, subset, draw-selection, and comparison controls.')
text=text.replace('private and community cards, the best five-card hand, the complete hand hierarchy, kickers, and ties.', 'hand rankings, combinatorics, exact draw probabilities, conditional probability, Bayes’ rule, and expected value.')
PAGE.write_text(text)
print('Rendered the Phase 2 mathematics block in',PAGE.name)
