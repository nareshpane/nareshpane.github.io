#!/usr/bin/env python3
"""Verify the page's mathematics using only Python's standard library.

Run: python3 research/math-behind-poker/verify_math.py
Enumerates all 2,598,960 five-card hands (no Monte Carlo), compares against
independently expressed counting formulas, and enumerates each draw's 1,081
unordered turn/river runouts. Writes verified_math.json next to this script.
"""
from collections import Counter
from fractions import Fraction
from itertools import combinations
from math import comb, factorial
from pathlib import Path
import json

SUITS = 'shdc'
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
DECK = [(rank, suit) for rank in range(2, 15) for suit in range(4)]
LABELS = ['High card', 'One pair', 'Two pair', 'Three of a kind', 'Straight',
          'Flush', 'Full house', 'Four of a kind', 'Straight flush']
IDS = ['high-card', 'one-pair', 'two-pair', 'three-of-a-kind', 'straight',
       'flush', 'full-house', 'four-of-a-kind', 'straight-flush']


def decode(code):
    return (RANKS.index(code[:-1]) + 2, SUITS.index(code[-1]))


def rank_hand(hand):
    """Lexicographic (category, tie-breaking ranks); ace-low is five-high."""
    ranks = sorted((r for r, s in hand), reverse=True)
    multiplicities = sorted(((n, r) for r, n in Counter(ranks).items()), reverse=True)
    flush = len({s for r, s in hand}) == 1
    distinct = set(ranks)
    straight = 5 if distinct == {14, 2, 3, 4, 5} else (
        max(ranks) if len(distinct) == 5 and max(ranks) - min(ranks) == 4 else 0)
    if flush and straight:
        return (8, straight)
    if multiplicities[0][0] == 4:
        return (7, multiplicities[0][1], multiplicities[1][1])
    if [n for n, r in multiplicities] == [3, 2]:
        return (6, multiplicities[0][1], multiplicities[1][1])
    if flush:
        return (5, *ranks)
    if straight:
        return (4, straight)
    if multiplicities[0][0] == 3:
        return (3, multiplicities[0][1], *sorted((r for n, r in multiplicities[1:]), reverse=True))
    if [n for n, r in multiplicities][:2] == [2, 2]:
        return (2, *sorted((r for n, r in multiplicities[:2]), reverse=True), multiplicities[2][1])
    if multiplicities[0][0] == 2:
        return (1, multiplicities[0][1], *sorted((r for n, r in multiplicities[1:]), reverse=True))
    return (0, *ranks)


def fraction_data(p):
    return {'fraction': str(p), 'percent': float(100 * p)}


def verify():
    total = comb(52, 5)
    assert total == 2598960 == 52 * 51 * 50 * 49 * 48 // factorial(5)
    formulas = {
        8: 10 * 4,
        7: 13 * 48,
        6: 13 * comb(4, 3) * 12 * comb(4, 2),
        5: 4 * comb(13, 5) - 40,
        4: 10 * 4**5 - 40,
        3: 13 * comb(4, 3) * comb(12, 2) * 4**2,
        2: comb(13, 2) * comb(4, 2)**2 * 11 * 4,
        1: 13 * comb(4, 2) * comb(12, 3) * 4**3,
        0: (comb(13, 5) - 10) * (4**5 - 4),
    }
    print('Enumerating all five-card hands…', flush=True)
    observed = Counter()
    royals = 0
    for hand in combinations(DECK, 5):
        score = rank_hand(hand)
        observed[score[0]] += 1
        royals += score == (8, 14)
    assert observed == formulas, (observed, formulas)
    assert sum(observed.values()) == total and royals == 4
    categories = []
    for category in range(8, -1, -1):
        count = observed[category]
        categories.append({'id': IDS[category], 'label': LABELS[category],
                           'count': count, **fraction_data(Fraction(count, total)),
                           'one_in': total / count})
    print('All nine categories agree with enumeration; total =', total, flush=True)

    scenarios = [
        {'id': 'flush', 'label': 'Flush draw', 'hole': ['Ah', 'Qh'], 'flop': ['8h', '3h', 'Kc'],
         'outs': [r + 'h' for r in RANKS if r not in ['A', 'Q', '8', '3']],
         'target': 'At least one more heart: enough for a heart flush.',
         'caution': 'Making this flush does not rule out an opponent’s full house or four of a kind.'},
        {'id': 'open', 'label': 'Open-ended straight draw', 'hole': ['8s', '9d'], 'flop': ['6c', '7h', 'Kh'],
         'outs': [r + s for r in ['5', '10'] for s in SUITS],
         'target': 'Any five or ten completes the current 6–7–8–9 sequence.',
         'caution': 'A straight-completing card may also complete an opponent’s flush or a higher straight.'},
        {'id': 'gutshot', 'label': 'Gutshot straight draw', 'hole': ['8s', '9d'], 'flop': ['5c', '7h', 'Kh'],
         'outs': ['6' + s for s in SUITS],
         'target': 'Any six fills the inside gap in 5–7–8–9.',
         'caution': 'The two-card result below counts hitting a six. A ten and a jack can also make a different straight without a six.'},
        {'id': 'set', 'label': 'Pocket pair → at least three eights', 'hole': ['8s', '8d'], 'flop': ['Kc', '7h', '2c'],
         'outs': ['8h', '8c'],
         'target': 'Either remaining eight makes a third eight; hitting both makes four.',
         'caution': 'This counts at least three eights, including full houses or quads, not the exclusive three-of-a-kind category.'},
    ]
    for s in scenarios:
        known = set(s['hole'] + s['flop'])
        all_codes = [r + suit for r in RANKS for suit in SUITS]
        remaining = [c for c in all_codes if c not in known]
        outs = set(s['outs'])
        assert len(known) == 5 and len(remaining) == 47 and not (known & outs)
        # Independently recover the one-card targets from the card patterns,
        # rather than merely checking the size of a manually supplied out list.
        def completes_target(code):
            six = list(map(decode, list(known) + [code]))
            if s['id'] == 'flush':
                return sum(suit == SUITS.index('h') for rank, suit in six) >= 5
            if s['id'] == 'set':
                return sum(rank == 8 for rank, suit in six) >= 3
            return any(rank_hand(hand)[0] in (4, 8) for hand in combinations(six, 5))
        assert {card for card in remaining if completes_target(card)} == outs
        o = len(outs)
        runouts = list(combinations(remaining, 2))
        hits = sum(bool(set(runout) & outs) for runout in runouts)
        probability = Fraction(hits, len(runouts))
        assert len(runouts) == comb(47, 2) == 1081
        assert probability == 1 - Fraction(comb(47 - o, 2), comb(47, 2))
        s.update({'out_count': o, 'runout_hits': hits, 'turn': fraction_data(Fraction(o, 47)),
                  'by_river': fraction_data(probability), 'rule_two_percent': 2 * o,
                  'rule_four_percent': 4 * o})
    assert [s['out_count'] for s in scenarios] == [9, 8, 4, 2]
    # Enumerate actual made-straight outcomes for the gutshot to verify the caveat.
    gutshot = scenarios[2]
    known = list(map(decode, gutshot['hole'] + gutshot['flop']))
    rest = [card for card in DECK if card not in known]
    made_straights = sum(any(rank_hand(h)[0] in (4, 8) for h in combinations(known + list(runout), 5))
                         for runout in combinations(rest, 2))
    assert made_straights > gutshot['runout_hits']

    seven_codes = ['As', 'Qs', 'Js', '10s', 'Kd', '9s', '2c']
    subsets = []
    for indices in combinations(range(7), 5):
        cards = [seven_codes[i] for i in indices]
        score = rank_hand(list(map(decode, cards)))
        subsets.append({'indices': list(indices), 'cards': cards, 'rank': list(score), 'category': LABELS[score[0]]})
    assert len(subsets) == comb(7, 5) == 21
    best = max(s['rank'] for s in subsets)
    assert best == [5, 14, 12, 11, 10, 9]
    best_index = next(i for i, s in enumerate(subsets) if s['rank'] == best)
    assert subsets[0]['rank'] == [4, 14]

    tree = {'hit_hit': Fraction(9, 47) * Fraction(8, 46),
            'hit_miss': Fraction(9, 47) * Fraction(38, 46),
            'miss_hit': Fraction(38, 47) * Fraction(9, 46),
            'miss_miss': Fraction(38, 47) * Fraction(37, 46)}
    assert sum(tree.values()) == 1
    assert sum(tree[k] for k in tree if k != 'miss_miss') == Fraction(378, 1081)
    posterior = Fraction(3, 4) * Fraction(1, 5) / (Fraction(3, 4) * Fraction(1, 5) + Fraction(1, 4) * Fraction(4, 5))
    assert posterior == Fraction(3, 7) == Fraction(150, 350)
    ev = Fraction(3, 10) * 20 + Fraction(7, 10) * (-5)
    assert ev == Fraction(5, 2)
    assert Fraction(9, 52) / Fraction(47, 52) == Fraction(9, 47)
    assert round(500 * formulas[7] / formulas[1], 3) == 0.284
    assert round(500 * formulas[8] / formulas[1], 3) == 0.018
    data = {'method': 'Exhaustive enumeration, independently compared with combinatorial formulas. No simulation.',
            'five_card_total': total, 'categories': categories, 'royal_flushes': royals,
            'scenarios': scenarios, 'seven_cards': seven_codes, 'subsets': subsets, 'best_subset_index': best_index,
            'flush_tree': {k: fraction_data(v) for k, v in tree.items()},
            'bayes_posterior': fraction_data(posterior), 'ev_units': float(ev),
            'gutshot_all_straight_runouts': made_straights}
    output = Path(__file__).with_name('verified_math.json')
    output.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    for c in categories:
        print(f"{c['label']:18} {c['count']:>9,} {c['percent']:10.6f}%")
    print('Verified four draw scenarios, all 21 subsets, the probability tree, Bayes, and EV.')
    print('Wrote', output.name)
    return data


if __name__ == '__main__':
    verify()
