#!/usr/bin/env python3
"""Enumerate the curated runouts and independently reproduce browser simulations.

Standard library only. Run after verify_math.py. Produces verified_lab.json;
the optional render_lab.py turns these results into static page fallbacks.
"""
from collections import Counter
from fractions import Fraction
from hashlib import sha256
from itertools import combinations
from math import comb, sqrt
from pathlib import Path
import json
import sys
sys.dont_write_bytecode = True
from verify_math import decode, rank_hand, RANKS, SUITS

HERE = Path(__file__).resolve().parent
SEEDS = [20260926, 17, 314159]
CHECKPOINTS = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
MASK = 2**32 - 1


def random_words(seed):
    state = seed & MASK
    while True:
        state = (state + 0x6D2B79F5) & MASK
        word = ((state ^ (state >> 15)) * (state | 1)) & MASK
        word ^= (word + (((word ^ (word >> 7)) * (word | 61)) & MASK)) & MASK
        yield (word ^ (word >> 14)) & MASK


def draw_index(words, size):
    limit = (2**32 // size) * size
    while True:
        word = next(words)
        if word < limit:
            return word % size


def as_probability(fraction):
    return {'fraction': f'{fraction.numerator}/{fraction.denominator}',
            'decimal': float(fraction), 'percent': float(100 * fraction)}


def main():
    previous = json.loads((HERE / 'verified_math.json').read_text())
    scenarios = previous['scenarios']
    scenarios.append({'id': 'overcards', 'label': 'Two overcards · pair an ace or king',
        'hole': ['As', 'Kd'], 'flop': ['Qh', '7c', '2s'],
        'outs': [r+s for r in ['A','K'] for s in SUITS if r+s not in ['As','Kd']],
        'target': 'Hit at least one remaining ace or king, pairing a hole-card rank.',
        'caution': 'Six nominal outs to this target. A pair of aces or kings need not beat an opponent; this is not an equity estimate.'})
    deck = [r+s for r in RANKS for s in SUITS]
    for scenario in scenarios:
        print('Enumerating and replaying:', scenario['id'], flush=True)
        known = scenario['hole'] + scenario['flop']
        assert len(set(known)) == 5
        remaining = [card for card in deck if card not in known]
        outs = set(scenario['outs'])
        assert outs <= set(remaining)
        o = len(outs)
        miss = '3d' if scenario['id'] in ['set','overcards'] else '2c'
        assert miss not in known and miss not in outs
        values, hit_count, score_lines = {}, 0, []
        category_counts = [0]*9
        for a, b in combinations(range(47), 2):
            seven = list(map(decode, known+[remaining[a], remaining[b]]))
            score = max(rank_hand(hand) for hand in combinations(seven, 5))
            values[(a,b)] = score
            category_counts[score[0]] += 1
            hit_count += bool({remaining[a],remaining[b]} & outs)
            score_lines.append(f'{a},{b}:'+','.join(map(str,score))+'\n')
        assert sum(category_counts) == comb(47,2) == 1081
        exact = Fraction(hit_count,1081)
        assert exact == 1-Fraction(comb(47-o,2),comb(47,2))
        scenario.update({'out_count':o, 'miss_turn':miss, 'unseen_flop':47,
            'next_flop':as_probability(Fraction(o,47)), 'by_river_flop':as_probability(exact),
            'next_turn':as_probability(Fraction(o,46)), 'exact_category_counts':category_counts,
            'exact_rank_digest':sha256(''.join(score_lines).encode()).hexdigest()})
        replays={}
        for seed in SEEDS:
            words=random_words(seed); hits=0; categories=[0]*9; points=[]
            for n in range(1,10001):
                a=draw_index(words,47); b=draw_index(words,46)
                if b>=a:b+=1
                runout=[remaining[a],remaining[b]]
                hit=bool(set(runout)&outs)
                hits+=hit
                score=values[tuple(sorted([a,b]))]
                categories[score[0]]+=1
                if n in CHECKPOINTS:
                    estimate=hits/n
                    points.append({'n':n,'hits':hits,'estimate':estimate,
                        'se':sqrt(estimate*(1-estimate)/n),'categories':categories.copy(),
                        'runout':runout,'rank':list(score)})
            replays[str(seed)]=points
        scenario['replays']=replays
    p=Fraction(378,1081)
    uncertainty=[{'n':n,'se':sqrt(float(p*(1-p))/n)} for n in [100,1000,10000]]
    pot={'pot_before_bet':100,'bet':25,'call':25,'reward':125,'final_pot':150,
         'reward_to_risk':5,'break_even':as_probability(Fraction(1,6)),
         'positive_ev':float(Fraction(1,5)*125-Fraction(4,5)*25),
         'negative_ev':float(Fraction(1,10)*125-Fraction(9,10)*25)}
    assert pot['positive_ev']==5 and pot['negative_ev']==-10
    sizes={'five_cards':comb(52,5),'seven_cards':comb(52,7),'subsets':comb(7,5),
           'flop_runouts':comb(47,2),'runout_and_opponent':comb(47,2)*comb(45,2)}
    assert sizes=={'five_cards':2598960,'seven_cards':133784560,'subsets':21,
                   'flop_runouts':1081,'runout_and_opponent':1070190}
    output={'method':'Exact enumeration plus independently reproduced seeded pseudorandom samples; no player-strength model.',
            'seeds':SEEDS,'checkpoints':CHECKPOINTS,'scenarios':scenarios,'pot_example':pot,
            'uncertainty':uncertainty,'state_sizes':sizes}
    (HERE/'verified_lab.json').write_text(json.dumps(output,indent=2,ensure_ascii=False)+'\n')
    print('Verified 5 × 1,081 runouts, their best-five ranks, 15 seeded 10,000-trial replays, pot EV, SE, and state sizes.')
    print('Wrote verified_lab.json')


if __name__=='__main__':main()
