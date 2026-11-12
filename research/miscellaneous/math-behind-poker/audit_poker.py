#!/usr/bin/env python3
"""Focused content/structure audit; no dependencies, network, or file mutations.

Run from any directory. verify_math.py and verify_lab.py supply the expensive
independent reference calculations; this script checks their rendered use.
"""
from collections import Counter
from fractions import Fraction
from html.parser import HTMLParser
from itertools import combinations
from math import comb, sqrt
from pathlib import Path
import json
import re
import subprocess
import sys
sys.dont_write_bytecode = True
from verify_math import rank_hand, decode

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
PAGE=HERE.parent/'math-behind-poker.html'
VOID={'meta','link','br','hr','img','input','source','wbr','area','base','col','embed','param','track'}


class Document(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root={'tag':'root','attrs':{},'children':[],'text':''}
        self.stack=[self.root];self.nodes=[]
    def handle_starttag(self,tag,attrs):
        node={'tag':tag,'attrs':dict(attrs),'children':[],'text':'','parent':self.stack[-1]}
        self.stack[-1]['children'].append(node);self.nodes.append(node)
        if tag not in VOID:self.stack.append(node)
    def handle_startendtag(self,tag,attrs):
        self.handle_starttag(tag,attrs)
        if tag not in VOID:self.handle_endtag(tag)
    def handle_endtag(self,tag):
        assert len(self.stack)>1 and self.stack[-1]['tag']==tag, f'Malformed close: {tag}'
        self.stack.pop()
    def handle_data(self,text):
        for node in self.stack:node['text']+=text
    def by_id(self,value):return next(n for n in self.nodes if n['attrs'].get('id')==value)


def descendants(node):
    for child in node['children']:
        yield child
        yield from descendants(child)


def has_class(node,name):return name in node['attrs'].get('class','').split()

def best(codes):return max(rank_hand(list(map(decode,h))) for h in combinations(codes,5))

def table_rows(node):
    return [[re.sub(r'\s+',' ',cell['text']).strip() for cell in row['children'] if cell['tag'] in ['th','td']]
            for row in descendants(node) if row['tag']=='tr']


def main():
    content=PAGE.read_text();doc=Document();doc.feed(content)
    assert len(doc.stack)==1
    ids=[n['attrs']['id'] for n in doc.nodes if 'id' in n['attrs']]
    assert len(ids)==len(set(ids)),'Duplicate IDs'
    for node in doc.nodes:
        for key in ['src','href']:
            url=node['attrs'].get(key,'')
            if not url or url.startswith(('https:','http:','data:')):continue
            assert url[1:] in ids if url.startswith('#') else (PAGE.parent/url).exists(),url
        if node['tag'] in ['select','input','progress']:
            identifier=node['attrs'].get('id')
            assert node['attrs'].get('aria-label') or any(n['tag']=='label' and n['attrs'].get('for')==identifier for n in doc.nodes),identifier
        if node['tag']=='button':
            assert node['text'].strip() and node['attrs'].get('type')=='button'
        if 'data-hand' in node['attrs']:
            codes=node['attrs']['data-hand'].split()
            assert len(codes)==len(set(codes)),codes
            for code in codes:decode(code)
    math_data=json.loads((HERE/'verified_math.json').read_text())
    lab=json.loads((HERE/'verified_lab.json').read_text())
    formulas=[10*4,13*48,13*comb(4,3)*12*comb(4,2),4*comb(13,5)-40,
              10*4**5-40,13*comb(4,3)*comb(12,2)*4**2,
              comb(13,2)*comb(4,2)**2*11*4,13*comb(4,2)*comb(12,3)*4**3,
              (comb(13,5)-10)*(4**5-4)]
    total=comb(52,5);assert total==2598960==sum(formulas)
    table=next(n for n in doc.nodes if has_class(n,'frequency-table'))
    rows=table_rows(table)[1:-1]
    for row,count,category in zip(rows,formulas,math_data['categories']):
        assert count==category['count']==int(row[1].replace(',',''))
        assert row[2]==f'{100*count/total:.6f}%'
        assert row[3]==f'{total/count:,.2f}'
    assert len(rows)==9 and math_data['royal_flushes']==4
    ladder=next(n for n in doc.nodes if has_class(n,'hand-ladder'))
    hands=[n['attrs']['data-hand'].split() for n in descendants(ladder) if 'data-hand' in n['attrs']]
    scores=[rank_hand(list(map(decode,h))) for h in hands]
    assert [score[0] for score in scores]==[8,8,7,6,5,4,3,2,1,0]
    assert scores[0]==(8,14) and all(a>b for a,b in zip(scores,scores[1:]))
    for n in doc.nodes:
        a=n['attrs']
        if 'data-hole' in a:
            available=(a['data-hole']+' '+a['data-board']).split()
            chosen=a['data-best'].split()
            assert len(set(available))==7 and set(chosen)<=set(available)
            assert rank_hand(list(map(decode,chosen)))==best(available)
        if has_class(n,'tie-row'):
            h=[d['attrs']['data-hand'].split() for d in descendants(n) if 'data-hand' in d['attrs']]
            assert len(h)==2 and rank_hand(list(map(decode,h[0])))>rank_hand(list(map(decode,h[1])))
    assert rank_hand(list(map(decode,'As 2d 3c 4h 5s'.split())))==(4,5)
    assert rank_hand(list(map(decode,'Qs Kd Ac 2h 3s'.split())))[0]==0
    for identifier,tie in [('board-plays',True),('kicker',False)]:
        hs=[n['attrs']['data-hand'].split() for n in descendants(doc.by_id(identifier)) if 'data-hand' in n['attrs']]
        board,a,b=hs;assert len(board)==5 and len(a)==len(b)==2 and len(set(board+a+b))==9
        if tie:assert best(board+a)==best(board+b)==rank_hand(list(map(decode,board)))
        else:assert best(board+a)>best(board+b)
    assert len(math_data['subsets'])==comb(7,5)==21
    approximate=next(n for n in doc.nodes if has_class(n,'approx-table'))
    for row,o in zip(table_rows(approximate)[1:],[9,8,4,2]):
        turn=Fraction(o,47);river=1-Fraction(comb(47-o,2),comb(47,2))
        assert row==[str(o),f'{100*turn:.3f}%',f'{2*o}%',f'{100*river:.3f}%',f'{4*o}%']
    assert round(float(Fraction(900,47)-18),3)==1.149
    assert round(float(36-Fraction(37800,1081)),3)==1.032
    assert sum(Fraction(item['fraction']) for item in math_data['flush_tree'].values())==1
    assert Fraction(3,4)*Fraction(1,5)/(Fraction(3,4)*Fraction(1,5)+Fraction(1,4)*Fraction(4,5))==Fraction(3,7)
    assert Fraction(3,10)*20-Fraction(7,10)*5==Fraction(5,2)
    for s in lab['scenarios']:
        o=len(s['outs']);assert o==s['out_count']
        assert Fraction(s['by_river_flop']['fraction'])==1-Fraction(comb(47-o,2),1081)
        assert Fraction(s['next_flop']['fraction'])==Fraction(o,47)
        assert Fraction(s['next_turn']['fraction'])==Fraction(o,46)
        assert sum(s['exact_category_counts'])==1081
        for replay in s['replays'].values():
            for point in replay:
                assert point['estimate']==point['hits']/point['n']
                assert sum(point['categories'])==point['n']
                assert abs(point['se']-sqrt(point['estimate']*(1-point['estimate'])/point['n']))<1e-14
    default=next(p for p in lab['scenarios'][0]['replays']['20260926'] if p['n']==1000)
    assert doc.by_id('sim-hits')['text']==str(default['hits'])
    assert doc.by_id('sim-estimate')['text']==f'{100*default["estimate"]:.3f}%'
    assert doc.by_id('sim-se')['text']==f'{100*default["se"]:.3f} pp'
    for row,point in zip(table_rows(doc.by_id('sim-checkpoints')),[p for p in lab['scenarios'][0]['replays']['20260926'] if p['n']<=1000]):
        assert row==[f'{point["n"]:,}',str(point['hits']),f'{point["estimate"]*100:.3f}%',f'{point["se"]*100:.3f}']
    for row,i in zip(table_rows(doc.by_id('sim-categories')),range(8,-1,-1)):
        assert row[1:]==[str(default['categories'][i]),f'{100*default["categories"][i]/1000:.3f}%',f'{100*lab["scenarios"][0]["exact_category_counts"][i]/1081:.3f}%']
    assert Fraction(25,100+25+25)==Fraction(1,6)
    assert Fraction(1,5)*125-Fraction(4,5)*25==5
    assert Fraction(1,10)*125-Fraction(9,10)*25==-10
    assert comb(52,7)==133784560 and comb(47,2)*comb(45,2)==1070190
    for point in lab['uncertainty']:
        assert abs(point['se']-sqrt(float(Fraction(378,1081)*Fraction(703,1081))/point['n']))<1e-14
    # Compare existing index entries against HEAD; no staging or history changes.
    index=(ROOT/'research.html').read_text()
    listing=re.search(r'<ul class="research-list">(.*?)</ul>',index,re.S)[1]
    entries=re.findall(r'<li>.*?</li>',listing,re.S)
    poker=[entry for entry in entries if 'research/math-behind-poker.html' in entry]
    assert len(poker)==1 and entries[0]==poker[0]
    old=subprocess.run(['git','show','HEAD:research.html'],cwd=ROOT,check=True,capture_output=True,text=True).stdout
    old_listing=re.search(r'<ul class="research-list">(.*?)</ul>',old,re.S)[1]
    old_entries=re.findall(r'<li>.*?</li>',old_listing,re.S)
    assert entries[1:]==old_entries,'Existing research entries changed'
    assert index.replace('        '+poker[0]+'\n\n','',1)==old,'Unrelated index changes'
    assert 'KICK_' not in content and '</div>_A' not in content
    print('PASS: hand counts, probabilities, 1-in-N values, all poker examples, outs, trees, approximations, Bayes, EV, simulation checkpoints, SE, and state sizes.')
    print('PASS: HTML nesting, IDs, navigation, local assets, control labels, and exactly one first index entry; prior index bytes preserved.')


if __name__=='__main__':main()
