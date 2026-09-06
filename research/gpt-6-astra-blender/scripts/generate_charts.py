"""Publication figures from actual city/experiment values; static matplotlib output."""
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/astra-city-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
C=json.loads((ROOT/'data/city.json').read_text())
A=json.loads((ROOT/'data/analysis.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
    'axes.spines.right':False,'axes.facecolor':'#fffdf9','figure.facecolor':'#fffdf9',
    'axes.edgecolor':'#bfb7a8','text.color':'#263b3e','axes.labelcolor':'#263b3e'})
def save(name):
    plt.savefig(ROOT/'renders'/f'{name}.png',dpi=160,bbox_inches='tight');plt.close()

fig,ax=plt.subplots(1,3,figsize=(13,3.5),layout='constrained')
ax[0].hist([b['height'] for b in C['buildings']],bins=12,color='#2d777b',edgecolor='#fffdf9')
ax[0].set(xlabel='Facade height parameter (m)',ylabel='Buildings',title='204 height parameters')
ks=sorted(int(k) for k in A['degree_distribution'])
ax[1].bar(ks,[A['degree_distribution'][str(k)] for k in ks],color='#c48143');ax[1].set(xticks=ks,xlabel='Road degree',ylabel='Intersections',title=f"{A['nodes']} junction degrees")
for i,hood in enumerate(['Old Quay','Garden Quarter','Civic Terrace','Innovation District']):
    heights=[b['height'] for b in C['buildings'] if b['neighbourhood']==hood]
    ax[2].barh(i,np.mean(heights),color=['#b76042','#609176','#bba062','#477b89'][i])
ax[2].set(yticks=range(4),yticklabels=['Old Quay','Garden Quarter','Civic Terrace','Innovation District'],xlabel='Mean facade height (m)',title='Neighbourhood variation')
save('city_statistics')

fig,ax=plt.subplots(1,3,figsize=(13,3.6),layout='constrained')
mc=A['monte_carlo'];xs=[r['p']*100 for r in mc];ys=[r['probability_disconnected']*100 for r in mc]
yerr=[[y-r['wilson_95'][0]*100 for r,y in zip(mc,ys)],[r['wilson_95'][1]*100-y for r,y in zip(mc,ys)]]
ax[0].errorbar(xs,ys,yerr=yerr,fmt='o-',capsize=5,color='#b75c40')
ax[0].set(ylabel='Any-node disconnection probability (%)',title='400 trials per probability\n95% Wilson intervals')
ax[1].plot(xs,[r['mean_route_increase_pct_conditional'] for r in mc],'o-',color='#347d87')
ax[1].set(ylabel='Mean route increase (%)',title='Example route detour\nConditional on connected endpoints')
ax[2].plot(xs,[r['mean_accessibility_loss_pct'] for r in mc],'o-',color='#967239')
ax[2].set(ylabel='Mean accessibility loss (%)',title=f"All {A['nodes']} origins\nUnreachable destinations contribute zero")
for a in ax:a.set_xlabel('Independent edge-failure probability (%)');a.set_xticks(xs);a.grid(alpha=.15)
save('monte_carlo')

fig,ax=plt.subplots(1,2,figsize=(11,4),layout='constrained')
ax[0].scatter(A['accessibility_before'],A['accessibility_after'],c=[n['x'] for n in C['nodes']],cmap='BrBG',s=35,edgecolor='#fffdf9')
lim=[min(A['accessibility_before'])*.97,max(A['accessibility_after'])*1.03]
ax[0].plot(lim,lim,'--',color='#8d8b80');ax[0].set(xlabel='Before: synthetic opportunity units',ylabel='After: synthetic opportunity units',title='Each dot is one origin node')
gains=A['accessibility_gain_pct'];n=C['nodes'];sc=ax[1].scatter([i['x'] for i in n],[i['y'] for i in n],c=gains,cmap='YlOrRd',s=80)
ax[1].axvspan(-17,17,color='#d4e7e7');ax[1].set(xlabel='Local x (m)',ylabel='Local y (m)',title='Accessibility increase by origin (%)',aspect='equal');fig.colorbar(sc,ax=ax[1],label='Increase (%)')
save('accessibility_statistics')

fig,ax=plt.subplots(figsize=(11,5),layout='constrained')
ax.axvspan(-17,17,color='#d1e5e6',label='River')
for e in C['roads']:
    a,b=[C['nodes'][i] for i in [e['u'],e['v']]]
    ax.plot([a['x'],b['x']],[a['y'],b['y']],color='#9baba9',lw=2,zorder=1)
for key,color,label in [('closure_route','#bf573e',f"After closure: {A['closure_route']['metres']:.2f} m"),('route','#cba02f',f"Baseline: {A['route']['metres']:.2f} m")]:
    nodes=[C['nodes'][i] for i in A[key]['nodes']]
    ax.plot([n['x'] for n in nodes],[n['y'] for n in nodes],color=color,lw=4,label=label,zorder=3)
for n in C['nodes']:
    ax.scatter(n['x'],n['y'],s=55,color='#255563',zorder=4)
    ax.annotate(str(n['id']),(n['x']+4,n['y']+4),fontsize=8)
ax.scatter(0,48,marker='x',s=170,color='#a42e24',zorder=5)
ax.set(xlabel='Synthetic local x (metres)',ylabel='Synthetic local y (metres)',aspect='equal',title='The exact road graph: node IDs and recomputed routes')
ax.legend(loc='upper center',bbox_to_anchor=(.5,-.18),ncol=3,fontsize=9);save('graph_map')
print('CHARTS_SUCCESS')
