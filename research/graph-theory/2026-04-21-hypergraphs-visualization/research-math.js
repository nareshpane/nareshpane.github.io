/* Exact, dependency-free research extensions. Browser and Node share these definitions. */
(function(root){
  'use strict';
  const M=typeof module!=='undefined'&&module.exports?require('./math.js'):root.HyperMath;
  const unique=xs=>[...new Set(xs)],intersection=(a,b)=>a.filter(x=>b.includes(x));
  const gallery=[
    M.create(['A','B','C','D'],[['A','B'],['B','C']]),
    M.create(['A','B','C','D','E'],[['A','B','C'],['B','D'],['C','D','E']]),
    M.create(['v1','v2','v3','v4','v5','v6','v7'],[['v1','v2','v3'],['v2','v3'],['v3','v5','v6'],['v4']])
  ];
  function atoms(h){const result=new Map();h.vertices.forEach(v=>{const mask=h.edges.reduce((m,e,i)=>m+(e.includes(v)?2**i:0),0);if(!result.has(mask))result.set(mask,[]);result.get(mask).push(v);});return [...result].sort((a,b)=>a[0]-b[0]).map(([mask,members])=>({mask,members}));}
  const pairSources=(h,pair)=>h.edges.flatMap((e,i)=>pair.every(v=>e.includes(v))?[i]:[]);
  const summary=h=>({order:h.vertices.length,size:h.edges.length,sizes:h.edges.map(e=>e.length),degrees:M.degrees(h),incidences:h.edges.reduce((s,e)=>s+e.length,0),pairs:M.twoSection(h),intersections:h.edges.flatMap((e,i)=>h.edges.slice(i+1).map((f,j)=>({i,j:i+j+1,members:intersection(e,f)}))).filter(x=>x.members.length)});
  function network(addresses,transactions){
    const vertices=unique(addresses);if(vertices.some(v=>typeof v!=='string'||!v.trim()))throw Error('Address labels must be nonempty strings.');
    const ids=new Set();const arcs=transactions.map((a,i)=>{const id=a.id||'E'+(i+1);if(ids.has(id))throw Error('Transaction IDs must be unique.');ids.add(id);const tail=unique(a.tail),head=unique(a.head);if(!tail.length||!head.length)throw Error('Each transaction needs at least one input and one output.');if([...tail,...head].some(v=>!vertices.includes(v)))throw Error('Every address must belong to the address set.');return {id,tail,head};});
    return {addresses:vertices,arcs};
  }
  const incidence=(n,role)=>n.addresses.map(v=>n.arcs.map(a=>+(a[role].includes(v))));
  const patterns=[['alpha','α','head','tail','Forward 2-chain'],['alphaPrime','α′','tail','head','Reverse 2-chain'],['beta','β','head','head','Inward 2-star'],['gamma','γ','tail','tail','Outward 2-star'],['L1','L₁','tail','head','Self-loop on E₁'],['L2','L₂','tail','head','Self-loop on E₂']];
  function motifs(a,b){const sets={alpha:intersection(a.head,b.tail),alphaPrime:intersection(a.tail,b.head),beta:intersection(a.head,b.head),gamma:intersection(a.tail,b.tail),L1:intersection(a.tail,a.head),L2:intersection(b.tail,b.head)};const active=patterns.filter(([k])=>sets[k].length).map(([k])=>k);return {sets,active,connected:['alpha','alphaPrime','beta','gamma'].some(k=>sets[k].length),pure:active.length===1};}
  function stb(a,b,exchanges=[]){
    const m=motifs(a,b),has=xs=>xs.some(v=>exchanges.includes(v)),only=keys=>m.active.length===keys.length&&keys.every(k=>m.active.includes(k));
    const forward=only(['alpha'])&&has(a.tail)&&has(b.head)&&!has(m.sets.alpha);
    const reverse=only(['alphaPrime'])&&has(a.head)&&has(b.tail)&&!has(m.sets.alphaPrime);
    const mixed=only(['alpha','alphaPrime']);
    const circularForward=mixed&&has(m.sets.alphaPrime)&&!has(m.sets.alpha);
    const circularReverse=mixed&&has(m.sets.alpha)&&!has(m.sets.alphaPrime);
    return {linear:forward!==reverse,circular:circularForward!==circularReverse,orientation:forward||circularForward?'E₁ → E₂':reverse||circularReverse?'E₂ → E₁':null,conditions:{forward,reverse,circularForward,circularReverse},motif:m};
  }
  const addressDegrees=n=>n.addresses.map(v=>({address:v,in:n.arcs.filter(a=>a.head.includes(v)).length,out:n.arcs.filter(a=>a.tail.includes(v)).length}));
  const allPairs=n=>n.arcs.flatMap((a,i)=>n.arcs.slice(i+1).map((b,j)=>({i,j:i+j+1,...motifs(a,b)})));
  // Distinct synthetic examples, never real addresses or blockchain records.
  const examples={
    one:{name:'Level 1 · One transaction',addresses:['A1','A2','A3','A4'],arcs:[{id:'E1',tail:['A1','A2'],head:['A3','A4']}],exchanges:[]},
    chain:{name:'Level 2 · A forward chain',addresses:['A1','A2','A3','A4','A5'],arcs:[{id:'E1',tail:['A1','A2'],head:['A3']},{id:'E2',tail:['A3'],head:['A4','A5']}],exchanges:[]},
    mixed:{name:'Level 3 · Reuse and simultaneous patterns',addresses:['A1','A2','A3','A4','A5','A6'],arcs:[{id:'E1',tail:['A1','A2'],head:['A2','A3']},{id:'E2',tail:['A2','A3'],head:['A1','A4']},{id:'E3',tail:['A4'],head:['A5','A6']}],exchanges:['A1','A6']},
    linear:{name:'Advanced · Linear STB',addresses:['X1','A1','A2','A3','X2'],arcs:[{id:'E1',tail:['X1','A1'],head:['A2','A3']},{id:'E2',tail:['A2','A3'],head:['X2']}],exchanges:['X1','X2']},
    circular:{name:'Advanced · Circular STB',addresses:['X1','A1','A2','A3'],arcs:[{id:'E1',tail:['X1','A1'],head:['A2','A3']},{id:'E2',tail:['A2','A3'],head:['X1']}],exchanges:['X1']}
  };
  const getExample=key=>{const e=examples[key];return {...network(e.addresses,e.arcs),exchanges:[...e.exchanges]};};
  // Fig. 2: exact consolidated quantities and identities in the supplied manuscript.
  const sourceFigure2={addresses:['src','a1','a2','a3','a4','a5','a6'],arcs:[{id:'T0',tail:['src'],head:['a3']},{id:'T1',tail:['src'],head:['a1','a2','a3']},{id:'T2',tail:['a2','a3'],head:['a3','a6']},{id:'T3',tail:['a1','a2'],head:['a4','a5']}],links:[['src','T0',50],['src','T1',50],['T0','a3',50],['T1','a1',10],['T1','a2',35],['T1','a3',5],['a1','T3',10],['a2','T3',7],['a2','T2',28],['a3','T2',55],['T2','a3',10],['T2','a6',73],['T3','a4',10],['T3','a5',7]]};
  const reportedRoles={L1:'1058M',L2:'1064M',alpha:'942M',alphaPrime:'927M',beta:'2147M',gamma:'1076M'};
  // Fig. 4 left: labels added to the ten otherwise unlabelled address circles.
  const sourceFigure4={addresses:['b1','b2','b3','c1','c2','c3','c4','d1','d2','d3'],arcs:[{id:'E1',tail:['b1','b2','b3','d1'],head:['b1','c2','c3','c4','d3']},{id:'E2',tail:['c1','c2','c3','b3'],head:['c1','d1','d2','d3','c3','b2']}],exchanges:['b1','b2','d1']};
  const api={gallery,atoms,pairSources,summary,intersection,network,incidence,patterns,motifs,stb,addressDegrees,allPairs,examples,getExample,sourceFigure2,sourceFigure4,reportedRoles};
  if(typeof module!=='undefined'&&module.exports)module.exports=api;else root.HyperResearchMath=api;
})(globalThis);
