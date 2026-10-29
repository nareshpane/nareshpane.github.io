/* Reusable SVG primitives for exact incidence and source-figure reconstructions. */
window.ResearchSVG=(()=>{
  'use strict';
  let serial=0;
  const escape=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const text=(x,y,label,cls='',anchor='middle')=>`<text x="${x}" y="${y}" text-anchor="${anchor}" class="${cls}">${escape(label)}</text>`;
  const svg=(content,label,w=300,h=300,cls='research-svg')=>`<svg class="${cls}" viewBox="0 0 ${w} ${h}" role="group" aria-label="${escape(label)}"><title>${escape(label)}</title>${content}</svg>`;
  const colours=['#ac653b','#3e6f9e','#477b39','#885792'];
  const roles={T1:'#b34036',H1:'#1677a2',T2:'#278447',H2:'#854ca6'};
  const roleNames={T1:'Inputs E₁',H1:'Outputs E₁',T2:'Inputs E₂',H2:'Outputs E₂'};
  function graph(nodes,links,{width=560,height=340,label='Directed incidence graph',prefix='graph',exchanges=[],active=null}={}){
    const id=prefix+'-'+(++serial),cs=[...new Set(links.map(l=>l.colour||'#527261'))];
    let out='<defs>'+cs.map((c,i)=>`<marker id="${id}-a${i}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10Z" fill="${c}"/></marker>`).join('')+'</defs>';
    const index=new Map(nodes.map(n=>[n.id,n]));
    links.forEach(l=>{const a=index.get(l.from),b=index.get(l.to);if(!a||!b)throw Error('Unknown figure endpoint');const dx=b.x-a.x,dy=b.y-a.y,d=Math.hypot(dx,dy)||1,nx=-dy/d,ny=dx/d,bend=l.bend||0;
      const start=(a.radius||18)+1,end=(b.radius||18)+3,x1=a.x+dx/d*start,y1=a.y+dy/d*start,x2=b.x-dx/d*end,y2=b.y-dy/d*end,cx=(x1+x2)/2+nx*bend,cy=(y1+y2)/2+ny*bend,c=l.colour||'#527261';
      out+=`<g class="directed-link" data-from="${escape(l.from)}" data-to="${escape(l.to)}" data-role="${l.role||''}" data-phase="${l.phase??0}" ${l.weight!==undefined?`data-weight="${l.weight}"`:''}><path d="M${x1} ${y1} Q${cx} ${cy} ${x2} ${y2}" fill="none" stroke="${c}" stroke-width="${active===l.role?3.6:1.9}" marker-end="url(#${id}-a${cs.indexOf(c)})"/>${l.label!==undefined||l.weight!==undefined?text((x1+x2)/2+nx*bend*.5,(y1+y2)/2+ny*bend*.5-6,l.label??l.weight,'quantity'):''}</g>`;
    });
    nodes.forEach(n=>{const exchange=exchanges.includes(n.id),fill=exchange?'#253a32':n.fill||'#fffdf9',stroke=n.stroke||'#527261';out+=`<g class="directed-node" data-node="${escape(n.id)}" data-type="${n.type||'address'}">`+(n.type==='transaction'?`<rect x="${n.x-17}" y="${n.y-15}" width="34" height="30" rx="3" fill="#fff1de" stroke="#a3653a" stroke-width="2"/>`:`<circle cx="${n.x}" cy="${n.y}" r="${n.radius||18}" fill="${fill}" stroke="${stroke}" stroke-width="2"/>`)+text(n.x,n.y+5,n.label||n.id,exchange?'exchange-label':'')+'</g>';});
    return svg(out,label,width,height,'research-svg directed-svg');
  }
  function network(n,{exchanges=[],label='Synthetic address-level directed hypergraph',active=null}={}){
    const height=Math.max(300,n.addresses.length*47+48,n.arcs.length*76+60),nodes=n.addresses.map((v,i)=>({id:v,x:72,y:38+i*(height-76)/Math.max(1,n.addresses.length-1),type:'address'}));
    n.arcs.forEach((a,j)=>nodes.push({id:a.id,x:420,y:50+j*(height-100)/Math.max(1,n.arcs.length-1),type:'transaction'}));
    const links=n.arcs.flatMap((a,j)=>a.tail.map(v=>({from:v,to:a.id,colour:j===0?roles.T1:roles.T2,role:'T'+(j+1),phase:j*2+1,bend:-10})).concat(a.head.map(v=>({from:a.id,to:v,colour:j===0?roles.H1:roles.H2,role:'H'+(j+1),phase:j*2+2,bend:-10}))));
    return graph(nodes,links,{width:500,height,label,exchanges,active});
  }
  function hull(points){const cross=(o,a,b)=>(a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0]);const ps=[...points].sort((a,b)=>a[0]-b[0]||a[1]-b[1]),lower=[],upper=[];for(const p of ps){while(lower.length>=2&&cross(lower.at(-2),lower.at(-1),p)<=0)lower.pop();lower.push(p);}for(const p of ps.toReversed()){while(upper.length>=2&&cross(upper.at(-2),upper.at(-1),p)<=0)upper.pop();upper.push(p);}return lower.slice(0,-1).concat(upper.slice(0,-1));}
  function contour(points,pad){return hull(points.flatMap(([x,y])=>Array.from({length:32},(_,i)=>[x+pad*Math.cos(i*Math.PI/16),y+pad*Math.sin(i*Math.PI/16)])));}
  const polygon=points=>'M'+points.map(p=>p.join(',')).join(' L')+' Z';
  return {escape,text,svg,colours,roles,roleNames,graph,network,hull,contour,polygon};
})();
