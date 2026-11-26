"use strict";
(() => {
  const {fmt,esc} = Atlas;
  const tool = text => '<div class="toolbar">' + text + '</div>';
  const svg = (title, content) => '<svg class="diagram" viewBox="0 0 560 205" role="img" aria-label="' +
    esc(title) + '">' + content + '</svg>';
  const output = '<p class="caption-text" aria-live="polite"></p>';

  function geometry() {
    const node = document.querySelector("#geometry-gallery");
    const vertex = (x,y) => '<circle class="vertex" cx="'+x+'" cy="'+y+'" r="4.5"/>';
    const label = (x,y,text) => '<text x="'+x+'" y="'+y+'" text-anchor="middle">'+text+'</text>';
    const examples = {
      "Point": ['<path class="guide" d="M125 90H272V153"/>'+vertex(272,90)+label(310,82,'p = (x, y)'),
        "p = (x, y)", "A hospital location is one coordinate pair."],
      "MultiPoint": [vertex(146,111)+vertex(271,58)+vertex(400,115)+label(146,137,'p₁')+label(271,44,'p₂')+label(400,141,'p₃'),
        "M = {p₁, p₂, p₃}", "One feature can group distinct facility candidates as disconnected points."],
      "LineString": ['<path class="trace" d="M81 127L173 64 281 107 385 50 475 98"/>'+
        [[81,127],[173,64],[281,107],[385,50],[475,98]].map(p=>vertex(...p)).join("")+label(81,150,'p₁')+label(477,123,'p₅'),
        "L = (p₁, p₂, p₃, p₄, p₅)", "A road centreline traces segments between ordered vertices."],
      "MultiLineString": ['<path class="trace" d="M77 118L148 54 224 87M302 123L396 60 480 127"/>'+
        [[77,118],[148,54],[224,87],[302,123],[396,60],[480,127]].map(p=>vertex(...p)).join("")+
        label(151,150,'LineString L₁')+label(398,150,'LineString L₂'),
        "M = {L₁, L₂}", "After clipping a road, the retained pieces may form disconnected LineStrings."],
      "Polygon": ['<path class="surface" d="M105 61L362 37 443 69 397 147 143 144Z M252 67L307 66 320 106 248 110Z" fill-rule="evenodd"/>'+
        label(283,94,'Hole')+label(183,111,'Interior')+label(467,55,'Exterior')+label(271,143,'Boundary'),
        "P = surface bounded by an outer ring and a hole", "A polygon’s boundary encloses its interior. An inner ring separates a hole from the polygon."],
      "MultiPolygon": ['<path class="surface" d="M76 59L195 45 227 122 111 139Z M314 64L434 38 488 117 400 141 316 127Z"/>'+
        label(145,94,'P₁')+label(399,100,'P₂'),
        "M = {P₁, P₂}", "The union of separated wildfire patches can have multiple polygon components."],
      "GeometryCollection": ['<path class="surface" d="M55 95L117 49 202 64 187 136 93 140Z"/><path class="trace" d="M249 127L321 61 376 89"/>'+
        vertex(473,87)+label(132,116,'Polygon P')+label(314,149,'LineString L')+label(473,114,'Point p'),
        "G = {P, L, p}", "A geometry collection may contain components of different dimensions."],
    };
    node.innerHTML = '<h4>Geometry types</h4><p class="geometry-instruction" id="geometry-instruction">Choose a geometry type to see its structure and an example of how it is used.</p>' +
      tool('<label>Geometry type<select aria-describedby="geometry-instruction">' +
      Object.keys(examples).map(k=>'<option>'+k+'</option>').join("")+'</select></label>') +
      '<div class="geometry-drawing"></div><p class="geometry-notation" aria-live="polite"></p>' + output;
    const select = node.querySelector("select");
    const draw = () => {
      const [shape, notation, caption] = examples[select.value];
      node.querySelector(".geometry-drawing").innerHTML = '<svg class="diagram" viewBox="0 0 560 179" role="img" aria-label="'+esc(select.value)+'">' +
        '<defs><pattern id="geometry-grid" width="24" height="24" patternUnits="userSpaceOnUse"><path d="M24 0H0V24" fill="none" stroke="#e3d2c4"/></pattern></defs>' +
        '<rect x="17" y="16" width="526" height="146" rx="4" fill="url(#geometry-grid)"/>'+shape+'</svg>';
      node.querySelector(".geometry-notation").textContent = notation;
      node.querySelector(".caption-text").textContent = caption;
    };
    select.onchange=draw; draw();
  }

  function predicates() {
    const node=document.querySelector("#predicate-lab");
    node.innerHTML='<h4>Move B relative to A</h4>'+tool('<label>Geometry B<select id="b-kind"><option>Rectangle</option><option>Line</option></select></label>'+
      '<label>Position<input id="b-position" type="range" min="15" max="365" value="224"></label>'+
      '<label>Construct geometry<select id="b-operation"><option>Inputs</option><option>Intersection</option><option>Union</option><option>Difference A − B</option><option>Buffer A</option></select></label>')+'<div></div>'+output;
    const kind=node.querySelector("#b-kind"), position=node.querySelector("#b-position"), op=node.querySelector("#b-operation");
    const render=()=>{
      const x=Number(position.value), rect=kind.value==="Rectangle";
      const ax=150, ay=43, aw=195, ah=120, edge=ax+aw, bw=104, by=76, bh=54;
      const overlap=Math.min(edge,x+bw)-Math.max(ax,x);
      const within=rect&&x>=ax&&x+bw<=edge;
      const predicates=[["intersects",overlap>=0],["touches",overlap===0],["within (B,A)",within||!rect&&x>=ax&&x+bw<=edge],
        ["contains (A,B)",within||!rect&&x>=ax&&x+bw<=edge],["overlaps",rect&&overlap>0&&!within],
        ["crosses",!rect&&overlap>0&&(x<ax||x+bw>edge)]];
      const a='<rect x="'+ax+'" y="'+ay+'" width="'+aw+'" height="'+ah+'" rx="0" fill="#afcfbd" stroke="#456f5b" stroke-width="2"/>';
      const b=rect?'<rect x="'+x+'" y="'+by+'" width="'+bw+'" height="'+bh+'" fill="#e6b998" fill-opacity=".75" stroke="#af734f" stroke-width="2"/>':
        '<path d="M'+x+' 103h'+bw+'" stroke="#aa6549" stroke-width="4"/>';
      const intercept=overlap>0?rect?'<rect x="'+Math.max(x,ax)+'" y="'+by+'" width="'+overlap+'" height="'+bh+'" fill="#416c59"/>':
        '<path d="M'+Math.max(x,ax)+' 103h'+overlap+'" stroke="#416c59" stroke-width="4"/>':overlap===0?
        rect?'<path d="M'+Math.max(x,ax)+' '+by+'v'+bh+'" stroke="#416c59" stroke-width="3"/>':
          '<circle cx="'+Math.max(x,ax)+'" cy="103" r="5" fill="#416c59"/>':"";
      const cut=rect?'<rect x="'+x+'" y="'+by+'" width="'+bw+'" height="'+bh+'" fill="black"/>':"";
      const extraLeft = x < ax ? '<path d="M'+x+' 103H'+Math.min(x+bw,ax)+'" stroke="#456f5b" stroke-width="4"/>' : '';
      const extraRight = x+bw > edge ? '<path d="M'+Math.max(x,edge)+' 103H'+(x+bw)+'" stroke="#456f5b" stroke-width="4"/>' : '';
      let union = a;
      if (!rect) union += extraLeft+extraRight;
      else if (overlap < 0) union += b.replace("#e6b998","#afcfbd").replace("#af734f","#456f5b");
      else if (!within) {
        const outer = x < ax ? 'M150 43H345V163H150V130H'+x+'V76H150Z' :
          'M150 43H345V76H'+(x+bw)+'V130H345V163H150Z';
        union = '<path d="'+outer+'" fill="#afcfbd" stroke="#456f5b" stroke-width="2"/>';
      }
      const shapes={
        Inputs:a+b, Intersection:'<g opacity=".24">'+a+b+'</g>'+intercept,
        Union:union,
        "Difference A − B":'<defs><mask id="difference-cut"><rect width="560" height="205" fill="white"/>'+cut+'</mask></defs><g mask="url(#difference-cut)">'+a+'</g>',
        "Buffer A":'<rect x="126" y="19" width="243" height="168" rx="24" fill="#d5e7dc" stroke="#648973" stroke-dasharray="4 3"/>'+a
      };
      node.querySelector("div:nth-of-type(2)").innerHTML=svg(op.value,shapes[op.value]+
        '<text x="158" y="62">A</text><text x="'+(x+4)+'" y="97">B</text>');
      node.querySelector("p").textContent=predicates.map(([name,good])=>name+" "+(good?"true":"false")).join(" · ")+
        " · distance "+Math.max(0,-overlap)+" schematic units.";
    };
    node.addEventListener("input",render);render();
  }

  function index() {
    const node=document.querySelector("#index-lab");
    node.innerHTML='<h4>Bounding-box candidates and exact hits</h4>'+tool('<label>Query position<input type="range" min="34" max="525" value="130"></label>')+'<div></div>'+output;
    const triangles=[[80,63,90],[124,100,79],[242,66,80],[315,84,105],[417,52,78]];
    const slider=node.querySelector("input");
    function draw() {
      const x=Number(slider.value), y=92, r=18;
      let candidates=0, hits=0;
      const marks=triangles.map(([a,b,v])=>{
        const c=x+r>=a&&x-r<=a+v&&y+r>=b&&y-r<=b+v;
        const distSegment=(px,py,qx,qy)=>{
          const vx=qx-px,vy=qy-py,t=Math.max(0,Math.min(1,((x-px)*vx+(y-py)*vy)/(vx*vx+vy*vy)));
          return Math.hypot(x-px-t*vx,y-py-t*vy);
        };
        const inside=x>=a&&y>=b&&x+y<=a+b+v;
        const exact=c&&(inside||Math.min(distSegment(a,b,a+v,b),distSegment(a+v,b,a,b+v),distSegment(a,b+v,a,b))<=r);
        candidates+=Number(c);hits+=Number(exact);
        return '<rect x="'+a+'" y="'+b+'" width="'+v+'" height="'+v+'" stroke="'+(c?"#b27642":"#b3c4ba")+
          '" fill="none" stroke-dasharray="4 3"/><path d="M'+a+' '+b+'h'+v+'l-'+v+' '+v+'Z" fill="'+
          (exact?"#4e8b76":c?"#e6c393":"#d5e0d9")+'"/>';
      }).join("");
      node.querySelector("div:nth-of-type(2)").innerHTML=svg("Indexed triangles and a moving circular query",marks+
        '<rect x="'+(x-r)+'" y="'+(y-r)+'" width="36" height="36" fill="none" stroke="#426c82" stroke-dasharray="3 2"/>'+
        '<circle cx="'+x+'" cy="'+y+'" r="'+r+'" fill="#83acc285" stroke="#315a71" stroke-width="2"/>');
      node.querySelector("p").textContent=candidates+" box candidates · "+hits+" exact hits · "+(candidates-hits)+" false-positive candidates.";
    }
    slider.oninput=draw;draw();
  }

  function probability() {
    const node=document.querySelector("#probability-lab");
    node.innerHTML='<h4>Flood magnitude over a planning horizon</h4>'+tool(
      '<label>Return period (years)<select>'+[20,50,100,200,500].map(n=>'<option '+(n===100?'selected':'')+'>'+n+'</option>').join("")+
      '</select></label><label>Planning horizon<input type="range" min="1" max="100" value="30"></label>')+output;
    const render=()=>{
      const t=Number(node.querySelector("select").value), years=Number(node.querySelector("input").value);
      const probability=100*(1-(1-1/t)**years);
      node.querySelector("p").textContent=years+" years · annual probability "+fmt(100/t,1)+"% · probability of at least one exceedance "+fmt(probability,1)+"%.";
    };
    node.addEventListener("input",render); render();
  }

  function network() {
    const node=document.querySelector("#network-lab");
    const coordinates={A:[53,106],B:[173,52],C:[176,155],D:[325,57],E:[327,151],F:[493,107]};
    const edges=[["A","B",3],["A","C",5],["B","C",3],["B","D",3],["C","E",3],["D","E",4],["D","F",2],["E","F",4]];
    node.innerHTML='<h4>Illustrative network · shortest path from A to F</h4>'+tool('<button type="button" data-step-path>Settle next node</button>'+
      '<button type="button" data-reset-path>Restart graph</button><label><input type="checkbox"> Close B–D</label>')+'<div></div>'+output;
    let settled,distance,parent,current;
    const close=node.querySelector("input");
    const key=(a,b)=>[a,b].sort().join("");
    const render=()=>{
      const route=new Set();
      if(settled.has("F")) for(let v="F";parent[v];v=parent[v]) route.add(key(v,parent[v]));
      const links=edges.map(([a,b,c])=>{
        const [x1,y1]=coordinates[a],[x2,y2]=coordinates[b], cut=close.checked&&key(a,b)==="BD";
        return '<line x1="'+x1+'" y1="'+y1+'" x2="'+x2+'" y2="'+y2+'" stroke="'+(cut?"#bc7f68":route.has(key(a,b))?"#396b57":"#acbcb2")+
          '" stroke-width="'+(route.has(key(a,b))?6:3)+'" '+(cut?'stroke-dasharray="6 5"':'')+'/><text x="'+
          ((x1+x2)/2+8)+'" y="'+((y1+y2)/2-6)+'">'+c+'</text>';
      }).join("");
      const dots=Object.entries(coordinates).map(([k,[x,y]])=>'<circle cx="'+x+'" cy="'+y+'" r="15" fill="'+
        (settled.has(k)?"#82b5a0":"#e1eae4")+'" stroke="#547c6d"/><text x="'+x+'" y="'+(y+4)+'" text-anchor="middle">'+k+'</text>').join("");
      node.querySelector("div:nth-of-type(2)").innerHTML=svg("Educational graph with weighted edges",links+dots);
      node.querySelector("p").textContent=(settled.has("F")?"Shortest cost "+distance.F+".":current?"Settled "+current+".":"Start at A.")+
        " Tentative costs: "+Object.keys(coordinates).map(k=>k+" "+(Number.isFinite(distance[k])?distance[k]:"∞")).join(" · ");
      node.querySelector("[data-step-path]").disabled=settled.has("F");
    };
    function reset(){settled=new Set();distance=Object.fromEntries(Object.keys(coordinates).map(k=>[k,k==="A"?0:Infinity]));parent={};current=null;render();}
    node.querySelector("[data-step-path]").onclick=()=>{
      current=Object.keys(coordinates).filter(k=>!settled.has(k)).sort((a,b)=>distance[a]-distance[b])[0];
      settled.add(current);
      for(const [a,b,weight] of edges){
        if(close.checked&&key(a,b)==="BD")continue;
        const other=a===current?b:b===current?a:null;
        if(other&&distance[current]+weight<distance[other]){distance[other]=distance[current]+weight;parent[other]=current;}
      }
      render();
    };
    node.querySelector("[data-reset-path]").onclick=reset;close.onchange=reset;reset();
  }
  for (const render of [geometry,predicates,index,probability,network]) render();
})();

