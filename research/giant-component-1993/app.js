/* Two experiments, one exact edge-evolution implementation. No graph libraries. */
'use strict';
const GC = (() => {
  function random(seed) {
    let state = Number(seed) >>> 0;
    const uint = () => {state = (state + 0x6D2B79F5) >>> 0; let t = state;
      t = Math.imul(t ^ t >>> 15, t | 1); t ^= t + Math.imul(t ^ t >>> 7, t | 61);
      return (t ^ t >>> 14) >>> 0;};
    return n => {const limit = Math.floor(4294967296 / n) * n; let x;
      do {x = uint();} while (x >= limit); return x % n;};
  }
  class DSU {
    constructor(n){this.p=Array.from({length:n},(_,i)=>i);this.v=Array(n).fill(1);this.e=Array(n).fill(0);this.count=n;this.complex=0;this.largest=1;}
    find(u){while(this.p[u]!==u){this.p[u]=this.p[this.p[u]];u=this.p[u];}return u;}
    add(u,v){let a=this.find(u),b=this.find(v);this.complex-=Number(this.e[a]>this.v[a]);
      if(a!==b){this.complex-=Number(this.e[b]>this.v[b]);if(this.v[a]<this.v[b])[a,b]=[b,a];
        this.p[b]=a;this.v[a]+=this.v[b];this.e[a]+=this.e[b]+1;this.count--;
      }else this.e[a]++;
      this.complex+=Number(this.e[a]>this.v[a]);this.largest=Math.max(this.largest,this.v[a]);}
    beta(u){const a=this.find(u);return this.e[a]-this.v[a]+1;}
  }
  function evolution(n,model,seed){const draw=random(seed),seen=new Set(),edges=[],d=new DSU(n);let firstFailure=null;
    while(d.count>1){let u=draw(n),v=draw(n);
      if(model==='simple'){if(u===v)continue;if(u>v)[u,v]=[v,u];const key=u*n+v;if(seen.has(key))continue;seen.add(key);}
      edges.push([u,v]);d.add(u,v);if(firstFailure===null&&d.complex>=2)firstFailure=edges.length;}
    return {edges,firstFailure};}
  const colours=['#237660','#2564a8','#9b6510','#9c3d78'];
  function colour(beta){return colours[Math.min(3,beta)];}
  const demos=[];
  class Demo {
    constructor(root){this.root=root;this.experiment=root.dataset.experiment;this.timer=null;this.m=0;this.generated=false;
      root.innerHTML=`<div class="controls">
      <label>Vertices (10–200)<input data-field="n" type="number" min="10" max="200" step="1" value="60"></label>
      <label>Seed (uint32)<input data-field="seed" type="number" min="0" max="4294967295" step="1" value="1993"></label>
      <label>Graph model<select data-field="model"><option value="simple">Simple graph</option><option value="multigraph">Multigraph</option></select></label>
      <label>Edges / second<input data-field="speed" type="number" min="1" max="60" value="8"></label>
      <label>Accepted edges<input data-field="m" type="number" min="0" value="0"></label></div>
      <div class="actions">${['Generate','Play','Pause','Next Edge','Previous Edge','Restart','Reset','Go to critical'].map(x=>`<button type="button" data-action="${x}">${x}</button>`).join('')}</div>
      <label class="timeline">Evolution timeline<input data-field="timeline" type="range" min="0" max="100" value="0"></label>
      <div class="critical-rail" aria-hidden="true"><span data-field="criticalMarker">│ critical</span></div>
      <p class="small" data-field="critical"></p>
      <label class="check"><input data-field="autoPause" type="checkbox" checked> Pause when two complex components first coexist</label>
      <div class="canvas-wrap"><canvas role="img" aria-label="Random graph components, coloured by cyclomatic number"></canvas></div>
      <div class="legend"><span style="--colour:#237660">Tree β = 0 · circle</span><span style="--colour:#2564a8">Unicyclic β = 1 · square</span><span style="--colour:#9b6510">Bicyclic β = 2 · diamond</span><span style="--colour:#9c3d78">Higher β ≥ 3 · triangle</span></div>
      <div class="stats"></div><div class="event" role="status" aria-live="polite"></div>
      <details class="components"><summary>Inspect connected components</summary><div class="component-table"></div></details>
      <p class="small">Every drawn edge belongs to the seeded process. Positions are fixed to keep changes legible; position does not determine edge probability. Labels are suppressed above 60 vertices. Timeline ends at connectivity, when the trajectory verdict is final.</p>`;
      this.canvas=root.querySelector('canvas');this.ctx=this.canvas.getContext('2d');
      root.querySelectorAll('[data-action]').forEach(b=>b.addEventListener('click',()=>this.action(b.dataset.action)));
      this.input('timeline').addEventListener('input',()=>{this.pause();this.seek(Number(this.input('timeline').value));});
      this.input('m').addEventListener('change',()=>{this.pause();this.seek(Number(this.input('m').value));});
      ['n','seed','model'].forEach(key=>this.input(key).addEventListener('change',()=>this.generate()));
      this.input('speed').addEventListener('change',()=>{if(this.timer){this.pause();this.play();}});
      this.resizeObserver=new ResizeObserver(()=>this.draw());this.resizeObserver.observe(this.canvas);this.generate();}
    input(key){return this.root.querySelector(`[data-field="${key}"]`);}
    generate(){this.pause();this.n=Math.max(10,Math.min(200,Math.floor(Number(this.input('n').value)||60)));this.input('n').value=this.n;
      this.seed=Number(this.input('seed').value)>>>0;this.input('seed').value=this.seed;this.model=this.input('model').value;
      const result=evolution(this.n,this.model,this.seed);this.edges=result.edges;this.firstFailure=result.firstFailure;
      const rng=random(this.seed^0x12345);this.positions=Array.from({length:this.n},(_,i)=>{
        const angle=i*Math.PI*(3-Math.sqrt(5)),r=.44*Math.sqrt((i+.5)/this.n);
        return [.5+r*Math.cos(angle),.5+r*Math.sin(angle)];});
      this.input('timeline').max=this.edges.length;this.input('m').max=this.edges.length;
      this.input('criticalMarker').style.left=`${Math.floor(this.n/2)/this.edges.length*100}%`;
      this.input('critical').textContent=`Critical marker: m = floor(n/2) = ${Math.floor(this.n/2)}. Connected at m = ${this.edges.length}. Browser seed uses Mulberry32; Python uses MT19937, so seeds reproduce within each implementation.`;
      this.seek(0);this.generated=true;}
    pause(){if(this.timer)clearInterval(this.timer);this.timer=null;this.buttons();}
    play(){if(this.timer||this.m===this.edges.length)return;const speed=Math.max(1,Math.min(60,Number(this.input('speed').value)||8));this.input('speed').value=speed;
      this.timer=setInterval(()=>{this.seek(this.m+1);if(this.m===this.edges.length||(this.m===this.firstFailure&&this.input('autoPause').checked))this.pause();},1000/speed);this.buttons();}
    action(action){switch(action){case 'Generate':this.generate();break;case 'Play':this.play();break;case 'Pause':this.pause();break;
      case 'Next Edge':this.pause();this.seek(this.m+1);break;case 'Previous Edge':this.pause();this.seek(this.m-1);break;
      case 'Restart':this.pause();this.seek(0);break;case 'Reset':this.input('n').value=60;this.input('seed').value=1993;this.input('model').value='simple';this.input('speed').value=8;this.input('autoPause').checked=true;this.generate();break;
      case 'Go to critical':this.pause();this.seek(Math.floor(this.n/2));break;}}
    seek(m){this.m=Math.max(0,Math.min(this.edges.length,Math.floor(m)||0));this.d=new DSU(this.n);this.everFailed=false;this.failureAt=null;
      for(let k=0;k<this.m;k++){this.d.add(...this.edges[k]);if(this.d.complex>=2){this.everFailed=true;if(this.failureAt===null)this.failureAt=k+1;}}
      this.input('m').value=this.m;this.input('timeline').value=this.m;
      const stats=[['Accepted edges',this.m],['Components',this.d.count],['Largest size',this.d.largest],['Complex now',this.d.complex],['Critical m',Math.floor(this.n/2)]];
      this.root.querySelector('.stats').innerHTML=stats.map(([k,v])=>`<div>${k}<strong>${v}</strong></div>`).join('');
      let snapshot=true;const components=[];for(let i=0;i<this.n;i++)if(this.d.p[i]===i){const beta=this.d.e[i]-this.d.v[i]+1;snapshot=snapshot&&beta<=2;components.push([i+1,this.d.v[i],this.d.e[i],beta]);}
      const event=this.root.querySelector('.event');const failed=this.experiment==='snapshot'?!snapshot:this.everFailed;
      event.classList.toggle('failed',failed);
      event.textContent=this.experiment==='snapshot'?`${this.m===Math.floor(this.n/2)?'Equation (11.7) snapshot':'Current snapshot (the target is m = floor(n/2))'}: ${snapshot?'all components have β ≤ 2':'a component has β ≥ 3'}.`:
        this.everFailed?`Historical FAILURE: two complex components first coexisted at edge ${this.failureAt}. There are ${this.d.complex} now; any later merger preserves failure.`:
        this.d.count===1?'Historical SUCCESS: connected, with no earlier coexistence. The verdict is final.':`No coexistence so far. Continue to connectivity to establish success. Complex components now: ${this.d.complex}.`;
      this.root.querySelector('.component-table').innerHTML='<table><thead><tr><th>Root label</th><th>v</th><th>e</th><th>β</th><th>e − v</th></tr></thead><tbody>'+components.map(([i,v,e,b])=>`<tr><td>${i}</td><td>${v}</td><td>${e}</td><td>${b}</td><td>${e-v}</td></tr>`).join('')+'</tbody></table>';
      this.buttons();this.draw();}
    buttons(){if(!this.edges)return;for(const b of this.root.querySelectorAll('[data-action]')){
      b.disabled=(b.dataset.action==='Pause'&&!this.timer)||(b.dataset.action==='Play'&&(!!this.timer||this.m===this.edges.length))||(b.dataset.action==='Next Edge'&&this.m===this.edges.length)||(b.dataset.action==='Previous Edge'&&this.m===0);}}
    draw(){if(!this.d)return;const rect=this.canvas.getBoundingClientRect();const ratio=window.devicePixelRatio||1;
      this.canvas.width=Math.round(rect.width*ratio);this.canvas.height=Math.round(rect.height*ratio);const c=this.ctx;c.setTransform(ratio,0,0,ratio,0,0);c.clearRect(0,0,rect.width,rect.height);
      const scale=Math.min(rect.width,rect.height)-34;const pt=i=>[(rect.width-scale)/2+this.positions[i][0]*scale,(rect.height-scale)/2+this.positions[i][1]*scale];
      const repetitions=new Map();for(let k=0;k<this.m;k++){const [u,v]=this.edges[k];const [x,y]=pt(u),[a,b]=pt(v);const key=Math.min(u,v)*this.n+Math.max(u,v),repeat=repetitions.get(key)||0;repetitions.set(key,repeat+1);
        c.strokeStyle=colour(this.d.beta(u));c.globalAlpha=.48;c.lineWidth=1.5;c.beginPath();
        if(u===v)c.arc(x+7+repeat*2,y-7-repeat*2,9+repeat*2,0,2*Math.PI);
        else if(repeat){const bend=repeat*7;c.moveTo(x,y);c.quadraticCurveTo((x+a)/2+(y-b)/Math.hypot(a-x,b-y)*bend,(y+b)/2+(a-x)/Math.hypot(a-x,b-y)*bend,a,b);}
        else{c.moveTo(x,y);c.lineTo(a,b);}c.stroke();}
      c.globalAlpha=1;for(let i=0;i<this.n;i++){const [x,y]=pt(i),beta=this.d.beta(i),r=this.n>100?3.5:5;c.fillStyle=colour(beta);c.strokeStyle='#fffdf8';c.lineWidth=1;c.beginPath();
        if(beta===0)c.arc(x,y,r,0,2*Math.PI);else if(beta===1)c.rect(x-r,y-r,r*2,r*2);else if(beta===2){c.moveTo(x,y-r-1);c.lineTo(x+r+1,y);c.lineTo(x,y+r+1);c.lineTo(x-r-1,y);c.closePath();}
        else{c.moveTo(x,y-r-2);c.lineTo(x+r+2,y+r);c.lineTo(x-r-2,y+r);c.closePath();}c.fill();c.stroke();
        if(this.n<=60){c.fillStyle='#223632';c.font='10px system-ui';c.fillText(String(i+1),x+7,y-5);}}
      this.canvas.setAttribute('aria-label',`${this.n} vertices, ${this.m} edges, ${this.d.count} components, ${this.d.complex} complex components`);}
  }
  function init(){document.querySelectorAll('.demo').forEach(r=>demos.push(new Demo(r)));
    document.querySelectorAll('.copy').forEach(button=>button.addEventListener('click',async()=>{const code=document.getElementById(button.dataset.code).textContent;
      try{await navigator.clipboard.writeText(code);button.textContent='Copied';}catch{button.textContent='Select code to copy';const range=document.createRange();range.selectNodeContents(document.getElementById(button.dataset.code));window.getSelection().removeAllRanges();window.getSelection().addRange(range);} }));
    const data=JSON.parse(document.getElementById('research-data').textContent);document.getElementById('data-status').textContent=`Validated embedded snapshot: ${data.rows.length} result cells, ${data.metadata.target_trials.toLocaleString()} trials per cell. Download JSON / CSV for exact values.`;}
  if(typeof document!=='undefined')document.addEventListener('DOMContentLoaded',init);
  return {DSU,random,evolution,demos};
})();
if(typeof module!=='undefined')module.exports=GC;
