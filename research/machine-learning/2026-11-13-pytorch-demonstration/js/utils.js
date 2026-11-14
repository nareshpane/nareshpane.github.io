export const colors=['#477da5','#b57a60','#8d78a5','#5a887b','#b96663'];
export const fmt=(x,n=4)=>Number(x).toLocaleString('en-US',{maximumFractionDigits:n,minimumFractionDigits:n});
export const $=(id)=>document.getElementById(id);
export const svg=(body,width=760,height=300,label='Scientific diagram',className='plot')=>`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="${label}" class="${className}">${body}</svg>`;
export const linePath=(points)=>points.map((p,i)=>`${i?'L':'M'}${p[0].toFixed(2)},${p[1].toFixed(2)}`).join(' ');
const chartRegistry=new Map();
export function chart(container,series,{xlabel='Index',ylabel='Value',ymin=null,ymax=null,title='',markers=false,height=300,compact=false,legend=true,cursorX=null}={}){
 chartRegistry.set(container,{series,options:{xlabel,ylabel,ymin,ymax,title,markers,height,compact,legend,cursorX}});
 const all=series.flatMap(s=>s.points).filter(p=>Number.isFinite(p[0])&&Number.isFinite(p[1]));
 if(!all.length){container.textContent='No finite observations.';return}
 let x0=Math.min(...all.map(p=>p[0])),x1=Math.max(...all.map(p=>p[0]));
 let y0=ymin??Math.min(0,...all.map(p=>p[1])),y1=ymax??Math.max(...all.map(p=>p[1]));
 if(x0===x1)x1=x0+1;if(y0===y1)y1=y0+1;
 if(ymax===null)y1+=(y1-y0)*.08;
 const width=Math.max(compact?280:320,Math.min(760,container.clientWidth||760)),left=62,right=width-(compact?17:25),top=compact?15:33,bottom=height-(compact?43:52),ticks=width<500?3:5;
 const X=x=>left+(x-x0)/(x1-x0)*(right-left),Y=y=>bottom-(y-y0)/(y1-y0)*(bottom-top);
 let b=`<title>${title||ylabel+' by '+xlabel}</title>`;
 for(let i=0;i<=4;i++){const y=y0+(y1-y0)*i/4; b+=`<path d="M${left} ${Y(y)}H${right}" stroke="#dce6ed"/><text x="${left-9}" y="${Y(y)+4}" text-anchor="end">${Math.abs(y)>=100?fmt(y,0):fmt(y,2)}</text>`}
 for(let i=0;i<=ticks;i++){const x=x0+(x1-x0)*i/ticks,decimals=Math.abs(x-Math.round(x))<1e-8?0:Math.abs(x1-x0)>=1?1:2;b+=`<text x="${X(x)}" y="${bottom+18}" text-anchor="middle">${fmt(x,decimals)}</text>`}
 b+=`<path d="M${left} ${top}V${bottom}H${right}" fill="none" stroke="#8ea8ba"/><text x="${(left+right)/2}" y="${height-6}" text-anchor="middle" class="axis-label">${xlabel}</text><text x="16" y="${(top+bottom)/2}" transform="rotate(-90 16 ${(top+bottom)/2})" text-anchor="middle" class="axis-label">${ylabel}</text>`;
 if(cursorX!==null&&cursorX>=x0&&cursorX<=x1)b+=`<path d="M${X(cursorX)} ${top}V${bottom}" stroke="#9e8cae" stroke-width="1.3" stroke-dasharray="4 3"/>`;
 series.forEach((s,i)=>{const color=s.color||colors[i%colors.length], points=s.points.map(p=>[X(p[0]),Y(p[1])]);
  if(!s.dots)b+=`<path d="${linePath(points)}" fill="none" stroke="${color}" stroke-width="2.2" ${s.dashed?'stroke-dasharray="5 4"':''}/>`;
  if(markers||s.dots)for(const [x,y]of points)b+=`<circle cx="${x}" cy="${y}" r="${s.dots?3:2}" fill="${color}"/>`;
 });
 container.innerHTML=svg(b,width,height,title||ylabel+' by '+xlabel,'plot responsive-plot')+(legend?`<div class="legend">${series.map((s,i)=>`<span style="--key:${s.color||colors[i%colors.length]}">${s.name}</span>`).join('')}</div>`:'');
}
export function alignNumericTables(root=document){for(const table of root.querySelectorAll('table')){if(table.closest('.matrix-block'))continue;const rows=[...table.tBodies].flatMap(b=>[...b.rows]),head=table.tHead?.rows[0];if(!rows.length)continue;const numeric=value=>/^[+−-]?[\d,]+(?:\.\d+)?(?:[eE][+−-]?\d+)?(?:\s*(?:%|×|s|ms|bytes))?$/.test(value.trim());for(let i=0;i<rows[0].cells.length;i++){let count=0;for(const row of rows){const cell=row.cells[i];if(!cell)continue;const yes=numeric(cell.textContent);cell.classList.toggle('numeric',yes);if(yes)count++}head?.cells[i]?.classList.toggle('numeric',count>rows.length/2)}}}
if(typeof addEventListener==='function'){let scheduled=false;addEventListener('resize',()=>{if(!scheduled){scheduled=true;requestAnimationFrame(()=>{scheduled=false;for(const[container,record]of chartRegistry)if(container.isConnected)chart(container,record.series,record.options)})}})}
export function readouts(container,items){container.innerHTML=items.map(([name,value])=>`<div><span>${name}</span><strong>${value}</strong></div>`).join('')}
export function typesetLive(container,tex,nativeMathml){
 const revision=String((+container.dataset.mathRevision||0)+1);container.dataset.mathRevision=revision;
 container.innerHTML=nativeMathml;container.dataset.mathRendered='native';
 const ready=window.MathJax?.startup?.promise;
 if(ready)ready.then(()=>window.MathJax.tex2svgPromise(tex,{display:true})).then(node=>{if(container.dataset.mathRevision===revision){container.replaceChildren(node);container.dataset.mathRendered='MathJax'}}).catch(()=>{if(container.dataset.mathRevision===revision)container.dataset.mathRendered='native'});
}
export function playback(lab,{count,render,interval=900}){
 let current=0,timer=null;
 const max=()=>typeof count==='function'?count():count;
 const stop=()=>{clearInterval(timer);timer=null;lab.querySelector('[data-action="play"]')?.setAttribute('aria-pressed','false')};
 const draw=()=>{current=Math.max(0,Math.min(max()-1,current));render(current);const range=lab.querySelector('[data-frame]');if(range){range.max=max()-1;range.value=current}};
 lab.querySelector('[data-action="play"]')?.addEventListener('click',()=>{
  stop();if(matchMedia('(prefers-reduced-motion: reduce)').matches){lab.querySelector('.motion-note')?.replaceChildren(document.createTextNode('Reduced motion is enabled. Use Next step to inspect each state.'));return}
  if(current>=max()-1)current=0;
  timer=setInterval(()=>{current++;draw();if(current>=max()-1)stop()},interval);
  lab.querySelector('[data-action="play"]').setAttribute('aria-pressed','true');draw();
 });
 for(const action of ['pause','next','previous','restart'])lab.querySelector(`[data-action="${action}"]`)?.addEventListener('click',()=>{stop();current=action==='next'?current+1:action==='previous'?current-1:action==='restart'?0:current;draw()});
 lab.querySelector('[data-frame]')?.addEventListener('input',e=>{stop();current=+e.target.value;draw()});
 document.addEventListener('visibilitychange',()=>{if(document.hidden)stop()});
 new IntersectionObserver(entries=>{if(!entries[0].isIntersecting)stop()},{threshold:0}).observe(lab);
 draw();return{render:draw,set(index){stop();current=index;draw()},stop,get index(){return current}};
}
export async function load(name){const response=await fetch(new URL('../data/'+name,import.meta.url));if(!response.ok)throw Error(`${name}: HTTP ${response.status}`);return response.json()}
export function protect(lab,callback){try{return callback()}catch(e){lab.insertAdjacentHTML('beforeend','<p class="lab-error" role="status">Laboratory unavailable. Serve the page over HTTP and see the reproduction guide.</p>');console.error(e)}}
