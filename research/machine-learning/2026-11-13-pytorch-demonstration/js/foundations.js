import{$,fmt,svg,chart,readouts,playback,typesetLive}from'./utils.js';
function matrix(name,values){return `<div class="matrix-block"><div class="matrix-caption">${name} · ${values.length} × ${values[0].length}</div><table aria-label="${name}"><tbody>${values.map(row=>`<tr>${row.map(v=>`<td>${Number.isInteger(v)?v:fmt(v,2)}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`}
function tensorLab(){const lab=$('tensor-lab');
 function update(){const n=+$('tensor-n').value,m=+$('tensor-m').value,r=+$('tensor-r').value,p=+$('tensor-p').value,op=$('tensor-op').value;
  const a=Array.from({length:n},(_,i)=>Array.from({length:m},(_,j)=>i*m+j+1));
  const b=Array.from({length:r},(_,i)=>Array.from({length:p},(_,j)=>i*p+j+1));let result=null,note='',error='';
  if(op==='matmul'){if(m!==r)error=`Invalid matrix product: the inner axes ${m} and ${r} differ.`;else{result=a.map(row=>b[0].map((_,j)=>row.reduce((sum,v,k)=>sum+v*b[k][j],0)));note=`(${n} × ${m}) @ (${r} × ${p}) → (${n} × ${p}); entry (1,1) is a row–column dot product.`}}
  if(op==='elementwise'||op==='broadcast'){if((n!==r&&n!==1&&r!==1)||(m!==p&&m!==1&&p!==1))error='Invalid broadcasting: align from the right; each pair of dimensions must agree or contain a 1.';else{result=Array.from({length:Math.max(n,r)},(_,i)=>Array.from({length:Math.max(m,p)},(_,j)=>op==='elementwise'?a[i%n][j%m]*b[i%r][j%p]:a[i%n][j%m]+b[i%r][j%p]));note=op==='broadcast'?'Addition repeats singleton axes logically; it does not copy a new full input array.':'Elementwise multiplication uses aligned entries, including singleton broadcasting.'}}
  if(op==='reduce'){result=[a[0].map((_,j)=>a.reduce((sum,row)=>sum+row[j],0))];note='sum(dim=0, keepdim=True) reduces the row axis; output shape is 1 × columns.'}
  if(op==='reshape'){if(n*m!==r*p)error=`Cannot reshape ${n*m} entries into ${r*p} entries.`;else{const flat=a.flat();result=Array.from({length:r},(_,i)=>flat.slice(i*p,(i+1)*p));note='Reshape preserves logical element order; it is a view only when strides permit.'}}
  if(op==='permute'){result=a[0].map((_,j)=>a.map(row=>row[j]));note='permute(1,0) swaps axes; the element at row i, column j moves to row j, column i. Right-hand dimensions are ignored.'}
  $('tensor-matrices').innerHTML=matrix('A',a)+(op==='reduce'||op==='permute'?'':matrix('B / requested shape',b))+(result?matrix('Result',result):'');
  $('tensor-status').className=error?'error-note':'stage-note';$('tensor-status').textContent=error||note;
  const bytes=+$('tensor-dtype').value,total=n*m;
  readouts($('tensor-readouts'),[['A elements',total],['Raw A storage',`${(total*bytes).toLocaleString()} bytes`],['Element width',`${bytes} bytes`],['Result elements',result?result.length*result[0].length:'Invalid']]);
 }
 lab.querySelectorAll('input,select').forEach(x=>x.addEventListener('input',update));update();
}
function scalarGraph(){const lab=$('graph-lab'),descriptions=[
 'Inputs: x=2 and target y=5 are observed. The trainable parameters begin at w=1 and b=0.',
 'Multiply: z=wx=1×2=2. The local derivatives are ∂z/∂w=x=2 and ∂z/∂x=w=1.',
 'Add bias: ŷ=z+b=2. The local sensitivities to z and b are both 1.',
 'Compare: r=ŷ−y=−3. The prediction derivative is 1; the target derivative is −1.',
 'Reduce to a scalar: L=½r²=4.5. The local derivative ∂L/∂r=r=−3 starts reverse mode.',
 'Reverse through subtraction and addition: ∂L/∂ŷ=−3 and ∂L/∂b=−3.',
 'Reverse through multiplication: ∂L/∂w=(−3)×2=−6. Contributions add if a parameter has several paths.',
 'Update both parameters simultaneously using their old gradients. Gradients describe sensitivity; the learning rate determines the update.',
 'Re-evaluate the entire forward graph at the updated parameters. A small enough step decreases this quadratic loss; an arbitrary step need not.'
 ];
 function draw(stage){const rate=+$('graph-rate').value,w=stage>=7?1+6*rate:1,b=stage>=7?3*rate:0,z=2*w,pred=z+b,res=pred-5,loss=.5*res*res;
  $('graph-rate-value').textContent=fmt(rate,2);
  const nodes=[{x:25,y:55,w:126,label:'Observed x',value:'2',active:0},{x:25,y:145,w:126,label:'Parameter w',value:fmt(w),active:0},
   {x:198,y:102,w:127,label:'Multiply wx',value:fmt(z),active:1},{x:198,y:210,w:127,label:'Parameter b',value:fmt(b),active:0},
   {x:373,y:102,w:126,label:'Add → prediction',value:fmt(pred),active:2},{x:546,y:102,w:126,label:'Residual ŷ − 5',value:fmt(res),active:3},
   {x:546,y:220,w:126,label:'Loss ½r²',value:fmt(loss),active:4}];
  let body='<defs><marker id="graph-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0L10 5L0 10" fill="#85a4ba"/></marker></defs>';
  for(const d of ['M151 85L198 126','M151 175L198 137','M325 135H373','M325 240L390 165','M499 135H546','M610 165V220'])body+=`<path d="${d}" fill="none" stroke="#9ab3c4" stroke-width="1.8" marker-end="url(#graph-arrow)"/>`;
  for(const node of nodes){const active=stage===node.active||(stage>=7&&[0,1,2,3,4].includes(node.active));body+=`<rect x="${node.x}" y="${node.y}" width="${node.w}" height="64" rx="8" fill="${active?'#e0edf8':'#f5f8fa'}" stroke="${active?'#477da5':'#c5d5df'}"/><text x="${node.x+node.w/2}" y="${node.y+22}" text-anchor="middle" style="font-size:11px">${node.label}</text><text x="${node.x+node.w/2}" y="${node.y+46}" text-anchor="middle" style="font-size:18px;fill:#26323b">${node.value}</text>`}
  if(stage>=5&&stage<=6){body+='<path d="M595 216V185H434V177M372 166H262V173M198 160H150" fill="none" stroke="#bd7973" stroke-width="3" stroke-dasharray="6 4"/>';body+='<text x="360" y="35" text-anchor="middle" style="fill:#9e5c57">Reverse sensitivities: dL/db = −3; dL/dw = −6</text>'}
  if(stage>=7)body+=`<text x="360" y="320" text-anchor="middle">w′=1+6η=${fmt(w)} · b′=3η=${fmt(b)} · L′=${fmt(loss,5)}</text>`;
  $('graph-diagram').innerHTML=svg(body,720,340,'Scalar forward graph with numerical values and reverse sensitivities');
  $('graph-stage').textContent=`${stage+1}/9 · ${descriptions[stage]}`;
  readouts($('graph-readouts'),[['Prediction',fmt(pred)],['Loss',fmt(loss,5)],['dL/dw (before update)','−6.0000'],['dL/db (before update)','−3.0000']]);
 }
 const player=playback(lab,{count:9,render:draw});$('graph-rate').addEventListener('input',()=>player.render());
}
function networkLab(){const weights=[[.8,-.4],[-.6,.9],[.5,.5]],bias=[.1,-.2,.3],out=[1.2,-.8,.6];
 function draw(){const inputs=[+$('network-x1').value,+$('network-x2').value],kind=$('network-activation').value,index=+$('network-neuron').value;
  const f=x=>kind==='relu'?Math.max(0,x):kind==='sigmoid'?1/(1+Math.exp(-x)):Math.tanh(x);
  const z=weights.map((w,i)=>w[0]*inputs[0]+w[1]*inputs[1]+bias[i]),h=z.map(f),prediction=h.reduce((s,v,i)=>s+v*out[i],-.1);
  let body='';const ix=105,hx=360,ox=640,iy=[110,235],hy=[55,170,285];
  weights.forEach((w,j)=>w.forEach((v,i)=>{const fraction=i===0?.32:.72,lx=ix+35+(hx-43-ix-35)*fraction,ly=iy[i]+(hy[j]-iy[i])*fraction-7;body+=`<line x1="${ix+35}" y1="${iy[i]}" x2="${hx-43}" y2="${hy[j]}" stroke="${v>0?'#8baec7':'#c49a8b'}" stroke-width="${1+Math.abs(v)*2}"/><rect x="${lx-15}" y="${ly-12}" width="30" height="17" rx="3" fill="#fcfeff" opacity=".94"/><text x="${lx}" y="${ly}" text-anchor="middle" style="font-size:11px">${v}</text>`}));
  h.forEach((v,j)=>body+=`<line x1="${hx+43}" y1="${hy[j]}" x2="${ox-43}" y2="170" stroke="${out[j]>0?'#8baec7':'#c49a8b'}" stroke-width="${1+Math.abs(out[j])*2}"/><text x="500" y="${(hy[j]+170)/2-4}" style="font-size:11px">${out[j]}</text>`);
  inputs.forEach((v,i)=>body+=`<circle cx="${ix}" cy="${iy[i]}" r="35" fill="#edf5fc" stroke="#91b0c7"/><text x="${ix}" y="${iy[i]+5}" text-anchor="middle">x${i+1}=${fmt(v,1)}</text>`);
  h.forEach((v,j)=>body+=`<circle cx="${hx}" cy="${hy[j]}" r="43" fill="${j===index?'#fff0e5':'#f4effa'}" stroke="${j===index?'#b57a60':'#c5b5d5'}" stroke-width="${j===index?2:1}"/><text x="${hx}" y="${hy[j]-5}" text-anchor="middle">h${j+1}</text><text x="${hx}" y="${hy[j]+17}" text-anchor="middle">${fmt(v,3)}</text>`);
  body+=`<circle cx="${ox}" cy="170" r="43" fill="#edf5fc" stroke="#91b0c7"/><text x="${ox}" y="162" text-anchor="middle">Output</text><text x="${ox}" y="184" text-anchor="middle">${fmt(prediction,3)}</text>`;
  $('network-diagram').innerHTML=svg(body,760,345,'Two inputs, three nonlinear hidden units, and one linear output with signed weights');
  const k=index+1,n=value=>Number(value).toFixed(4),activation=kind==='relu'?'\\operatorname{ReLU}':kind==='sigmoid'?'\\operatorname{sigmoid}':'\\tanh';
  const tex=`\\begin{aligned}z_{${k}}&=(${n(weights[index][0])})(${n(inputs[0])})+(${n(weights[index][1])})(${n(inputs[1])})+(${n(bias[index])})=${n(z[index])},\\\\h_{${k}}&=${activation}(z_{${k}})=${n(h[index])},\\\\\\widehat y&=1.2h_1-0.8h_2+0.6h_3-0.1=${n(prediction)}.\\end{aligned}`;
  const sub=(name,i)=>`<msub><mi>${name}</mi><mn>${i}</mn></msub>`,mn=v=>`<mn>${n(v)}</mn>`,paren=v=>`<mo>(</mo>${mn(v)}<mo>)</mo>`,row=v=>`<mtr><mtd><mrow>${v}</mrow></mtd></mtr>`;
  const native=`<math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mtable columnalign="left">${row(sub('z',k)+'<mo>=</mo>'+paren(weights[index][0])+paren(inputs[0])+'<mo>+</mo>'+paren(weights[index][1])+paren(inputs[1])+'<mo>+</mo>'+paren(bias[index])+'<mo>=</mo>'+mn(z[index]))}${row(sub('h',k)+'<mo>=</mo><mi mathvariant="normal">'+(kind==='relu'?'ReLU':kind)+'</mi><mo>(</mo>'+sub('z',k)+'<mo>)</mo><mo>=</mo>'+mn(h[index]))}${row('<mover><mi>y</mi><mo>^</mo></mover><mo>=</mo><mn>1.2</mn>'+sub('h',1)+'<mo>-</mo><mn>0.8</mn>'+sub('h',2)+'<mo>+</mo><mn>0.6</mn>'+sub('h',3)+'<mo>-</mo><mn>0.1</mn><mo>=</mo>'+mn(prediction))}</mtable></math>`;
  typesetLive($('network-calculation'),tex,native);
  $('network-x1-value').textContent=fmt(inputs[0],1);$('network-x2-value').textContent=fmt(inputs[1],1);
 }
 $('network-lab').querySelectorAll('input,select').forEach(e=>e.addEventListener('input',draw));draw();
}
function derivativeLab(){
 function draw(){
  const power=+$('difference-step').value,step=10**(-power);
  const estimate=(Math.sin(.7+step)-Math.sin(.7-step))/(2*step);
  $('difference-step-value').textContent=`h = 10^−${power}`;
  readouts($('difference-readouts'),[
   ['Central difference',fmt(estimate,10)],
   ['Exact cos(0.7)',fmt(Math.cos(.7),10)],
   ['Absolute error',Math.abs(estimate-Math.cos(.7)).toExponential(3)]
  ]);
  const points=Array.from({length:12},(_,i)=>{
   const h=10**(-i-1);
   const error=Math.abs((Math.sin(.7+h)-Math.sin(.7-h))/(2*h)-Math.cos(.7));
   return[i+1,Math.log10(Math.max(1e-18,error))];
  });
  chart($('difference-plot'),[{name:'log₁₀ absolute error',points}],{xlabel:'−log₁₀(h)',ylabel:'log₁₀ absolute error',ymin:-16,ymax:0,markers:true});
 }
 $('difference-step').addEventListener('input',draw);draw();
}
function responsibilityLab(){const rows=[['Scalar or matrix evaluation','NumPy expressions already suffice. No derivative code is required.','Tensor syntax adds little for this isolated calculation.'],['Statistical loss with 72 parameters','Write the vectorized likelihood and its analytical gradient; use a SciPy optimizer.','Write the same likelihood with supported tensors; autograd supplies the gradient.'],['Change a composed model','Update the forward formula and every affected derivative or external differentiation interface.','Edit the forward model; check the graph remains connected and gradients remain valid.'],['Repeated accelerator training','Coordinate kernels, device transfers, derivative logic, batching, and optimizer state, or use another framework.','Tensor devices, autograd, modules, optimizers, and data utilities coordinate these tasks. Profiling still decides performance.']];function draw(){const i=+$('responsibility-task').value;$('responsibility-manual').textContent=rows[i][1];$('responsibility-torch').textContent=rows[i][2]}$('responsibility-task').addEventListener('input',draw);draw()}
export function initFoundations(){tensorLab();scalarGraph();networkLab();derivativeLab();responsibilityLab()}
