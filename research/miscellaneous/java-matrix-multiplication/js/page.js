/* Java owns numerical examples; this file owns their presentation and controls. */
(() => {
  "use strict";
  const ROOT = "java-matrix-multiplication/traces/";
  const $ = id => document.getElementById(id);
  const motion = matchMedia("(prefers-reduced-motion: reduce)");
  const C = {paper:"#fffaf0", ink:"#302d25", muted:"#6c604d", line:"#d6c6aa", teal:"#35685d", terra:"#a34930", gold:"#8b641f", green:"#e3ecdf", peach:"#fae6d6", yellow:"#f6edc8"};
  const players = [];
  const fmt = n => n.toLocaleString("en-US");
  const el = (tag, cls, text) => { const n = document.createElement(tag); if(cls)n.className=cls; if(text!==undefined)n.textContent=text; return n; };

  function player(name, length, render, delay=900) {
    const controls=document.querySelector('[data-player="'+name+'"]');
    let index=0, playing=false, timer=0, visible=true;
    const buttons={};
    ["Play","Pause","Previous step","Next step","Reset"].forEach(label=>{
      const b=el("button","",label); b.type="button";
      b.setAttribute("aria-label",name+" · "+label); controls.append(b); buttons[label]=b;
    });
    const label=el("label","","Speed"); label.htmlFor=name+"Speed";
    const speed=el("input"); speed.type="range"; speed.min=".5";speed.max="3";speed.step=".5";speed.value="1";speed.id=name+"Speed";
    const out=el("output","","1×");out.htmlFor=speed.id;controls.append(label,speed,out);
    function sync() {
      buttons.Play.setAttribute("aria-pressed",String(playing));
      buttons.Pause.setAttribute("aria-pressed",String(!playing));
      buttons["Previous step"].disabled=index===0;
      buttons["Next step"].disabled=index===length-1;
    }
    function schedule() {
      clearTimeout(timer);
      visible=isVisible();
      if(playing&&visible&&!document.hidden)timer=setTimeout(()=>{
        if(index<length-1){index++;render(index,true);sync();schedule();}
        else {playing=false;sync();}
      },delay/Number(speed.value));
    }
    function pause(){playing=false;clearTimeout(timer);sync();document.getAnimations().forEach(a=>{if(controls.closest("section,.visual")?.contains(a.effect?.target))a.finish();});render(index,false);}
    function go(i,animate=true){pause();index=Math.max(0,Math.min(length-1,i));render(index,animate);sync();}
    buttons.Play.onclick=()=>{if(index===length-1){index=0;render(index,false);}playing=true;sync();schedule();};
    buttons.Pause.onclick=pause;
    buttons["Previous step"].onclick=()=>go(index-1);
    buttons["Next step"].onclick=()=>go(index+1);
    buttons.Reset.onclick=()=>go(0,false);
    speed.oninput=()=>{out.value=speed.value+"×";schedule();};
    document.addEventListener("visibilitychange",schedule);
    function isVisible(){const r=controls.parentElement.getBoundingClientRect();return r.bottom>0&&r.top<innerHeight;}
    function checkVisibility(){if(isVisible()!==visible)schedule();}
    document.addEventListener("scroll",checkVisibility,{passive:true});
    window.addEventListener("resize",checkVisibility);
    const api={go,pause,get index(){return index;}};players.push(api);sync();render(0,false);return api;
  }
  motion.addEventListener("change",()=>players.forEach(p=>p.pause()));

  function fly(node, from, to, animate=true) {
    if(!animate||motion.matches||!node||!from||!to)return;
    const a=from.getBoundingClientRect(),b=to.getBoundingClientRect(),r=node.getBoundingClientRect();
    node.animate([
      {transform:"translate("+(a.left+a.width/2-r.left-r.width/2)+"px,"+(a.top+a.height/2-r.top-r.height/2)+"px)",opacity:.65},
      {transform:"translate("+(b.left+b.width/2-r.left-r.width/2)+"px,"+(b.top+b.height/2-r.top-r.height/2)+"px)",opacity:1}
    ],{duration:480,easing:"cubic-bezier(.2,.7,.3,1)"});
  }
  function matrix(id,values) {
    const box=$(id);box.replaceChildren();box.style.setProperty("--cols",values[0].length);
    const cells=values.map((row,i)=>row.map((v,j)=>{
      const cell=el("span","matrix-cell",v===null?"·":v);cell.dataset.row=i;cell.dataset.col=j;
      cell.setAttribute("aria-label","row "+(i+1)+", column "+(j+1)+": "+(v===null?"empty":v));box.append(cell);return cell;
    }));
    return {box,cells};
  }
  function setCell(cell,value,classes="") {
    cell.textContent=value===null?"·":value;cell.className="matrix-cell "+classes+(value===null?" empty":"");
    cell.setAttribute("aria-label","row "+(+cell.dataset.row+1)+", column "+(+cell.dataset.col+1)+": "+(value===null?"empty":value));
  }
  function canvas(id,w,h) {
    const c=$(id),d=Math.min(devicePixelRatio||1,2);c.width=w*d;c.height=h*d;
    const ctx=c.getContext("2d");ctx.scale(d,d);return ctx;
  }
  function text(ctx,t,x,y,size=16,color=C.ink,align="center") {
    ctx.font="500 "+size+"px Inter, sans-serif";ctx.fillStyle=color;ctx.textAlign=align;ctx.textBaseline="middle";ctx.fillText(t,x,y);
  }
  function wrappedText(ctx,value,x,y,width,size=16,color=C.ink) {
    ctx.font="500 "+size+"px Inter, sans-serif";
    const words=value.split(" ");let row="",offset=0;
    for(const word of words){const next=row?row+" "+word:word;if(ctx.measureText(next).width>width&&row){text(ctx,row,x,y+offset,size,color);offset+=size*1.5;row=word;}else row=next;}
    text(ctx,row,x,y+offset,size,color);
  }
  function rect(ctx,x,y,w,h,fill=C.paper,stroke=C.line,r=7) {
    ctx.beginPath();ctx.roundRect(x,y,w,h,r);ctx.fillStyle=fill;ctx.fill();
    if(stroke){ctx.strokeStyle=stroke;ctx.lineWidth=1.5;ctx.stroke();}
  }
  function line(ctx,x,y,xx,yy,color=C.line,width=1.5) {
    ctx.beginPath();ctx.moveTo(x,y);ctx.lineTo(xx,yy);ctx.strokeStyle=color;ctx.lineWidth=width;ctx.stroke();
  }
  function arrow(ctx,x,y,xx,yy,color=C.teal,width=3) {
    line(ctx,x,y,xx,yy,color,width);const angle=Math.atan2(yy-y,xx-x);
    line(ctx,xx,yy,xx-11*Math.cos(angle-.45),yy-11*Math.sin(angle-.45),color,width);
    line(ctx,xx,yy,xx-11*Math.cos(angle+.45),yy-11*Math.sin(angle+.45),color,width);
  }
  function tween(owner,render,animate) {
    if(owner.frame)cancelAnimationFrame(owner.frame);
    if(!animate||motion.matches){render(1);return;}
    const start=performance.now();
    function tick(now){const t=Math.min(1,(now-start)/480);render(1-(1-t)**3);if(t<1)owner.frame=requestAnimationFrame(tick);}
    owner.frame=requestAnimationFrame(tick);
  }

  function theatre(trace) {
    const a=matrix("theatreA",trace.a),b=matrix("theatreB",trace.b),c=matrix("theatreC",trace.result.map(r=>r.map(()=>null)));
    const pairs=[];
    for(let k=0;k<3;k++){
      const lane=el("div","pair-lane");lane.append(el("small","","PAIR k = "+k));
      const values=el("div","pair-values"),left=el("span","pair-value","?"),right=el("span","pair-value right","?");
      values.append(left,el("span","","×"),right);const product=el("span","product","·");
      lane.append(values,product);$("pairLanes").append(lane);pairs.push({lane,left,right,product});
    }
    const states=[{phase:"ready",e:trace.events[0],completed:0}];
    for(let cell=0;cell<4;cell++){
      const events=trace.events.slice(cell*3,cell*3+3),e=events[0];
      ["row","column","detach"].forEach(phase=>states.push({phase,e,completed:cell}));
      events.forEach(e=>["pair","multiply","travel","sum"].forEach(phase=>states.push({phase,e,completed:cell})));
      states.push({phase:"store",e:events[2],completed:cell});
    }
    player("theatre",states.length,(index,animate)=>{
      const s=states[index],e=s.e,phase=s.phase,after=["sum","store"].includes(phase);
      const prior=e.partial-e.product,partial=after?e.partial:(["pair","multiply","travel"].includes(phase)?prior:0);
      a.cells.forEach((row,i)=>row.forEach((cell,k)=>setCell(cell,trace.a[i][k],i===e.i&&phase!=="ready"?"row-on "+(k===e.k&&["pair","multiply","travel","sum"].includes(phase)?"selected":""):"")));
      b.cells.forEach((row,k)=>row.forEach((cell,j)=>setCell(cell,trace.b[k][j],j===e.j&&!["ready","row"].includes(phase)?"col-on "+(k===e.k&&["pair","multiply","travel","sum"].includes(phase)?"selected":""):"")));
      c.cells.forEach((row,i)=>row.forEach((cell,j)=>{
        const done=i*2+j<s.completed||(phase==="store"&&i===e.i&&j===e.j);
        setCell(cell,done?trace.result[i][j]:null,done?"completed":i===e.i&&j===e.j?"active-output":"");
      }));
      const detached=!["ready","row","column"].includes(phase);
      pairs.forEach((p,k)=>{
        const productShown=detached&&(k<e.k||(k===e.k&&["multiply","travel","sum","store"].includes(phase)));
        p.lane.className="pair-lane"+(detached?" revealed":"")+(detached&&k===e.k?" active":"");
        p.left.textContent=detached?trace.a[e.i][k]:"?";p.right.textContent=detached?trace.b[k][e.j]:"?";
        p.product.textContent=productShown?trace.events[s.completed*3+k].product:"·";
        if(phase==="detach"){fly(p.left,a.cells[e.i][k],p.left,animate);fly(p.right,b.cells[k][e.j],p.right,animate);}
      });
      $("accValue").textContent=partial;
      $("accCode").textContent="c["+e.i+"]["+e.j+"] += a["+e.i+"][k] * b[k]["+e.j+"]";
      if(phase==="travel")fly(pairs[e.k].product,pairs[e.k].product,$("accValue"),animate);
      if(phase==="store")fly(c.cells[e.i][e.j],$("accValue"),c.cells[e.i][e.j],animate);
      const messages={
        ready:"C is empty. Choose a row and a column to begin the first output entry.",
        row:"Select row "+(e.i+1)+" of A: ["+trace.a[e.i].join(", ")+"].",
        column:"Select column "+(e.j+1)+" of B: ["+trace.b.map(r=>r[e.j]).join(", ")+"].",
        detach:"The selected row and column extend into three matching pairs. Their order stays fixed.",
        pair:"Pair k="+e.k+": a["+e.i+"]["+e.k+"]="+e.left+" with b["+e.k+"]["+e.j+"]="+e.right+".",
        multiply:e.left+" × ("+e.right+") = "+e.product+". The accumulator has not changed yet.",
        travel:"The product "+e.product+" travels toward the accumulator, currently "+prior+".",
        sum:prior+" + ("+e.product+") = "+e.partial+". "+(e.cellComplete?"The dot product is complete.":"Advance k to the next pair."),
        store:"Store "+e.partial+" in C"+(e.i+1)+(e.j+1)+" ↔ c["+e.i+"]["+e.j+"]. "+(s.completed===3?"All four entries are complete.":"Now choose the next output entry.")
      };
      $("theatreStatus").textContent="Step "+index+"/"+(states.length-1)+" · "+messages[phase];
      $("pairLabel").textContent=detached?"Row "+(e.i+1)+" × column "+(e.j+1)+" · align their shared index k":"Select → extend → pair → multiply → add → store";
      $("theatreMath").textContent="C"+(e.i+1)+(e.j+1)+" = row "+(e.i+1)+" · column "+(e.j+1);
      $("theatreJava").textContent="c["+e.i+"]["+e.j+"]";
      $("theatreMachine").textContent=phase==="store"?"write completed value "+e.partial:"partial value = "+partial;
    },500);
    document.querySelector('[data-player="theatre"] button').click();
  }

  function representation(trace) {
    let unfolded=false,selected=0;
    const buttons=trace.a.flat().map((value,k)=>{
      const button=el("button");button.type="button";button.append(el("b","",value),el("small",""));
      button.onclick=()=>{selected=k;render();};$("representationCells").append(button);return button;
    });
    function render(){
      $("representationCells").classList.toggle("unfolded",unfolded);
      buttons.forEach((b,k)=>{
        const i=Math.floor(k/3),j=k%3;b.querySelector("small").textContent=unfolded?"A["+i+"]["+j+"]":"A"+(i+1)+(j+1);
        b.setAttribute("aria-pressed",String(k===selected));b.setAttribute("aria-label","A"+(i+1)+(j+1)+" equals "+trace.a[i][j]+", Java A["+i+"]["+j+"]");
      });
      const i=Math.floor(selected/3),j=selected%3;
      $("representationStatus").textContent="Mathematical A"+(i+1)+(j+1)+" = "+trace.a[i][j]+" ↔ Java A["+i+"]["+j+"] = "+trace.a[i][j]+". Row "+(i+1)+" becomes index "+i+"; column "+(j+1)+" becomes index "+j+".";
      $("representationToggle").textContent=unfolded?"Fold back into the matrix":"Unfold into Java rows";
      $("representationToggle").setAttribute("aria-pressed",String(unfolded));
    }
    $("representationToggle").onclick=()=>{
      const old=buttons.map(b=>b.getBoundingClientRect());unfolded=!unfolded;render();
      if(!motion.matches)buttons.forEach((b,k)=>{const now=b.getBoundingClientRect();b.animate([{translate:(old[k].left-now.left)+"px "+(old[k].top-now.top)+"px"},{translate:"0px 0px"}],{duration:550,easing:"ease"});});
    };render();
  }

  function loops(trace) {
    const a=matrix("loopA",trace.a),b=matrix("loopB",trace.b),c=matrix("loopC",trace.result.map(r=>r.map(()=>0)));
    const tracks=[["iTrack",2],["jTrack",2],["kTrack",3]].map(([id,n])=>Array.from({length:n},(_,k)=>{const t=el("i","",k);$(id).append(t);return t;}));
    const states=[{line:"allocate",e:trace.events[0],done:0}];
    trace.events.forEach((e,k)=>{
      if(e.j===0&&e.k===0)states.push({line:"i",e,done:k});
      if(e.k===0)states.push({line:"j",e,done:k});
      states.push({line:"k",e,done:k},{line:"add",e,done:k+1});
    });states.push({line:"return",e:trace.events.at(-1),done:12});
    player("loops",states.length,index=>{
      const s=states[index],e=s.e;
      tracks.forEach((ts,n)=>ts.forEach((t,k)=>t.classList.toggle("active",k===[e.i,e.j,e.k][n])));
      a.cells.forEach((row,i)=>row.forEach((cell,k)=>setCell(cell,trace.a[i][k],i===e.i?"row-on "+(k===e.k?"selected":""):"")));
      b.cells.forEach((row,k)=>row.forEach((cell,j)=>setCell(cell,trace.b[k][j],j===e.j?"col-on "+(k===e.k?"selected":""):"")));
      const result=trace.result.map(r=>r.map(()=>0));trace.events.slice(0,s.done).forEach(v=>result[v.i][v.j]=v.partial);
      c.cells.forEach((row,i)=>row.forEach((cell,j)=>setCell(cell,result[i][j],i===e.i&&j===e.j?"selected":"")));
      document.querySelectorAll("#javaCode [data-line]").forEach(l=>l.classList.toggle("active",l.dataset.line===s.line));
      $("loopArithmetic").textContent=e.left+" × ("+e.right+") → c["+e.i+"]["+e.j+"] = "+result[e.i][e.j];
      $("loopCounts").textContent=s.done+" products · "+s.done+" additions";
      const action={allocate:"Allocate C; Java initializes all entries to 0.",i:"The outer i loop selects output row "+e.i+".",j:"The j loop selects output column "+e.j+".",k:"The inner k loop selects matching position "+e.k+".",add:"Multiply "+e.left+" × ("+e.right+"), then add "+e.product+" to c["+e.i+"]["+e.j+"].",return:"Return the completed array [[0, 21], [4, 23]]."};
      $("loopsStatus").textContent="State "+(index+1)+"/"+states.length+" · i="+e.i+", j="+e.j+", k="+e.k+". "+action[s.line];
    },850);
  }

  function geometry(g) {
    let order="ab",step=0,current=[[1,0],[0,1]],currentVector=[1,1];const owner={};
    function draw(m,v){
      const w=$("geometryCanvas").clientWidth<400?440:600;
      const ctx=canvas("geometryCanvas",w,440);const ox=w/2-5,oy=230,u=55;
      const pt=p=>[ox+p[0]*u,oy-p[1]*u];
      const mv=p=>[m[0][0]*p[0]+m[0][1]*p[1],m[1][0]*p[0]+m[1][1]*p[1]];
      ctx.save();ctx.beginPath();ctx.rect(10,10,w-20,420);ctx.clip();
      for(let k=-6;k<=6;k++){
        let a=pt(mv([k,-6])),b=pt(mv([k,6]));line(ctx,...a,...b,"#c6cbb0",1);
        a=pt(mv([-6,k]));b=pt(mv([6,k]));line(ctx,...a,...b,"#c6cbb0",1);
      }
      line(ctx,15,oy,w-15,oy,C.muted);line(ctx,ox,15,ox,425,C.muted);
      for(let k=-4;k<=4;k++)if(k){text(ctx,k,ox+k*u,oy+15,11,C.muted);text(ctx,k,ox-15,oy-k*u,11,C.muted);}
      ctx.setLineDash([5,5]);arrow(ctx,ox,oy,ox+u,oy-u,C.muted,2);ctx.setLineDash([]);
      [[1,0],[0,1]].forEach((e,k)=>{const end=pt(mv(e));arrow(ctx,ox,oy,...end,k?C.gold:C.teal,2);text(ctx,"e"+(k+1),end[0]+(k?17:-16),end[1]+18,14,k?C.gold:C.teal);});
      const end=pt(v);arrow(ctx,ox,oy,...end,C.terra,4);rect(ctx,end[0]-42,end[1]-37,84,25,C.paper,C.terra,5);text(ctx,"("+v.map(x=>Math.round(x*10)/10).join(", ")+")",end[0],end[1]-24,13,C.terra);
      ctx.restore();text(ctx,"x₁",w-30,oy-16,15,C.muted);text(ctx,"x₂",ox+18,25,15,C.muted);
    }
    function render(animate){
      const I=[[1,0],[0,1]],m=step===0?I:step===1?(order==="ab"?g.b:g.a):g[order];
      const vec=(step===0?g.x:step===1?g[order==="ab"?"bx":"ax"]:g[order+"x"]).flat();
      const start=current.map(r=>r.slice()),startV=currentVector.slice();
      tween(owner,t=>{current=m.map((r,i)=>r.map((v,j)=>start[i][j]+(v-start[i][j])*t));currentVector=vec.map((v,i)=>startV[i]+(v-startV[i])*t);draw(current,currentVector);},animate);
      document.querySelectorAll("#geometryOrder button").forEach(b=>b.setAttribute("aria-pressed",String(b.dataset.order===order)));
      document.querySelectorAll("#geometrySteps button").forEach(b=>b.setAttribute("aria-pressed",String(+b.dataset.step===step)));
      const descriptions=order==="ab"?["Start with x = (1, 1).","B rotates x to (−1, 1).","A shears Bx to (0, 1).","The product AB sends x directly to (0, 1)."]:["Start with x = (1, 1).","A shears x to (2, 1).","B rotates Ax to (−1, 2).","The product BA sends x directly to (−1, 2)."];
      $("geometryStatus").textContent=descriptions[step]+" "+(step===3?"Same endpoint as the two actions.":"");
      $("geometryCode").textContent=step===0?"double[][] x = {{1}, {1}};":step===1?"multiply("+(order==="ab"?"B":"A")+", x)":step===2?"multiply("+(order==="ab"?"A, multiply(B, x)":"B, multiply(A, x)")+")":"multiply(multiply("+order.toUpperCase().split("").join(", ")+"), x)";
    }
    document.querySelectorAll("#geometryOrder button").forEach(b=>b.onclick=()=>{order=b.dataset.order;render(true);});
    document.querySelectorAll("#geometrySteps button").forEach(b=>b.onclick=()=>{step=+b.dataset.step;if(step===3){current=[[1,0],[0,1]];currentVector=[1,1];}render(true);});window.addEventListener("resize",()=>render(false));render(false);
  }

  function scale(data) {
    let selected=0,api;
    const buttons=data.map((d,k)=>{const b=el("button","",d.n+"×"+d.n);b.type="button";b.onclick=()=>{selected=k;api.go(0);};$("scaleSizes").append(b);return b;});
    api=player("scale",25,index=>{
      const narrow=$("scaleCanvas").clientWidth<500,w=narrow?450:900;
      const d=data[selected],n=d.n,ctx=canvas("scaleCanvas",w,narrow?565:340),size=270/n,x=narrow?90:100,y=35,progress=index/24;
      buttons.forEach((b,k)=>b.setAttribute("aria-pressed",String(k===selected)));
      for(let i=0;i<n;i++)for(let j=0;j<n;j++){
        const active=(i*n+j)/d.cells<=progress;
        ctx.fillStyle=active?((i+j+index)%7===0?C.terra:C.teal):"#e8dfc9";
        ctx.fillRect(x+j*size,y+i*size,Math.max(1,size-1),Math.max(1,size-1));
        if(n<=4)text(ctx,"·",x+(j+.5)*size,y+(i+.5)*size,20,active?C.paper:C.muted);
      }
      text(ctx,n+" × "+n+" result",x+135,322,16);
      const cx=narrow?225:645,dy=narrow?300:0;
      text(ctx,"ONE OUTPUT ENTRY",cx,58+dy,14,C.muted);
      for(let k=0;k<Math.min(n,16);k++){const left=cx-155+k%8*40,top=95+dy+Math.floor(k/8)*40;rect(ctx,left,top,30,30,k<=Math.floor(progress*n)?C.yellow:C.paper,C.gold,4);if(n<=8)text(ctx,"×",left+15,top+15,16,C.gold);}
      text(ctx,n<=16?n+" products along k":n+" products · 16 sampled here",cx,200+dy,18,C.terra);
      if(!narrow)arrow(ctx,515,235,775,235,C.gold,2);
      text(ctx,"i × j choose a cell; k fills it.",cx,narrow?540:273,16,C.muted);
      $("scaleCells").textContent=fmt(d.cells);$("scaleTerms").textContent=fmt(n);$("scaleProducts").textContent=fmt(d.products);
      $("scaleStatus").textContent=n+"×"+n+": "+fmt(d.cells)+" entries × "+n+" products = "+fmt(d.products)+". "+(index===0?"Play to sweep a sampled field.":"Field sample "+index+"/24; not measured CPU activity.");
    },180);
    window.addEventListener("resize",()=>api.go(api.index,false));
  }

  function tileGrid(id,values) {
    const box=$(id);box.replaceChildren();
    const cells=values.flat().map((v,k)=>{const cell=el("span","",v);cell.style.setProperty("--bi",Math.floor(k/8));cell.style.setProperty("--bj",Math.floor(k%4/2));box.append(cell);return cell;});
    return {box,cells};
  }
  function tiles(trace) {
    const a=tileGrid("tileA",trace.a),b=tileGrid("tileB",trace.b),c=tileGrid("tileC",trace.result.map(r=>r.map(()=>0)));
    let tiled=false;
    for(let k=0;k<256;k++){const s=el("span");s.style.setProperty("--tx",Math.floor(k%16/4));s.style.setProperty("--ty",Math.floor(k/64));$("denseTileGrid").append(s);}
    const miniBlocks=["mainMemory","cacheMemory","processorMemory"].map(id=>{const block=el("div","memory-blocks");$(id).append(block);return block;});
    function layout(value){tiled=value;[a,b,c].forEach(g=>g.box.classList.toggle("tiled",tiled));$("denseTileGrid").classList.toggle("tiled",tiled);$("tileLayout").setAttribute("aria-pressed",String(tiled));$("tileLayout").textContent=tiled?"Show ordinary cell grid":"Reorganize into tiles";}
    $("tileLayout").onclick=()=>layout(!tiled);
    player("tiles",trace.events.length*3+1,(index,animate)=>{
      const eventIndex=Math.max(0,Math.floor((index-1)/3)),phase=index===0?-1:(index-1)%3,e=trace.events[eventIndex];
      layout(index>0);
      const partial=index===0?trace.result.map(r=>r.map(()=>0)):phase===2?e.partial:eventIndex?trace.events[eventIndex-1].partial:trace.result.map(r=>r.map(()=>0));
      [[a,e.aTile,trace.a],[b,e.bTile,trace.b],[c,e.outputTile,partial]].forEach(([grid,tile,values])=>grid.cells.forEach((cell,k)=>{
        const i=Math.floor(k/4),j=k%4;cell.textContent=values[i][j];cell.className=index>0&&Math.floor(i/2)===tile[0]&&Math.floor(j/2)===tile[1]?"on":"";
      }));
      ["mainMemory","cacheMemory","processorMemory"].forEach((id,k)=>$(id).classList.toggle("active",k===phase));
      $("memoryTile").textContent="A("+e.aTile+") + B("+e.bTile+")";
      $("cacheTile").textContent="2 × 2 entries per tile";
      $("processorTile").textContent="C("+e.outputTile+") += A × B";
      miniBlocks.forEach((block,k)=>{
        block.replaceChildren();
        const tiles=k===2?[[partial,e.outputTile]]:[[trace.a,e.aTile],[trace.b,e.bTile]];
        tiles.forEach(([m,where])=>{const group=el("div","mini-tile");for(let i=0;i<2;i++)for(let j=0;j<2;j++)group.append(el("span","",m[where[0]*2+i][where[1]*2+j]));block.append(group);});
        block.style.opacity=phase < k ? 0.25 : 1;
      });
      if(phase===1)fly(miniBlocks[1],miniBlocks[0],miniBlocks[1],animate);
      if(phase===2)fly(miniBlocks[2],miniBlocks[1],miniBlocks[2],animate);
      document.querySelectorAll("#blockedCode [data-line]").forEach(n=>n.classList.toggle("active",index>0&&n.dataset.line===(phase===2?"compute":"select")));
      $("tilesStatus").textContent=index===0?"Begin with the ordinary grid. Reorganize it into 2×2 tiles, then follow the Java loop order.":"Tile event "+(eventIndex+1)+"/8 · ii="+e.ii+", kk="+e.kk+", jj="+e.jj+". "+["Select A and B tiles in the ii → kk → jj loop order.","Illustrate keeping those tiles nearby for reuse.","Java has accumulated this block product into C; the displayed partial values are recorded by the kernel."][phase];
    },1000);
  }

  function sparse(data) {
    let faded=false;
    data.matrix.flat().forEach(v=>$("sparseGrid").append(el("span",v===0?"zero":"",v)));
    const paths=[["0 × B[0][j] → 0","zero"],["4 × B[1][j] → useful contribution","useful"],["0 × B[2][j] → 0","zero"],["0 × B[3][j] → 0","zero"],["0 × B[4][j] → 0","zero"],["0 × B[5][j] → 0","zero"]];
    paths.forEach(([s,c])=>$("sparsePaths").append(el("div",c,s)));
    [["values",data.values],["columns",data.columns],["rowOffsets",data.rowOffsets]].forEach(([name,v])=>$("csrData").append(el("div","",name+" = ["+v.join(", ")+"]")));
    function render(){["sparseGrid","sparsePaths"].forEach(id=>$(id).classList.toggle("fade",faded));$("sparseToggle").setAttribute("aria-pressed",String(faded));$("sparseToggle").textContent=faded?"Restore all zero products":"Fade zero products";$("sparseStatus").textContent=faded?"First output row: only 4 × B[1][j] contributes. Across all six rows, CSR visits 7 stored entries instead of 36 positions.":"First output row: five of its six possible products are zero. The full matrix contains 29 zero entries.";}
    $("sparseToggle").onclick=()=>{faded=!faded;render();};render();
  }

  function parallel(trace) {
    const c=tileGrid("parallelC",trace.result.map(r=>r.map(()=>"·")));c.box.classList.add("tiled");
    const lanes=trace.tasks.map(t=>{
      const lane=el("div","worker-lane"),track=el("div","worker-track"),tile=el("span","task-tile","C("+t.row/2+","+t.col/2+")");
      lane.append(el("span","","Worker "+(t.lane+1)));track.append(tile);lane.append(track);$("workerLanes").append(lane);return tile;
    });
    player("parallel",5,index=>{
      lanes.forEach((tile,k)=>{
        tile.className="task-tile"+(index===2?" computing":index>=3?" returned":"");
        tile.textContent=index===0?"queued":index>=3?trace.tasks[k].result.flat().join(","):"C("+trace.tasks[k].row/2+","+trace.tasks[k].col/2+")";
      });
      c.cells.forEach((cell,k)=>{cell.textContent=index>=3?trace.result[Math.floor(k/4)][k%4]:"·";cell.className=index>=3?"done":"";});
      $("parallelStatus").textContent=[
        "Four distinct output tiles are queued as RecursiveAction tasks.",
        "Illustrative assignment: one result tile per worker. Every task reads A and B.",
        "Workers sweep the full k dimension inside their own tiles. Inputs are shared; output writes are disjoint.",
        "Computed tiles return their values to the corresponding C regions. These values were produced by Java’s ForkJoinPool.",
        "join() waits for completion; the parallel result "+(trace.verified?"matches":"does not match")+" the classical result. The pool is shut down in finally."
      ][index];
    },1300);
  }

  function strassen(trace) {
    let previous=0;const owner={};
    const products=trace.products.map(p=>{const card=el("div");card.append(el("span","",p.name),el("b","","·"));$("sevenProducts").append(card);return card;});
    function draw(stage,t,from) {
      const narrow=$("strassenCanvas").clientWidth<500,w=narrow?450:900;
      const ctx=canvas("strassenCanvas",w,narrow?500:450),center=w/2;
      const ratio=from===0&&stage===1?t:stage===0?0:1;
      const split=stage>=1?12:0;
      [["a","b","c","d"],["e","f","g","h"]].forEach((labels,g)=>{
        const ox=g?center+28:center-146;
        labels.forEach((v,k)=>{const x=ox+(k%2)*(38+split),y=26+Math.floor(k/2)*(32+split);rect(ctx,x,y,36,30,g?C.yellow:C.green,g?C.gold:C.teal,4);text(ctx,v,x+18,y+15,16);});
      });
      text(ctx,"×",center-2,60,24,C.terra);text(ctx,"Four quadrants in each input",center,137,15,C.muted);
      const count=stage===0?8:7,spread=narrow?390:760,start=narrow?30:70,box=narrow?42:70;
      const normal=["ae","bg","af","bh","ce","dg","cf","dh"];
      for(let k=0;k<(ratio>0&&ratio<1?8:count);k++){
        const x8=start+k*spread/7,x7=start+k*spread/6,x=x8+(x7-x8)*ratio;
        if(k===7&&ratio>0){ctx.globalAlpha=1-ratio;}
        const shown=stage>=2&&k<=stage-2,selected=stage>=2&&stage<=8&&k===stage-2;
        line(ctx,center,153,x,218,shown?C.terra:C.line,selected?3:1.5);
        rect(ctx,x-box/2,218,box,49,shown?C.peach:C.paper,selected?C.terra:C.gold);
        text(ctx,stage===0?normal[k]:"M"+(k+1),x,242,narrow?14:16,shown?C.terra:C.ink);
        if(stage>=10&&k<7){
          for(let j=0;j<7;j++){const xx=x-box/2+j*box/6;line(ctx,x,267,xx,318,C.gold,1);rect(ctx,xx-2,318,4,8,C.gold,null,1);}
          if(stage>=11)for(let j=0;j<49;j++){const xx=x-box/2+j*box/49;ctx.fillStyle=C.teal;ctx.fillRect(xx,345,Math.max(.6,box/60),5);}
        }
        ctx.globalAlpha=1;
      }
      const bottom=stage===0?"8 block multiplications · two per output quadrant":stage===1?"Reorganize sums and differences → 7 products":stage<=8?trace.products[stage-2].name+" = "+trace.products[stage-2].formula+" = "+trace.products[stage-2].value:stage===9?"Recombine → C = [[7, 5], [8, 20]]":stage===10?"1 → 7 → 49 recursive subproblems":"1 → 7 → 49 → 343 · half the width at each level";
      wrappedText(ctx,bottom,center,stage>=10?385:320,w-35,narrow?17:19,C.terra);
      wrappedText(ctx,stage>=10?"T(n) = 7T(n/2) + O(n²)":"Products feed linear combinations of the four output blocks.",center,narrow?452:428,w-30,15,C.muted);
    }
    const api=player("strassen",12,(stage,animate)=>{
      const from=previous;previous=stage;tween(owner,t=>draw(stage,t,from),animate&&from===0&&stage===1);
      products.forEach((card,k)=>{card.className=stage>=2&&k<=stage-2?"revealed":"";if(stage===k+2)card.classList.add("active");card.querySelector("b").textContent=stage>=2&&k<=stage-2?trace.products[k].value:"·";});
      $("strassenStatus").textContent=stage===0?"Ordinary block multiplication: four output quadrants each need two products. Advance to reorganize the eight branches.":stage===1?"Seven new branches multiply linear combinations. The extra additions and subtractions make the saving possible.":stage<=8?trace.products[stage-2].name+" = "+trace.products[stage-2].formula+" = "+trace.products[stage-2].value+" for A=[[3,1],[2,4]], B=[[2,0],[1,5]].":stage===9?"Java recombines the seven products to [[7,5],[8,20]], matching classical multiplication.":"Apply the same seven-product rule recursively to each half-size subproblem. "+(stage===10?"At depth two: 49 children instead of 64.":"At depth three: 343 leaves instead of 512. The exponent becomes log₂7 ≈ 2.8074.");
    },1300);
    window.addEventListener("resize",()=>api.go(api.index,false));
    $("strassenVerified").textContent="Java verification: "+JSON.stringify(trace.result)+" "+(trace.verified?"equals the classical product.":"FAILED.")+" Tests also compare deterministic 1×1, 2×2, 4×4, 8×8, and 16×16 inputs.";
    const slider=$("growthSize");function growth(){const p=+slider.value,n=2**p,a=8**p,b=7**p;$("growthSizeOut").value=fmt(n);$("classicCount").textContent=fmt(a);$("strassenCount").textContent=fmt(b);$("classicBar").style.width="100%";$("strassenBar").style.width=b/a*100+"%";}slider.oninput=growth;growth();
  }

  function precision(data) {
    let reverse=false;
    function render(animate){
      const old=[...$("precisionTerms").children].filter(n=>n.tagName==="SPAN").map(n=>n.getBoundingClientRect());
      $("precisionTerms").replaceChildren();
      const terms=reverse?[...data.terms].reverse():data.terms;
      terms.forEach((v,k)=>{if(k)$("precisionTerms").append(el("i","","+"));const chip=el("span","",v);$("precisionTerms").append(chip);
        if(animate&&!motion.matches&&old[2-k]){const r=chip.getBoundingClientRect();chip.animate([{transform:"translateX("+(old[2-k].left-r.left)+"px)"},{transform:"translateX(0)"}],{duration:500,easing:"ease"});}
      });
      $("precisionCode").textContent=reverse?"sum(terms, true)":"sum(terms, false)";$("precisionValue").textContent=reverse?data.reverse:data.forward;
      $("precisionToggle").textContent=reverse?"Restore forward order":"Reverse the accumulation";$("precisionToggle").setAttribute("aria-pressed",String(reverse));
      $("precisionStatus").textContent="Java double result, "+(reverse?"last term first":"first term first")+": "+(reverse?data.reverse:data.forward)+". The displayed digits come from Java’s generated trace.";
    }
    $("precisionToggle").onclick=()=>{reverse=!reverse;render(true);};render(false);
  }

  function recap() {
    const names=["Scalar product","Output entry","Matrix","Java loops","Computation field","Tiles","Workers","Strassen","ω"];
    const messages=[
      "Two input entries produce one scalar product. Java: a[i][k] * b[k][j].",
      "Products accumulate into a single output entry, c[i][j].",
      "Every row–column pair gives an entry; together they form C.",
      "i chooses a row, j chooses a column, k traverses the dot product.",
      "At scale, n² entries each need n products: n³ classical multiplications.",
      "Group the field into tiles to reuse nearby data. Arithmetic count stays cubic.",
      "Assign independent result tiles to Java tasks and join their computed values.",
      "Reorganize eight block multiplications into seven, then recurse.",
      "Ask the wider question: what arithmetic exponent is achievable in principle?"
    ];
    const owner={};let previous=0,api;
    const buttons=names.map((name,k)=>{const b=el("button","",name);b.type="button";b.onclick=()=>api.go(k);$("recapSteps").append(b);return b;});
    function position(stage,k) {
      if(stage===0)return {x:350+(k%2)*200,y:155+Math.floor(k/2)*.5,s:7};
      if(stage===1)return {x:418+(k%8)*8,y:130+Math.floor(k/8)*8,s:6};
      if(stage===2||stage===4)return {x:350+(k%8)*25,y:70+Math.floor(k/8)*25,s:stage===2?20:8};
      if(stage===3)return {x:240+Math.floor(k/22)*210,y:85+(k%22)*7,s:5};
      if(stage===5)return {x:330+(k%8)*25+Math.floor(k%8/4)*20,y:60+Math.floor(k/8)*25+Math.floor(k/32)*20,s:18};
      if(stage===6)return {x:290+(k%16)*20,y:80+Math.floor(k/16)*50,s:12};
      if(stage===7)return {x:90+(k%7)*120+(Math.floor(k/7)%3-1)*14,y:100+Math.floor(k/7)*17,s:6};
      const a=k/64*Math.PI*1.7+.65;return {x:450+90*Math.cos(a),y:165+75*Math.sin(a),s:8};
    }
    api=player("recap",9,(stage,animate)=>{
      const from=previous;previous=stage;buttons.forEach((b,k)=>b.setAttribute("aria-pressed",String(k===stage)));
      tween(owner,t=>{
        const narrow=$("recapCanvas").clientWidth<500,w=narrow?450:900,center=w/2;
        const ctx=canvas("recapCanvas",w,350),x=v=>center+(v-450)*(narrow?.5:1);
        const place=(s,k)=>{const p=position(s,k);p.x=center+(p.x-450)*(narrow?([3,6,7].includes(s)?.5:1.2):1);return p;};
        if(stage===3){["i","j","k"].forEach((v,k)=>text(ctx,v,x(240+k*210),48,25,C.terra));text(ctx,"for → for → for",center,285,20,C.teal);}
        if(stage===6)for(let k=0;k<4;k++)text(ctx,"W"+(k+1),x(205),85+k*50,16,C.muted);
        if(stage===7)for(let k=0;k<7;k++)line(ctx,center,30,x(90+k*120),100,C.gold,2);
        for(let k=0;k<64;k++){const p=place(from,k),q=place(stage,k);rect(ctx,p.x+(q.x-p.x)*t,p.y+(q.y-p.y)*t,p.s+(q.s-p.s)*t,p.s+(q.s-p.s)*t,k%3===0?C.terra:k%3===1?C.teal:C.gold,null,2);}
        if(stage===0){text(ctx,"2",center-(narrow?120:85),120,35,C.teal);text(ctx,"×",center,165,32,C.muted);text(ctx,"5",center+(narrow?120:115),120,35,C.gold);text(ctx,"= 10",center,260,24,C.terra);}
        if(stage===1)text(ctx,"c[i][j]",center,250,22,C.teal);
        if(stage===8){text(ctx,"ω",center,160,58,C.terra);text(ctx,"2 ≤ ω < 2.371177",center,287,22,C.ink);}
        if(![0,1,3,8].includes(stage))text(ctx,names[stage],center,310,22,C.ink);
      },animate);
      $("recapStatus").textContent=(stage+1)+"/9 · "+messages[stage];
    },1500);
    window.addEventListener("resize",()=>api.go(api.index,false));
  }

  async function load(file,init,ids) {
    try{const r=await fetch(ROOT+file+".json");if(!r.ok)throw new Error("HTTP "+r.status);init(await r.json());}
    catch(error){
      ids.forEach(id=>{const status=$(id);status.textContent="This Java trace could not load ("+error.message+"). Use the local HTTP preview; the explanation and source remain available.";status.classList.add("error");});
      console.error(file,error);
    }
  }
  Promise.all([
    load("row-column",d=>{theatre(d);representation(d);loops(d);},["theatreStatus","representationStatus","loopsStatus"]),
    load("blocked",tiles,["tilesStatus"]),
    load("parallel",parallel,["parallelStatus"]),
    load("strassen",strassen,["strassenStatus","strassenVerified"]),
    load("learning",d=>{geometry(d.geometry);scale(d.scale);sparse(d.sparse);precision(d.precision);},["geometryStatus","scaleStatus","sparseStatus","precisionStatus"])
  ]);
  recap();
  document.querySelectorAll(".equation, pre").forEach(n=>n.tabIndex=0);
})();
