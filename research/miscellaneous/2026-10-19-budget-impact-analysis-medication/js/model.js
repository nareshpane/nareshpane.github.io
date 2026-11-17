/* Pure calculation module. CAD, 52-week years, half-open intervals. No DOM. */
export const defaults = Object.freeze({n:120,growth:.10,shares:[.30,.40,.50],secPrice:882.59,adaPrice:471.27,frequency:4,escalation:0,discount:0,adaDiscount:0,inflation:0,fee:0,markup:0,admin:0,publicPay:1,discontinuation:0,mortality:0,loss:0,back:0,initiations:0,delay:0});
export const historical = Object.freeze({adult:4478159,prevalence:.01,diagnosed:.19,severity:.56,biologic:.035,coverage:.73});
export function population(p){let n=p.adult;return [n,...['prevalence','diagnosed','severity','biologic','coverage'].map(k=>n*=p[k])];}
export function schedule(drug,frequency=4,until=160){
 const out=[];if(drug==='S'){for(let w=0;w<=4;w++)out.push({week:w,units:2,loading:true});for(let w=4+frequency;w<until;w+=frequency)out.push({week:w,units:2,loading:false});}
 else {out.push({week:0,units:4,loading:true},{week:2,units:2,loading:true});for(let w=4;w<until;w++)out.push({week:w,units:1,loading:false});}return out;
}
export function courseCost(drug,start,from,to,p,year=0){
 const calc=f=>{let units=0,events=0,loading=0;for(const d of schedule(drug,f,Math.max(0,to-start)+1)){const w=start+d.week;if(w>=from-1e-8&&w<to-1e-8){units+=d.units;events++;if(d.loading)loading+=d.units;}}
 const unit=(drug==='S'?p.secPrice*(1-p.discount):p.adaPrice*(1-p.adaDiscount))*Math.pow(1+p.inflation,year);
 const acquisition=units*unit,extras=acquisition*p.markup+events*(p.fee+p.admin);
 return {units,events,loading,acquisition,extras,total:acquisition+extras,induction:loading*unit*(1+p.markup)};};
 const a=calc(p.frequency);if(drug!=='S'||p.frequency===2)return a;const b=calc(2);return Object.fromEntries(Object.keys(a).map(k=>[k,a[k]*(1-p.escalation)+b[k]*p.escalation]));
}
export function annualCosts(p){return {secFirst:courseCost('S',0,0,52,p),secMaintenance:courseCost('S',0,52,104,p),adaFirst:courseCost('A',0,0,52,p),adaMaintenance:courseCost('A',0,52,104,p)};}
const sum=(a,f)=>a.reduce((s,x)=>s+f(x),0);
function take(cohorts,drug,amount){const total=sum(cohorts,x=>x.drug===drug?x.n:0),taken=[];if(total<=0)return taken;const q=Math.min(1,Math.max(0,amount/total));for(const c of cohorts){if(c.drug===drug){const n=c.n*q;c.n-=n;taken.push({...c,n});}}return taken;}
function bill(cohorts,from,to,p,year){return sum(cohorts,c=>c.n*courseCost(c.drug,c.start,Math.max(from,c.start),to,p,year).total);}
export function budget(input={}){
 const p={...defaults,...input,shares:[...(input.shares||defaults.shares)]};let ref=[{drug:'A',start:-104,n:p.n}],neo=[{drug:'A',start:-104,n:p.n}],prev=p.n;const rows=[];
 for(let t=0;t<3;t++){
  const begin=t*52,end=begin+52;const survival=t?((1-p.mortality)*(1-p.loss)*(1-p.discontinuation)):1;
  const removed=prev*(1-survival);for(const c of [...ref,...neo])c.n*=survival;
  const entrants=t?prev*p.growth+p.initiations:0;
  if(entrants){ref.push({drug:'A',start:begin,n:entrants});neo.push({drug:'A',start:begin,n:entrants});}
  const n=prev*survival+entrants,oldS=sum(neo,c=>c.drug==='S'?c.n:0),target=n*p.shares[t];
  // Returners leave S at the year boundary; adalimumab is reloaded (explicit teaching assumption).
  const returned=Math.max(oldS*p.back,oldS-target),backs=take(neo,'S',returned);
  let refCost=bill(ref,begin,end,p,t),newCost=0;
  const remainingS=sum(neo,c=>c.drug==='S'?c.n:0),switchN=Math.max(0,target-remainingS);
  const switched=take(neo,'A',switchN),when=begin+p.delay;
  for(const c of backs){neo.push({drug:'A',start:begin,n:c.n});}
  newCost+=bill(switched,begin,when,p,t);
  for(const c of switched)neo.push({drug:'S',start:when,n:c.n});
  newCost+=bill(neo,begin,end,p,t);
  const secCost=bill(neo.filter(c=>c.drug==='S'),begin,end,p,t),sec=sum(neo,c=>c.drug==='S'?c.n:0);
  rows.push({year:t+1,n,entrants,removed,sec,ada:n-sec,newSwitch:sum(switched,c=>c.n),returned:sum(backs,c=>c.n),secPY:remainingS+sum(switched,c=>c.n)*(52-p.delay)/52,referenceGross:refCost,newGross:newCost,secGross:secCost,reference:refCost*p.publicPay,newCost:newCost*p.publicPay,impact:(newCost-refCost)*p.publicPay});prev=n;
 }
 return {p,rows,total:sum(rows,r=>r.impact),reference:sum(rows,r=>r.reference),newCost:sum(rows,r=>r.newCost)};
}
export function payer(charge,{deductible=650,maximum=900,copay=.30,privateRate=0}={}){const d=Math.max(0,deductible),m=Math.max(d,maximum);const residual=Math.min(charge,m,Math.min(charge,d)+Math.max(0,charge-d)*copay);const privatePaid=residual*privateRate;return {charge,public:charge-residual,private:privatePaid,patient:residual-privatePaid};}
export function records(){
 // Disjoint membership cells: D only 40; M only 1000; N only 60; DM 60; DN 10; MN 120; DMN 60.
 const cells=[['D',40],['M',1000],['N',60],['DM',60],['DN',10],['MN',120],['DMN',60]],r=[];let i=0;
 for(const [sources,count]of cells)for(let j=0;j<count;j++){const id=i++;r.push({id:`PAT-${String(id+1).padStart(3,'0')}`,sources,msp:sources.includes('M')?(id%3?2:1):0,confirmed:id%5===0,resident:id%19!==0,adult:id%13!==0,window:id%17!==0,severe:id%3!==0,failed:id%4!==0,biologic:id%5!==1,coverage:id%7!==0,current:id%6!==0});}return r;
}
export function cohort(data,definition='broad',filters=[]){const qualifies=x=>definition==='broad'||(definition==='repeated'?(x.sources.includes('D')||x.msp>=2):x.confirmed);let selected=data.filter(qualifies),stages=[{label:'Case definition',n:selected.length}];for(const key of filters){selected=selected.filter(x=>x[key]);stages.push({label:key,n:selected.length});}return {selected,stages};}
export function rng(seed=20261019){let a=seed>>>0;return ()=>{a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return ((t^t>>>14)>>>0)/4294967296;};}
function normal(r){return Math.sqrt(-2*Math.log(Math.max(1e-12,r())))*Math.cos(2*Math.PI*r());}
function gamma(k,r){if(k<1)return gamma(k+1,r)*Math.pow(r(),1/k);const d=k-1/3,c=1/Math.sqrt(9*d);for(;;){const x=normal(r),v=Math.pow(1+c*x,3);if(v<=0)continue;const u=r();if(u<1-.0331*x*x*x*x||Math.log(u)<.5*x*x+d*(1-v+Math.log(v)))return d*v;}}
export function beta(mean,k,r){if(mean<=0||mean>=1)return mean;const a=gamma(mean*k,r),b=gamma((1-mean)*k,r);return a/(a+b);}
export function monteCarlo(p,count=500,seed=20261019,k=60,cv=.15){const r=rng(seed),values=[];for(let i=0;i<count;i++){const sigma=Math.sqrt(Math.log(1+cv*cv));const sampled={...p,n:p.n*Math.exp(.2*normal(r)-.02),secPrice:p.secPrice*Math.exp(sigma*normal(r)-sigma*sigma/2),adaPrice:p.adaPrice*Math.exp(sigma*normal(r)-sigma*sigma/2),shares:p.shares.map(s=>beta(s,k,r)).sort((a,b)=>a-b),publicPay:beta(p.publicPay,k,r)};values.push(budget(sampled).total);}const ordered=[...values].sort((a,b)=>a-b),q=f=>ordered[Math.floor(f*(count-1))];return {values,median:q(.5),p5:q(.05),p95:q(.95),positive:values.filter(x=>x>0).length/count,mean:sum(values,x=>x)/count,halfMean:sum(values.slice(0,Math.floor(count/2)),x=>x)/Math.floor(count/2)};}
export function decomposition(p){const base={...p,n:p.n,growth:0,initiations:0,discontinuation:0,mortality:0,loss:0,frequency:4,escalation:0,discount:0,adaDiscount:0,secPrice:p.adaPrice*52/26};const states=[0,budget(base).total,budget({...base,secPrice:p.secPrice,discount:p.discount,adaDiscount:p.adaDiscount}).total,budget({...base,secPrice:p.secPrice,discount:p.discount,adaDiscount:p.adaDiscount,frequency:p.frequency,escalation:p.escalation}).total,budget(p).total];return states.slice(1).map((v,i)=>({label:['Switching / induction at equal maintenance price','Price / discounts','Maintenance dosing / escalation','Volume / exits'][i],value:v-states[i],from:states[i],to:v}));}
