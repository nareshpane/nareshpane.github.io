/* Pure numerical functions shared by the page and Node validation. */
(function(root){'use strict';
const z={90:1.6448536269514722,95:1.959963984540054,99:2.5758293035489004};
function rng(seed){let a=seed>>>0;return()=>{a+=0x6D2B79F5;let t=a;t=Math.imul(t^t>>>15,t|1);t^=t+Math.imul(t^t>>>7,t|61);return((t^t>>>14)>>>0)/4294967296;};}
function normal(r){return Math.sqrt(-2*Math.log(Math.max(1e-12,r())))*Math.cos(2*Math.PI*r());}
function interval(x,se,level=95){if(!Number.isFinite(x)||!Number.isFinite(se)||se<0||!z[level])throw Error('Invalid interval inputs');const margin=z[level]*se;return {lo:x-margin,hi:x+margin,margin};}
function difference(a,b,sa,sb,rho,level=95){if(sa<0||sb<0||Math.abs(rho)>1)throw Error('Invalid covariance inputs');const variance=Math.max(0,sa*sa+sb*sb-2*rho*sa*sb),se=Math.sqrt(variance),delta=b-a;return{delta,variance,se,covariance:rho*sa*sb,z:se>0?delta/se:null,...interval(delta,se,level)};}
function moments(xs,truth){const mean=xs.reduce((a,b)=>a+b,0)/xs.length,variance=xs.reduce((a,b)=>a+(b-mean)**2,0)/xs.length,bias=mean-truth;return{mean,variance,bias,sd:Math.sqrt(variance),rmse:Math.sqrt(variance+bias*bias)};}
function adjustedCV(p,se,q){return se/p*Math.sqrt(p/(1-p))*Math.sqrt((1-q)/q);}
function choose(N,n,r){const map=new Map(),out=[];for(let j=0;j<n;j++){const k=j+Math.floor(r()*(N-j)),v=map.has(k)?map.get(k):k;map.set(k,map.has(j)?map.get(j):j);out.push(v);}return out;}
function sampling(p,n,reps,scheme,seed){const N=12000,K=Math.round(N*p),K0=Math.round(K*.25),K1=K-K0,r=rng(seed),values=[];const unemployed=i=>i<6000?i<K0:i-6000<K1;
for(let b=0;b<reps;b++){let ids;if(scheme==='srs')ids=choose(N,n,r);else if(scheme==='stratified')ids=choose(6000,n/2,r).concat(choose(6000,n/2,r).map(i=>i+6000));else ids=choose(600,n/20,r).flatMap(c=>Array.from({length:20},(_,j)=>20*c+j));values.push(ids.reduce((s,i)=>s+unemployed(i),0)/n);}return{values,truth:K/N,...moments(values,K/N)};}
function bootstrap(B,seed){const r=rng(seed),U=[[2,4,3,7],[6,10,8,12]],values=[];for(let b=0;b<B;b++){let u=0,l=0;for(let h=0;h<2;h++)for(let j=0;j<3;j++){u+=4/3*U[h][Math.floor(r()*4)];l+=4/3*50;}values.push(u/l);}return{values,truth:52/400,...moments(values,52/400)};}
const api={z,rng,normal,interval,difference,moments,adjustedCV,choose,sampling,bootstrap};if(typeof module!=='undefined')module.exports=api;else root.LFSStats=api;
})(typeof window!=='undefined'?window:globalThis);
