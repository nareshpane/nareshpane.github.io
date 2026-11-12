/* Standalone browser implementation; no vendor IV or pricing library. */
(function(root){'use strict';
const phi=x=>Math.exp(-x*x/2)/Math.sqrt(2*Math.PI);
function N(x){ // Abramowitz–Stegun 26.2.17, maximum absolute error about 7.5e-8
 if(x===0)return .5;const a=Math.abs(x),t=1/(1+.2316419*a);const tail=phi(a)*t*(.319381530+t*(-.356563782+t*(1.781477937+t*(-1.821255978+t*1.330274429))));return x>0?1-tail:tail;
}
function price(S,K,T,r,v,q=0,kind='call'){
 if(![S,K,T,r,v,q].every(Number.isFinite)||S<=0||K<=0||T<0||v<0)throw Error('Invalid input');
 const z=kind==='call'?1:-1,a=S*Math.exp(-q*T),b=K*Math.exp(-r*T);
 if(!T||!v)return Math.max(z*(a-b),0);
 const d1=(Math.log(S/K)+(r-q+v*v/2)*T)/(v*Math.sqrt(T)),d2=d1-v*Math.sqrt(T);
 return z*(a*N(z*d1)-b*N(z*d2));
}
function greeks(S,K,T,r,v,q=0,kind='call'){
 const d1=(Math.log(S/K)+(r-q+v*v/2)*T)/(v*Math.sqrt(T)),d2=d1-v*Math.sqrt(T),a=Math.exp(-q*T),b=Math.exp(-r*T),z=kind==='call'?1:-1;
 return {delta:z*a*N(z*d1),gamma:a*phi(d1)/(S*v*Math.sqrt(T)),vega:S*a*phi(d1)*Math.sqrt(T)/100,theta:(-S*a*phi(d1)*v/(2*Math.sqrt(T))+z*q*S*a*N(z*d1)-z*r*K*b*N(z*d2))/365,rho:z*K*T*b*N(z*d2)/100};
}
function rng(seed){let s=seed>>>0;return ()=>{s+=0x6D2B79F5;let t=s;t=Math.imul(t^(t>>>15),t|1);t^=t+Math.imul(t^(t>>>7),t|61);return ((t^(t>>>14))>>>0)/4294967296;};}
function paths(seed=1729,mu=.06,v=.25,n=8){const u=rng(seed),normal=()=>Math.sqrt(-2*Math.log(Math.max(1e-12,u())))*Math.cos(2*Math.PI*u());return Array.from({length:n},()=>{let s=100;return [s,...Array.from({length:252},()=>s*=Math.exp((mu-v*v/2)/252+v/Math.sqrt(252)*normal()))];});}
root.BSM={N,phi,price,greeks,paths};if(typeof module!=='undefined')module.exports=root.BSM;
})(typeof window==='undefined'?globalThis:window);
