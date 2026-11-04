/* Pure teaching calculations; shared by the browser labs and numerical checks. */
(function(root,factory){const api=factory();if(typeof module==='object'&&module.exports)module.exports=api;else root.RCTMath=api;})(typeof globalThis!=='undefined'?globalThis:this,function(){
  'use strict';
  function cdf(x){if(x===0)return .5;if(x>9)return 1;if(x< -9)return 0;const z=Math.abs(x);let term=z,sum=z;for(let k=1;k<500;k++){term*=z*z/(2*k+1);sum+=term;if(term<sum*1e-16)break;}const area=sum*Math.exp(-z*z/2)/Math.sqrt(2*Math.PI);return Math.max(0,Math.min(1,.5+Math.sign(x)*area));}
  const quantiles=new Map();
  function quantile(p){if(quantiles.has(p))return quantiles.get(p);let lo=-10,hi=10;for(let i=0;i<70;i++){const m=(lo+hi)/2;if(cdf(m)<p)lo=m;else hi=m;}const z=(lo+hi)/2;quantiles.set(p,z);return z;}
  function planning(n,d,p0,alpha=.05){const p1=p0-d,avg=(p0+p1)/2,s0=Math.sqrt(2*avg*(1-avg)/n),sA=Math.sqrt((p0*(1-p0)+p1*(1-p1))/n),critical=quantile(1-alpha/2),mu=-d/s0,sigma=sA/s0;const power=cdf((-critical-mu)/sigma)+1-cdf((critical-mu)/sigma);return {p1,s0,sA,critical,mu,sigma,power,beta:1-power};}
  function effects(a,c,n=100){const p1=a/n,p0=c/n,rd=p1-p0,rr=p1/p0,se=Math.sqrt(1/a+1/c-2/n),z=quantile(.975);return {p1,p0,rd,rr,rrr:1-rr,or:(a/(n-a))/(c/(n-c)),lo:Math.exp(Math.log(rr)-z*se),hi:Math.exp(Math.log(rr)+z*se)};}
  return {cdf,quantile,planning,effects};
});
