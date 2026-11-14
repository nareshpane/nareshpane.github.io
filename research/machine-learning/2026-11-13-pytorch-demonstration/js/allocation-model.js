// Original preference optimization and hard-capacity realization; added readouts are derived only.
export const softmax=row=>{const maximum=Math.max(...row),values=row.map(v=>Math.exp(v-maximum)),sum=values.reduce((s,v)=>s+v,0);return values.map(v=>v/sum)};
export function dispatch(demand,policy,capacity,cost){let remaining=capacity.map(Math.floor),flow=demand.map(()=>[0,0,0]),unserved=Array(8).fill(0);
 const scores=policy.map(row=>row.map((p,j)=>capacity[j]>0?-Math.log(p+1e-15):1000));
 const priority=Array.from({length:8},(_,i)=>i).sort((a,b)=>{const x=[...scores[a]].sort((a,b)=>a-b),y=[...scores[b]].sort((a,b)=>a-b);return(y[1]-y[0])-(x[1]-x[0])});
 for(const i of priority){let left=demand[i];const stations=[0,1,2].sort((a,b)=>scores[i][a]-scores[i][b]||a-b);for(const j of stations){const take=Math.min(left,remaining[j]);flow[i][j]+=take;remaining[j]-=take;left-=take}unserved[i]=left}
 const total=demand.reduce((s,v)=>s+v,0),lost=unserved.reduce((s,v)=>s+v,0),serviceCost=flow.reduce((s,row,i)=>s+row.reduce((v,f,j)=>v+f*cost[i][j],0),0),shortageCost=lost*15,expense=serviceCost+shortageCost;
 const districtCosts=flow.map((row,i)=>(row.reduce((s,f,j)=>s+f*cost[i][j],0)+15*unserved[i])/Math.max(1,demand[i])).filter((_,i)=>demand[i]>0),mean=districtCosts.reduce((s,v)=>s+v,0)/Math.max(1,districtCosts.length);
 const disparity=Math.sqrt(districtCosts.reduce((s,v)=>s+(v-mean)**2,0)/Math.max(1,districtCosts.length));
 const loads=capacity.map((v,j)=>v-remaining[j]);
 return{flow,unserved,cost:expense/Math.max(1,total),loads,lost,total,served:total-lost,serviceCost,shortageCost,totalCost:expense,disparity,violations:loads.filter((v,j)=>v>capacity[j]).length};
}
export function browserPolicy(d,capacity,risk,fairness){const train=d.training_scenarios,cost=d.cost,I=8,J=3,S=train.length,alpha=.9,rho=1.5,capSum=capacity.reduce((s,v)=>s+v,0);let logits=cost.map(row=>row.map((v,j)=>capacity[j]>0?-v:-20)),eta=2.5,history=[];
 for(let step=0;step<100;step++){const a=logits.map(softmax),q=a.map((r,i)=>r.reduce((s,v,j)=>s+v*cost[i][j],0)),meanq=q.reduce((s,v)=>s+v,0)/I,ga=a.map(()=>Array(J).fill(0));let mean=0,tail=0,penalty=0,tailCount=0;
  for(const demand of train){const total=demand.reduce((s,v)=>s+v,0),L=demand.reduce((s,v,i)=>s+v*q[i],0)/total,excess=capacity.map((cap,j)=>Math.max(0,demand.reduce((s,v,i)=>s+v*a[i][j],0)-cap));mean+=L/S;tail+=Math.max(0,L-eta)/S;if(L>eta)tailCount++;
   penalty+=excess.reduce((s,v)=>s+v*v,0)/S/Math.max(1,capSum);
   for(let i=0;i<I;i++)for(let j=0;j<J;j++)ga[i][j]+=(1+risk*(L>eta?1:0)/(1-alpha))/S*demand[i]/total*cost[i][j]+rho*2*demand[i]*excess[j]/S/Math.max(1,capSum);
  }
  const disparity=q.reduce((s,v)=>s+(v-meanq)**2,0)/I,objective=mean+risk*(eta+tail/(1-alpha))+fairness*disparity+rho*penalty;
  history.push({step,objective,policy:a});for(let i=0;i<I;i++){for(let j=0;j<J;j++)ga[i][j]+=fairness*2/I*(q[i]-meanq)*cost[i][j];const weighted=ga[i].reduce((s,v,j)=>s+v*a[i][j],0);for(let j=0;j<J;j++)logits[i][j]-=.18*a[i][j]*(ga[i][j]-weighted)}eta-=.04*risk*(1-tailCount/S/(1-alpha));
 }
 return history;
}
export function empiricalCvar(values,alpha=.9){const ordered=[...values].sort((a,b)=>b-a),mass=(1-alpha)*ordered.length,whole=Math.floor(mass+1e-12),remainder=Math.max(0,mass-whole);return(ordered.slice(0,whole).reduce((s,v)=>s+v,0)+(whole<ordered.length?remainder*ordered[whole]:0))/mass}
