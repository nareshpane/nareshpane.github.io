(function(L){
 'use strict';
 L.knownCounts=[0,1,2,4,11,34,156,1044,12346];
 L.decodeGraph6=function(s){s=s.trim();const n=s.charCodeAt(0)-63;if(n<1||n>32)throw Error('Only small graph6 orders are supported.');let bits=s.slice(1).split('').flatMap(c=>L.range(6).map(i=>((c.charCodeAt(0)-63)>>(5-i))&1)),k=0,e=[];for(let j=1;j<n;j++)for(let i=0;i<j;i++)if(bits[k++])e.push([i,j]);return L.graph(n,e);};
 L.generate=function(n,method='canonical',progress=()=>{}){
   if(!Number.isInteger(n)||n<1||n>5)throw Error('Live exhaustive generation supports orders 1–5. Orders 6–8 use verified catalogues.');
   let total=2**(n*(n-1)/2),reps=[],keys=new Map(),work=0,groups=[];
   for(let mask=0;mask<total;mask++){
     let g=L.fromMask(n,mask),index=-1;
     if(method==='naive'){for(let i=0;i<reps.length;i++){let r=L.isomorphisms(g,reps[i]);work+=r.nodes;if(r.maps.length){index=i;break;}}}
     else {let c=L.canonical(g);work+=c.nodes;if(!c.complete)throw Error('Incomplete canonical search.');if(keys.has(c.code))index=keys.get(c.code);else keys.set(c.code,reps.length);}
     if(index<0){index=reps.length;reps.push(g);groups.push([]);}groups[index].push(mask);
     if(mask%32===0)progress({processed:mask+1,total,classes:reps.length});
   }
   return {n,method,total,work,reps,groups,count:reps.length,complete:true};
 };
})(GraphLab);
