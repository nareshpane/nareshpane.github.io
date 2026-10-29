(function(L){
 'use strict';
 L.isomorphisms=function(g,h,{all=false,limit=250000,coloursG=null,coloursH=null}={}){
   if(g.n!==h.n)return {maps:[],complete:true,nodes:0,reason:'Different vertex counts.'};
   let ig=L.invariants(g),ih=L.invariants(h);
   for(const k of ['edges','degrees','components','triangles','bipartite','cliques4'])if(JSON.stringify(ig[k])!==JSON.stringify(ih[k]))return {maps:[],complete:true,nodes:0,reason:'Invariant mismatch: '+k+'.'};
   let cg=coloursG||L.colours(L.refine(g,[L.range(g.n)]).cells,g.n),ch=coloursH||L.colours(L.refine(h,[L.range(h.n)]).cells,h.n);
   let p=Array(g.n).fill(-1),used=Array(g.n).fill(false),maps=[],nodes=0,complete=true;
   function dfs(depth){
     if(++nodes>limit){complete=false;return true;}if(depth===g.n){maps.push(p.slice());return !all;}
     let best=-1,cands=null;
     for(let u=0;u<g.n;u++)if(p[u]<0){let cs=L.range(g.n).filter(v=>!used[v]&&cg[u]===ch[v]&&L.degree(g,u)===L.degree(h,v)&&p.every((w,x)=>w<0||g.a[u][x]===h.a[v][w]));if(cands===null||cs.length<cands.length){best=u;cands=cs;}if(!cs.length)break;}
     for(const v of cands){p[best]=v;used[v]=true;if(dfs(depth+1)){p[best]=-1;used[v]=false;return true;}p[best]=-1;used[v]=false;}return false;
   }
   dfs(0);return {maps,complete,nodes,reason:maps.length?'A verified adjacency-preserving bijection was found.':complete?'Exact search exhausted all compatible bijections.':'Search limit reached; result is undetermined.'};
 };
})(GraphLab);
