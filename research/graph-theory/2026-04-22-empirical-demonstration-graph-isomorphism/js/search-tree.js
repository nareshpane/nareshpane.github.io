(function(L){
 'use strict';
 L.searchTree=function(g,{initial=[L.range(g.n)],limit=500,prune=false,automorphisms=[],strategy='smallest'}={}){
   let nodes=[],complete=true;
   function visit(seq,cells,parent=null,refinement=null,reason=null){
     if(nodes.length>=limit){complete=false;return null;}
     const id=nodes.length,target=L.target(cells,strategy),node={id,parent,seq:seq.slice(),cells,target,refinement,reason,children:[],leaf:cells.length===g.n};nodes.push(node);
     if(reason||node.leaf)return id;
     let done=[];
     for(const v of target){
       let witness=null,representative=null;
       if(prune)for(const u of done){witness=automorphisms.find(p=>seq.every(w=>p[w]===w)&&p[u]===v);if(witness){representative=u;break;}}
       let r=L.refine(g,L.individualize(cells,v),[[v]]);
       let why=witness?{type:'PC',permutation:witness,representative,vertex:v,text:`P_C: ${L.cycles(witness)} fixes the prefix pointwise and maps ${representative+1} to ${v+1}. The earlier sibling has an equivalent subtree.`}:null;
       const child=visit([...seq,v],r.cells,id,r,why);if(child!==null)node.children.push(child);if(!why)done.push(v);
       if(!complete)break;
     }
     return id;
   }
   const r=L.refine(g,initial);visit([],r.cells,null,r);return {nodes,complete};
 };
 L.leafAutomorphism=function(orderA,orderB){const p=[];orderA.forEach((v,i)=>p[v]=orderB[i]);return p;};
 L.traversal=function(tree,mode='dfs'){let out=[],queue=[0];while(queue.length){let i=mode==='dfs'?queue.pop():queue.shift(),n=tree.nodes[i];out.push(i);if(mode==='dfs')queue.push(...n.children.slice().reverse());else queue.push(...n.children);}return out;};
})(GraphLab);
