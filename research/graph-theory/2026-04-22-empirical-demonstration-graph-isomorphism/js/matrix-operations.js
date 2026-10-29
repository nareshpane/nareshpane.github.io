/* Rows are source vertices, columns are target vertices: P[i][f(i)] = 1.
   Thus (P B P^T)[i][j] = B[f(i)][f(j)]. All values are computed. */
(function(L){
 'use strict';
 L.transpose=a=>a[0].map((_,j)=>a.map(r=>r[j]));
 L.multiply=(a,b)=>{if(a[0].length!==b.length)throw Error('Incompatible matrix dimensions.');return a.map(r=>b[0].map((_,j)=>r.reduce((s,x,k)=>s+x*b[k][j],0)));};
 L.subtract=(a,b)=>a.map((r,i)=>r.map((x,j)=>x-b[i][j]));
 L.permutationMatrix=p=>{L.permutation(p);return p.map(v=>L.range(p.length).map(w=>Number(v===w)));};
 L.pullback=(a,p)=>L.multiply(L.multiply(L.permutationMatrix(p),a),L.transpose(L.permutationMatrix(p)));
 L.matrixEqual=(a,b)=>a.length===b.length&&a.every((r,i)=>r.length===b[i].length&&r.every((v,j)=>v===b[i][j]));
 L.matrixCheck=(g,h,p)=>{const P=L.permutationMatrix(p),product=L.multiply(L.multiply(P,h.a),L.transpose(P)),difference=L.subtract(g.a,product);return {P,product,difference,ok:L.matrixEqual(g.a,product),pairs:L.edgesOfComplete(g.n).map(([i,j])=>({i,j,fi:p[i],fj:p[j],left:g.a[i][j],right:h.a[p[i]][p[j]],ok:g.a[i][j]===h.a[p[i]][p[j]]}))};};
 L.edgesOfComplete=n=>L.range(n).flatMap(i=>L.range(n-i-1).map(j=>[i,i+j+1]));
 L.matrixInvariants=g=>{const a2=L.multiply(g.a,g.a),a3=L.multiply(a2,g.a);return {degrees:g.a.map(r=>r.reduce((s,x)=>s+x,0)),a2,a3,trace3:a3.reduce((s,r,i)=>s+r[i],0),triangles:a3.reduce((s,r,i)=>s+r[i],0)/6};};
 L.quotient=(g,cells)=>cells.map(c=>cells.map(w=>w.reduce((s,v)=>s+g.a[c[0]][v],0)));
 L.fourCliques=g=>{let out=[];for(let a=0;a<g.n;a++)for(let b=a+1;b<g.n;b++)for(let c=b+1;c<g.n;c++)for(let d=c+1;d<g.n;d++){const xs=[a,b,c,d];if(L.edgesOfComplete(4).every(([i,j])=>g.a[xs[i]][xs[j]]))out.push(xs);}return out;};
 L.prismMaps=m=>{let maps=[];for(let swap=0;swap<2;swap++)for(let sign of [1,-1])for(let turn=0;turn<m;turn++)maps.push({swap,sign,turn,p:L.range(2*m).map(v=>((Math.floor(v/m)+swap)%2)*m+((sign*(v%m)+turn)%m+m)%m)});return maps;};
 L.prismWitness=[0,4,5,2,1,3];
})(GraphLab);
