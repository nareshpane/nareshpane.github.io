/* One source of truth for the three introductory uniform families. */
(function(root,factory){const api=factory();if(typeof module==='object'&&module.exports)module.exports=api;else root.UniformExamples=api;})(typeof globalThis!=='undefined'?globalThis:this,()=>{
 'use strict';
 const vertices=['A','B','C','D','E','F'];
 const examples=[{k:2,edges:[['A','B'],['B','C'],['D','E']]},{k:3,edges:[['A','B','C'],['C','D','E'],['A','E','F']]},{k:4,edges:[['A','B','C','D'],['B','C','D','E'],['C','D','E','F']]}];
 function summarize(example){const degrees=vertices.map(v=>example.edges.reduce((n,e)=>n+Number(e.includes(v)),0)),sizes=example.edges.map(e=>e.length);return {degrees,sizes,incidences:sizes.reduce((a,b)=>a+b,0),degreeSum:degrees.reduce((a,b)=>a+b,0),uniform:sizes.every(s=>s===example.k)};}
 return {vertices,examples,summarize};
});
