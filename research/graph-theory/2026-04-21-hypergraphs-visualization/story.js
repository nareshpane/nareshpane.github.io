/* Pure timeline: every frame can be reconstructed directly, in either time direction.
   The example comes only from HyperMath.example(); builder edits never mutate it. */
(function(root) {
  'use strict';
  const M=typeof module!=='undefined'&&module.exports?require('./math.js'):root.HyperMath;
  const colours=['#ac653b','#3e6f9e','#477b39','#885792','#8a7521','#327b88','#9a4b55','#607e30','#6d65a3','#a26429','#41766c','#8b5371'];
  const stages=[
    {start:0,end:1.5,edges:0,title:'Six vertices, no memberships yet',narration:'Keep A–F fixed while relationships change.',transition:.45},
    {start:1.5,end:3.5,edges:1,title:'One triple or three pairs?',narration:'The triple is one group; its projection has three pairwise edges.',active:0,transition:.6},
    {start:3.5,end:5.5,edges:2,title:'Add a pair hyperedge',narration:'e₂ has two members; e₁ has three. This family is non-uniform.',active:1,transition:.5},
    {start:5.5,end:8,edges:3,title:'Groups can share vertices',narration:'e₃ shares C with e₁ and D with e₂.',active:2,transition:.6},
    {start:8,end:10.5,edges:4,title:'Complete the running example',narration:'Four groups on six vertices have eleven distinct memberships.',active:3,transition:.6},
    {start:10.5,end:13,edges:4,title:'Read memberships as a matrix',narration:'A highlighted column and its coloured spokes record the same members.',matrix:true,transition:.35},
    {start:13,end:15.5,edges:4,title:'Project each group to pairs',narration:'Ten distinct pairs remain. Any repeated pair would be counted only once.',matrix:true,projection:true,transition:.6},
    {start:15.5,end:18,edges:4,title:'Same projection, different groups',narration:'One triple and three separate pairs both project to K₃. Group identity is lost.',matrix:true,counterexample:true,transition:.4}
  ];
  const example=M.example();
  const snapshots=stages.map(stage=>{
    const h=M.create(example.vertices,example.edges.slice(0,stage.edges));
    const matrix=M.incidence(h),degrees=M.degrees(h),sizes=h.edges.map(e=>e.length);
    return {h,matrix,degrees,sizes,pairs:M.twoSection(h),total:sizes.reduce((a,b)=>a+b,0)};
  });
  // Freeze cached mathematical snapshots, including their nested arrays.
  function freeze(value){if(value&&typeof value==='object'){Object.values(value).forEach(freeze);Object.freeze(value);}return value;}
  freeze(snapshots);freeze(stages);freeze(colours);
  const duration=stages[stages.length-1].end;
  function at(seconds,reducedMotion=false) {
    const time=Math.max(0,Math.min(duration,Number.isFinite(seconds)?seconds:0));
    const index=Math.max(0,stages.findIndex(s=>time<s.end));
    const stageIndex=time===duration?stages.length-1:index,stage=stages[stageIndex];
    const progress=Math.min(1,(time-stage.start)/stage.transition);
    const reveal=reducedMotion?1:progress*progress*(3-2*progress);
    const active=stage.matrix&&!stage.counterexample
      ? Math.min(3,Math.floor((time-stage.start)/(stage.end-stage.start)*4))
      : stage.active??null;
    return {...snapshots[stageIndex],time,stageIndex,stage,active,reveal};
  }
  const api={stages,duration,colours,at};
  if(typeof module!=='undefined'&&module.exports)module.exports=api;else root.HyperStory=api;
})(globalThis);
