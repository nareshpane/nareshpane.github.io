/* Progressive enhancement: all dataset and evaluation rows already exist in HTML. */
(() => {
 'use strict';
 const $=id=>document.getElementById(id);
 const models=[...document.querySelectorAll('details.model')];
 const update=d=>{d.querySelector('.expand-label').textContent=d.open?'− Collapse':'+ Expand';};
 models.forEach(d=>d.addEventListener('toggle',()=>update(d)));
 $('expand-all').addEventListener('click',()=>models.forEach(d=>{d.open=true;update(d)}));
 $('collapse-all').addEventListener('click',()=>models.forEach(d=>{d.open=false;update(d)}));
 const rows=[...$('dataset-body').rows];
 const ids=['dataset-search','filter-year','filter-exporter','filter-destination','filter-sector'];
 function filter(){
   const tokens=$('dataset-search').value.trim().toLowerCase().split(/\s+/).filter(Boolean);
   let n=0;
   rows.forEach(row=>{
     const a=row.dataset;
     const ok=tokens.every(t=>a.search.includes(t)) && (!$(ids[1]).value||a.year===$(ids[1]).value) && (!$(ids[2]).value||a.exporter===$(ids[2]).value) && (!$(ids[3]).value||a.destination===$(ids[3]).value) && (!$(ids[4]).value||a.sector===$(ids[4]).value);
     row.hidden=!ok;if(ok)n++;
   });
   $('dataset-count').textContent=`${n} displayed / 336 total observations`;
 }
 ids.forEach(id=>$(id).addEventListener(id==='dataset-search'?'input':'change',filter));
 $('clear-filters').addEventListener('click',()=>{ids.forEach(id=>$(id).value='');filter();});
 document.querySelectorAll('.prediction-controls').forEach(control=>{
   const body=$(control.dataset.body), r=[...body.rows];
   const input=control.querySelector('input'),select=control.querySelector('select'),count=control.querySelector('.prediction-count');
   const apply=()=>{let n=0;const tokens=input.value.toLowerCase().trim().split(/\s+/).filter(Boolean);r.forEach(row=>{const ok=(!select.value||row.dataset.year===select.value)&&tokens.every(t=>row.dataset.search.includes(t));row.hidden=!ok;if(ok)n++;});count.textContent=`${n} displayed / ${r.length} evaluation predictions`;};
   input.addEventListener('input',apply);select.addEventListener('change',apply);
   control.querySelector('button').addEventListener('click',()=>{input.value='';select.value='';apply();});apply();
 });
 const errors=[];window.addEventListener('error',e=>errors.push(e.message));
 window.chapterUI={filter,models,errors};
})();
