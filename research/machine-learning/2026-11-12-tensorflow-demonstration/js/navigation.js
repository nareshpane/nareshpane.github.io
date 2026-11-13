/* Static chapter links remain available if enhancement fails. */
export function initNavigation(){
 const nav=document.querySelector('#toc-tree'), chapters=[...document.querySelectorAll('section.chapter')];
 if(!nav||!chapters.length)return;
 nav.replaceChildren();
 for(const [i,chapter] of chapters.entries()){
  const group=document.createElement('details'); group.dataset.chapter=chapter.id;group.open=i===0;
  const summary=document.createElement('summary'),a=document.createElement('a');a.href='#'+chapter.id;a.textContent=`${i+1}. ${chapter.dataset.short}`;summary.append(a);group.append(summary);summary.addEventListener('click',e=>{if(!e.target.closest('a'))group.dataset.userClosed=String(group.open)});
  const subs=document.createElement('div');subs.className='subsections';
  chapter.querySelectorAll('.subsection').forEach(section=>{const link=document.createElement('a');link.href='#'+section.id;link.textContent=section.dataset.tocLabel||section.querySelector('h3').textContent;subs.append(link)});
  group.append(subs);nav.append(group);
 }
 document.querySelector('#expand-toc').addEventListener('click',()=>nav.querySelectorAll('details').forEach(d=>{d.dataset.userClosed='false';d.open=true}));
 document.querySelector('#collapse-toc').addEventListener('click',()=>nav.querySelectorAll('details').forEach(d=>{d.dataset.userClosed='true';d.open=false}));
 const toggle=document.querySelector('#menu-toggle'),sidebar=document.querySelector('#sidebar'),backdrop=document.querySelector('#drawer-backdrop');
 function close(){sidebar.classList.remove('open');backdrop.classList.remove('open');toggle.setAttribute('aria-expanded','false')}
 toggle.addEventListener('click',()=>{const open=!sidebar.classList.contains('open');sidebar.classList.toggle('open',open);backdrop.classList.toggle('open',open);toggle.setAttribute('aria-expanded',String(open))});
 backdrop.addEventListener('click',close);document.addEventListener('keydown',e=>{if(e.key==='Escape'){close();toggle.focus()}});
 nav.addEventListener('click',e=>{if(e.target.closest('a')){e.stopPropagation();if(innerWidth<=800)close()}});
 const sections=[...document.querySelectorAll('.chapter,.subsection')], progress=document.querySelector('#reading-progress');let pending=false,lastId='';
 function update(){
  pending=false;const line=innerWidth<=800?84:40;let current=sections[0];
  for(const s of sections){if(s.getBoundingClientRect().top<=line)current=s;else break}
  if(current&&current.id!==lastId){lastId=current.id;const chapter=current.matches('.chapter')?current:current.closest('.chapter');
   nav.querySelectorAll('details').forEach(d=>{const active=d.dataset.chapter===chapter.id;d.classList.toggle('active-chapter',active);if(active&&d.dataset.userClosed!=='true')d.open=true});
   nav.querySelectorAll('a').forEach(a=>{const active=a.hash==='#'+current.id;a.classList.toggle('active',active);if(active)a.setAttribute('aria-current','location');else a.removeAttribute('aria-current')});
   const link=nav.querySelector(`a[href="#${current.id}"]`),box=nav.getBoundingClientRect();
   if(link&&link.getClientRects().length){const r=link.getBoundingClientRect();if(r.top<box.top+8)nav.scrollTop-=box.top+8-r.top;else if(r.bottom>box.bottom-8)nav.scrollTop+=r.bottom-box.bottom+8}
  }
  const range=document.documentElement.scrollHeight-innerHeight;progress.style.width=`${range>0?Math.min(100,100*scrollY/range):0}%`;
 }
 function schedule(){if(!pending){pending=true;requestAnimationFrame(update)}}
 addEventListener('scroll',schedule,{passive:true});addEventListener('resize',schedule);document.addEventListener('toggle',schedule,true);update();
}
