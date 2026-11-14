export function initNavigation(){
 const tree=document.querySelector('#toc-tree'),sections=[...document.querySelectorAll('.chapter,.subsection,.references-section')],groups=[...tree.querySelectorAll('details')];
 const sidebar=document.querySelector('#sidebar'),toggle=document.querySelector('#menu-toggle'),backdrop=document.querySelector('#drawer-backdrop'),main=document.querySelector('main');
 function close(returnFocus=false){sidebar.classList.remove('open');backdrop.classList.remove('open');toggle.setAttribute('aria-expanded','false');main.inert=false;sidebar.inert=innerWidth<=820;sidebar.setAttribute('aria-hidden',String(innerWidth<=820));if(returnFocus)toggle.focus()}
 function open(){sidebar.inert=false;sidebar.setAttribute('aria-hidden','false');sidebar.classList.add('open');backdrop.classList.add('open');toggle.setAttribute('aria-expanded','true');main.inert=true;sidebar.querySelector('.drawer-close').focus()}
 toggle.addEventListener('click',()=>sidebar.classList.contains('open')?close(true):open());
 sidebar.querySelector('.drawer-close').addEventListener('click',()=>close(true));backdrop.addEventListener('click',()=>close(true));
 document.addEventListener('keydown',event=>{if(!sidebar.classList.contains('open'))return;if(event.key==='Escape'){event.preventDefault();close(true)}
  if(event.key==='Tab'){const focus=[...sidebar.querySelectorAll('button,a,summary')].filter(e=>e.getClientRects().length&&(!e.closest('.subsections')||e.closest('details').open));const first=focus[0],last=focus.at(-1);if(event.shiftKey&&document.activeElement===first){event.preventDefault();last.focus()}else if(!event.shiftKey&&document.activeElement===last){event.preventDefault();first.focus()}}
 });
 tree.addEventListener('click',e=>{const anchor=e.target.closest('a');if(anchor){e.stopPropagation();if(innerWidth<=820)close();requestAnimationFrame(update)}});
 for(const group of groups)group.querySelector('summary').addEventListener('click',e=>{if(!e.target.closest('a'))group.dataset.userClosed=String(group.open)});
 document.querySelector('#expand-toc').addEventListener('click',()=>groups.forEach(g=>{g.dataset.userClosed='false';g.open=true}));
 document.querySelector('#collapse-toc').addEventListener('click',()=>groups.forEach(g=>{g.dataset.userClosed='true';g.open=false}));
 let pending=false,last='';
 function update(){pending=false;let current=sections[0];const activation=innerWidth<=820?72:32;
  for(const section of sections){if(section.getBoundingClientRect().top<=activation)current=section;else break}
  if(scrollY+innerHeight>=document.documentElement.scrollHeight-2)current=sections.at(-1);
  if(current&&current.id!==last){last=current.id;const chapter=current.closest('.chapter');
   for(const group of groups){const active=group.dataset.chapter===chapter?.id;group.classList.toggle('active-chapter',active);if(active&&group.dataset.userClosed!=='true')group.open=true}
   tree.querySelectorAll('a').forEach(a=>{const active=a.hash==='#'+current.id;a.classList.toggle('active',active);if(active)a.setAttribute('aria-current','location');else a.removeAttribute('aria-current')});
   const link=tree.querySelector(`a[href="#${current.id}"]`);if(link?.getClientRects().length){const r=link.getBoundingClientRect(),box=tree.getBoundingClientRect();if(r.top<box.top+8)tree.scrollTop-=box.top+8-r.top;else if(r.bottom>box.bottom-8)tree.scrollTop+=r.bottom-box.bottom+8}
  }
  const range=document.documentElement.scrollHeight-innerHeight;document.querySelector('#reading-progress').style.width=(range>0?100*scrollY/range:0)+'%';
 }
 function schedule(){if(!pending){pending=true;requestAnimationFrame(update)}}
 addEventListener('scroll',schedule,{passive:true});addEventListener('resize',()=>{if(innerWidth>820)close();else if(!sidebar.classList.contains('open')){sidebar.inert=true;sidebar.setAttribute('aria-hidden','true')}schedule()});document.addEventListener('toggle',schedule,true);close();update();
}
