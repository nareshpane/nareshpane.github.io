import {chapters,duration,locateScene,sceneArt} from './storyboard.js?v=20261026-3';
export function createVideo(getResult,getBudgetMode=()=>'illustration'){const $=id=>document.getElementById(id);let seconds=0,playing=false,last=0,lastPaint=0,token=0,currentScene=-1;const reduced=matchMedia('(prefers-reduced-motion: reduce)').matches;
 $('seek').max=duration;$('video-duration').textContent=`${duration} seconds at 1× · ${chapters.length} scenes`;
 $('scene-nav').innerHTML=chapters.map((c,i)=>`<button type="button" data-scene="${i}" aria-label="Scene ${i+1}: ${c[0]}"><span>${String(i+1).padStart(2,'0')}</span> ${c[0]}</button>`).join('');
 function draw(){const q=locateScene(seconds);
  if(currentScene!==q.index){if(!reduced)$('scene-art').animate?.([{opacity:.55},{opacity:1}],{duration:280,easing:'ease-out'});$('scene-title').textContent=`${q.index+1} / ${chapters.length} - ${chapters[q.index][2]}`;$('scene-caption').textContent=chapters[q.index][3];currentScene=q.index;}
  $('scene-art').innerHTML=sceneArt(q.index,getResult(),reduced?.65:q.progress,getBudgetMode());$('seek').value=seconds;$('seek-label').textContent=`${seconds.toFixed(1)} / ${duration} s`;$('intro').classList.toggle('motion',playing&&!reduced);$('play').disabled=playing;$('pause').disabled=!playing;
  document.querySelectorAll('[data-scene]').forEach(el=>{const current=Number(el.dataset.scene)===q.index;el.classList.toggle('active',current);el.setAttribute('aria-current',current?'step':'false');});
 }
 function frame(now,id){if(!playing||id!==token)return;seconds=Math.min(duration,seconds+(now-last)/1000*Number($('speed').value));last=now;if(seconds>=duration)playing=false;if(now-lastPaint>=80||!playing){draw();lastPaint=now;}if(playing)requestAnimationFrame(t=>frame(t,id));}
 function start(){if(playing)return;if(seconds>=duration)seconds=0;playing=true;last=performance.now();const id=++token;draw();requestAnimationFrame(t=>frame(t,id));}
 function pause(){playing=false;token++;draw();}
 $('play').onclick=start;$('pause').onclick=pause;$('replay').onclick=()=>{pause();seconds=0;start();};$('seek').oninput=()=>{seconds=Math.max(0,Math.min(duration,Number($('seek').value)||0));last=performance.now();draw();};document.querySelectorAll('[data-scene]').forEach(el=>el.onclick=()=>{seconds=chapters.slice(0,Number(el.dataset.scene)).reduce((s,c)=>s+c[1],0);last=performance.now();draw();});
 document.addEventListener('visibilitychange',()=>{if(document.hidden)pause();});draw();return {update:draw,pause};
}
