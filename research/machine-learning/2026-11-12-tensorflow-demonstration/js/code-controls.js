async function copyText(text){
 try{if(navigator.clipboard&&isSecureContext){await navigator.clipboard.writeText(text);return}}catch{/* Some local browser permissions deny Clipboard API. */}
 const t=document.createElement('textarea');t.value=text;t.style.position='fixed';t.style.left='-9999px';document.body.append(t);t.select();t.setSelectionRange(0,text.length);
 const copied=document.execCommand('copy');t.remove();if(!copied)throw Error('Select the source text or use the download link.');
}
export function initCodeControls(){
 document.querySelectorAll('.code-panel').forEach(panel=>{
  const code=panel.querySelector('pre code'),copy=panel.querySelector('.copy-code'),expand=panel.querySelector('.expand-code');
  copy?.addEventListener('click',async()=>{const label=copy.textContent;try{await copyText(code.textContent);copy.textContent='Copied';copy.setAttribute('aria-label',`${panel.querySelector('.language')?.textContent||'Code'} source copied`)}catch(error){copy.textContent='Copy unavailable';panel.querySelector('.code-note').textContent=error.message}setTimeout(()=>{copy.textContent=label;copy.removeAttribute('aria-label')},2000)});
  expand?.addEventListener('click',()=>{const open=panel.querySelector('pre').classList.toggle('expanded');expand.textContent=open?'Limit code height':'Expand code fully';expand.setAttribute('aria-expanded',String(open))});
 });
}
