async function copy(text){
 try{if(navigator.clipboard&&isSecureContext){await navigator.clipboard.writeText(text);return}}catch{}
 const field=document.createElement('textarea');field.value=text;field.style.position='fixed';field.style.left='-9999px';document.body.append(field);field.select();const okay=document.execCommand('copy');field.remove();if(!okay)throw Error('Use the source download or select the text.');
}
export function initCodeControls(){document.querySelectorAll('.code-panel').forEach(panel=>{
 const source=panel.querySelector('pre code'),status=panel.querySelector('.copy-status');
 panel.querySelector('.copy-code').addEventListener('click',async()=>{try{await copy(source.textContent);status.textContent='Copied exact source'}catch(e){status.textContent=e.message}});
 panel.querySelector('.expand-code').addEventListener('click',e=>{const full=panel.querySelector('pre').classList.toggle('full-height');e.target.textContent=full?'Limit code height':'Expand code fully';e.target.setAttribute('aria-expanded',String(full))});
})}
