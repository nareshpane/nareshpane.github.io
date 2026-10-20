/* Rasterize original vector artwork for visual QA when a browser cannot launch.
 * SHARP_MODULE points to an existing development installation, not a site dependency.
 */
import {createRequire} from 'node:module';
import {readFileSync,writeFileSync,mkdirSync} from 'node:fs';
import {sceneArt,chapters} from '../js/storyboard.js';
import {budget} from '../js/model.js';
import {clinicalDecisionArt,renderDoseMath} from '../js/education.js';
const req=createRequire(import.meta.url),sharp=req(process.env.SHARP_MODULE||'sharp');
const out=new URL('../figures/qa/',import.meta.url);mkdirSync(out,{recursive:true});
const geometry=readFileSync(new URL('../figures/canada-provinces.svg',import.meta.url),'utf8');
const images=[];
for(let i=0;i<chapters.length;i++){
 let vector=sceneArt(i,budget(),.65).replace('<title>','<style>text{font-family:Segoe UI,sans-serif;font-size:16px;fill:#233c35}</style><title>');
 vector=vector.replace(/<use href="figures\/canada-provinces.svg#([A-Z]+)"([^>]*)\/>/g,(_,code,attrs)=>`<g ${attrs}>${new RegExp('<g id="'+code+'"[\\s\\S]*?</g>').exec(geometry)[0]}</g>`);
 const png=await sharp(Buffer.from(vector)).resize(1120,530).png().toBuffer();
 images.push({input:await sharp(png).resize(560,265).toBuffer(),left:i%3*560,top:Math.floor(i/3)*265});
 if([0,4,9,11].includes(i))writeFileSync(new URL(`scene-${i+1}.png`,out),png);
}
await sharp({create:{width:1680,height:Math.ceil(chapters.length/3)*265,channels:4,background:'#fff7ed'}}).composite(images).png().toFile(new URL('storyboard-contact.png',out).pathname.replace(/^\/([A-Za-z]:)/,'$1'));
console.log('Rendered all SVG scenes; this is vector QA, not browser/HTML verification.');
await sharp(Buffer.from(clinicalDecisionArt())).flatten({background:'#fffaf0'}).png().toFile(new URL('clinical-decision.png',out).pathname.replace(/^\/([A-Za-z]:)/,'$1'));
const nodes=new Map();globalThis.document={getElementById(id){if(!nodes.has(id))nodes.set(id,{value:'first',innerHTML:'',textContent:''});return nodes.get(id);}};
renderDoseMath(budget().p,budget());
await sharp(Buffer.from(nodes.get('dose-equation').innerHTML)).flatten({background:'#fffaf0'}).png().toFile(new URL('dose-equation.png',out).pathname.replace(/^\/([A-Za-z]:)/,'$1'));
