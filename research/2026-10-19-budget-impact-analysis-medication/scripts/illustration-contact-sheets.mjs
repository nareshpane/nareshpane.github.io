/* Screenshot contact sheets for inspection; no site dependency. */
import {createRequire} from 'node:module';
import {fileURLToPath} from 'node:url';
const require=createRequire(import.meta.url),sharp=require(process.env.SHARP_MODULE||'sharp');
const root=new URL('../figures/qa/illustrations/',import.meta.url);
for(const name of ['desktop','mobile']){
 const tiles=[];
 for(let i=0;i<13;i++)tiles.push({input:await sharp(fileURLToPath(new URL(`${name}-scene-${i+1}.png`,root))).resize(560,265,{fit:'contain',background:'#fff'}).toBuffer(),left:i%3*560,top:Math.floor(i/3)*265});
 await sharp({create:{width:1680,height:1325,channels:3,background:'#fff'}}).composite(tiles).png().toFile(fileURLToPath(new URL(`${name}-contact.png`,root)));
}

for(const name of ['desktop','mobile']){
 const sections=['severity-clinician','clinical-tree','prevalence-methods','database-grid','sa-illustration','pharmacy-art','dose-equation','limitations','approval-story','care-art','patient-icons','linkage-art'];
 const tiles=[];
 for(let i=0;i<sections.length;i++)tiles.push({input:await sharp(fileURLToPath(new URL(`${name}-${sections[i]}.png`,root))).resize(480,400,{fit:'contain',background:'#fff'}).toBuffer(),left:i%3*480,top:Math.floor(i/3)*400});
 await sharp({create:{width:1440,height:1600,channels:3,background:'#fff'}}).composite(tiles).png().toFile(fileURLToPath(new URL(`${name}-sections.png`,root)));
}
