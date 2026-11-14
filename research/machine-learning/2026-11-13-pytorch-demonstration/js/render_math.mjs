// Build-time only. Uses the same MathJax 3 engine; no second browser renderer is shipped.
// node js/render_math.mjs <temporary directory containing node_modules/mathjax-full>
import {createRequire} from 'node:module';
import {readFileSync,writeFileSync} from 'node:fs';
import {resolve} from 'node:path';
const require=createRequire(import.meta.url),root=process.argv[2];
if(!root)throw Error('Pass a temporary MathJax installation directory. See README.');
const from=name=>require(resolve(root,'node_modules/mathjax-full/js',name));
const {mathjax}=from('mathjax.js'),{TeX}=from('input/tex.js'),{SVG}=from('output/svg.js');
const {liteAdaptor}=from('adaptors/liteAdaptor.js'),{RegisterHTMLHandler}=from('handlers/html.js');
const {AllPackages}=from('input/tex/AllPackages.js'),{STATE}=from('core/MathItem.js');
const {SerializedMmlVisitor}=from('core/MmlTree/SerializedMmlVisitor.js');
RegisterHTMLHandler(liteAdaptor());
const doc=mathjax.document('',{InputJax:new TeX({packages:AllPackages.filter(name=>!['noerrors','noundefined'].includes(name))}),OutputJax:new SVG({fontCache:'none'})});
const visitor=new SerializedMmlVisitor(),source=JSON.parse(readFileSync(new URL('../data/math-source.json',import.meta.url),'utf8')),expressions={};
for(const row of source.expressions){
 const tree=doc.convert(row.tex,{display:row.display,end:STATE.COMPILED});
 const mathml=visitor.visitTree(tree);
 if(mathml.includes('<merror'))throw Error('Invalid TeX: '+row.tex+'\n'+mathml);
 expressions[row.key]={tex:row.tex,display:row.display,mathml};
}
writeFileSync(new URL('../data/math-render-cache.json',import.meta.url),JSON.stringify({engine:'MathJax 3.2.2',purpose:'Native MathML fallback generated from exactly the displayed TeX; no prose substitutes',expressions},null,2)+'\n');
console.log(`Compiled ${Object.keys(expressions).length} unique mathematical expressions without errors.`);
