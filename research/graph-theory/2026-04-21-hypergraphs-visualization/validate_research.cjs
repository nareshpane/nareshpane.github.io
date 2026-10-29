/* Independent finite-set and bit-mask oracles; no external package needed. */
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const M=require('./math.js'),R=require('./research-math.js');
let checks=0;const eq=(a,b,label)=>{assert.deepEqual(a,b,label);checks++;};
const expected=[{degrees:[1,2,1,0],sizes:[2,2],total:4,pairs:['AB','BC']},{degrees:[1,2,2,2,1],sizes:[3,2,3],total:8,pairs:['AB','AC','BC','BD','CD','CE','DE']},{degrees:[1,2,3,1,1,1,0],sizes:[3,2,3,1],total:9,pairs:['v1v2','v1v3','v2v3','v3v5','v3v6','v5v6']}];
R.gallery.forEach((h,i)=>{const s=R.summary(h),x=expected[i];eq(s.degrees,x.degrees);eq(s.sizes,x.sizes);eq(s.incidences,x.total);eq(s.pairs.map(p=>p.join('')),x.pairs);eq(s.degrees.reduce((a,b)=>a+b,0),s.incidences);const atoms=R.atoms(h);eq(atoms.flatMap(a=>a.members).sort(),[...h.vertices].sort());for(const atom of atoms)for(const v of atom.members)eq(h.edges.map(e=>+e.includes(v)),h.edges.map((_,j)=>(atom.mask>>j)&1));});
eq(R.intersection(...R.gallery[0].edges),['B']);eq(R.gallery[1].vertices.filter(v=>R.gallery[1].edges.every(e=>e.includes(v))),[]);eq(R.gallery[2].vertices.filter(v=>R.gallery[2].edges.slice(0,3).every(e=>e.includes(v))),['v3']);
eq(R.pairSources(R.gallery[2],['v2','v3']),[0,1]);eq(R.summary(R.gallery[2]).degrees[3],1);eq(M.twoSection(R.gallery[2]).filter(p=>p.includes('v4')),[]);eq(R.summary(R.gallery[2]).degrees[6],0);
const dup=R.network(['A','B','C'],[{id:'E1',tail:['A','A'],head:['A','B','B']},{id:'E2',tail:['B'],head:['C']}]);eq(dup.arcs[0].tail,['A']);eq(dup.arcs[0].head,['A','B']);
assert.throws(()=>R.network(['A'],[{tail:[],head:['A']}]));checks++;assert.throws(()=>R.network(['A'],[{tail:['A'],head:['B']}]));checks++;assert.throws(()=>R.network(['A'],[{id:'E1',tail:['A'],head:['A']},{id:'E1',tail:['A'],head:['A']}]));checks++;
for(const key of Object.keys(R.examples)){const n=R.getExample(key);for(const role of ['tail','head']){const b=R.incidence(n,role);eq(b.map(row=>row.reduce((a,x)=>a+x,0)),R.addressDegrees(n).map(d=>role==='tail'?d.out:d.in));eq(b.flat().reduce((a,x)=>a+x,0),n.arcs.reduce((s,a)=>s+a[role].length,0));}}
const linear=R.getExample('linear'),circular=R.getExample('circular');eq(R.stb(...linear.arcs,linear.exchanges).linear,true);eq(R.stb(...linear.arcs,linear.exchanges).circular,false);eq(R.stb(...circular.arcs,circular.exchanges).circular,true);eq(R.stb(...circular.arcs,circular.exchanges).linear,false);
eq(R.stb(...linear.arcs,['X1','X2','A2']).linear,false,'Intermediate exchange must invalidate STB');eq(R.stb(...linear.arcs,['X1']).linear,false,'Both outer endpoints required');
const loop={...linear.arcs[0],head:[...linear.arcs[0].head,'X1']};eq(R.stb(loop,linear.arcs[1],linear.exchanges).linear,false,'Loop must invalidate');
eq(R.stb(linear.arcs[1],linear.arcs[0],linear.exchanges).conditions.reverse,true);eq(R.stb(circular.arcs[1],circular.arcs[0],circular.exchanges).conditions.circularReverse,true);
eq(R.stb(...circular.arcs,['X1','A2']).circular,false,'Exchanges in both cross-intersections invalidate');
eq(R.motifs({tail:['A'],head:['A']},{tail:['B'],head:['B']}).connected,false,'Disjoint loops do not connect transactions');
eq(R.motifs({tail:['A'],head:['B']},{tail:['C'],head:['D']}).active,[]);

// Every possible ordered pair on three addresses, using independent integer masks.
const masks=[];for(let t=1;t<8;t++)for(let h=1;h<8;h++)masks.push([t,h]);
const vs=['A1','A2','A3'],toSet=m=>vs.filter((_,i)=>m&(1<<i));
const totals={orderedPairs:0,connected:0,connectedPure:0,linear:0,circular:0,signatures:{}};
for(const [t1,h1]of masks)for(const [t2,h2]of masks){
 const a={tail:toSet(t1),head:toSet(h1)},b={tail:toSet(t2),head:toSet(h2)};
 const bits=[h1&t2,t1&h2,h1&h2,t1&t2,t1&h1,t2&h2],signature=bits.map(x=>+!!x).join('');
 const actual=R.stb(a,b,['A1','A3']),m=actual.motif;
 eq(R.patterns.map(([key])=>m.sets[key]),bits.map(toSet),'Independent bitwise intersection oracle');
 const connected=!!(bits[0]|bits[1]|bits[2]|bits[3]);eq(m.connected,connected);
 const pure=bits.filter(Boolean).length===1;eq(m.pure,pure);
 const ex=5,onlyForward=signature==='100000',onlyReverse=signature==='010000',onlyMixed=signature==='110000';
 const linF=onlyForward&&!!(t1&ex)&&!!(h2&ex)&&!(bits[0]&ex),linR=onlyReverse&&!!(h1&ex)&&!!(t2&ex)&&!(bits[1]&ex);
 const cirF=onlyMixed&&!!(bits[1]&ex)&&!(bits[0]&ex),cirR=onlyMixed&&!!(bits[0]&ex)&&!(bits[1]&ex);
 eq(actual.linear,linF!==linR,'Independent linear oracle');eq(actual.circular,cirF!==cirR,'Independent circular oracle');
 totals.orderedPairs++;totals.connected+=+connected;totals.connectedPure+=+(connected&&pure);totals.linear+=+actual.linear;totals.circular+=+actual.circular;totals.signatures[signature]=(totals.signatures[signature]||0)+1;
}
eq(totals.orderedPairs,2401);assert.ok(totals.linear>0&&totals.circular>0);checks++;
const f=R.sourceFigure2;eq(f.links.length,14);const sourceNet=R.network(f.addresses,f.arcs);eq(R.incidence(sourceNet,'tail').flat().reduce((a,b)=>a+b,0)+R.incidence(sourceNet,'head').flat().reduce((a,b)=>a+b,0),14);
for(const a of sourceNet.arcs){const incoming=f.links.filter(l=>l[1]===a.id),outgoing=f.links.filter(l=>l[0]===a.id);eq(incoming.map(l=>l[0]).sort(),[...a.tail].sort());eq(outgoing.map(l=>l[1]).sort(),[...a.head].sort());eq(incoming.reduce((s,l)=>s+l[2],0),outgoing.reduce((s,l)=>s+l[2],0),'Fig. 2 quantity conservation');}
eq(f.links.find(l=>l[0]==='T1'&&l[1]==='a2')[2],7+20+8);eq(f.links.find(l=>l[0]==='a2'&&l[1]==='T2')[2],20+8);eq(f.links.find(l=>l[0]==='a3'&&l[1]==='T2')[2],5+50);
const generic=R.network(R.sourceFigure4.addresses,R.sourceFigure4.arcs),gm=R.motifs(...generic.arcs);
eq(generic.addresses.length,10);eq(generic.arcs.reduce((s,a)=>s+a.tail.length+a.head.length,0),19);
eq(R.patterns.map(([k])=>gm.sets[k]),[['c2','c3'],['b2','d1'],['c3','d3'],['b3'],['b1'],['c1','c3']],'Fig. 4 full-resolution source role intersections');
eq(R.sourceFigure4.exchanges,['b1','b2','d1']);eq(R.stb(...generic.arcs,R.sourceFigure4.exchanges).linear,false);eq(R.stb(...generic.arcs,R.sourceFigure4.exchanges).circular,false);
const protectedFiles=JSON.parse(fs.readFileSync(path.join(__dirname,'source-inventory.json'),'utf8')).files.filter(f=>f.path==='hypergraph_bitcoin.pdf'||f.path.startsWith('Verbatim/')||/^hypergraph_.*\.png$/.test(f.path));
const crypto=require('node:crypto');for(const f of protectedFiles)eq(crypto.createHash('sha256').update(fs.readFileSync(path.join(__dirname,f.path))).digest('hex'),f.sha256,'Protected source '+f.path);
const report={status:'PASS',checks,gallery:R.gallery.map(R.summary),syntheticEnumeration:{universe:vs,exchangeSet:['A1','A3'],method:'All 49 possible nonempty tail/head pairs for each of two indexed transactions; ordered transaction pairs, address-set equality permitted.',...totals},sourceFigures:'Figure 2 identities, 14 directed incidences, quantities, and conservation checked',protectedFiles:protectedFiles.length};
if(process.argv.includes('--write')){fs.mkdirSync(path.join(__dirname,'qa'),{recursive:true});fs.writeFileSync(path.join(__dirname,'qa','research-math-audit.json'),JSON.stringify(report,null,2));fs.writeFileSync(path.join(__dirname,'synthetic-transactions.json'),JSON.stringify({kind:'Synthetic educational examples; no real Bitcoin data',examples:Object.fromEntries(Object.keys(R.examples).map(k=>[k,R.getExample(k)])),sourceFigure2:{kind:'Transcribed mathematical figure, not blockchain observations',...R.sourceFigure2},sourceFigure4:{kind:'Source topology, address labels added, not blockchain observations',...R.sourceFigure4}},null,2));}
console.log(JSON.stringify(report,null,2));
