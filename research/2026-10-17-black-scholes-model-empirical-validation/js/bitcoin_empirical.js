/* Verified academic observations: aggregates plus twenty limited, attributed examples. */
(function(){'use strict';
const D=window.BITCOIN_RESULTS,colors={H30:'#205492',H60:'#087f80',B:'#a87327'},names={H30:'H30 historical sigma',H60:'H60 historical sigma',B:'Earlier calibrated sigma'};
if(!D||D.status!=='computed')return;
const money=s=>s.replace(/\(([^,]+), ([^\]]+)\]/g,(_,a,b)=>Number(a).toFixed(2)+'-'+Number(b).toFixed(2));
function axisDates(c){[...c.svg.querySelectorAll('.axis-x-tick')].forEach((t,i)=>{const v=c.xDomain[0]+i/5*(c.xDomain[1]-c.xDomain[0]);t.textContent=new Date(v*86400000).toISOString().slice(0,7);});}
const day=s=>Date.parse(s)/86400000;
const history=D.daily.filter(d=>d.price!==null),hc=lineChart('btc-history', [{color:colors.B,points:history.map(d=>[day(d.date),d.price])}], 'UTC calendar date','Supplied daily close ($/BTC)');axisDates(hc);$('btc-history').parentElement.querySelector('figcaption').textContent+=' Daily observations: '+num(history.length)+'.';
const vh=D.daily.filter(d=>d.H30!==null&&d.H60!==null),vc=lineChart('btc-volatility',['H30','H60'].map(m=>({color:colors[m],points:vh.map(d=>[day(d.date),d[m]*100])})),'UTC completed day','Annualized historical sigma (%)');axisDates(vc);$('btc-volatility').parentElement.querySelector('figcaption').textContent+=' Common valid daily estimates: '+num(vh.length)+'.';
function groupedBars(id,groups,key,label){
 const vals=groups.flatMap(g=>Object.keys(colors).map(m=>g.models[m][key])),c=chart(id,[0,Math.max(1,groups.length)],[0,Math.max(...vals,1)*1.2],label,key==='mae'?'MAE (USD / BTC notional)':'RMSE (USD / BTC notional)');
 c.svg.querySelectorAll('.axis-x-tick').forEach(t=>t.remove());
 groups.forEach((g,i)=>{Object.keys(colors).forEach((m,j)=>{const v=g.models[m][key],b=el('rect',{x:c.x(i+.15+j*.23),y:c.y(v),width:c.x(.18)-c.x(0),height:c.y(0)-c.y(v),fill:colors[m],tabindex:0,'aria-label':`${g.label}, ${names[m]}: ${stat(v)} USD; n=${num(g.n)}`},c.plot);el('title',{},b).textContent=b.getAttribute('aria-label');});txt(c.svg,c.x(i+.48),300,money(g.label),{'text-anchor':'middle','font-size':11});});
}
function scatter(g,m){
 const target=$('btc-market');target.replaceChildren();const pts=g.scatter[m],v=Math.max(...pts.flatMap(p=>[p.market,p.model]),1)*1.05;
 const svg=el('svg',{viewBox:'0 0 760 760',role:'img','aria-label':'Observed versus theoretical binned price means with equality line'},target),x=z=>75+z/v*620,y=z=>680-z/v*620;
 for(let i=0;i<=5;i++){const z=i*v/5;el('path',{d:`M75 ${y(z)}h620 M${x(z)} 60v620`,stroke:'#e2e2d5',fill:'none'},svg);txt(svg,x(z),704,stat(z,0),{'text-anchor':'middle'});txt(svg,65,y(z)+4,stat(z,0),{'text-anchor':'end'});}
 el('path',{d:'M75 680L695 60',stroke:'#718c80','stroke-dasharray':'6 5',fill:'none'},svg);
 pts.forEach(p=>{const info=`Mean observed $${stat(p.market)}; mean ${names[m]} $${stat(p.model)}; n=${num(p.n)} transactions; mean residual $${stat(p.bias)}`;const dot=el('circle',{cx:x(p.market),cy:y(p.model),r:5,fill:colors[m],tabindex:0,'aria-label':info},svg);el('title',{},dot).textContent=info;});
 txt(svg,385,746,'Observed transaction premium (USD equivalent)',{'text-anchor':'middle'});txt(svg,18,370,'Model premium (same USD units)',{transform:'rotate(-90 18 370)','text-anchor':'middle'});
}
function render(){
 const side=$('btc-emp-side').value,term=$('btc-emp-term').value,m=$('btc-emp-model').value,g=D.groups[side+'|'+term];
 if(!g)return;
 $('btc-emp-count').textContent=`${num(g.n)} common held-out transactions | ${num(g.dates)} UTC dates | ${names[m]}. Calls and puts are per one BTC notional; all monetary errors are USD equivalent.`;
 $('btc-metrics').innerHTML=table(['Configuration','n','MAE $','Median |e| $','RMSE $','Bias $','Mean |e| / S (bp)'],Object.keys(colors).map(k=>{const v=g.metrics[k];return [names[k],num(v.n),stat(v.mae),stat(v.median_absolute),stat(v.rmse),stat(v.bias),stat(v.spot_bps)];}));
 $('btc-metrics-extra').innerHTML=table(['Model','MAE (BTC)','Median signed error $','Mean |e| / premium','Eligible denominator','|e| ≤ $5','|e| ≤ 1 spot bp'],Object.keys(colors).map(k=>{const v=g.metrics[k];return [k,stat(v.mae_btc,6),stat(v.median_signed),(v.relative_price_error===null?'N/A':stat(v.relative_price_error*100,1)+'%'),num(v.relative_price_n),stat(v.within_5_usd*100,1)+'%',stat(v.within_1_spot_bp*100,1)+'%'];}));
 scatter(g,m);
 const bins=g.residual[m],c=chart('btc-residuals',[bins[0].lo,bins.at(-1).hi],[0,Math.max(...bins.map(b=>b.n),1)*1.1],'Theoretical - transaction premium ($)','Transaction count');
 bins.forEach(b=>{const bar=el('rect',{x:c.x(b.lo),y:c.y(b.n),width:Math.max(.5,c.x(b.hi)-c.x(b.lo)-.5),height:c.y(0)-c.y(b.n),fill:b.hi<=0?coral:colors[m],tabindex:0},c.plot);el('title',{},bar).textContent=`[${stat(b.lo)}, ${stat(b.hi)}] USD: n=${num(b.n)}`;});
 const smile=g.smile,hi=Math.max(...smile.flatMap(v=>[v.iv,v.H30,v.H60]),1)*110,sc=chart('btc-smile',[.8,1.2],[0,hi],'K/S; discrete pooled groups','Annualized sigma (%)');
 smile.forEach(v=>{sc.dot(v.m,v.iv*100,v.side==='call'?blue:teal,`${v.side}, ${money(v.label)}, n=${num(v.n)} valid IV; median IV ${stat(v.iv*100,1)}%; H30 ${stat(v.H30*100,1)}%; H60 ${stat(v.H60*100,1)}%`);});
 groupedBars('btc-money',g.money,'mae','K/S buckets');groupedBars('btc-term',g.maturity,'mae','Calendar days remaining');groupedBars('btc-side-bars',g.side,'mae','Option side');
 const mg=Object.keys(colors).map(k=>({label:k,n:g.n,models:{[k]:g.metrics[k]}}));
 const max=Math.max(...Object.values(g.metrics).map(v=>v.rmse))*1.2,bc=chart('btc-model-bars',[0,3],[0,max],'H30 / H60 / calibrated; MAE / RMSE','USD equivalent error');bc.svg.querySelectorAll('.axis-x-tick').forEach(t=>t.remove());
 Object.keys(colors).forEach((k,i)=>{['mae','rmse'].forEach((metric,j)=>{const v=g.metrics[k][metric],b=el('rect',{x:bc.x(i+.15+j*.35),y:bc.y(v),width:bc.x(.28)-bc.x(0),height:bc.y(0)-bc.y(v),fill:colors[k],opacity:j?0.45:1,tabindex:0},bc.plot);el('title',{},b).textContent=`${names[k]} ${metric.toUpperCase()}: $${stat(v)}, n=${num(g.n)}`;});txt(bc.svg,bc.x(i+.5),300,k,{'text-anchor':'middle'});});
 $('btc-paired').innerHTML=table(['Historical minus calibrated absolute error','Mean Delta $','Median Delta $','Trades improved','Aggregate MAE reduction'],['H30','H60'].map(h=>{const p=g.paired[h];return [h+' vs B',stat(p.mean),stat(p.median),stat(p.improved_observations_percent,1)+'%',stat(p.aggregate_mae_reduction_percent,1)+'%'];}));
 for(const id of ['btc-market','btc-residuals','btc-smile','btc-money','btc-term','btc-model-bars','btc-side-bars']){const cap=$(id).parentElement.querySelector('figcaption');if(!cap.dataset.base)cap.dataset.base=cap.textContent;cap.textContent=cap.dataset.base+' Selected evaluation transactions: '+num(g.n)+'.';}
 const cells=D.heatmap;
 $('btc-error-cells').innerHTML=table(['Maturity (days)','K/S','n',names[m]+' MAE $'],cells.map(v=>[money(v.term),money(v.money),num(v.n),`<span class="btc-error-cell" style="--shade:${Math.min(.8,.12+v.models[m].mae/Math.max(...cells.map(t=>t.models[m].mae))*.6)}">${stat(v.models[m].mae)}</span>`]));
}
on(['btc-emp-side','btc-emp-term','btc-emp-model'],render);
$('btc-emp-reset').addEventListener('click',()=>{['btc-emp-side','btc-emp-term','btc-emp-model'].forEach(id=>$(id).selectedIndex=0);render();});$('reset').addEventListener('click',()=>{sortField='timestamp';ascending=true;render();samples();});
$('btc-filter-counts').innerHTML=table(['Sequential stage','Retained','Removed at stage'],D.quality_flow.map(v=>[v.stage.replaceAll('_',' '),num(v.remaining),num(v.removed)]));
let sortField='timestamp',ascending=true;
function samples(){let rows=D.examples.filter(r=>$('btc-example-side').value==='all'||r.kind===$('btc-example-side').value);rows.sort((a,b)=>(typeof a[sortField]==='number'?a[sortField]-b[sortField]:String(a[sortField]).localeCompare(String(b[sortField])))*(ascending?1:-1));
 const cols=[['timestamp','Trade time (UTC)'],['contract','Option identifier'],['kind','Side'],['strike','Strike ($)'],['expiration','Expiry (UTC)'],['spot','Recorded index ($/BTC)'],['observed_btc','Observed premium (BTC)'],['observed_usd','Observed premium ($)'],['H30','H30 BSM ($)'],['H60','H60 BSM ($)'],['B','Calibrated BSM ($)'],['H30_error','H30 - observed ($)']];
 $('btc-examples').innerHTML='<div class="table-wrap"><table><thead><tr>'+cols.map(([k,l])=>`<th scope="col"><button type="button" data-btc-sort="${k}" title="${D.sample_columns[k]}">${l}</button></th>`).join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+cols.map(([k])=>`<td>${r[k]===null?'N/A':typeof r[k]==='number'?stat(r[k],k==='observed_btc'?6:k==='strike'?8:Math.abs(r[k])<.01?6:Math.abs(r[k])<1?3:Math.abs(r[k])<10?2:1):r[k]}</td>`).join('')+'</tr>').join('')+'</tbody></table></div>';
 $('btc-example-count').textContent=rows.length+' of 20 genuine illustrative transactions; selected separately from the aggregate figures.';
 $('btc-examples').querySelectorAll('button[data-btc-sort]').forEach(b=>b.onclick=()=>{ascending=sortField===b.dataset.btcSort?!ascending:true;sortField=b.dataset.btcSort;samples();});
}
on(['btc-example-side'],samples);$('btc-example-clear').onclick=()=>{$('btc-example-side').selectedIndex=0;sortField='timestamp';ascending=true;samples();};samples();render();

function comparisonBars(id,title,rows){
 const target=$(id);if(!target)return;
 const width=580,height=278,left=150,right=510,top=68,max=320;
 const svg=el('svg',{viewBox:`0 0 ${width} ${height}`,role:'img','aria-label':title+': mean absolute error divided by spot in basis points; zero to 320 bp'},target);
 txt(svg,20,28,title,{'font-size':18,'font-weight':650});
 [0,100,200,300].forEach(v=>{const x=left+v/max*(right-left);el('path',{d:`M${x} ${top-14}V${height-45}`,stroke:'#d6e1d8',fill:'none'},svg);txt(svg,x,height-23,stat(v),{'text-anchor':'middle','font-size':12});});
 rows.forEach(([label,value,color],i)=>{const y=top+i*56;txt(svg,left-12,y+17,label,{'text-anchor':'end','font-size':13});const bar=el('rect',{x:left,y,width:value/max*(right-left),height:24,rx:3,fill:color,tabindex:0,'data-bps':value,'aria-label':`${title}, ${label}: ${stat(value)} bp`},svg);el('title',{},bar).textContent=bar.getAttribute('aria-label');txt(svg,left+value/max*(right-left)+8,y+17,stat(value)+' bp',{'font-size':13,'font-weight':650});});
 txt(svg,332,height-3,'Mean absolute error / spot (bp)',{'text-anchor':'middle','font-size':12});
}
comparisonBars('compare-spxw-bars','SPXW · same-date contract holdout',[
 ['Historical 20',E.metrics.A.spot_bps,'#205492'],['Calibrated',E.metrics.B.spot_bps,'#087f80']]);
comparisonBars('compare-bitcoin-bars','Bitcoin · later-date holdout',[
 ['Historical H30',D.metrics.H30.spot_bps,'#bd9348'],['Historical H60',D.metrics.H60.spot_bps,'#a87327'],['Earlier calibrated',D.metrics.B.spot_bps,'#665035']]);
})();
