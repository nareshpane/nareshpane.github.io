/* Mathematical teaching controls only. No market data, IV fitting or network. */
(function(){'use strict';
const ids=['btc-spot','btc-strike','btc-premium','btc-future','btc-days','btc-vol','btc-rate'];
function bitcoinIllustration(){
 const S=+$('btc-spot').value,K=+$('btc-strike').value,p=+$('btc-premium').value,ST=+$('btc-future').value,T=+$('btc-days').value/365,
 sigma=+$('btc-vol').value/100,r=+$('btc-rate').value/100,kind=$('btc-kind').value,z=kind==='call'?1:-1;
 ids.forEach(id=>{$(id+'-out').textContent=fmt(+$(id).value,id==='btc-premium'?3:id==='btc-rate'?1:0);});
 const payoff=x=>Math.max(z*(x-K),0),terminal=payoff(ST),xmin=Math.min(K*.5,ST*.8),xmax=Math.max(K*1.5,ST*1.2),xs=range(xmin,xmax,121);
 $('btc-btc-node').textContent=fmt(p,5)+' BTC';$('btc-conversion-node').textContent='× $'+fmt(S,0)+' per BTC now';$('btc-usd-node').textContent='$'+fmt(p*S);
 $('btc-currency-result').innerHTML=`<div><strong>$${fmt(p*S)}</strong><span>Chosen initial premium: ${fmt(p,5)} BTC × $${fmt(S,0)}/BTC</span></div><div><strong>${fmt(terminal/ST,6)} BTC</strong><span>Terminal payoff: $${fmt(terminal)} ÷ $${fmt(ST,0)}/BTC; not profit</span></div>`;
 const usd=lineChart('btc-payoff-usd',[{color:blue,points:xs.map(x=>[x,payoff(x)])}],'Expiration reference ($/BTC)','Illustrative payoff (USD equivalent)');usd.dot(ST,terminal,blue,`Illustrative ${kind}: S_T=${fmt(ST)}, payoff $${fmt(terminal)}; not market data.`);
 const btc=lineChart('btc-payoff-btc',[{color:teal,points:xs.map(x=>[x,payoff(x)/x])}],'Expiration reference ($/BTC)','Illustrative terminal payoff (BTC)');btc.dot(ST,terminal/ST,teal,`Illustrative terminal payoff ${fmt(terminal/ST,6)} BTC. Initial conversion spot is not used here.`);
 const call=BSM.price(S,K,T,r,sigma,0,'call'),put=BSM.price(S,K,T,r,sigma,0,'put');
 const d1=(Math.log(S/K)+(r+sigma*sigma/2)*T)/(sigma*Math.sqrt(T)),d2=d1-sigma*Math.sqrt(T);
 $('btc-model-result').innerHTML=`<div><strong>Call $${fmt(call,4)}</strong><span>${fmt(call/S,6)} BTC per unit notional</span></div><div><strong>Put $${fmt(put,4)}</strong><span>${fmt(put/S,6)} BTC per unit notional</span></div>`;
 $('btc-worked-steps').textContent=`Calculated teaching example: S=$${fmt(S,0)}, K=$${fmt(K,0)}, T=${fmt(T,6)} years, r=${fmt(r*100,2)}%, σ=${fmt(sigma*100,1)}%, q=0. σ√T=${fmt(sigma*Math.sqrt(T),6)}; d₁=${fmt(d1,6)}; d₂=${fmt(d2,6)}; N(d₁)=${fmt(BSM.N(d1),6)}; N(d₂)=${fmt(BSM.N(d2),6)}. Call = $${fmt(S*BSM.N(d1),4)} − $${fmt(K*Math.exp(-r*T)*BSM.N(d2),4)} = $${fmt(call,4)}. Divide by S, not S_T, to express this initial model value in BTC.`;
 const inBTC=$('btc-display-unit').value==='btc';lineChart('btc-benchmark-chart',['call','put'].map((side,i)=>({color:i?teal:blue,points:range(S*.5,S*1.5,121).map(x=>[x,BSM.price(x,K,T,r,sigma,0,side)/(inBTC?x:1)])})),
 'Current reference S ($/BTC)',inBTC?'Benchmark value (BTC per unit)':'Benchmark value (USD equivalent)');
}
function resetBitcoin(){ids.forEach(id=>{$(id).value=$(id).defaultValue;});$('btc-kind').selectedIndex=0;$('btc-display-unit').selectedIndex=0;bitcoinIllustration();}
on([...ids,'btc-kind','btc-display-unit'],bitcoinIllustration);$('btc-reset').addEventListener('click',resetBitcoin);
$('reset').addEventListener('click',bitcoinIllustration);bitcoinIllustration();
})();
