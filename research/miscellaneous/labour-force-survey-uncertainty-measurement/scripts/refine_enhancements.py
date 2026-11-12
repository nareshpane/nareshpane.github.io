from pathlib import Path
p=Path('research/labour-force-survey-uncertainty-measurement/js/enhancements.js')
s=p.read_text(encoding='utf-8')
s=s.replace('function panel(x,y,w,h,title){return', '''function panel(x,y,w,h,title){const max=Math.floor((w-32)/9),words=title.split(' '),lines=[''];for(const word of words){const i=lines.length-1;if(lines[i]&&(lines[i]+' '+word).length>max)lines.push(word);else lines[i]+=(lines[i]?' ':'')+word;}const label='<text x="'+(x+16)+'" y="'+(y+30)+'" class="scene-title">'+lines.map((line,i)=>'<tspan x="'+(x+16)+'" dy="'+(i?19:0)+'">'+esc(line)+'</tspan>').join('')+'</text>';return''')
s=s.replace("${text(x+16,y+30,title,'scene-title')}","${label}")
# Leaders are added outside the map callback so they are not overwritten by concatenation.
s=s.replace("s+='<g class=\"map-labels\">'+geos.map", "s+='<g class=\"map-labels\">'+[['NB',492,354],['NS',510,385],['PE',512,327]].map(([code,x,y])=>{const g=geos.find(g=>g[0]===code);return `<path d=\"M${g[2]} ${g[3]}L${x-6} ${y-4}\" class=\"map-leader\"/>`;}).join('')+geos.map")
s=s.replace('s+=`<path d="M${x} ${y}L${a[0]-6} ${a[1]-4}" class="map-leader"/>`;','')
# Preserve keyboard focus when the map re-renders after selection.
s=s.replace("function drawMap(){const geometry", "function drawMap(){const focusedCode=document.activeElement?.dataset?.geo;const geometry")
s=s.replace("$('journey-map').innerHTML=s;}","$('journey-map').innerHTML=s;if(focusedCode)$('journey-map').querySelector('[data-geo=\"'+focusedCode+'\"]')?.focus({preventScroll:true});}")
# Tighten sampling and office scenes without excessive empty areas.
s=s.replace("height=narrow?640:400", "height=narrow?640:300").replace("pw,narrow?164:290", "pw,narrow?164:210").replace("y+268", "y+190")
s=s.replace("height=narrow?720:440;s+=text(20,32,'ANALYSTS", "height=narrow?640:440;s+=text(20,32,'ANALYSTS")
# Accessible motion: people arrive; records flow to processing or dissemination.
s=s.replace("if(stage===3){height", "if(stage===3){height")
s=s.replace("person(x+102,y+68,'#8d5b43','#315e85','calm',.7)", "'<g id=\"journey-visitor\">'+person(x+102,y+68,'#8d5b43','#315e85','calm',.7)+'</g>'")
s=s.replace("if(stage===6){height", "if(stage===6){height")
s=s.replace("$('journey-scene').setAttribute('viewBox'", "if(stage===6)s+='<g id=\"journey-release-record\"><rect width=\"32\" height=\"40\" rx=\"4\" fill=\"#fff4df\" stroke=\"#a34925\"/><path d=\"M6 10H26M6 18H25M6 26H22\" stroke=\"#28685d\" stroke-width=\"2\"/></g>';\n$('journey-scene').setAttribute('viewBox'")
s=s.replace("if(stage===4){const record", "const within=Math.min(1,(elapsed-stage*12)/12);if(stage===3)$('journey-visitor')?.setAttribute('transform',`translate(${-20*(1-within)} 0)`);if(stage===6)$('journey-release-record')?.setAttribute('transform',`translate(${30+within*(mobile.matches?400:900)} ${113+Math.sin(within*Math.PI)*8})`);if(stage===0)$('journey-map').querySelector('.map-environments')?.setAttribute('opacity',String(.5+.5*within));\nif(stage===4){const record")
p.write_text(s,encoding='utf-8')
# Current authority also confirms integrated household weighting.
p=Path('research/labour-force-survey-uncertainty-measurement.html');h=p.read_text(encoding='utf-8')
h=h.replace('The 2017 method derives <strong>composite calibration</strong> and integrated weighting, with a common final person weight within a household.', 'The 2025 guide explicitly confirms <strong>composite calibration</strong> and integrated weighting, with a common final person weight within a household; the 2017 method supplies deeper derivations.')
p.write_text(h,encoding='utf-8')
