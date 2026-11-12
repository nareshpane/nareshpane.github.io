/* Original SVG artwork. Symbols describe units, never production sample locations. */
(() => {
  'use strict';
  const palette=['#26766b','#b96643','#678450','#87729e','#507a9b','#aa8740'];
  const text=(x,y,s,size=18,extra='')=>`<text x="${x}" y="${y}" font-size="${size}" ${extra}>${s}</text>`;
  function human(x,y,i=0,s=1,mood='calm') {
    const skin=['#b87a55','#edc39d','#865941','#d69b78'][i%4],coat=palette[i%6];
    return `<g transform="translate(${x} ${y}) scale(${s})"><path d="M-12 27Q0 19 12 27L16 61H-16Z" fill="${coat}"/><path d="M-8 60L-9 84M8 60L10 84" stroke="#415653" stroke-width="7" stroke-linecap="round"/><path d="M-12 30L-24 49M12 30L24 43" stroke="${skin}" stroke-width="6" stroke-linecap="round"/><circle cy="8" r="13" fill="${skin}"/><path d="M-13 4Q-13-10 1-8Q16-8 13 8L8 0Q1 5-8 0L-13 8" fill="${i%2?'#634c3c':'#353c3a'}"/><path d="M-6 8h2M4 8h2" stroke="#493c34" stroke-width="1.7"/>${mood==='surprise'?'<ellipse cy="15" rx="2.4" ry="3" fill="#754f3e"/>':`<path d="${mood==='thoughtful'?'M-4 16L4 14':'M-4 14Q0 18 4 14'}" fill="none" stroke="#754f3e"/>`}${i===4?'<path d="M-11 7H-1V12H-11ZM1 7H11V12H1ZM-1 9H1" fill="none" stroke="#415653"/>':''}</g>`;
  }
  function tree(x,y,s=1,pine=false){return `<g transform="translate(${x} ${y}) scale(${s})"><path d="M0-18V8" stroke="#997351" stroke-width="5"/>${pine?'<path d="M0-70L-26-28H-17L-31-10H31L17-28H26Z" fill="#799b7d"/>':'<circle cy="-40" r="25" fill="#a9bd89"/><circle cx="-16" cy="-30" r="17" fill="#8fa978"/><circle cx="14" cy="-34" r="19" fill="#9cb481"/>'}</g>`;}
  function building(x,y,type='house',s=1,id=''){
    const tall=type==='tower',mid=type==='midrise',apt=tall||mid,duplex=type==='duplex',town=type==='townhouse';
    const w=apt?96:town?150:100,h=tall?205:mid?122:duplex?80:70;
    let art=`<g transform="translate(${x} ${y}) scale(${s})" ${id?`id="${id}"`:''}><ellipse cx="${w/2+7}" cy="4" rx="${w/2+14}" ry="9" fill="#506a5720"/>`;
    if(!apt)art+=`<path d="M-7-${h-8}L${w/2}-${h+40}L${w+7}-${h-8}Z" fill="${duplex?'#9c7e6b':'#bc7a58'}" stroke="#896c56" stroke-width="2"/>`;
    art+=`<rect y="-${h}" width="${w}" height="${h}" rx="3" fill="${tall?'#c0cfd0':mid?'#cebca6':town?'#ddcfad':'#f0dbb8'}" stroke="#8c9f93" stroke-width="2"/>`;
    const rows=tall?7:mid?4:2,cols=town?5:3;
    for(let r=0;r<rows;r++)for(let c=0;c<cols;c++)art+=`<rect data-dwelling="${r*cols+c}" x="${9+c*(w-12)/cols}" y="${-h+10+r*(h-20)/rows}" width="${apt?17:town?17:18}" height="${apt?15:14}" rx="1.5" fill="#e9f3ef" stroke="#789b9a"/>`;
    for(let j=0;j<(town?3:duplex?2:1);j++)art+=`<rect x="${(j+.5)*w/(town?3:duplex?2:1)-7}" y="-25" width="14" height="25" fill="#537d74"/><circle cx="${(j+.5)*w/(town?3:duplex?2:1)+3}" cy="-12" r="1.5" fill="#f6d38b"/>`;
    if(tall)art+='<path d="M-4-205H100" stroke="#6d8e89" stroke-width="6"/>';
    return art+'</g>';
  }
  function landscape(kind='urban',id=''){
    let s=`<g ${id?`id="${id}"`:''}><rect width="1000" height="540" rx="12" fill="#edf1e5"/><path d="M0 90Q160 20 310 95T680 72T1000 100V180H0Z" fill="#dce7d9"/>`;
    if(kind==='rural')s+='<path d="M0 140H1000V540H0Z" fill="#e7e1b6"/><path d="M0 235H1000M0 350H1000M140 150L65 540M780 140L930 540" stroke="#d3cd97" stroke-width="18"/><path d="M0 420Q550 410 1000 250" stroke="#c7cabc" stroke-width="32" fill="none"/>';
    else s+='<path d="M0 337H1000M510 140V540" stroke="#c9d0c8" stroke-width="46"/><path d="M0 337H1000M510 140V540" stroke="#f9faf4" stroke-width="2" stroke-dasharray="12 14"/><path d="M0 310H1000M0 364H1000M481 140V540M539 140V540" stroke="#f9f7ef" stroke-width="7"/>';
    const positions=kind==='rural'?[[130,245,'house'],[705,325,'duplex'],[375,430,'house']]:[[90,280,kind==='urban'?'tower':'house'],[240,280,kind==='urban'?'midrise':'duplex'],[590,280,'townhouse'],[795,280,'duplex'],[95,492,'house'],[275,492,'duplex'],[595,492,'house'],[800,492,kind==='urban'?'midrise':'house']];
    positions.forEach(([x,y,t],i)=>s+=building(x,y,t,1,id?`${id}-b${i}`:''));
    [[35,295],[415,278],[755,500],[945,295],[35,505],[940,505],[740,195]].forEach(([x,y],i)=>s+=tree(x,y,.8,kind==='rural'||i%3===0));
    s+='<rect x="372" y="400" width="86" height="100" rx="16" fill="#cddfbb"/><path d="M384 414Q440 444 382 488" stroke="#f4edd9" stroke-width="8" fill="none"/>'+tree(425,446,.6);
    if(kind!=='rural')s+=human(430,265,2,.45)+human(755,385,1,.5)+human(785,405,3,.32);
    return s+'</g>';
  }
  function monitor(x,y,kind=0){return `<g transform="translate(${x} ${y})"><rect width="92" height="59" rx="5" fill="#426c66"/><rect x="5" y="5" width="82" height="46" rx="2" fill="#f4f7ec"/>${kind===0?'<path d="M12 39L25 25L39 32L54 18L76 12" fill="none" stroke="#b96643" stroke-width="3"/>':kind===1?'<path d="M18 19H71M18 36H71M30 15V23M60 32V40" stroke="#507a9b" stroke-width="2"/><circle cx="43" cy="19" r="3" fill="#b96643"/><circle cx="52" cy="36" r="3" fill="#26766b"/>':'<path d="M14 14H76M14 25H76M14 36H76M32 10V43M56 10V43" stroke="#98b2a0"/>'}<path d="M46 59V72M28 72H64" stroke="#698579" stroke-width="4"/></g>`;}
  function summary(kind){
    let s='';
    if(kind===0)s=human(45,22,0,.55)+human(102,22,1,.55)+human(157,22,4,.55)+text(40,87,'E',14)+text(96,87,'U',14)+text(151,87,'O',14);
    if(kind===1)s=building(10,83,'house',.42)+building(78,83,'duplex',.42)+building(150,83,'midrise',.42)+'<rect x="74" y="32" width="48" height="54" rx="5" fill="none" stroke="#b96643" stroke-width="3"/>'+text(85,101,'w',14);
    if(kind===2)s='<path d="M15 84H195M22 83C57 83 64 17 104 17S149 83 188 83" fill="none" stroke="#26766b" stroke-width="3"/><path d="M76 81V38M132 81V38" stroke="#b96643" stroke-width="2"/><path d="M76 95H132" stroke="#b96643" stroke-width="4"/>';
    if(kind===3)s='<path d="M25 60H185M25 49V71M185 49V71" stroke="#507a9b" stroke-width="3"/><circle cx="103" cy="60" r="7" fill="#b96643"/>'+text(18,95,'lower',12)+text(80,30,'estimate',12)+text(158,95,'upper',12);
    if(kind===4)s='<path d="M12 37L36 13L58 23L81 10L97 26L124 12L148 28L192 40L182 74L146 86L122 66L92 83L65 64L36 74Z" fill="#dce7d9" stroke="#98ad9b"/><path d="M45 40H100M117 64H181M45 33V47M100 33V47M117 57V71M181 57V71" stroke="#507a9b" stroke-width="3"/><circle cx="70" cy="40" r="4" fill="#b96643"/><circle cx="158" cy="64" r="4" fill="#26766b"/>';
    if(kind===5)s='<rect x="30" y="18" width="115" height="76" rx="5" fill="#f0e5d1" stroke="#bfa98b"/><path d="M40 35H133M40 53H133M40 72H133M71 25V88" stroke="#9ab2a0"/><circle cx="128" cy="50" r="28" fill="#edf3eccc" stroke="#507a9b" stroke-width="4"/><path d="M149 71L176 96" stroke="#507a9b" stroke-width="8" stroke-linecap="round"/>'+text(117,57,'?',22);
    return `<svg viewBox="0 0 210 110" aria-hidden="true" focusable="false">${s}</svg>`;
  }
  window.LFSArt={palette,text,human,tree,building,landscape,monitor,summary};
})();
