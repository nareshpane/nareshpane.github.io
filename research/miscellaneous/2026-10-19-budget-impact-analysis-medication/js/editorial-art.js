/* Original generated editorial artwork; inventory and prompt in figures/illustrations/. */
export const artworkNames=['research','residents','pharmacy','hospital','physician','ed','lab','programmer','linkage','dispensing','economist','policy'];
const descriptions=[
 'Research analyst at a computer with reports and financial charts',
 'Adults of different ages and appearances in everyday clothing',
 'Pharmacist dispensing medication to a patient at a stocked pharmacy counter',
 'Nurse caring for an adult in a hospital bed; general healthcare-service illustration',
 'Physician reviewing clinical findings with a seated adult patient',
 'Emergency entrance, ambulance and uniformed paramedic',
 'Laboratory scientist with microscope and sample tubes',
 'Programmer working at two displays with code and record tables',
 'Data analyst reviewing linked relational tables',
 'Pharmacist and patient exchanging a prescription at a second pharmacy',
 'Health economist with spreadsheet, calculator and financial charts',
 'Four professionals discussing a briefing at a meeting table'
];
// Each viewport selects a complete tile. Static raster subjects move as a whole.
export function artwork(name,x,y,w,h){
 const i=artworkNames.indexOf(name);if(i<0)throw new Error('Unknown illustration: '+name);
 const aw=1448,ah=1086,tw=aw/4,th=ah/3,side=Math.min(w,h),left=x+(w-side)/2,top=y+(h-side)/2;
 return `<g class="editorial-art" data-artwork="${name}"><rect x="${x}" y="${y}" width="${w}" height="${h}" rx="12" fill="#fff"/><svg x="${left}" y="${top}" width="${side}" height="${side}" viewBox="${i%4*tw} ${Math.floor(i/4)*th} ${tw} ${th}" preserveAspectRatio="xMidYMid meet" overflow="hidden" role="img" aria-label="${descriptions[i]}"><title>${descriptions[i]}</title><image href="2026-10-19-budget-impact-analysis-medication/figures/illustrations/professional-cast.png" width="${aw}" height="${ah}"/></svg></g>`;
}
