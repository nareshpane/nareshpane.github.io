/* Accessible expansions for compact SVG labels; no analytical values. */
export const abbreviations={
 'CDA-AMC':"Canada’s Drug Agency / L’Agence des médicaments du Canada",
 CADTH:'Canadian Agency for Drugs and Technologies in Health (former name)',
 pCPA:'pan-Canadian Pharmaceutical Alliance',
 DAD:'Discharge Abstract Database',MSP:'Medical Services Plan',
 NACRS:'National Ambulatory Care Reporting System',PLIS:'Provincial Laboratory Information Solution',
 BIA:'Budget impact analysis',BI:'Budget impact',HS:'Hidradenitis suppurativa',
 BC:'British Columbia',CAD:'Canadian dollars; CAD $000 means thousands of Canadian dollars',
 ICD:'International Classification of Diseases',
 'ICD-9':'International Classification of Diseases, ninth revision',
 'ICD-9-CM':'ICD-9, Clinical Modification; not interchangeable with truncated physician-billing codes',
 'ICD-10':'International Classification of Diseases, tenth revision',
 'ICD-10-CA':'ICD-10, Canadian enhancement',
 DIN:'Drug Identification Number',SQL:'Structured Query Language',
 CIHI:'Canadian Institute for Health Information',
 CMDB:'Canadian Management Information System Database',
 PPV:'Positive predictive value',QALYs:'Quality-adjusted life-years',
 CRP:'C-reactive protein',ALT:'Alanine aminotransferase',
 q4w:'Every four weeks',q2w:'Every two weeks'
};
export function svgTerminology(body){
 return Object.entries(abbreviations).filter(([key])=>new RegExp('\\b'+key+'\\b').test(body)).map(([key,value])=>key+': '+value).join('; ');
}
