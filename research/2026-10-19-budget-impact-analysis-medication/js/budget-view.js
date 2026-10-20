/* One source for the requested illustrative expenditure example, in CAD.
 * It never changes model inputs, calculated sensitivities or model CSV exports. */
const rows=Object.freeze([
 Object.freeze({year:1,reference:2900000,newCost:3700000,impact:800000}),
 Object.freeze({year:2,reference:3200000,newCost:4200000,impact:1000000}),
 Object.freeze({year:3,reference:3500000,newCost:4900000,impact:1400000})
]);
export const policyIllustration=Object.freeze({rows,reference:9600000,newCost:12800000,total:3200000,illustrative:true});
export function budgetView(result,mode='illustration'){return mode==='live'?result:policyIllustration;}
export const budgetViewLabel=mode=>mode==='live'?'Live calculated scenario':'Illustrative policy example';
