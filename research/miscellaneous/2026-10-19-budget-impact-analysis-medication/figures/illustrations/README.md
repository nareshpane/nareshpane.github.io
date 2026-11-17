# Illustration inventory

## Published asset

`professional-cast.png` — 1448 × 1086 pixels, opaque RGB PNG, 2.04 MiB. Original AI-generated artwork created for this revision using the built-in imagegen tool. Twelve 362 × 362 tile viewports share one file; browser caching avoids duplicate downloads. No supplied screenshot was embedded, traced or recoloured. No third-party republication licence is asserted for the references. No attribution or watermarks were removed.

White illustration surfaces intentionally frame the opaque artwork on the peach/green page. Complete tile viewports preserve aspect ratio; independent SVG maps, equations, labels and charts remain authoritative. Raster chart marks are decorative, not data. Hospital scenes represent general service use, not typical HS severity.

| Tile (row, column) | Name | Reference concepts | Appearances |
|---|---|---|---|
| 1, 1 | research | 3, 5 | Scene 1; authorization/evidence review; prevalence; limitations |
| 1, 2 | residents | 2, 15–17 | Scene 2; prevalence; varied compact patient figures remain SVG |
| 1, 3 | pharmacy | 18, 19 | Scenes 3, 7, 8; PharmaNet source panel |
| 1, 4 | hospital | 1, 2, 15–17 | Scene 4; DAD panel |
| 2, 1 | physician | 8, 12, 13 | Scenes 4, 9; clinical pathway, severity assessment, decision tree, Special Authority, limitations |
| 2, 2 | ed | 1, 2 (general clinical style) | Scene 4; NACRS panel; original paramedic/ambulance composition |
| 2, 3 | lab | 7, 8 (professional style) | Scene 4; PLIS panel; research/authorization |
| 2, 4 | programmer | 3, 9 | Scene 5; bottom-up prevalence |
| 3, 1 | linkage | 6, 10 | Scenes 6, 7; cohort/coding uncertainty |
| 3, 2 | dispensing | 18–20 | Scenes 3, 7; public/private/patient payments; reimbursement limitations |
| 3, 3 | economist | 3, 5 | Scene 11; provincial financial appraisal |
| 3, 4 | policy | 3–5 | Scenes 12–13; coverage decision |

## Reference audit

All 20 supplied local screenshots were accessible and visually inspected. They are opaque images with substantial white backgrounds, ranging from 343–496 px wide and 184–443 px high. `reference-inventory.json` records each file’s dimensions, subject, category, alpha status, and nonwhite bounds. Image 3 is an already truncated wide scene; no attempt was made to invent missing original detail.

## Implementation

`js/editorial-art.js` centrally maps named roles to complete atlas tiles, with descriptions and square contained SVG viewports. `js/illustrations.js` retains editable maps, charts, documents, medication packages and six compact patient silhouettes with different hair, build, clothing and stance. The atlas is static: dispensing packets, SQL selection, data transfer, patient switches and financial bars are independent SVG elements. No nonexistent raster layers are animated.

Generation specification: `generation-prompt.txt`. No external runtime packages are used by the page.
