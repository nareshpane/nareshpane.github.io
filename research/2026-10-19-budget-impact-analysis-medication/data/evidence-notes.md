# Evidence review and model conventions

Prepared 2026-10-19. No patient-level administrative data were supplied.

## Original supplied source

`SR0781-Cosentyx_Combined_Review CADTH Pharmacoeconomic Review - BIA.pdf`

SHA-256: `A457422B67B5792E57854BE310084DAA53D0D183D325B48B72E671A3B0439A45`.
The original was read, not edited. Printed page numbers coincide with the PDF page indices used by links.

- Pages 9–12: final monograph added inadequate response to conventional systemic therapy and q4w maintenance, with q2w escalation based on response. The source's phrase “twice-weekly” on p. 11 conflicts with the schedules specified elsewhere; the tool uses **every two weeks**, not twice a week.
- Pages 36–39: clinical diagnosis, Hurley stages, treatment pathway, IL-17A binding, adalimumab adult dosing (160 mg, 80 mg two weeks later, then 40 mg weekly from week 4). Adjunct treatments are not identical substitutes.
- Pages 143–148: original economic submission predates final dosing revisions and revised submitted price. Indirect comparison with adalimumab is uncertain. This website does not perform a cost-effectiveness analysis.
- Pages 165–167, Table 8: secukinumab $882.5900/150 mg unit, two units for 300 mg; adalimumab biosimilar $471.2700/40 mg. Secukinumab price is the highest sponsor-submitted provincial price (including BC); comparator prices generally Ontario formulary, accessed June 28–July 10, 2023. Product listing agreements are not reflected. Do not label these current BC net prices.
- Pages 173–174, Table 17: BC adults 4,478,159; prevalence 1%; diagnosed 19%; moderate/severe 56%; sponsor biologic uptake 14%; initial BC coverage factor 41%. Sponsor national incremental total $76,542,993. Reference biologic market 100% adalimumab biosimilars.
- Pages 175–178: historical expert estimate around 100 publicly funded BC biologic users plus 20 past discontinuers; biologic uptake revised to 3.5%, BC population coverage factor 73%, biologic-use growth 10% annually. These are not contemporary observations.
- Page 178, Table 18: existing-user switching 54/59/61% revised to 10/20/30%; biologic-growth comparison labelled 2.5% versus 10%. Table 17 separately labels population growth 2.85%; retain the distinction.
- Page 179, Tables 19–20: combined CDA-AMC national reanalysis $9,547,349. Separate revisions (after correction) are not sequential additive contributions. No population-share BC allocation is valid here.

## Current official context

- BC September 2025 PharmaCare newsletter p. 10: HS limited coverage effective August 13, 2025. The website is an educational **retrospective counterfactual**, not a prediction about a currently unlisted drug.
- BC HLTH 5485 (2025/07/24 version, checked 2026-10-19), pp. 1–2: common HS criteria for adalimumab biosimilars and secukinumab; age ≥18; Hurley II/III; recent continuous 90-day optimally dosed oral-antibiotic failure; ≥3 abscesses/nodules, ≥2 anatomical areas, ≥1 draining sinus/fistula. Dermatologist and body map required. Initial 6 months, renewal 1 year with maintained response; one biologic at a time. Use current form rather than national recommendation criteria as BC criteria.
- BC MSP code descriptions and Teleplan Chapter 2: ICD-9-based billing; minimum first three ICD-9 characters. Do not assume complete ICD-9-CM 705.83 is stored in every extract; decimal handling and detail require the specific dictionary. A broad 705 code cannot uniquely identify HS.
- CIHI latest NACRS closed-year quality document, updated August 2026, 2025–2026 data: partial mandate in BC; estimated ED coverage **75.0%**, Table 1 p. 9. BC expected-visit denominator comes from previous-year CMDB. Reporting level changes diagnosis availability. This is encounter coverage, not sensitivity of an HS phenotype.
- BC Health Data Central states PLIS repository not currently available for access requests. It cannot be promised as an accessible case-finding dataset. HS diagnosis is clinical.
- BC About PharmaNet, updated 2026-06-29: community-pharmacy dispensing network, medication histories and limited coverage information. Patient-facing records omit dispense costs. Authorized research financial extracts require definitions; residual non-public charge is not necessarily final patient burden.
- BC Fair PharmaCare regular assistance levels, 2026-01-06: income-related deductible/maximum, 70% public payment after deductible, 100% after family maximum. This page's patient illustration assumes all charges eligible and simplified coordination of benefits; it is not actual adjudication.
- 2023 phenotype-development/validation study (PMC10334943): supports the need for local case-definition validation. No foreign-system accuracy estimate is assigned to BC.

Official links are versioned in `js/assumptions.js`. Some BC HTML endpoints returned gateway errors; official indexed text and the current downloadable shared HS form were available. No source claim about a 2026 observed BC HS patient count is made.

## Explicit numerical choices

- Half-open 52-week intervals: first treatment year [0,52), next [52,104). Secukinumab q4w: 32 units first year, 26 maintenance; q2w: 56 and 52. Adalimumab: 54 and 52 units of 40 mg. These differ from daily/365-day annualizations and are independently calculated.
- Starting biologic cohort is established adalimumab use; existing users do not reload annually. New courses load once. Switching later in a year truncates prior exposure and begins a new event schedule.
- Default N=120 and cumulative shares 30/40/50% are requested teaching assumptions, separate from the historical 121.739648 top-down expected count and synthetic bottom-up count.
- Growth=10% from Year 2 is a historical scenario assumption, not BC population growth. Expected-count modelling permits fractional stocks. Record explorer counts are discrete integers.
- Optional shared exits act at annual boundaries. Returners reload adalimumab and are protected from immediate reswitch; other A users can replace them. Achieved share is capacity constrained.
- Payer example is separate from aggregate BIA. It partitions each charge once into public/private/final patient amounts. Non-drug costs are excluded.
- Monte Carlo seed=20261019; assumed beta/lognormal distributions, not fitted evidence. Sorted sampled shares change their marginals. Repeatability and half-sample mean diagnostic are reported, not claimed as proof of convergence.
