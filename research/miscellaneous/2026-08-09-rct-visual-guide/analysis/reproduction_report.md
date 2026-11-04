# Statistical reproduction and reconciliation

Executed deterministically on the unchanged Dryad CSV. SHA-256: `7358326b071203978fb575e7bc4eb4fa8eb9985bdda1a2eb1dd5154afb52076a`. Source version 21098. All 4,556 deposited rows retained, 2,278 per assignment. TX=0 placebo, TX=1 synbiotic per README. Denominators correspond to the randomized population; a complete binary indicator does not prove complete prospective follow-up.

## Methods

Crude arm risks; unpooled Wald RD interval; Katz log-RR interval; uncorrected Pearson chi-square and two-sided Fisher exact tests. These are fixed-sample, unadjusted intervals. Protocol pp.31–33 prespecifies ITT, chi-square/Fisher tests and interim monitoring. Reporting summary identifies SAS 9.4. No original SAS program or exact published CI procedure is deposited; matching rounded values is numerical replication, not proof of identical software or sequential inference.

## Published Table 2 versus released-data calculations

| Endpoint | Published placebo/synbiotic events | Deposited placebo/synbiotic events | Published RR (95% CI) | Reproduced RR (95% CI) | RR minus published |
|---|---|---|---|---|---|
| Sepsis or death | 206/123 | 206/123 | 0.60 (0.48, 0.74) | 0.59709 (0.48126, 0.74079) | -0.00291 |
| Sepsis composite | 202/117 | 202/117 | 0.58 (0.46, 0.72) | 0.57921 (0.46474, 0.72186) | -0.00079 |
| Culture-positive sepsis | 27/6 | 27/6 | 0.22 (0.09, 0.53) | 0.22222 (0.09193, 0.53720) | +0.00222 |
| Culture-negative sepsis B | 36/19 | 36/19 | 0.53 (0.30, 0.92) | 0.52778 (0.30367, 0.91728) | -0.00222 |
| pSBI–LRTI | 139/92 | 139/92 | 0.66 (0.51, 0.88) | 0.66187 (0.51190, 0.85578) | +0.00187 |
| Death | 4/6 | 4/6 | 1.50 (0.42, 5.31) | 1.50000 (0.42386, 5.30836) | +0.00000 |
| Diarrhoea | 59/12 | 59/12 | 0.20 (0.11, 0.38) | 0.20339 (0.10964, 0.37731) | +0.00339 |
| Local infection | 33/16 | 33/16 | 0.48 (0.27, 0.88) | 0.48485 (0.26763, 0.87838) | +0.00485 |
| Abscess | 11/5 | 11/5 | 0.45 (0.16, 1.33) | 0.45455 (0.15819, 1.30613) | +0.00455 |
| Omphalitis | 13/3 | 13/3 | 0.23 (0.07, 0.81) | 0.23077 (0.06585, 0.80872) | +0.00077 |
| Gram-negative sepsis | 16/4 | 16/4 | 0.25 (0.08, 0.75) | 0.25000 (0.08371, 0.74663) | +0.00000 |
| Gram-positive sepsis | 11/2 | 11/2 | 0.18 (0.04, 0.82) | 0.18182 (0.04035, 0.81936) | +0.00182 |

Primary risks: 0.05399473 versus 0.09043020; RD -0.03643547 (95% CI [np.float64(-0.051430185383848), np.float64(-0.021440754036696347)]); RR 0.59708738 (95% CI [0.4812609515655235, 0.740790077758002]); Pearson p=2.0273265e-06; Fisher p=2.41148232e-06. ARR=0.03643547; reciprocal ARR=27.44578 (Wald-derived interval [np.float64(19.44383424902172), np.float64(46.64015072830353)]). Published NNT 27 [19,47]; the reciprocal point estimate is 27.45 and rounds up to 28 under a ceiling convention. Do not force it to 27.

## Data integrity and unresolved discrepancies

- 2 repeated-ID excess rows, 4 rows in repeated-ID groups; 0 exactly duplicated records. Retained all records; no private IDs displayed. Distinct clinical records cannot safely be collapsed by identifier alone.
- Primary indicator equals SEPSIS OR DEATH for every row; sepsis equals CULTPOS + SEPSISB + SEPSISC. Culture-positive sepsis equals GRAMPOS + GRAMNEG. Binary coding and dose/time ranges verified.
- Dataset LOSTFU combines lost/withdrawn: placebo 82, synbiotic 129. Published Fig.1 lost/withdrawn sums are 89 and 131. Original Completed_Study values also differ from STATUS. Preserve these separate measures; do not harmonize silently.
- Published Table 1 birth weight is 2799.9 vs 2803.7 g. MOMAGE and MOMPAR have 14 and 15 missing values in the release, despite the published table's general no-missing footnote. Report variable-specific denominators.
- Corrigendum: 88 suspected cases comprise culture-positive and culture-negative infants; 94 additional cultures were negative. Dataset CULTPOS=33, SEPSISB=55 (sum 88). Do not call all 88 culture-positive; the additional 94 lack a dedicated sample/testing indicator in this deposit.
- Published LRTI interval upper endpoint is 0.88; our Katz upper endpoint is 0.85578. Culture-positive and abscess upper endpoints also differ at the published precision (0.537 vs 0.53; 1.306 vs 1.33). Do not invent a cause: original interval code is unavailable. All differences remain visible in the table.
- The release contains 95 distinct four-character Study_ID prefixes; the published setting is 149 villages. This cannot independently verify the reported number of villages and is not substituted for it.
- TIME is age of a recorded morbidity event, not an individual censoring time for everyone. Two primary-event ages missing. No defensible exact Kaplan–Meier reconstruction: loss dates and enrollment ages are not supplied. The case-age histogram is descriptive and is not a survival curve.
- PODRR is entirely missing; POP* fields are described only as subsamples. They are not used to guess primary denominators. Numeric `.` treated as missing; no other endpoint transformations.

## Sensitivity (exploratory, not recovered outcomes)

| Analysis | Synbiotic events/n | Placebo events/n | RR (95% CI) |
|---|---|---|---|
| all_assigned | 123/2278 | 206/2278 | 0.5971 (0.4813, 0.7408) |
| exclude_lostfu | 123/2149 | 206/2196 | 0.6101 (0.4920, 0.7567) |
| block_positions_1_2_3 | 103/1864 | 150/1592 | 0.5865 (0.4605, 0.7468) |
| block_positions_1_2 | 74/1262 | 97/1067 | 0.6450 (0.4820, 0.8631) |
| all_flagged_events | 252/2278 | 288/2278 | 0.8750 (0.7465, 1.0256) |
| flagged_synbiotic_only | 252/2278 | 206/2278 | 1.2233 (1.0272, 1.4568) |
| flagged_placebo_only | 123/2278 | 288/2278 | 0.4271 (0.3486, 0.5232) |

Flagged-outcome scenarios add assumed events among LOSTFU rows; these are deliberately hypothetical and are not observations. Excluding flagged rows is a complete-case descriptive comparison, not automatically unbiased. Block-position restrictions use supplied indicators and reduce precision; they do not recreate concealed assignment records. Published supplement restrictions were 0.59 [0.46,0.75] and 0.65 [0.48,0.86]; compare computed results above.

## Exploratory effect modification

Male RR / female RR = 0.8856, 95% CI [0.5698028120688136, 1.3764287781048856], interaction p=0.58923. Wald contrast of independent log-RRs, not a difference in within-sex significance. No subgroup efficacy claim; not prespecified, no multiplicity adjustment, no sequential adjustment.

## Validation assertions and limits

Shape, checksum, arm coding/counts, binary ranges, composite consistency, dose ranges and event ages pass. Duplicate IDs, ambiguous subsamples, inconsistent follow-up summaries and missing censoring dates are reported, not suppressed. This pipeline reproduces aggregate binary estimates; it does not validate event adjudication, allocation concealment, all adverse events, protocol execution or a biological mechanism. Original publication full text was subscription-restricted; public tables, figure, abstract, corrigendum, supplements and the deposited 52-page protocol were reviewed. Publisher PDFs were held only in temporary review storage; CC0 deposited protocol and README are preserved locally.
