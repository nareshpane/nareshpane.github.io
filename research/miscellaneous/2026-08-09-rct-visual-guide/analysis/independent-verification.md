# Independent statistical recheck — November 4, 2026

A second implementation reads the original CSV using Python csv, not pandas, and does not import the existing reproduction functions. It uses standard-library means/SDs, direct chi-square arithmetic and a two-sided hypergeometric Fisher test. Source bytes and all rows are retained. Published values are transcribed from corrected Table 2; the preceding webpage is preserved separately.

| Measure | Current webpage (underlying RR) | Published RR [CI] | Independently recalculated RR [CI] | Status |
|---|---|---|---|---|
| SEPSISDEATH | 0.597087 | 0.60 [0.48, 0.74] | 0.597087 [0.481261, 0.740790] | Current agrees; rounded published point agrees |
| SEPSIS | 0.579208 | 0.58 [0.46, 0.72] | 0.579208 [0.464743, 0.721865] | Current agrees; rounded published point agrees |
| CULTPOS | 0.222222 | 0.22 [0.09, 0.53] | 0.222222 [0.091926, 0.537200] | Current agrees; rounded published point agrees; CI endpoint discrepancy retained |
| SEPSISB | 0.527778 | 0.53 [0.30, 0.92] | 0.527778 [0.303669, 0.917278] | Current agrees; rounded published point agrees |
| SEPSISC | 0.661871 | 0.66 [0.51, 0.88] | 0.661871 [0.511900, 0.855778] | Current agrees; rounded published point agrees; CI endpoint discrepancy retained |
| DEATH | 1.500000 | 1.50 [0.42, 5.31] | 1.500000 [0.423860, 5.308359] | Current agrees; rounded published point agrees |
| DIARRHEA | 0.203390 | 0.20 [0.11, 0.38] | 0.203390 [0.109637, 0.377312] | Current agrees; rounded published point agrees |
| LOCINF | 0.484848 | 0.48 [0.27, 0.88] | 0.484848 [0.267626, 0.878381] | Current agrees; rounded published point agrees |
| ABSCESS | 0.454545 | 0.45 [0.16, 1.33] | 0.454545 [0.158185, 1.306135] | Current agrees; rounded published point agrees; CI endpoint discrepancy retained |
| OMPHA | 0.230769 | 0.23 [0.07, 0.81] | 0.230769 [0.065850, 0.808723] | Current agrees; rounded published point agrees |
| GRAMNEG | 0.250000 | 0.25 [0.08, 0.75] | 0.250000 [0.083709, 0.746633] | Current agrees; rounded published point agrees |
| GRAMPOS | 0.181818 | 0.18 [0.04, 0.82] | 0.181818 [0.040346, 0.819360] | Current agrees; rounded published point agrees |

## Hypothesis-test reconciliation

| Endpoint | Published p / footnote | Independent Pearson p | Independent two-sided Fisher p |
|---|---|---|---|
| SEPSISDEATH | <0.001 / chi-square | 2.0273265e-06 | 2.4114823e-06 |
| SEPSIS | <0.001 / chi-square | 8.015518e-07 | 9.452729e-07 |
| CULTPOS | <0.001 / chi-square | 0.00024355598 | 0.00030903293 |
| SEPSISB | 0.021 / chi-square | 0.021097099 | 0.029049418 |
| SEPSISC | 0.002 / chi-square | 0.0015041523 | 0.0018381198 |
| DEATH | 0.526 / Fisher | 0.52663519 | 0.75363576 |
| DIARRHEA | <0.001 / chi-square | 1.8892237e-08 | 1.0565983e-08 |
| LOCINF | 0.015 / chi-square | 0.014616773 | 0.020609051 |
| ABSCESS | 0.133 / chi-square | 0.13293168 | 0.20930795 |
| OMPHA | 0.014 / chi-square | 0.012265878 | 0.021051725 |
| GRAMNEG | 0.007 / chi-square | 0.0071625512 | 0.011640047 |
| GRAMPOS | 0.012 / chi-square | 0.012429158 | 0.022277291 |

## Findings

- Mortality: published Table 2 gives p=0.526 and labels Fisher exact; independent two-sided Fisher is 0.7536358, while Pearson is 0.5266352. The label/value pair is not reproduced.
- Omphalitis: published p=0.014 versus uncorrected Pearson 0.0122659. Cause not identified from released software documentation.
- Published culture-positive, LRTI and abscess CI upper endpoints differ from Katz reconstruction at shown precision.
- Published LRTI p=0.002 versus Pearson 0.0015042 is compatible with rounding; some borderline displayed rounding conventions differ.
- LOSTFU versus published flow dispositions, repeated identifiers, missing maternal covariates and ID-prefix/village counts remain unresolved.

The primary reconstruction remains RR 0.5970874 [0.4812610, 0.7407901], RD −0.03643547 [−0.05143019, −0.02144075], Pearson p=2.0273265e−6. This verifies an unadjusted fixed-sample reconstruction, not the unavailable original interval software or complete sequential procedure. A numerical discrepancy is not repaired by changing records or relabeling a published method.

Protocol pp.31–33 documents ITT, binary chi-square/Fisher tests, planning risks and interim monitoring; stepwise regression was proposed if imbalance was observed. No released SAS program identifies exact CI code or establishes which adjusted model was actually fitted. The corrected 88-case interpretation remains in place. Survival risk sets are not recoverable from event ages alone.

Checks: 4,556 records, 74 fields, 2,278 per arm; 12 published outcome pairs, six continuous baseline mean/SD checks, three block/complete-case checks. Two repeated-ID excess records are retained; no exactly duplicated row. Endpoint indicators have no numeric missing values, but clinical follow-up is not complete. No original CSV bytes changed.

Sources: https://www.nature.com/articles/nature23480/tables/2 ; https://doi.org/10.1038/nature25006 ; deposited README and protocol. The main article body remains subscription-restricted. Displayed method equations in the revised page are our explanatory reconstructions, not purported quotations of equations from an unavailable body text.
