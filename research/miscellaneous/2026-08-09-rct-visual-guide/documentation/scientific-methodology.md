# Scientific methodology and review

The estimand is an assigned-strategy contrast for the day-60 sepsis/pSBI-or-death composite among randomized eligible infants. Treatment coding and endpoint variables come from the deposited README, not guesses from column names. The main denominator is 2,278 per assigned group. All rows remain, including repeated identifiers, because the release has no exactly duplicated records or authority to merge differing records.

## Computation and transformations

1. Verify raw CSV MD5 against Dryad metadata. Read with pandas; parse SAS `.` and blank fields as missing. Keep original bytes unchanged.
2. Assert shape, arm counts, binary coding, dose and age ranges, composite consistency and absence of primary events among LOSTFU-flagged records.
3. Summarize baseline cleaned BWT, MOMAGE and MOMPAR plus documented infant sex, gestational category and delivery place. Report n per variable. SMD divides mean difference by root-mean-square sample SD; no baseline hypothesis tests or data-driven adjustment.
4. Compute crude event proportions, unpooled Wald RD intervals, Katz log-RR intervals, OR, reciprocal ARR where positive, Pearson chi-square without continuity correction and two-sided Fisher exact p-values. No zero-cell correction is used. Positive event counts are required for the displayed RR intervals.
5. Generate JSON/CSV, static SVGs and a detailed report. Build static HTML tables from the same aggregate outputs. Formulas in the manuscript use the verified primary inputs.
6. Explore two block-position restrictions, complete-case exclusion and explicit missing-event scenarios. These are diagnostics or assumptions, not recovered outcomes. An exploratory sex interaction uses a contrast of independent subgroup log-RRs with its Wald uncertainty. No multiplicity adjustment or prespecified claim.

## Boundaries of reproduction

The protocol prescribes binary tests and ITT but does not supply the original SAS code or a precise reproducible CI algorithm. Agreement at rounded primary precision is a numerical reproduction. Intervals are not adjusted for sequential stopping, village dependence or selection from missing ascertainment. The original allocation blocks cannot be fully audited from clinical outcome records. No individual counterfactual, causal mechanism, CACE or universal efficacy is inferred.

TIME records morbidity age, not last-contact dates for all records. Only a case-age histogram is shown. Exact survival curves, delayed-entry risk sets and informative censoring sensitivity cannot be recovered from this field alone. Two primary-event ages are missing.

The reported corrected outcome categories and released indicators are explained separately. The additional 94 negative culture evaluations are not assigned to individuals or assumed equal to the 55 SEPSISB cases. Mortality has ten events and very wide uncertainty; morbidity benefit does not establish mortality benefit.

## Final conceptual review

- Event coding and the sign of benefit are explicit: favorable migraine response versus adverse infant endpoint.
- Randomization gives balance and unbiasedness over assignments, not identical realized groups.
- Complete randomization derivation is labeled separately from actual village-blocked assignment.
- Potential-outcome consistency, no interference and treatment versions are stated.
- Frequentist coverage is a property of a procedure across repetitions, not a posterior probability of one fixed interval.
- Fisher sharp-null inference is distinct from a mean-null test; synthetic permutations do not purport to replicate the trial design.
- Relative and absolute effects are accompanied by risks and denominators. NNT rounding and zero-crossing intervals are explained.
- Missing outcomes and complete numerical coding are distinguished; opposite-arm missingness assumptions can reverse a point estimate.
- Biological mechanisms, component hypotheses, exploratory interactions and clinical applicability remain qualified.
- Public tables, supplements, reporting summary, corrigendum and deposited protocol are cited. Full subscription article text and original software remain unavailable.

See `analysis/reproduction_report.md` for exact computed reconciliation, and `documentation/validation-results.json` for automated checks.

## Second independent verification

`analysis/independent_verification.py` uses csv and standard-library arithmetic, independently of the original pandas/SciPy functions. It reproduces all 12 outcome numerator pairs, six baseline mean/SD pairs and three alternate populations. For mortality, the published Fisher label accompanies p=0.526; two-sided Fisher is 0.7536358 while Pearson is 0.5266352. Omphalitis also has an unresolved published p-value discrepancy. Existing CI endpoint differences remain explicit. No raw records are changed. See `analysis/independent-verification.md` and `data/independent_verification.json`.

The expanded browser labs are synthetic: complete 1:1 assignments, a known-SE normal coverage model with SE proportional to 1/sqrt(n), and fixed-sample two-sided normal planning. The shared `statistical-math.js` calculation is checked against SciPy at varying significance levels. These are explanatory mathematical reconstructions, not the unavailable original SAS programs.
