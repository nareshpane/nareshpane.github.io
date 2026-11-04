# Final implementation report

Updated November 4, 2026. Scope is limited to the named HTML page and its dedicated asset directory. Nothing staged, committed or pushed. Pre-existing changes in `research.html` and unrelated untracked directories are untouched.

## Preservation and expansion

- All 30 original lettered conceptual sections A–AD remain, plus the original two cohort visuals and eight laboratory/conduct/extension sections: **40 preserved components**. Original HTML is archived for comparison. The guide remains recognizable through its serif headings, warm paper, blue/terracotta arm colors, person symbols, conceptual table, flow diagrams and controls.
- Laboratories moved next to relevant theory. Trial bridges distinguish actual individual assignment in village blocks from unrestricted synthetic randomization, cluster and factorial alternatives.
- Six parts organize the experiment, causal foundations, effect/inference theory, empirical replication, conduct/extensions, interpretation/reproducibility.
- Added trial formulation and hypothesized mechanism, verified protocol and amendments, original CONSORT-style redraw, corrected sepsis definitions, full data audit, baseline analysis, principal binary reconstruction, secondary-component forest, event-age feasibility, exploratory sex interaction, missingness and block-position sensitivity, sources and replication links.
- Added derivations for design-based unbiasedness and finite-population variance, binary risks/effects, delta-method log-RR uncertainty, RD/NNT uncertainty, pooled z/Pearson testing, planning power and Kaplan–Meier risk-set requirements. Assumptions and unavailable quantities are explicit.

## Figures and interactions

Eight editable SVG files: intervention mechanism, participant flow, corrected synthetic CI and subgroup scales, generated real baseline balance, principal risk comparison, component/secondary forest and event-age histogram. Existing person-grid, SMD, counterfactual, coverage, Fisher, power, conduct, cluster, factorial, threat-map and pipeline visuals remain.

Retained seeded allocation/reset, potential-outcome reveal, effect tabs, baseline reroll, dual-world reveal, repeated interval simulation, Fisher simulation/reset and power controls. Improvements: exact normal draws for the fixed-SE synthetic coverage model; plus-one Monte Carlo p estimate; baseline-risk/sample-size/effect controls for power, checked against SciPy; finite allocation transition with reduced-motion support. Added synthetic 2×2 sandbox and differential missing-event scenario controls with resets. No autoplay or duplicate timeline controls.

Navigation includes six collapsible parts, all chapters, selected deeper subsections, expand/collapse-all, active-section and active-parent state, an independent scrollbar and active-link visibility management. A mobile drawer supports Escape dismissal and keyboard-operable controls. Anchor headings receive focus. Tables use header scopes; generated images have textual alternatives; original quantitative examples are labeled synthetic.

Metadata closely follows the supplied 280px pale-green reference, five rows and corresponding label colors. Creation date comes from Git (August 9, 2026), modified November 4, 2026. Harness is verified as Codex on Windows. Exact model/reasoning attribution is unavailable and shown as “Not recorded”; screenshot values are not invented as session facts.

## Empirical deliverables

All three deposited source files downloaded and preserved with original names and byte-for-byte Dryad MD5 matches: CSV, 52-page protocol and README DOCX. Public metadata archived. Publisher supplements, reporting summary, public tables, flow figure, abstract and corrigendum reviewed; copyrighted materials linked rather than assumed to be CC0. Full subscription article body could not be reviewed.

Python scripts, dependency ranges, data dictionary for all 74 variables, aggregate JSON/JS/CSV, generated SVGs and reconciliation report are included. Browser scripts contain aggregates only. No identifiers appear in page tables or charts.

Primary result: 123/2,278 versus 206/2,278; RR **0.597087**, Katz interval **0.481261–0.740790**; RD **−3.64355 percentage points** with unpooled Wald interval **−5.14302 to −2.14408 pp**. Pearson p approximately **2.0273×10⁻⁶**. Main rounded published result reproduced. Other published component numerators also match.

Discrepancies remain visible: repeated identifiers, follow-up classifications, maternal age/parity missingness, 95 deposited ID prefixes versus a published 149-village setting, and some secondary CI endpoints. NNT reciprocal is 27.4458; published 27 versus ceiling convention 28 is explained. The sex interaction is exploratory (ratio of RRs 0.886; interval 0.570–1.376; p≈0.589). Missingness scenarios are assumptions and can reverse the point estimate under sufficiently differential assumptions.

## Verification

Automated validation checks raw checksums, shape, coding, arm totals, composite identities, ranges, all published outcome numerator pairs, independent primary formulas, participant-flow arithmetic, event-age histogram totals, local links/anchors, unique IDs, alt text, valid SVG XML, JavaScript syntax, preservation and JavaScript power versus SciPy. Exact results are recorded in `validation-results.json`.

Browser testing covered the initial 1280px desktop layout, metadata, real-data primary figure, navigation tracking, all retained interactive labs, all new sliders/reset controls, extreme parameter values and mathematical rendering. At 390px, a discovered retained-image overflow was repaired. Dense charts scroll inside their panels on small screens; no page-level horizontal overflow remains. Final checks at 320, 390, 768, 1280 and 1440 pixels show no page-level horizontal overflow. Navigation expand/collapse works by mouse and keyboard. All nine non-navigation disclosure controls were activated; rendered mathematical content has no MathJax errors. No browser console errors or warnings were recorded. Direct file URL testing was rejected by the browser protocol policy; the supported HTTP preview is the verified route.

## Remaining limitations

No exact survival reconstruction: two event ages missing, last-contact dates and individual entry ages absent. No original SAS code, exact published interval algorithm or complete interim data/DSMB record: reconstructed inference is unadjusted and fixed-sample. No validation of concealment in practice, full safety adjudication, biological mediation or transport to a broader clinical population. No claim that subgroup findings are confirmatory. External MathJax and optional Google Fonts need network access; static figures and tables survive without JavaScript, and system fonts/raw TeX are fallbacks. This is not a clinical recommendation.


## Final validation record

- 157 unique HTML IDs; 119 local resource/anchor references checked, with no missing target.
- All 8 linked images have alt text; all 8 source SVGs parse as XML and load in the browser.
- All 3 original Dryad file MD5s match; all 12 published outcome count pairs match.
- Independent primary formulas, flow arithmetic and event-age histogram totals pass.
- 17 buttons, 7 range controls; retained labs, effect tabs, new resets, all parameters and disclosure panels exercised.
- Four JavaScript power scenarios agree with SciPy with maximum absolute difference 1.1064e-7.
- JavaScript syntax checks and scoped git whitespace checks pass.
- Browser viewport override reset to the user's normal size after testing.
- A final unchanged-code rebuild produced 14 byte-identical analytical, figure and HTML outputs, confirming deterministic regeneration in the recorded environment.
