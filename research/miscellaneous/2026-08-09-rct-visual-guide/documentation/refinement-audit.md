# Refinement audit — November 4, 2026

Inspection precedes edits in this pass. The complete existing page, CSS, retained lab script, page interactions, build source, analytical pipeline, SVG sources and aggregate outputs were reviewed. `pre-refinement-guide.html`, `pre-refinement-summary.json` and `refinement-baseline-manifest.json` preserve the starting state.

## Preserve

Keep all six parts, 30 A–AD sections, the 40 original components, trial/corrigendum distinctions, source deposits, full data dictionary, static empirical figures, navigation and seeded laboratories. The existing warm paper, serif hierarchy, blue/terracotta arm colors and explicit evidence labels are effective. No new framework or external service is needed.

## Targeted changes

| Component | Finding | Refinement | Final evidence |
|---|---|---|---|
| Full-width prose | `.chapter>p` caps text at 88ch; inherited `.intro` adds 850px; part summaries at 78ch. Their available article width can be substantially greater. | Remove those particular caps, retain appropriate shrinkable grid tracks; expand desktop page ceiling moderately. | Browser width measurements before/after |
| Header | Large title competes with introduction; source authors only appear at the end. | Reduce title about 12%; insert full citation immediately below header, with all authors and three source links. | Desktop and mobile screenshots |
| Metadata | Placeholders contradict new explicit user-supplied labels. | Set exact labels GPT-6.1 Sol / High, preserve five-row geometry and other fields. | DOM assertions |
| Related panels | Padding mostly consistent, but cards/flow blocks use the same neutral tint, and some narrow cards overflow inline math. | Shared radius/padding, equal-width shrinkable tracks, semantic faint palette, local equation scrolling. | Grid symmetry and overflow checks |
| Clinical context | Existing schematic explains mechanism but contains no clinical actors or treatment timeline. | Original SVG compositions for eligibility/care, balanced oral assignment and follow-up, microbiology classification, infant counterfactual and reproducibility. | SVG sources; no reference screenshot copied |
| Baseline lab | Fixed 540-person sample and one allocation at a time. | Extend same lab with sample size, 200 seeded repeats, cumulative imbalance histogram and reset. | Count/balance/seed checks |
| 2×2 sandbox | Correct text/table outputs; visual link to risks and formulas is weak. | Add linked icon arrays, risk bars and substituted equations using same two existing event sliders. | Numeric table/plot agreement |
| Interval coverage | Correct fixed-SE synthetic normal model, only reroll control. | Confidence-level and illustrative sample-size controls; hold normal draws fixed when changing interval width. | 90/95/99% critical values and coverage checks |
| Power/test lab | Three planning parameters but fixed alpha; no distribution or rejection-region figure. | Alpha control and linked null/alternative densities, critical values, alpha/beta/power; reuse existing parameters. | JS/SciPy checks including varying alpha |
| Empirical forest | Correct log scale, limited count/context information. | Event columns, interval endcaps, direction labels and explicit pointwise uncertainty. | Generated from same aggregate data |
| Methods attribution | Sources distinguish unavailable original CI code, but educational equations could be tied more explicitly to documented methods. | Methods crosswalk, Pearson/Fisher derivations, sample-size equation and sequential-boundary explanation, all labeled reconstructions. | Protocol/table locators and explicit attribution |
| Verification | Primary result is correct; earlier report did not fully reconcile reported p-values. | Independent CSV/math implementation, current/published/recomputed comparison, exact Fisher check and retained discrepancies. | independent-verification.md/json |

## Reference use

The 16 supplied screenshots were inspected as aesthetic references. Their clinical/academic outlines, pale backgrounds and limited accents inform original compositions. No screenshot, tracing, patient identity, adult infusion scene or copyrighted reference asset is inserted. All new illustrations use editable SVG and historical study context; oral administration is labeled explicitly.

The final report records implementation status, changed files, testing and remaining source limitations.
