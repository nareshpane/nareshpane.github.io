# October 20, 2026 revision

This is an in-place revision of the existing publication. The original PDF and calculation engine are unchanged. No screenshots were attached to this revision turn; the existing page and detailed written requirements supplied the visual reference.

## Preserved numerical foundation

- `js/model.js` SHA-256 before and after: `4AD3930F3EEB7BC11177BD47C97731935F0718EAEFFACC3A64C7D7886DDA91C0`.
- Original PDF SHA-256: `A457422B67B5792E57854BE310084DAA53D0D183D325B48B72E671A3B0439A45`.
- Baseline three-year reference CAD 9,757,551.096; new scenario CAD 9,885,787.744285714; incremental CAD 128,236.64828571305.
- Published aggregate display: **9,758 / 9,886 / 128 CAD $000**. CSV retains full dollar-level precision.
- Headcounts are displayed as integers; patient-years remain separately labelled fractional exposure. Independently rounded rows need not sum exactly.
- New prevalence arithmetic and Special Authority checklists are independent teaching examples, not new estimated BC administrative observations or changes to the BIA cohort. Applying either existing cohort estimate to the BIA still requires its explicit “Use” button.
- Payer-mode and regular/enhanced assistance examples pass explicit parameters to the existing `payer()` function. They never silently change the aggregate BIA public cost weight.

## Visual and explanatory changes

- Exact requested title, matching browser/Open Graph/Twitter titles, longer modest green subtitle, four compact metadata rows. Existing verified GPT-6 (Codex) description retained; reasoning unavailable.
- Twelve-scene, 94-second video. Scene selection precedes the canvas; playback controls precede a separate explanation panel. The synchronized clock controls map zoom, moving records, SQL typing/highlight, patient movement, progress, captions and scene selection. Reduced-motion playback uses static per-scene artwork.
- Original editorial humans, care facilities, pharmacies, medication packages, relational tables, SQL workstations and a multi-person policy briefing.
- Canada’s 13 province/territory geometries extracted without distortion from the existing Natural Earth v5.1.2 reference asset. Public-domain attribution retained; no download dependency.
- Clinical decision tree, non-graphic severity schematics, six-stage regulatory/reimbursement storyboard, matched-age prevalence illustration, Special Authority teaching checklist, dose-cost equation, Fair PharmaCare spending phases, payer-plan examples and eight illustrated limitation cards.
- Aggregate spending, national comparison, waterfall, tornado, Monte Carlo outputs and video financial values use CAD $000 with zero decimals. Unit acquisition prices and individual patient costs remain CAD dollars.

## Official checks, October 20, 2026

- **HLTH 5485** downloaded form: version July 24, 2025. Initial: 6 months; switch: separate 6-month request; renewal: 1 year. Initial adult/Hurley/antibiotic/lesion requirements and continued response requirements confirmed on pp. 1–2. The teaching checklist does not make an official determination.
- **Health Canada** NOC page and trial-sponsor guidance: distinguishes regulated trial authorization, manufacturer-sponsored research and marketing authorization based on safety, efficacy and quality.
- **CDA-AMC** reimbursement-review index returned HTTP 403. Official patient-input guidance p. 9 and official review procedures describe advice, subsequent negotiations and public-plan decision-making. A working alternate guidance link appears beside the index citation.
- **pCPA** official Brand Process Guidelines support negotiation and confidentiality of sensitive pricing information. No current net price is asserted.
- **BC** Special Authority policy, drug-review decision list, regular Fair PharmaCare assistance levels, Plan C and Plan W documentation checked. Plan W’s correct resident-page URL is `.../who-we-cover/first-nations-health-authority-clients`; it is funded by FNHA. Coverage applies to eligible people/products and pricing rules, not every medicine.
- BC plan overview endpoints were intermittently unavailable; individual official plan pages and official indexed text supported the qualified explanation. Current source URLs and dates are in `js/assumptions.js`.
- The original CDA-AMC review’s pp. 173–179 were re-read. Source-derived clinical and dosing material remains linked to printed PDF pages. Historical national results remain distinct from BC illustrative results and the retrospective pre-listing framing.

## Validation and honest limits

- Baseline tests ran before editing: 2,769 numerical checks and 22 original presentation checks passed.
- Final numerical checks, presentation/controller checks, revision checks, static audit, JavaScript parsing and HTTP asset checks have their own machine-readable reports in `data/`.
- The presentation test includes seeded Monte Carlo repeatability, CSV raw-number checks, plan examples, prevalence arithmetic, eligibility examples and simulated-clock playback across all 12 scenes for 94 seconds. It checks speed, pause/resume, scrub, replay, direct jumps and final-frame hold.
- All 12 SVG scenes were rasterized and inspected as a contact sheet; clinical decision-tree and dose-equation previews were also inspected. This caught and corrected overlaps and source-specific table examples. These are **vector-artwork previews, not browser screenshots**. Rasterization emitted font-cache warnings and used available fonts; actual browser typography is unverified.
- An actual Chrome/Playwright launch was attempted and rejected by the sandbox with `spawn EPERM`. Therefore desktop/mobile browser rendering, browser console, real keyboard/touch usability and actual download/playback are not claimed as verified. The updated browser script is supplied for execution outside this sandbox.

No staging, committing, pushing, index changes or unrelated repository edits were performed. Pre-existing index changes and unrelated untracked project folders were preserved.
