# Context, animation and budget illustration revision — October 20, 2026

## Files changed

- Main `2026-10-19-budget-impact-analysis-medication.html`: exact new title, subtitle and badge; new broad-context section; broader case-study heading with original question as supporting text; historical notice moved immediately before references with its listing link retained; synchronized budget-view selectors; labelled live calculation audit; static timeline metadata updated.
- `css/revision.css`: balanced title, compact context hub with eight original inline SVG icons and responsive domain cards, case-study subtitle and context-note styling.
- `js/storyboard.js`: revised Canada evidence reports, clean BC composition, expanded administrative-system names, explicit ICD-9/ICD-10 context, existing-medication heading, shared illustrative forecast, new policy-options frame and consistent closing briefing.
- `js/video.js`: dynamic scene count and selected budget view passed to rendering. Existing play/pause/replay, speed, seek, scene navigation and reduced-motion behavior retained.
- New `js/budget-view.js`: single immutable source for the requested expenditure illustration, independent of calculation-engine inputs.
- `js/app.js`: shared budget-view state and synchronized selectors; annual/cumulative charts and summary cards use that view. Editing BIA assumptions selects live calculation automatically. Model audit, sensitivity and model CSV remain calculated outputs.
- New `figures/bc-centred.svg` and `scripts/recenter-bc-map.py`: recenter existing Natural Earth Lambert conic geometry from a 96°W to 126°W central meridian. This is a rigid rotation of about −27.0224°, with unchanged shape and aspect ratio. Supporting care-setting images now sit outside the map.
- `scripts/test-revision.mjs`, `scripts/test-presentation.mjs`, `scripts/test-browser.mjs`: updated meaningful regression checks for 13 scenes, new content, timing, illustration arithmetic and both budget views.
- `scripts/review-illustrations.mjs`, `scripts/illustration-contact-sheets.mjs`, `scripts/review-storyboard.mjs`: dynamic/new scene captures and contact sheets; new context, budget and historical-note captures.
- `README.md`, validation JSON reports under `data/`, and browser screenshots/contact sheets under `figures/qa/`.

## Budget values

All amounts below are CAD millions; the page consistently displays aggregate amounts as CAD $000.

| Year | Reference | New scenario | Increment |
|---|---:|---:|---:|
| 1 | 2.9 | 3.7 | 0.8 |
| 2 | 3.2 | 4.2 | 1.0 |
| 3 | 3.5 | 4.9 | 1.4 |
| Total | 9.6 | 12.8 | 3.2 |

The default policy illustration is intentionally labelled as an illustration. The forecast scene, closing briefing, later budget cards, annual bars and cumulative lines all read the same data. Switching either selector to “Live calculated scenario” synchronizes all of those displays. The illustration does not silently replace model assumptions, sensitivity results or exported model calculations. Existing default model impact remains CAD 128,236.64828571305; the engine is byte-identical.

## Animation timing and scenes

Retained the existing twelve-scene sequence and inserted “Three policy choices” between the forecast and closing meeting, producing thirteen scenes. The price-renegotiation option is highlighted as an illustrative choice, not a current recommendation or an automatic model conclusion.

Each existing chapter duration is half its former duration. The new frame occupies four seconds at 1×, giving 51 seconds total at 1× and 102 seconds at 0.5×. Thus 0.5× restores the former per-scene pacing, with the additional frame included. Dropdown multipliers remain 0.5×, 1×, 1.5× and 2×. Slider, scene navigation and progress labels use the revised real-time 1× timeline.

The user’s named scenes correspond to the retained timeline as follows: national evidence (1), BC population (2), encounter sources (4), coding (5), current alternatives (8), expenditure forecast (11), new policy choices (12), closing briefing (13). Existing pharmacy, linkage, treatment-option and stock/flow scenes remain in their original relative order.

## Source and reference handling

All eight supplied screenshots were accessible and inspected. They depict the existing publication and were used to locate and refine the requested elements. Existing original artwork is retained; no new raster illustrations or third-party screenshot assets were added.

The short cross-sector orientation distinguishes health-specific BIA terminology from related financial-impact questions in other sectors. It cites [ISPOR good-practice guidance](https://www.ispor.org/publications/journals/value-in-health/abstract/Volume-17--Issue-1/budget-impact-analysis-principles-of-good-practice-report-of-the-ispor-2012-budget-impact-analysis-good-practice-ii-task-force) and [HM Treasury’s Green Book](https://www.gov.uk/government/publications/the-green-book--2/the-green-book-2026), checked for this revision. Existing medical citations and the historical listing notice were preserved.

## Validation

- Numerical suite: 2,769 checks pass; original model and CDA-AMC PDF hashes unchanged.
- Revision suite: 111 checks pass, including annual and cumulative illustration arithmetic, timeline boundaries, all thirteen scenes, live-view propagation, heading and notice placement.
- Presentation suite: 46 checks pass, including both budget selectors, 0.5× timing, play/pause/resume/replay/scrubbing, scene jumps, complete timeline traversal and original CSV values.
- Actual Chrome regression: desktop 1440×1000, tablet 820×1180, mobile 390×844; budget-view consistency, headings, real playback pace, calculation controls, export, seeded Monte Carlo, keyboard focus and zero document overflow.
- Browser captures: all thirteen scenes, including both horizontal ends on mobile; new context hub, moved historical note and selected budget charts captured alongside existing affected panels. Scene contact sheet and full context/budget/notice images inspected for readability and overlaps.
- Local static path, DOM-ID and ES-module checks pass. All artwork and map references remain project-relative.

No files outside the project folder were modified. No staging, commit or push was performed. Touch hardware and a screen-reader session were not tested.
