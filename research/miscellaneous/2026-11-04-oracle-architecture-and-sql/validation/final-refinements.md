# Final teaching, layout and research-index refinements

Completed November 4, 2026 (America/Los_Angeles). This record covers the two successive final refinement requests. Earlier implementation checks remain documented in `report.md`; the navigation and visual details below supersede earlier descriptions of visible progress/motion controls.

## Files and artwork

- Edited `../2026-11-04-oracle-architecture-and-sql.html` and the repository's `research.html`.
- Edited `css/refinements.css` and `js/monograph.js`; removed unused progress/checkbox rules from `css/monograph.css`, retaining system reduced-motion rules.
- Added 26 original assets in `svg/illustrations/`: six dimension scenes, five SQL processing scenes, five transaction scenes, six ETL scenes, three Oracle memory/process/file scenes, and `opening-ecosystem.svg`. There are now 50 illustrations and two self-contained overview SVGs.
- Added `validation/build_teaching_illustrations.py`; updated `validation/build_diagrams.py` and `svg/national-architecture.svg`. The workload overview remains byte-identical.
- Updated asset and illustration READMEs, the source register and validation evidence. Added `references/oracle-refinement-notes.md`, with official Oracle 19c references and the scope of the technical review.

## Header and spacing

At a 1440px viewport, the first title component remains **42.96px**. The italic component is **34.96px**, exactly **8px = 6pt smaller**. All 17 chapter/reference headings receive the same **6.6667px = 5pt reduction** from their respective existing responsive bases. Smaller card, simulation, table and navigation headings are unaffected. Before/after computed values are saved in `round3-typography-before.json` and `round3-typography-after.json`.

The exact subtitle spans the full header and occupies one line at 1440px, two lines at 768px and natural additional lines on phones. Metadata increases from 11.52px to **14.1867px**, equivalent to **+2pt**, in a 320px spring-green panel that contracts on mobile. All four rows remain single-line, including labels and values. SQL wrappers are inset **26px per side**, an increase of 10px; mobile retains 4px. Measurements across 31 rendered code wrappers showed less than 0.001px left/right asymmetry.

The opening chapter places its database/SQL ecosystem beside the existing heading, subtitle and paragraph on wide screens, and stacks it on narrower screens. Only selected architecture/star fills are adjusted; neighboring explanatory content is preserved.

## Teaching behavior

- Patient cards have six separate faint blue, peach, rose, lavender, ivory and cool-gray identities. Active cards retain their own hue, with a stronger border/inset outline and emphasized step number. Their pale-green dynamic explanation retains its requested outline. The instruction is directly above the cards.
- Direct selection, keyboard activation and playback share the existing state. Focus survives card rerendering; each player exposes exactly one pressed card and a synchronized step counter.
- The Oracle explorer distinguishes the client, shared SGA, associated server process/private PGA, background processes and persistent database files. Both the dropdown and diagram buttons select the same component. Expanded names and responsibilities include DBWn, LGWR, CKPT, SMON and PMON/CLMN/CLnn. The Windows thread qualification and existing detailed prose remain.
- Five illustrated SQL stages explicitly describe the expanded hard-parse path. Suitable cursor reuse can bypass optimization; plan reuse is not result reuse. The dynamic explanation is larger with a muted-red outline.
- Five illustrated transaction stages distinguish undo, redo, dirty buffers and ordinary synchronous redo acknowledgment. DBWn may write before or after commit; the final card is not a required post-commit flush. The illustrative commit mode and configured-storage qualification are explicit.
- ETL/ELT introductions distinguish the general concepts from the six-stage teaching pipeline. The source-to-target pictorial overview complements the larger interactive cards. Stage-specific panels explain lineage, validation, replay, mappings, publication and reconciliation.
- The isolated ETL receipt ledger reuses one fixture line. Valid first receipt: **1 = 1 new + 0 duplicate + 0 rejected**. Identical replay: **2 = 1 new + 1 duplicate + 0 rejected**, with one fact represented. Previously loaded scenario: first receipt is duplicate; replay yields two duplicates and one preexisting fact. Unknown-code scenario: two receipts yield two rejections and zero facts. Stage selection does not add receipts. A receipt audit trail is expandable, and ledger reset is separate from playback restart.
- Correction/version-aware processing is explained rather than implemented by this fixed-payload ledger. Source update and ingestion clocks are explicitly labeled not supplied in the fixture; no source timestamps were invented or changed.

## Final navigation and source flow

The visible reading-progress label, percentage, bar and Reduce Motion checkbox were removed, including the mobile percentage. Expand/collapse, sticky/scrolling sidebar, anchor navigation, active links and mobile Escape behavior remain. CSS still honors `prefers-reduced-motion`; JavaScript follows that operating-system query and stops active players when it changes. The OS preference itself was not toggled during QA.

The national diagram measures card geometry with `ResizeObserver` and redraws four smooth source branches, a common junction and one final arrow into Secure Transmission. On the two-column mobile layout, upper branches use the gutter between the lower cards. Desktop and mobile screenshots confirm no text crossings. The downloadable overview also has four branches and one final arrow.

## Research index

The preexisting Oracle entry was consolidated from Miscellaneous into the first research item, using the exact requested title, description and relative URL. CPI is second. The actual index previously had TAOCP ahead of CPI; TAOCP is now the first of the remaining entries, followed by their existing order. Existing non-Oracle titles, descriptions and links are preserved. The now-empty Miscellaneous wrapper was removed. There is one Oracle link.

The index mutation was prepared and diff-reviewed first, then applied only if its inspected SHA-256 still matched, preserving intervening user changes. No files were staged, committed, pushed or deployed.

## Validation performed

- Browser QA at **320×812, 390×844, 768×1024 and 1440×1000**: no page-level horizontal overflow; metadata has four rows and no overflowing label/value text; diagrams and code remain contained. See `final-responsive.json`.
- All **20 laboratories** initialize; all **138 options across 29 dropdowns** exercised with no console warnings/errors. These counts include five speed selectors and the architecture failure-path selector.
- All five sequence players tested for last/direct selection, Previous, Restart, Next, keyboard Enter, speed, Play/Pause and exactly one active card. Focus remained on the activated card. Patient playback reached step 6 of 6 and stopped.
- All four Oracle component selections checked; clicking the PGA diagram updates its explanation. All three ETL scenarios and explicit replays checked, including mobile use. Patient color/active states checked individually.
- Expand All opens all 12 groups; Collapse All closes all 12. Mobile selection closes the menu; Escape closes it. Active chapter highlighting remains functional after removing progress UI.
- SQL copy reported **Copied** after its write promise resolved. Exact operating-system clipboard contents were not independently verified.
- Static page checker, JavaScript syntax check, all 52 SVG XML parses and JSON parses pass. All local page references exist and HTTP checks pass.
- Independent verification of all **14 fixture tables and 12 exercise outputs** passes. Hash comparisons confirm **13 data/SQL files and every existing static SQL block are unchanged**. See `round3-preservation-results.json`. Synthetic identifiers and numeric references remain intact.
- The research index was inspected in-browser: Oracle first, CPI second, TAOCP third; the Oracle relative link opens the monograph correctly.

Representative proof images: `final-header-desktop.jpg`, `final-header-mobile.jpg`, `final-patient-journey.jpg`, `final-dimensions.jpg`, `final-architecture-desktop.jpg`, `final-architecture-mobile.jpg`, `final-oracle-instance.jpg`, `final-oracle-mobile.jpg`, `final-query-execution.jpg`, `final-etl.jpg`, `final-research-index.jpg`.

## Limits

No Oracle instance was available: SQL parsing/execution, real commit/recovery behavior and optimizer performance were not tested against Oracle. Browser simulations are local teaching models. Performance illustrations remain conceptual. Direct file-protocol browsing is blocked by the in-app browser; browser verification used the local HTTP server and relative GitHub Pages structure. No production integration or deployment was performed.
