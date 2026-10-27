# Directed k-tree page refinement — October 27, 2026

The existing page and its four research sections were retained. All changes are confined to the primary HTML and its associated asset directory. The root research.html and the original Verbatim images/PDF are unchanged.

## Presentation

- Header: thin sage border; balanced title lines; the requested full italic paper subtitle and author. The compact metadata uses the four supplied emojis, coloured labels, October 27, 2026, Codex on Windows, GPT-6.1 Sol and Medium.
- Introduction: four compact SVG cards show valid k=1,2,3 forests and an invalid directed cycle. Each valid card has five vertices and exactly 5-k selected arcs. The existing three comparison diagrams use smaller viewBoxes, node coordinates and panel dimensions, with peach borders.
- Notation: one default-collapsed semantic details/summary section contains the original and additional notation. The remaining columns are Source symbol, Meaning and intuition, and Role; first-appearance values were removed.
- Verbatim: all four panels identify Chen's paper, use 19px Garamond-style text at line-height 1.5, and are centred at a maximum width of 1000px. Visible image filenames, artificial separators and scan galleries were removed. Sentences and the long tree list flow across former image boundaries. The final cropped paragraph is completed from the supplied PDF, visually verified on printed p.555, with a disclosure outside its panel.

## Layout causes and corrections

The previous global 84ch paragraph cap left unused space inside wider components. Text now uses its containing reading column; figure captions use the card width. Grid/flex children can shrink, and intentional figure dimensions replace oversized general diagram boxes.

Unconstrained inline MathJax runs in the original theorem text caused mobile page overflow. Nine long formula runs now have local horizontal scrolling. Ordinary prose retains natural word wrapping.

Equation containers have natural height, sufficient padding, no fixed height and no vertical scrolling. Graph SVG styles remain scoped away from MathJax. Browser bounds checks found no vertically clipped formula SVGs, including the original 4x4 reduced Laplacian with determinant 12.

## New mathematical teaching components

1. Arrow-following demonstration reuses the three existing diagrams. A selected starting vertex is traced to a reference vertex or a repeated vertex. Previous/next, play/pause, restart and speed controls remain synchronized.
2. Three-vertex one-root/two-root comparisons precede the general definition.
3. All four outgoing-choice assignments of a symmetric triangle demonstrate repeated-edge rejection and three valid trees before abstract algebra.
4. A full 3x3 outgoing Laplacian becomes a 2x2 root-deleted minor. Four arithmetic steps evaluate its determinant, and a gallery independently enumerates its three trees.
5. A graph → full Laplacian → reduced minor explorer highlights the +1 diagonal and -1 off-diagonal contributions of an inspected arc. Add/remove/reset controls recompute both the exact determinant and direct enumeration. Arcs into and out of the reference vertex illustrate different effects of root deletion.
6. A small k=1,2,3 explorer produces exact counts 3,2,1; its final state shows three isolated roots and no selected arcs.
7. Existing generation animation stages now highlight concrete intermediate selections. Original Figure 1 labels around the central crossing are separated, and a closing synthesis connects roots, factors, cancellation and minors.

## Validation

- Existing Python audit: 643 exact graph/root-set cases, zero discrepancies.
- Existing Node audit: 463 exhaustive/source cases, all 18 printed source products and multiple-cycle cancellation, zero discrepancies.
- Added learning tests: 15 path/start combinations, all six triangle arc contributions, four primer examples, invalid inputs and root-set counts 3,2,1.
- Added transcription audit: all original transcription tokens retained in order in all four panels, apart from removing the editorial figure-placement annotation and repositioning its caption. The supplied excerpt files and raw transcription JSON remain unchanged.
- Static checks: balanced HTML, unique IDs and 26 article-local links/anchors; root research.html unchanged from HEAD. All 13 original source-image hashes match.
- Browser: widths 1920, 1440, 1366, 1024, 768 and 390 px; no page-wide horizontal overflow, no vertical equation scrollbars, no vertically clipped equation SVGs and no MathJax errors. The four-card row becomes two columns on tablets and one on mobile.
- Browser controls: notation keyboard expansion; path termination/cycle detection/playback; determinant steps and tree gallery; matrix edge selection/add/remove/reset; tiny-k counts; original animation endpoints; original laboratory presets, roots, completion, edge editing and reset. The original source counts remain 12 trees and 6 two-trees. No browser console errors or warnings were observed.

Machine-readable records are in data/refinement-browser-qa.json, data/learning-test-results.json and data/transcription-check-results.json. Preview images are figures/refined-overview.jpg and figures/matrix-lesson-preview.jpg.

## Remaining limits

MathJax still requires its CDN; graphs and controls require JavaScript. Long source formulas may scroll horizontally on small screens. Experiments remain bounded, exact, unit-weight checks and are not general proofs. The paper's operational algebra interpretation remains explicitly disclosed. Reduced-motion manual controls are implemented; browser media emulation was unavailable for a live preference-switch test.

## Files

Modified: primary HTML; css/style.css; js/app.js; validation/build_page.py, check_site.py and README.md; content-inventory and site-check results.

Added: js/learning.js; validation/learning_content.py, test_learning.js, test_transcription.py and this report; scoped result JSON and two preview JPGs.

Nothing was staged, committed or pushed. Unrelated pre-existing untracked projects were preserved.

## Relevant Git status

     M 2026-10-27-directed-k-trees-digraph-empirical-demonstration.html
     M 2026-10-27-directed-k-trees-digraph-empirical-demonstration/css/style.css
     M 2026-10-27-directed-k-trees-digraph-empirical-demonstration/data/content-inventory.json
     M 2026-10-27-directed-k-trees-digraph-empirical-demonstration/data/site-check-results.json
     M 2026-10-27-directed-k-trees-digraph-empirical-demonstration/js/app.js
     M 2026-10-27-directed-k-trees-digraph-empirical-demonstration/validation/README.md
     M 2026-10-27-directed-k-trees-digraph-empirical-demonstration/validation/build_page.py
     M 2026-10-27-directed-k-trees-digraph-empirical-demonstration/validation/check_site.py
    ?? 2026-10-27-directed-k-trees-digraph-empirical-demonstration/data/learning-test-results.json
    ?? 2026-10-27-directed-k-trees-digraph-empirical-demonstration/data/refinement-browser-qa.json
    ?? 2026-10-27-directed-k-trees-digraph-empirical-demonstration/data/transcription-check-results.json
    ?? 2026-10-27-directed-k-trees-digraph-empirical-demonstration/figures/matrix-lesson-preview.jpg
    ?? 2026-10-27-directed-k-trees-digraph-empirical-demonstration/figures/refined-overview.jpg
    ?? 2026-10-27-directed-k-trees-digraph-empirical-demonstration/js/learning.js
    ?? 2026-10-27-directed-k-trees-digraph-empirical-demonstration/validation/REFINEMENT_REPORT.md
    ?? 2026-10-27-directed-k-trees-digraph-empirical-demonstration/validation/learning_content.py
    ?? 2026-10-27-directed-k-trees-digraph-empirical-demonstration/validation/test_learning.js
    ?? 2026-10-27-directed-k-trees-digraph-empirical-demonstration/validation/test_transcription.py
