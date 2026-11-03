# Quantization refinement — November 03, 2026

This report describes the current page after the targeted layout and introductory-animation refinement. All 36 numbered sections, the 76 heading anchors, mathematical derivations, numerical examples, references, navigation hierarchy and substantive scientific experiments are retained.

## Layout corrections

The repeated right-side gaps came from a shared `max-width:79ch` rule on direct paragraphs, lists and callouts. The subtitle separately had `max-width:40ch`, and the title contained a hard-coded `<br>`. These source rules were corrected; no section-by-section width patches were added.

Ordinary prose and reference lists now share a responsive 110ch maximum, using the available central column at the requested viewport sizes. Compact equations use intrinsic width and automatic inline margins. Explanations, observations and optional derivations use a separate intrinsic panel rule with a responsive reading-measure limit. Long inline mathematics retains internal scrolling instead of creating page overflow.

The exact title and subtitle fit one line at 1280, 1440 and 1600 px, with the preserved five-row metadata on the right. Tablet and phone headers wrap naturally. The specified “How to read this page” box and its contents were removed completely. Peach is softened to #FFF6EE and interactive blue to #F0F6FD; existing semantic green, ivory and lavender treatments remain.

The componentwise vector display is a semantic table with aligned columns and unchanged calculations, signs and error colors. Tensor/channel/group displays use shared two-channel presentation grids, collapsing on narrow screens. Compact demonstration panels have deliberate maximum widths; full scientific plots retain their available width. The common per-tensor scale remains prominent.

## Animation status

The original university animation was subsequently rebuilt using the five supplied references, then resized without changing its content or timing. The current runtime uses intro-story.js and university-ensemble.svg. See [ANIMATION-VALIDATION.md](ANIMATION-VALIDATION.md) for the current storyboard, resize measurements, playback checks and asset-cleanup record.

## Files

Retained scientific-layout captures: validation-desktop.jpg, validation-distortion.jpg and validation-panels.jpg. Current introductory-animation captures are linked from ANIMATION-VALIDATION.md. Superseded animation previews were removed after checking references.

Updated: the primary HTML, quantization.css, presentation markup in quantization.js, research.js, verify-math.cjs, this report and review screenshots.

New runtime assets: intro-story.js and university-ensemble.svg. No new library, framework, video service or font was introduced. The numerical core quantization-math.js is unchanged by this refinement.

## Verification

- Numerical verification: 6,716 checks pass, including the new classroom and affine-story examples, plus the existing threshold, saturation, signed-tie, 1–16-bit grid, zero/constant/one-sided calibration, uniform MSE and Lloyd–Max checks.
- JavaScript syntax checks pass for all changed scripts.
- HTML/SVG audit: 225 unique IDs, 76 heading targets, five local script/style dependencies, one document/body closing pair and six valid reusable character fragments.
- Responsive measurements at 1600, 1440, 1280, 1024, 768 and 390 px show no page-level horizontal overflow and centered compact equations. At desktop widths the title/subtitle each occupy one line, metadata remains to their right, and opening prose fills the central content width.
- Browser inspection covers the requested early sections, formal definitions, rate–distortion discussion, vector and per-tensor panels, references, and all seven illustrated scenes. Detailed mathematical displays retain intentional internal scrolling on phones.
- Playback, pause/restart, chapter seeking, keyboard scrubbing, speed changes, caption synchronization and stable caption heights were exercised. Full 60-second playback at 1× was checked.
- Existing scientific controls and reset paths were checked after the presentation refactor; console and MathJax errors were inspected.

Reproduce numerical checks from the mathematics directory:

```powershell
node 2026-08-21-quantization/verify-math.cjs
node --check 2026-08-21-quantization/intro-story.js
node --check 2026-08-21-quantization/research.js
node --check 2026-08-21-quantization/quantization.js
git diff --check -- 2026-08-21-quantization.html 2026-08-21-quantization
```

## Limits and repository safety

The supplied attachment contained only the written brief, without the referenced screenshots. Refinements were checked against the actual rendered page. Browser testing used the available Chromium-based in-app browser; other browser engines and physical touch/screen-reader devices were not independently tested. OS reduced-motion emulation was not exercised; its pause/static-transcript behavior was inspected in code.

Existing Google Fonts and MathJax CDN dependencies remain. All new animation assets and computations are local, but an uncached offline load still depends on the existing CDN for TeX typesetting. SVG assets should be viewed through the site's normal HTTP hosting or a local static server.

These layout checks predate the final animation resize and publication commit. No page move, research.html edit, or unrelated research-file modification was performed. The pre-existing research.html modification and unrelated untracked projects were preserved.
