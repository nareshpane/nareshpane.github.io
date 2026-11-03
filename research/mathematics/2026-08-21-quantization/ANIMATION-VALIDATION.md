# University introduction: illustration and playback refinement

Date: 2026-11-03. Scope: the introductory animation in `2026-08-21-quantization.html` and its dedicated assets. The final resize and supporting Quantization page files are included in the publication commit.

## Reference interpretation

All five supplied PNGs were inspected before implementation. They informed new SVG artwork rather than being pasted into the film as backgrounds.

| Reference | Adaptation |
| --- | --- |
| Lecture hall | Teaching board, window grid, foreground writing desk, professor explaining and students moving from curiosity to recognition. |
| Campus | Pediment, columns, central entrance, steps and softly rounded trees; students bring a numerical question into class. The establishing shot lasts six seconds. |
| Small-group discussion | Seated students and professor around an oval table, laptop, notebook and cup; gestures coincide with the coarse/fine comparison. |
| Presentation board | Freestanding board with a number line, original and reconstructed values, drawn error bracket and adjacent-level step bracket. The reference's orange/yellow was reinterpreted as muted peach and spring green. |
| Collaborative technical discussion | A shared laboratory display, schematic network, weight labels, code/reconstruction cards and a raw-bit budget comparison. |

The recurring cast has consistent hair, clothing, skin tones and facial proportions: a gray-haired professor with glasses, an ochre-clad student and a lavender-clad student. The new `university-ensemble.svg` supplies separate head and torso fragments. Articulated SVG arms, legs, brows, eyes and mouths in `intro-story.js` provide gestures, walking, thinking, recognition and restrained nodding. Furniture and architecture are original vector geometry in that script.

## Revised sequence

All seven previous scenes were rebuilt or replaced. The mathematical idea remains a roughly one-minute introduction.

| Story time | Scene and meaningful developments |
| --- | --- |
| 0–6 | Campus question, students approach, measurement `2.4` and proposed answer `2 ?` appear. |
| 6–15 | Six inputs appear promptly; mappings resolve about 0.8 seconds apart; the three-result summary and student recognition follow. |
| 15–24 | Input and reconstruction are located on the ruler; a point moves to 2; the red error bracket and equation appear; the unit step is drawn. |
| 24–33 | The same input is shown on two synchronized grids; the fine grid and its closer reconstruction appear; students compare absolute errors. |
| 33–42 | Four learned weights appear on a small schematic; the display changes to weight → 4-bit code → reconstruction, then signed error and decoding relation. |
| 42–52 | Equal-N 16-bit and 4-bit budgets assemble tile by tile; the quarter-size result appears; a separate explanation distinguishes raw weight codes from total runtime memory and inference speed. |
| 52–60 | Measurement and weight cards appear in sequence; the precision/error tradeoff and method/model/task qualification follow; the final prompt leads into the article's six-number example. |

New reveals and mathematical developments generally occur every 1–3 seconds. Ordinary scene crossfades last 450 ms, begin just before the next boundary and finish exactly at it. A paused chapter jump therefore displays its requested chapter immediately. Camera push is restrained to 0.9%; character actions and all mathematical reveals use the same story clock.

## Numerical checks

- Rounding: `(0.7, 1.2, 1.8, 2.4, 2.9, 3.3) → (1, 1, 2, 2, 3, 3)`; exact ties go toward the larger integer.
- Error convention: `e = x − Q(x)`; `2.4 → 2` gives `+0.4`.
- Study-group example: nearest-level steps of 1 and 0.25 give 2 and 2.5. Absolute errors are 0.4 and 0.1 **for this input**.
- Affine example: calibration range `[−1.24, 2.61]`, scale `3.85/15 = 0.256666…`, zero-point 5; `0.73 → q=8 → 1000 → 0.77`, signed error `−0.04`. The code and reconstruction cards use the existing mathematical core.
- For the same N weights, `4N/16N = ¼` of the raw code bits. Each tile represents N bits. No proportional runtime-memory or inference-speed guarantee is made.

## Functional and visual validation

- JavaScript syntax check passed.
- `node 2026-08-21-quantization/verify-animation.cjs`: **487 checks passed**. The actual controller and SVG-building code execute against a small DOM adapter in four configurations: desktop/mobile × ordinary/reduced motion. Checks cover arbitrary forward/backward scrubs, exact boundaries, identical geometry after revisiting a timestamp, maximum visible scene count, a complete 60-second frame-clock run, completion, all speed choices, pause, restart and resolution of every local character fragment.
- Existing `verify-math.cjs`: **6,716 numerical checks passed**.
- In-app Chromium preview: an uninterrupted 60-second sequence played at 1× and stopped with `1:00 / 1:00` and Play restored.
- The complete sequence also played to 60 seconds at the phone width at 1.5× and stopped correctly. Opening the static transcript exposed all seven parts, including the runtime-memory and speed caveat.
- Browser speed measurements were approximately 1.49 seconds of story time per wall-clock second at 1.5×, and 0.49 at 0.5×. The small difference reflects measurement and frame scheduling.
- Pausing left both the timestamp and serialized SVG geometry unchanged across subsequent observations.
- Dragging the single timeline slider moved from the campus to 34.7 seconds and immediately showed the laboratory caption. Keyboard seeking and repeated chapter jumps also worked; nine nonsequential chapter jumps each showed exactly one fully opaque scene at the requested boundary.
- Desktop viewport override: 1440 × 1000. Phone overrides: 390 × 900 and 320 × 850. The browser's content widths exclude its scrollbar. No horizontal page overflow was observed; captions fit at all inspected widths.
- Explanatory height stayed constant across sampled chapters: 109 px on desktop, 173 px at the larger phone width and 196 px at the narrowest tested width. Caption dimensions are measured from all captions after font loading and resize.
- Browser warning/error log was empty. HTML, story script and the new SVG asset returned HTTP 200; the SVG parsed successfully as XML.
- Label spacing was revised after visual QA: input/error labels, the step bracket, the 16N count and the final qualification no longer collide with other content or characters.

## Current review captures

- [Compact desktop presentation](validation-animation-desktop.jpg)
- [Responsive phone presentation](validation-animation-mobile.jpg)

Earlier full-size animation previews were superseded by these two current captures.

## Scope preservation and limits

Hashes taken before this refinement match afterward for the HTML outside the introductory section, the entire stylesheet, all other demonstration scripts, the shared mathematical core and `research.html`. Existing unrelated working-tree modifications and untracked files were preserved. No dependencies, remote image runtime, staging, commit, push or deployment were introduced.

Reduced motion disables crossfades, camera push, walking, swaying, nodding, blinking and smoothly interpolated reveals; mathematical states change discretely on the same timeline. A seven-part static transcript includes the full numerical and memory qualifications. The reduced-motion branch was exercised by the automated controller test; OS-level preference emulation was not available in the browser tool. Browser QA used Chromium viewport overrides, not physical phones, Firefox or Safari. This is a silent SVG film with synchronized text rather than narrated audio.
