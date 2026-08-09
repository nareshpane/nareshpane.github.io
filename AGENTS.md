# Repository guidance

## Purpose and scope

- This repository publishes Naresh Pane's personal website through GitHub Pages.
- Do not modify existing, unrelated website or research files unless explicitly asked.
- AI-assisted, agentic, mathematical, computational, and graph-theory research normally belongs under `research/`.
- Make the smallest coherent set of changes necessary. Do not reorganize unrelated parts of the site.
- If the existing structure conflicts with these rules, explain the conflict before any broad restructuring.

## New research-topic layout

For a topic with slug `example-topic`, normally use:

```text
research/
├── example-topic.html
└── example-topic/
    ├── analysis.py
    ├── another_script.py
    ├── figure_01.png
    ├── figure_02.jpeg
    └── supporting files
```

- The primary page is `research/example-topic.html`, directly inside `research/`.
- Never put the primary HTML page in `research/example-topic/`; that directory is for supporting materials.
- Put project-generated or project-used Python, PNG, JPG/JPEG, animation/video, and other computational artifacts in the topic-specific subdirectory.
- Retain useful Python source used to generate figures or simulations.
- Use descriptive filenames; avoid generic names such as `image1.png` and `script1.py`.
- From `research/example-topic.html`, use relative asset paths such as `example-topic/figure_01.png` and `example-topic/simulation.py`. Never put machine-specific absolute paths (for example, `/home/naresh/Documents/...`) in website HTML.

## Existing research

- Older projects use differing conventions. Do not rename, move, reorganize, or normalize them merely to match this guidance.
- These rules govern new work unless migration of an older project is explicitly requested.
- Do not overwrite `research/erdos-renyi-1960.html` unless explicitly asked.

## Canonical research-page reference

Before creating a substantial research page, inspect `research/erdos-renyi-1960.html`. Adapt its visual and structural language to the new question; do not mechanically copy its content or topic-specific sections. Preserve where appropriate:

- cream/off-white background, centered card-style content, restrained blue accent, and rounded panels;
- Cormorant Garamond major headings, Inter body text, and Source Code Pro where appropriate;
- MathJax notation, equation containers, analysis boxes, styled tables, responsive media containers, responsive/mobile behavior, and footer navigation to Home and Research.

## Technology and research standards

- Prefer static HTML, CSS, and JavaScript; use Python for research, simulations, and figure generation when needed.
- Keep pages suitable for direct GitHub Pages hosting. Do not introduce React, Vue, Tailwind, Bootstrap, npm dependencies, or a web framework unless explicitly requested.
- Where appropriate, state definitions precisely; include notation, equations, algorithms, methodology, informative visualizations, and explanations of what figures show.
- Distinguish theoretical claims from empirical simulation results, and give images descriptive alt text.
- Do not invent mathematical results or citations. If factual claims from a paper or source are not locally available, request the needed source before treating them as verified.

## Git workflow

- Before substantive changes, inspect `git status` and do not overwrite unrelated uncommitted work.
- After changes, inspect `git status` and relevant diffs; summarize created, modified, and deleted files, and verify HTML asset paths are relative and correct.
- Do not automatically commit, push, force-push, or rewrite history.
- When explicitly asked to commit, include only files relevant to the requested task unless instructed otherwise.

# Publishing new research pages

- `research.html` is at the repository root and serves as the public index of research projects.
- Primary research pages normally live directly under `research/` as `research/<topic-slug>.html`.
- Supporting `.py`, `.png`, `.jpg`/`.jpeg`, `.mp4`, and related files normally live in `research/<topic-slug>/`.
- A newly created research HTML page should not automatically be listed in `research.html` while it is still a draft.
- When explicitly told that a research project is ready to publish, update the root-level `research.html` as part of the publishing workflow.
- Add the new project as the first `<li>` inside the existing `<ul class="research-list">`.
- Use the existing `project-title` and `project-desc` markup.
- Link to it with `research/<topic-slug>.html`.
- Give it a concise project title and description based on the completed page.
- Preserve every existing item below it and preserve their current order.
- Do not rewrite, reorder, restyle, or correct older entries unless explicitly asked.
- Verify the main page, supporting assets, relative links, and `research.html` link before considering the project ready to publish.
- Do not commit or push unless explicitly requested.
