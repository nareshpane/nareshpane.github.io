# Repository Guidance

## Site Model

- This is a dependency-free static GitHub Pages site. Root-level HTML files are standalone public pages; no build, test, lint, or deployment command is configured.
- Make the smallest coherent change and do not alter unrelated pages or research projects. If existing structure conflicts with these rules, explain the conflict before broad restructuring.
- Keep pages suitable for direct GitHub Pages hosting. Do not introduce React, Vue, Tailwind, Bootstrap, npm dependencies, or another web framework unless explicitly requested.
- AI-assisted, agentic, mathematical, computational, and graph-theory work normally belongs in `research/`.

## New Research Projects

- Create the primary page as `research/<topic-slug>.html`; place generated figures, videos, and supporting Python in `research/<topic-slug>/`. Do not place the primary page in that asset directory.
- Link assets from the page with relative paths such as `<topic-slug>/figure_01.png`; never use machine-specific absolute paths.
- Retain scripts that generate published figures or simulations and use descriptive filenames.
- Existing projects use older, inconsistent asset layouts. These conventions apply to new work; do not move, rename, or normalize existing projects unless explicitly asked.
- Do not overwrite `research/erdos-renyi-1960.html` unless explicitly asked. For a substantial new research page, inspect it first and adapt its cream/card visual language, typography, responsive treatment, MathJax/equation presentation, and footer navigation rather than copying topic-specific content.
- State definitions and methodology precisely where relevant, distinguish proven claims from simulation results, and give images descriptive alt text.
- Do not present unverified paper claims or citations as factual; ask for the source if it is not available locally.

## Publishing Research

- Treat a new research page as a draft unless explicitly told it is ready to publish; do not add draft pages to `research.html`.
- When publishing, add the new entry as the first `<li>` in `research.html`'s `ul.research-list`, using the existing `project-title` and `project-desc` markup and an `href` of `research/<topic-slug>.html`.
- Preserve all existing research entries below it, including their order and wording; do not restyle or correct them unless explicitly asked.
- Before finishing, verify every changed HTML asset path and, for a published page, its `research.html` link.

## Version Control

- Check `git status` before and after substantive work; preserve unrelated uncommitted changes.
- Review relevant diffs after changes.
- Do not commit, push, force-push, or rewrite history unless explicitly requested. When asked to commit, include only files relevant to the request.
