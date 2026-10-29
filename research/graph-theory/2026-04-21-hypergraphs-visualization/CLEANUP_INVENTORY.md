# Verified cleanup inventory

Prepared before deleting any file. All paths are relative to the dedicated asset folder. Only these 21 historical raster previews are candidates. Source PDF, nine Verbatim screenshots, ten original gallery PNGs, runtime scripts, reproduction scripts, datasets, reports, and historical JSON audits are retained.

| Candidate | Current replacement | Why unnecessary |
|---|---|---|
| `preview-1440.jpg` | `qa/refinement-preview-1440.jpg` | Superseded generated preview; no runtime or reproduction input dependency. |
| `preview-390.jpg` | `qa/refinement-preview-390.jpg` | Superseded generated preview; no runtime or reproduction input dependency. |
| `preview-768.jpg` | `qa/refinement-preview-768.jpg` | Superseded generated preview; no runtime or reproduction input dependency. |
| `preview-builder.jpg` | `qa/refinement-preview-builder.jpg` | Superseded generated preview; no runtime or reproduction input dependency. |
| `preview-colouring.jpg` | `qa/refinement-preview-colouring.jpg` | Superseded generated preview; no runtime or reproduction input dependency. |
| `preview-sperner.jpg` | `qa/refinement-preview-sperner.jpg` | Superseded generated preview; no runtime or reproduction input dependency. |
| `refinement-counterexample.png` | `qa/refinement-counterexample.png` | Superseded generated preview; no runtime or reproduction input dependency. |
| `refinement-examples.png` | `qa/refinement-examples.png` | Superseded generated preview; no runtime or reproduction input dependency. |
| `refinement-incidence-1440.png` | `qa/refinement-incidence-1440.png` | Superseded generated preview; no runtime or reproduction input dependency. |
| `refinement-incidence-390.png` | `qa/refinement-incidence-390.png` | Superseded generated preview; no runtime or reproduction input dependency. |
| `refinement-incidence-768.png` | `qa/refinement-incidence-768.png` | Superseded generated preview; no runtime or reproduction input dependency. |
| `refinement-preview-1440.jpg` | `qa/refinement-preview-1440.jpg` | Superseded generated preview; no runtime or reproduction input dependency. |
| `refinement-preview-390.jpg` | `qa/refinement-preview-390.jpg` | Superseded generated preview; no runtime or reproduction input dependency. |
| `refinement-preview-768.jpg` | `qa/refinement-preview-768.jpg` | Superseded generated preview; no runtime or reproduction input dependency. |
| `refinement-preview-builder.jpg` | `qa/refinement-preview-builder.jpg` | Superseded generated preview; no runtime or reproduction input dependency. |
| `refinement-preview-colouring.jpg` | `qa/refinement-preview-colouring.jpg` | Superseded generated preview; no runtime or reproduction input dependency. |
| `refinement-preview-sperner.jpg` | `qa/refinement-preview-sperner.jpg` | Superseded generated preview; no runtime or reproduction input dependency. |
| `refinement-story-1440.png` | `qa/refinement-story-1440.png` | Superseded generated preview; no runtime or reproduction input dependency. |
| `refinement-story-390.png` | `qa/refinement-story-390.png` | Superseded generated preview; no runtime or reproduction input dependency. |
| `refinement-story-768.png` | `qa/refinement-story-768.png` | Superseded generated preview; no runtime or reproduction input dependency. |
| `refinement-story-desktop.png` | `qa/refinement-story-1440.png` | Superseded generated preview; no runtime or reproduction input dependency. |

Reference audit: the HTML, CSS, runtime scripts, generator, and validation scripts were searched. The old names occur as output filenames in tests now redirected to qa/, and in the historical Git-status listing in REFINEMENT_REPORT.md. None is consumed as an input. The historical listing is retained as a record of that earlier state. Source hashes and planned replacement paths are recorded in cleanup-inventory.json.

Removal is performed only after every replacement exists and validation passes. No directory is recursively removed. The legacy hypergraph_demo.html is absent, so no redirect deletion is necessary. research.html already links directly to the canonical page and is unchanged.

Completed: all 21 listed files were removed individually after hash and replacement checks. Removed 2,282,483 bytes. No source, runtime, or reproduction input was deleted.
