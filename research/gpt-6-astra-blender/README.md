# A procedural city as a spatial laboratory

The public page is `../gpt-6-astra-blender.html`. All authored/generated project
assets live in this directory. No website build step is required. The research
index entry is prepared locally; no commit or push was made in this session.

## Measured scene

Blender 4.5.13 LTS, seed 260905. There are 204 individually named building
objects referencing twelve shared architectural meshes, 64 road/path junctions,
108 undirected road and footway edges, three existing transit edges and one proposed edge.
Read `data/scene_validation.json` for actual object/mesh/triangle counts and
file sizes, and `data/analysis.json` for all calculated results. Counts include
hidden analytical geometry unless explicitly marked as visible-only.

`city_master.blend` contains the architectural model, analytical collections,
three cameras and studio lighting. The default view is the complete city.
The geometry is stylized and not a watertight fabrication model.

## Edit a building

Select a named `B###` object in collection `04 Buildings`. Position and scale
are independent for each building. The custom properties identify its
archetype, neighbourhood and nearest graph node. Topology is shared: make
the mesh single-user (Object > Relations > Make Single User > Object & Data)
before editing only one building's vertices. Otherwise linked siblings update.

Ordinary road markings, intersections and sidewalks are batched. Each road
edge has its own named object. Two crossing-specific batches allow complete
crossing removal. Seven tapered, branching tree meshes serve 162 instances. Six original clothing
families serve 48 pedestrians; twelve have four-phase walking shape keys and
sidewalk root motion. The web export substitutes simplified static poses and
lower-detail foliage without animation payloads. Lamps share one mesh.

## Reproduce

Run from the repository root, with Blender and FFmpeg/libx264 installed:

```bash
python3 -m venv /tmp/astra-city-env
/tmp/astra-city-env/bin/pip install -r research/gpt-6-astra-blender/requirements.txt
/tmp/astra-city-env/bin/python research/gpt-6-astra-blender/scripts/generate_data.py
/tmp/astra-city-env/bin/python research/gpt-6-astra-blender/scripts/generate_charts.py
/tmp/astra-city-env/bin/python research/gpt-6-astra-blender/scripts/run_blender_pipeline.py \
  --blender "$HOME/Documents/astra-blender-test/blender-4.5.13-linux-x64/blender"
/tmp/astra-city-env/bin/python research/gpt-6-astra-blender/scripts/package_assets.py
/tmp/astra-city-env/bin/python research/gpt-6-astra-blender/scripts/build_page.py
python3 -m http.server 8000 --bind 127.0.0.1
```

Open http://127.0.0.1:8000/research/gpt-6-astra-blender.html.
To run browser QA in another terminal:

```bash
/tmp/astra-city-env/bin/playwright install chromium
/tmp/astra-city-env/bin/python research/gpt-6-astra-blender/scripts/qa_site.py
/tmp/astra-city-env/bin/python research/gpt-6-astra-blender/scripts/package_assets.py
```

The last packaging pass refreshes the checksummed inventory after QA.
The page is static HTML with local CSS/JS, local MathJax 3.2.2, local
model-viewer 4.1.0, and the site's usual Google Fonts stylesheet. No npm,
React or build framework is required. Vendor license files are retained.
All geometry is original procedural output; no external 3D assets are used.

## Analytical modes

`render_views.py` opens the master afresh for each mode. From Blender's Python
console, add this project's `scripts/` directory to `sys.path`, then:

```python
from render_views import open_master, setup_mode
open_master()
setup_mode('betweenness')
```

Other modes: `hero`, `detail`, `overlay`, `graph`, `degree`, `route`, `closure`,
`access_before`, `access_after`, `resilience`. Restore the master before
switching between analytical modes: material changes are intentionally
performed on a fresh read, not accumulated across modes.

The animation .blend files contain the actual keyframes (frames 1–224 at
16 fps, 14 seconds, four perspective camera shots each). The shortest-path animation follows node sequences from analysis.json.
The bridge experiment restarts the example journey when the crossing is
removed; it does not reroute a traveler already mid-journey. Accessibility
material interpolation is a transition between two solves, not a series of
new economic equilibria. MP4 files use 960 × 540 H.264/yuv420p, BT.709 and fast-start metadata.
Run `render_animations.py -- --test` through Blender and inspect the samples
before `-- --render`. Lightweight stage labels are added during encoding.

## Cinematic hero reel

The separate [camera scene](blender/city_cinematic.blend) opens the existing
master and adds five perspective cameras and five animated aim targets. No
city mesh or building transform is changed. Cinematic-only presentation uses
Eevee, 16 samples, warm key/cool fill, one directional shadow source and an
extended studio floor. There is no expensive ray tracing, depth of field,
motion blur, or atmospheric volume. Current-world stills share this lighting.

The 24-second [silent MP4](video/astra_city_cinematic.mp4) contains 576 frames
at 24 fps, 1280 × 720, H.264 constrained baseline / yuv420p with fast-start
metadata. Five shots cover an aerial approach (0–5 s), Innovation District
sweep (5–10 s), low transit-street tracking (10–15 s), northern bridge and
river (15–20 s), and full-city pullback (20–24 s). Auto-clamped Bezier keys
ease camera position, aim and focal length. Four deliberate cuts separate
the shots, selected by camera timeline markers; the final pose matches the opening for
looping. No graph overlay was added to this architectural showcase.

Reproduce this addition independently, without rebuilding the city or rerunning
any simulation. Run the benchmark and inspect its temporary frames before the
full render; adjust samples if your hardware requires it:

```bash
blender --background --threads 6 --python-exit-code 1 \
  --python research/gpt-6-astra-blender/scripts/render_cinematic.py -- \
  --test --output /tmp/astra-cinematic-benchmark
blender --background --threads 6 --python-exit-code 1 \
  --python research/gpt-6-astra-blender/scripts/render_cinematic.py -- \
  --render --output /tmp/astra-cinematic-final
blender --background --python-exit-code 1 \
  --python research/gpt-6-astra-blender/scripts/validate_cinematic.py
python3 research/gpt-6-astra-blender/scripts/encode_cinematic.py \
  --frames /tmp/astra-cinematic-final
python3 research/gpt-6-astra-blender/scripts/package_assets.py --keep-images
```

Use the portable Blender executable shown above if `blender` is not on PATH.
[Rendering](scripts/render_cinematic.py), [encoding](scripts/encode_cinematic.py)
and [saved-scene validation](scripts/validate_cinematic.py) are separate scripts.
Raw PNG sequences stay in `/tmp`; only the editable scene, video and
[poster](renders/cinematic/poster.webp) are published. Affected still images were regenerated in their existing article positions;
no redundant gallery was added.

Measured results are in [benchmark](data/cinematic_benchmark.json),
[per-frame runtime](data/cinematic_runtime.json),
[scene validation](data/cinematic_scene_validation.json), and
[media validation](data/cinematic_media_validation.json). The browser suite
also checks native muted autoplay (without bypassing browser policy), a loop
across the end boundary, pause/resume, decoded poster, mobile viewport and
the original technical videos and interactive models. It releases the hero
decoder only in unrelated headless tests to avoid software-decoder contention.
Headless QA uses Chromium's software video decoder: this machine's VA-API
decoder disconnected when combined with SwiftShader. No autoplay-policy
bypass or page-side workaround is used.

## World revision and validation

The parcel locations, widths, depths and facade-height parameters are preserved.
Twelve architectural families add restrained variation in roofs, setbacks,
storefronts, cornices and balconies. Seven original tree archetypes use tapered
branching and asymmetric foliage clusters; no separate leaf objects exist.

The actual walking graph adds two park diagonals and eight outer junctions with
promenade/feeder links: 64 nodes, 108 edges and cycle rank 45 (previously 37).
There is no new walking river crossing. Both crossings together remain a cut.
All shortest paths, centrality, accessibility, Monte Carlo trials and figures
were recomputed. B171 now snaps to node 59 rather than node 40; that derived
assignment and its opportunity allocation were recomputed, not manually fixed.

`validate_world_revision.py` replays all seeded generation and all 1,200 trials
using Dijkstra distance matrices instead of the generator's Floyd–Warshall.
See `data/world_calculation_validation.json` and
`data/world_revision_comparison.json`. The latter includes a same-OD comparison
and clearly distinguishes non-paired Monte Carlo aggregates. Saved analytical
keyframes are checked by `validate_animations.py`; video bytes/frames by the
encoder and `package_assets.py`; browser behavior by the existing full QA suite.

The master now has 676 objects / 670 meshes, 676,260 instance-expanded triangles;
the web city has 257,840 triangles and is 1,366,104 bytes. This is more geometry
than the earlier primitive vegetation, but still a small download. Shared meshes
save storage, not all rendering work; see measured counts in the validation JSON.

## Model assumptions

- Local Cartesian metres, no real geographic CRS or elevation dataset.
- Road/footway costs: Euclidean edge length at 80 m/min, rounded to the nearest
  integer second before routing. Original grid edges are 48 m; park diagonals
  and irregular promenade links have different lengths. Distances sum lengths,
  not rounded times converted back to metres.
- Transit: 240 m/min plus 30 seconds on every edge, including successive
  segments. No route-continuation or waiting-state model.
- Accessibility: nonnegative synthetic opportunities, lambda = 0.18/min,
  self-destinations included, unreachable destinations contribute zero.
- Building accessibility inherits the nearest intersection; no within-block
  walking links. Node distance is retained in each building record.
- Monte Carlo: 400 trials at each independent road-edge deletion probability
  0.03, 0.08 and 0.15. Transit excluded. All trials retained in JSON.
- Disconnection means any graph component split. Pair-specific detours
  condition on connected endpoints; all-origin losses include disconnected
  cases. Wilson intervals describe finite Monte Carlo sampling uncertainty.
- Synthetic opportunities and fixed-cost network changes are illustrative,
  not estimates of causal transit policy effects or real economic welfare.

## Provenance and review

The full user-supplied brief is in `data/original_prompt.txt`, including its
requested OpenCode/GPT workflow note. The page attributes that note to the
brief and distinguishes it from the local execution record. Automated tests
and agent visual inspections were performed; author review remains pending.

The preliminary test's portable Blender executable was reused. Its six-building
scene, GLB and image were not copied into this project. The new city's larger
population uses fewer objects through deliberate shared meshes and batching.
