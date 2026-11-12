# The Algorithms Behind City Traffic Lights

Static teaching page: `../traffic-control-algorithms.html`.

## Preview

From the repository root:

```sh
python3 -m http.server 8000 --bind 127.0.0.1
```

Open http://127.0.0.1:8000/research/traffic-control-algorithms.html.
No build or installed runtime dependency is required.

## Files

- `page.css`: warm editorial layout, provenance card and responsive styles.
- `simulation.js`: pure deterministic models, shared with Node checks.
- `page.js`: native SVG/canvas rendering, controls and animation scheduling.
- `favicon.svg`: locally drawn signal icon.
- `checks.cjs`: model invariants, worked arithmetic and static resource checks.

Run the checks from the repository root:

```sh
node research/traffic-control-algorithms/checks.cjs
```

## Model boundaries

All simulations are synthetic. No outputs represent measured city performance.

The opening model uses right-hand traffic, four straight lanes, fixed separation,
constant moving speed, no new yellow entries and explicit clearances. Its 68-second
explanatory sequence includes an exclusive pedestrian interval and a schematic
network transition. Its queues count all vehicles not yet beyond the stop point,
including arrivals approaching that queue and vehicles outside the viewport.
Restart reinitializes every arrival clock and vehicle. The 1/60-second simulation
step is independent of display rate and playback speed.

The binary conflict matrix is a conservative compatibility model, not a geometric
certification. Dedicated turn lanes are assumed. Opposing through pairs, opposing
protected left pairs and same-origin through/left pairs are compatible; all other
distinct pairs are excluded. Right turns and pedestrians are outside the matrix.

The flow/capacity experiments use fluid queues and cycle-averaged service, omitting
within-cycle fluctuations in the capacity plot. The fixed timing plot retains the
red/green periods. Both omit start-up losses and are labeled accordingly.

The actuated comparison uses exactly the same prescribed integer arrivals for both
policies. Queue area is accumulated at one-second resolution; it is not an empirical
average trip delay. The MPC model enumerates 32 five-block switch/hold sequences
and executes the first block. Each 10-second block serves up to six vehicles if
held, or three after a five-second clearance on switching. Arrival forecasts are
periodic and known exactly. The objective sums end-of-block queue totals.

The city lab has nine nodes and 18 approach queues. It is a directed network with
southbound and eastbound traffic only, not a complete four-way microscopic model.
Each receiving approach can store 20 vehicles including reserved in-transit slots.
Transfers take four seconds and departures are no more than one vehicle per two
seconds of green. Boundary overflow is retained outside the network. Conservation:

```
generated = internal queues + in transit + external backlog + exited
```

All policies enforce 3 seconds yellow and 2 seconds all-red. Actuated and
pressure-based policies have minimum 8-second green, and switch at 28 seconds if
the other direction has demand. Actuation also switches when its queue is empty
and the arrival gap is at least three seconds. The pressure variant compares its
own queue minus the next approach's queued/reserved vehicles. It is a teaching
variant, not the original max-pressure formulation or its theoretical guarantee.

Metrics are current queue per approach; mean completed queue-visit waiting time;
stop episodes when joining a red/nonempty queue; and cumulative boundary exits.
Waiting vehicles are excluded from the completed-wait mean, so high congestion
must be interpreted using queue/backlog counts as well. Simulation speed changes
wall-clock playback only. Other lab controls restart the scenario.

The corridor uses positions in meters, nine-meter center spacing and stop gates.
Its time-space lines show desired unimpeded travel; the cars themselves obey red
signals. Bandwidth is the longest admissible departure interval sampled at 0.1 s
within the first 28-second green.

## Accessibility and performance

Controls are native buttons, sliders, inputs and selects. The conflict matrix
supports hover, keyboard focus and click. Signal labels supplement colors. All
significant animation uses one requestAnimationFrame scheduler and visible-region
checks via IntersectionObserver. Hidden tabs do not accumulate simulation time.
Reduced-motion preference starts every animation paused; time sliders and step
buttons are available. Pause-all provides an additional page-level control.

Equations use native HTML text/subscripts/superscripts instead of a remotely loaded
math renderer. Drawings are original programmatic vectors/canvas; there are no
stock images, videos, remote fonts, API calls, or runtime libraries. Sources are
linked in the article; source pages are not fetched while viewing it.
