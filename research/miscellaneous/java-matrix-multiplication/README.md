# Inside Matrix Multiplication — a visual investigation with Java

The standalone page is `../java-matrix-multiplication.html`.
The four acts follow arithmetic → Java representation → geometry → scaling,
locality, sparse structure, parallel tasks and Strassen → floating-point and
algebraic complexity. The visual family follows the site's fMRI research page.

## Build and preview

From the repository root:

```bash
./research/java-matrix-multiplication/build-media.sh
python3 -m http.server 8001 --bind 127.0.0.1
```

Open http://127.0.0.1:8001/research/java-matrix-multiplication.html.
Use HTTP: browsers restrict JSON fetches on file URLs.

The build uses only the Java standard library and ffmpeg. A JDK 17+ is
recommended; the sources also compile with the installed javac 8 and execute
on the installed Java 21 runtime. It compiles, runs correctness checks,
generates five JSON files, draws two PNGs plus 600 Java2D frames (1280×720),
and encodes 30 seconds of H.264/yuv420p MP4. Temporary frames are removed
after successful encoding; ignored class files remain under `build/`.
No npm, site build, framework, server application, or runtime Java installation
is needed by visitors. Google Fonts and MathJax are optional external page
resources; local styles retain fallback fonts and the mathematical source.

## What is computed, and what is illustrated

- `src/MatrixAlgorithms.java` preserves the classical, blocked, recursive
  Strassen, CSR-times-dense, and ForkJoinPool kernels. Optional observers on
  the actual classical and blocked kernels emit their intermediate states.
  Tile events follow the implementation's **ii → kk → jj** order.
- `src/LearningExamples.java` is a complete executable class with the
  page's `double[][]` example, validated multiplication method, and
  order-sensitive double summation.
- `src/TraceWriter.java` generates:
  - `row-column.json`: 12 observed products and partial sums.
  - `blocked.json`: 8 observed tile updates and complete partial C snapshots.
  - `strassen.json`: seven products evaluated by the same helper the kernel
    uses, plus an actual comparison with classical multiplication.
  - `parallel.json`: four distinct tiles computed by ForkJoinPool and
    verified against classical multiplication. Lane assignment is deliberately
    illustrative, **not recorded thread scheduling**.
  - `learning.json`: Java-computed geometry/composition endpoints,
    power-of-two operation counts, CSR arrays, and double precision results.
- `js/page.js` animates those values using DOM and Canvas. It performs no
  large matrix multiplication. Interpolated geometry, sampled work pulses,
  memory movement, recursion diagrams, and the recap are explanatory layouts.
  They are not CPU instruction traces, cache measurements, or benchmarks.
- `src/MatrixAnimationGenerator.java` draws the warm poster, recap, and
  film. The final film result is computed from the actual worked A and B.
- `css/page.css` supplies the cream/peach/yellow theme and narrow layouts.
  Motion never autoplays; every scene has explicit controls. Reduced-motion
  users get instantaneous state changes instead of spatial transitions.

The small integer examples avoid overflow. The integer implementation itself
does not offer arbitrary precision or overflow checks. Strassen accepts equal
square power-of-two matrices. These are teaching kernels, not production BLAS.

## Verification

`MatrixTests` checks the known worked answer, fixed-seed equality across
algorithms, Strassen sizes 1/2/4/8/16, rectangular inputs with tile sizes
1/2/4/9, sparse zero rows, invalid inputs, callback order/count, and actual Java
double summation-order differences. Seed: `20260922L`.

To compile and check without rendering media, from the repository root:

```bash
mkdir -p research/java-matrix-multiplication/build/classes
javac -encoding UTF-8 -d research/java-matrix-multiplication/build/classes research/java-matrix-multiplication/src/*.java
java -cp research/java-matrix-multiplication/build/classes MatrixTests
java -cp research/java-matrix-multiplication/build/classes LearningExamples
java -cp research/java-matrix-multiplication/build/classes TraceWriter research/java-matrix-multiplication/traces
```

`verify_page.py` is an optional development check using externally installed
Python Playwright and Chromium, not a website dependency:

```bash
python3 research/java-matrix-multiplication/verify_page.py http://127.0.0.1:8001
```

It checks controls and final values, AB versus BA, formulas, local links and
assets, video metadata, browser errors, overflow at 1440/768/390/360 widths,
and reduced motion. Screenshots go to `/tmp/matrix-redesign-review/`.
Inspect the screenshots as well as the automated assertions.

The dated August 2026 exponent statement links to the specific arXiv v1
record. It is an asymptotic upper bound, not a measured property of these
Java kernels.
