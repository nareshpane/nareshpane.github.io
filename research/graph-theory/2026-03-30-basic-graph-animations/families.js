/* Comparison plates use the same edge lists for drawing and counting. */
(() => {
  "use strict";

  function polygon(n, cx = 120, cy = 71, radius = 52, start = -Math.PI / 2) {
    return Array.from({ length: n }, (_, i) => {
      const angle = start + (2 * Math.PI * i) / n;
      return [cx + radius * Math.cos(angle), cy + radius * Math.sin(angle)];
    });
  }

  function path(n) {
    return Array.from({ length: n - 1 }, (_, i) => [i, i + 1]);
  }

  function cycle(n, offset = 0) {
    return Array.from({ length: n }, (_, i) => [offset + i, offset + ((i + 1) % n)]);
  }

  function complete(n) {
    const edges = [];
    for (let v = 1; v < n; v++) {
      for (let u = 0; u < v; u++) edges.push([u, v]);
    }
    return {points: polygon(n), edges};
  }

  function star(leaves) {
    return {
      points: [[120,71], ...polygon(leaves, 120, 71, 54, leaves % 2 ? -Math.PI / 2 : 0)],
      edges: Array.from({ length: leaves }, (_, i) => [0, i + 1])
    };
  }

  function completeBipartite(left, right) {
    const part = (size, x, top, bottom) => Array.from({length: size}, (_, i) =>
      [x, top + i * (bottom - top) / (size - 1)]);
    const edges = [];
    for (let u = 0; u < left; u++) {
      for (let j = 0; j < right; j++) edges.push([u, left + j]);
    }
    return {
      points: [...part(left, 56, 20, 118), ...part(right, 184, 28, 125)],
      edges, groups: [...Array(left).fill(0), ...Array(right).fill(1)],
      labels: [...Array.from({length: left}, (_, i) => "abc"[i]), ...Array.from({length: right}, (_, i) => "xyz"[i])],
      title: `Two columns of ${left} and ${right} vertices. Every vertex in the first column joins every vertex in the second.`
    };
  }

  function show(container, graph, heading, tone, metric, note) {
    const card = document.createElement("article");
    card.className = "plate tone-" + tone;
    const title = document.createElement("h4");
    title.textContent = heading;
    const figure = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    GraphAtlas.draw(figure, graph);
    const stats = document.createElement("p");
    stats.className = "metrics";
    stats.textContent = `n = ${graph.points.length} · m = ${graph.edges.length} · ${metric}`;
    const text = document.createElement("p");
    text.className = "plate-note";
    text.textContent = note;
    card.append(title, figure, stats, text);
    document.getElementById(container).append(card);
  }

  // Keep vertex 0 at the right of the original octagon and count counterclockwise.
  const originalRing = polygon(8,120,71,60,0).map(([x,y]) => [x,142-y]);
  const chain = Array.from({length:8}, (_,i) => [15+30*i,71]);
  const bowtie = {
    points:[[35,28],[35,114],[120,71],[205,28],[205,114]],
    edges:[[0,1],[1,2],[2,0],[2,3],[3,4],[4,2]], highlightVertices:[2]
  };
  const chainOrder = [3,4,0,5,1,6,2,7];
  const comparisons = {
    complement: {...complete(8), points:originalRing},
    wedge: {points:originalRing, edges:[[0,1],[1,2]], highlightEdges:[[0,1],[1,2]], highlightVertices:[1]},
    chain: {points:chain, edges:path(8), highlightVertices:[0,7]},
    bowtie,
    'double-star': {
      points:[[82,71],[158,71],[30,30],[20,71],[30,112],[210,30],[220,71],[210,112]],
      edges:[[0,1],[0,2],[0,3],[0,4],[1,5],[1,6],[1,7]], highlightVertices:[0,1]
    },
    spanning: {points:originalRing,edges:path(8)},
    bipath: {points:chain,edges:path(8), labels:chainOrder,groups:chainOrder.map(i=>Number(i>=4))},
    'broken-triangle': {points:originalRing,edges:path(8),highlightVertices:[0,7]},
    'bowtie-cut': bowtie
  };
  document.querySelectorAll('[data-companion]').forEach(card => {
    const graph = comparisons[card.dataset.companion];
    const {n,m,components} = GraphAtlas.stats(graph);
    GraphAtlas.draw(card.querySelector('svg'), {...graph,title:card.querySelector('h3').textContent});
    card.querySelector('.metrics').textContent = 'n = '+n+' · m = '+m+' · '+components.length+
      (components.length === 1 ? ' component' : ' components');
  });

  const completeNames = ["Triangle · K₃", "Four vertices · K₄", "Five vertices · K₅", "Six vertices · K₆"];
  const completeNotes = [
    "The last vertex closes the triangle by joining both earlier vertices.",
    "Three new edges connect the fourth vertex to a copy of K₃.",
    "The fifth vertex adds four edges, bringing the total to ten.",
    "The five edges at the sixth vertex account for the jump from ten to fifteen."
  ];
  const tones = ["teal", "peach", "blue", "lavender"];
  for (let n = 3; n <= 6; n++) {
    const graph = complete(n);
    graph.highlightVertices = [n - 1];
    graph.labels = graph.points.map((_, i) => i === n - 1 ? String(n) : "");
    graph.highlightEdges = graph.edges.filter(([, v]) => v === n - 1);
    graph.title = `Complete graph on ${n} vertices. The final vertex and its ${n - 1} incident edges are highlighted.`;
    show("complete-plates", graph, completeNames[n - 3], tones[n - 3],
      "deg = " + GraphAtlas.stats(graph).degrees[0], completeNotes[n - 3]);
  }

  const trees = [
    {
      heading: "Path · P₇", tone: "blue",
      points: [[18,85],[52,57],[86,85],[120,57],[154,85],[188,57],[222,85]],
      edges: path(7),
      note: "The endpoints are six edges apart, the largest possible distance in a seven-vertex tree."
    },
    {
      heading: "Star · K₁,₆", tone: "peach",
      ...star(6),
      note: "Any two leaves reach one another in two steps through the hub."
    },
    {
      heading: "Branching tree", tone: "teal",
      points: [[120,20],[64,69],[176,69],[32,119],[96,119],[144,119],[208,119]],
      edges: [[0,1],[0,2],[1,3],[1,4],[2,5],[2,6]],
      note: "From the root at the top, every leaf lies two edges away. Leaves on different branches are four apart."
    }
  ];
  for (const tree of trees) {
    const leaves = [];
    GraphAtlas.stats(tree).degrees.forEach((degree, i) => { if (degree === 1) leaves.push(i); });
    tree.highlightVertices = leaves;
    tree.title = tree.heading + " with all " + leaves.length + " degree-one leaves highlighted.";
    show("tree-plates", tree, tree.heading, tree.tone, leaves.length + " leaves", tree.note);
  }

  const odd = {
    points: polygon(5), edges: cycle(5), highlightEdges: [[4, 0]],
    title: "Pentagon. The highlighted closing edge joins the two ends of a four-edge path."
  };
  const even = {
    points: polygon(6, 120, 71, 54, 0), edges: cycle(6), groups: [0,1,0,1,0,1],
    title: "Hexagon. Alternate vertices form two independent sets."
  };
  const triangles = {
    points: [...polygon(3, 66, 71, 40), ...polygon(3, 174, 71, 40)],
    edges: [...cycle(3), ...cycle(3, 3)],
    title: "Two separate triangles. Every vertex has degree two."
  };
  function showRegular(graph, heading, tone, note) {
    show("cycle-plates", graph, heading, tone, GraphAtlas.stats(graph).degrees[0] + "-regular", note);
  }
  showRegular(odd, "Odd cycle · C₅", "peach",
    "Alternate two colours along the path. Its endpoints receive the same colour, so the closing edge forces a third.");
  showRegular(even, "Even cycle · C₆", "teal",
    "Alternate two colours and the sixth edge closes the cycle with different colours at its ends.");
  showRegular(triangles, "Two triangles · 2C₃", "lavender",
    "Six vertices of degree two also fit into two components. Their degree sequence matches the hexagon’s.");

  for (const leaves of [3, 5]) {
    const graph = star(leaves);
    graph.highlightVertices = [0];
    graph.title = `Star on ${leaves + 1} vertices with its degree-${leaves} hub highlighted.`;
    show("star-plates", graph, leaves === 3 ? "Three leaves · K₁,₃" : "Five leaves · K₁,₅",
      leaves === 3 ? "teal" : "peach", "hub degree " + GraphAtlas.stats(graph).degrees[0],
      leaves === 3 ? "Deleting the hub of this three-leaf star, also called a claw, leaves three isolated vertices."
        : "Deleting a single leaf leaves a smaller star. Deleting the hub separates all five leaves.");
  }

  const utility = completeBipartite(3, 3);
  for (const [graph, heading, tone, note] of [
    [completeBipartite(2, 3), "Unequal parts · K₂,₃", "teal",
      "Each left vertex has three neighbours. Each right vertex has two, so the degrees record the opposite part’s size."],
    [utility, "Equal parts · K₃,₃", "lavender",
      "Balancing the parts gives every vertex the same degree. A matching of three edges can cover all six vertices."]
  ]) {
    const left = graph.groups.filter(value => value === 0).length;
    show("bicomplete-plates", graph, heading, tone, `${left} × ${graph.points.length - left} edges`, note);
  }

  const square = {
    points: polygon(4, 120, 71, 56, Math.PI / 4), edges: cycle(4),
    groups: [0,1,0,1], labels: ["A","B","A","B"],
    title: "Square with alternating colour classes A, B, A, B. No pair of neighbours of a vertex is adjacent."
  };
  const pentagon = {
    points: polygon(5), edges: cycle(5), groups: [0,1,0,1,2], labels: ["A","B","A","B","C"],
    title: "Pentagon with a proper three-colouring. Its only cycle uses five edges."
  };
  show("trianglefree-plates", square, "Triangle-free · C₄", "teal", "2 colours",
    "The two vertices labelled A share both B neighbours. Adding an edge between the A vertices creates two triangles.");
  show("trianglefree-plates", pentagon, "Triangle-free · C₅", "peach", "3 colours",
    "Its only cycle has length five. This triangle-free graph still contains an odd cycle, which rules out a bipartition.");

  const prism = {
    points: [[120,13],[31,128],[209,128],[120,59],[90,102],[150,102]],
    edges: [...cycle(3), ...cycle(3,3), [0,3],[1,4],[2,5]],
    title: "Triangular prism graph with two triangular ends and three joining edges."
  };
  const pentaclique = complete(5);
  pentaclique.title = "Five vertices with every pair joined. Each vertex meets the other four.";
  for (const [graph, heading, tone, note] of [
    [even, "Degree two · C₆", "teal", "Following one of the two ways out of a vertex and continuing around the rim visits all six vertices."],
    [prism, "Degree three · Prism", "blue", "Each vertex has two neighbours on its triangular end and one across the prism. Adding these degrees counts each edge twice."],
    [pentaclique, "Degree four · K₅", "lavender", "Each vertex meets all four others. On five vertices, degree four forces the entire edge set."]
  ]) {
    const degrees = GraphAtlas.stats(graph).degrees;
    const sum = degrees.reduce((total, d) => total + d, 0);
    show("regular-plates", graph, heading, tone, "Σ deg = " + sum, note);
  }

  for (const rim of [4, 5]) {
    const graph = star(rim);
    graph.edges.push(...cycle(rim, 1));
    const rimClasses = Array.from({length: rim}, (_, i) => i % 2);
    if (rim % 2) rimClasses[rim - 1] = 2;
    const rimColours = new Set(rimClasses).size;
    graph.groups = [rimColours, ...rimClasses];
    graph.labels = graph.groups.map(colour => "ABCD"[colour]);
    graph.highlightVertices = [0];
    graph.title = `Wheel with a ${rim}-vertex rim. Letters A through ${graph.labels[0]} name distinct colours; the highlighted hub meets every rim vertex.`;
    show("wheel-plates", graph, rim === 4 ? "Even rim · W₅" : "Odd rim · W₆",
      rim === 4 ? "blue" : "peach", (rimColours + 1) + " colours",
      rim === 4 ? "The square rim alternates A and B. The hub touches both colours, so it needs colour C."
        : "The pentagonal rim needs A, B and C. The hub touches every rim vertex, forcing a fourth colour D.");
  }

  const crossed = complete(4);
  crossed.points = [[65,21],[175,21],[175,121],[65,121]];
  crossed.title = "Four labelled vertices on a rectangle with both diagonals present. The diagonals cross once.";
  const tetrahedron = {
    ...crossed, points: [[120,17],[32,119],[208,119],[120,82]],
    title: "The same four vertices and six edges, with vertex 3 inside the triangle formed by vertices 0, 1 and 2."
  };
  show("plane-plates", crossed, "Crossed drawing · K₄", "lavender", "1 crossing",
    "Edges 0–2 and 1–3 cross inside the square. Their intersection carries no vertex.");
  show("plane-plates", tetrahedron, "Plane drawing · K₄", "peach", "0 crossings",
    "Move vertex 3 inside triangle 0–1–2. All six pairs stay adjacent and the crossing disappears.");
  const cube = {
    points: [[56,15],[184,15],[184,127],[56,127],[89,44],[151,44],[151,98],[89,98]],
    edges: [...cycle(4), ...cycle(4,4), [0,4],[1,5],[2,6],[3,7]],
    title: "The cube graph drawn as nested rectangles with corresponding corners joined. All edges are disjoint except at shared endpoints."
  };
  show("plane-plates", cube, "Cube graph · Q₃", "blue",
    (2 - cube.points.length + cube.edges.length) + " faces",
    "Count the outer region as a face. The six regions give Euler’s identity n − m + f = 2 for this connected plane drawing.");
  show("obstruction-plates", utility, "Complete bipartite · K₃,₃", "lavender",
    "2n − 4 = " + (2 * utility.points.length - 4),
    "A simple bipartite planar graph on six vertices has at most eight edges. All nine cross-part edges make a crossing unavoidable.");
  show("obstruction-plates", pentaclique, "Complete · K₅", "blue",
    "3n − 6 = " + (3 * pentaclique.points.length - 6),
    "A simple planar graph on five vertices has at most nine edges. The tenth edge of K₅ rules out every plane drawing.");
})();
