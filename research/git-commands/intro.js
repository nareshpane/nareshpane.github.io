/* One persistent SVG world, one virtual clock. Every visible state is derived
   from t, including refs, content copies, typing, camera and facial expression.
   No timers, irreversible scene mutations, network requests or Git execution. */
'use strict';
(() => {
  const $ = id => document.getElementById(id);
  const stage = $('film-stage');
  if (!stage) return;
  const duration = 90;
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  const narrow = matchMedia('(max-width: 700px)');
  const clamp = x => Math.max(0, Math.min(1, x));
  const ease = x => {
    x = clamp(x);
    return x * x * (3 - 2 * x);
  };
  const progress = (t, at, span = .8) => reduced.matches ? Number(t >= at + span) : ease((t - at) / span);
  const interval = (t, a, b) => progress(t, a, .35) * (1 - progress(t, b - .35, .35));
  const mix = (a, b, p) => a + (b - a) * p;
  const esc = text => String(text).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
  const stamp = t => `${Math.floor(t/60)}:${String(Math.floor(t%60)).padStart(2,'0')}`;
  const chapters = [
    [0, '01', 'A folder', 'Files change. Nothing is recording history yet.', 'Watch the code lines change inside index.html. Saving a working file is not a commit; the three documents stay on their workbench.'],
    [6, '02', 'git init', 'A hidden layer, beneath the visible project.', 'git init creates repository infrastructure. HEAD names the unborn main branch. The .git listing is conceptual; the index appears when needed.'],
    [12, '03', 'Working tree', 'Inspect the files before choosing a snapshot.', 'Status detects the three untracked files. Before their first add, edited files are still untracked—not yet tracked modifications.'],
    [18, '04', 'Staging', 'Capture a content state. Keep the original file.', 'git add stages the file’s current content. A ghost copy travels to the index; index.html remains on the workbench. The other files are staged next.'],
    [25, '05', 'Inside the object database', 'Content receives an address; names live in trees.', 'Add has stored blobs and updated index entries. The camera reveals abbreviated illustrative object IDs. A tree and commit are still a preview until commit runs. Git supports SHA-1 and SHA-256 repository formats.'],
    [32, '06', 'First commits', 'A snapshot becomes a node with a parent link.', 'Commit writes the indexed tree and root commit A. A second edit, add and commit creates B → A. Main advances; HEAD keeps symbolically naming main.'],
    [42, '07', 'Branching', 'New pointer. New work. The same object store.', 'Experiment begins at B without copying files or history. HEAD switches to it; C is created. Switching back and committing D on main produces genuine divergence.'],
    [52, '08', 'Merging', 'A common base explains two different changes.', 'The inset shows an alternative fast-forward: only a ref advances. In our diverged graph, B is the base and C/D are tips. Resolve the CSS conflict; M then acquires two parent links.'],
    [64, '09', 'Remote repositories', 'Objects cross the network; the working files stay.', 'Origin is a separate repository. Push sends needed objects and requests its main ref update to M. Git and the complete local history existed before the hosting service appeared.'],
    [72, '10', 'Fetch and pull', 'Learning remote history is separate from integration.', 'A colleague clones, commits N and pushes. Fetch receives N and updates origin/main while main stays at M. Pull --ff-only fetches again, then advances main; afterward a private draft P is committed locally.'],
    [80, '11', 'Recovery', 'The name moved. The object can still be here.', 'A soft reset moves main from P back to N, preserving files and index. P fades from branch ancestry. Reflog reveals the old position; a rescue branch names P without overwriting working files.'],
    [86, '12', 'The complete system', 'Construct. Name. Compare. Navigate.', 'Git is not a pile of commands. It is a system for constructing, naming, comparing and moving through snapshots. Working files, the index, objects and references now have visible places.']
  ];
  // Command timings share the visual clock. Output is intentionally concise and
  // illustrative; letters and short IDs in the world are not real Git hashes.
  const commands = [
    [0, 'editor index.html', 'Edit the heading; save the file.', 'local'],
    [6, 'git init -b main', 'Initialized empty Git repository in git-demo/.git/', 'local'],
    [12, 'git status --short', '?? index.html\n?? styles.css\n?? README.md', 'local'],
    [18, 'git add index.html', '', 'local'],
    [23.5, 'git add styles.css README.md', '', 'local'],
    [25, 'git hash-object index.html', 'a41e7c…  (illustrative abbreviated object ID)', 'local'],
    [32, 'git commit -m "Create landing page"', '[main (root-commit) …] Create landing page\n3 files changed', 'local'],
    [37.3, 'git add index.html', '', 'local'],
    [39, 'git commit -m "Add navigation"', '[main …] Add navigation', 'local'],
    [42, 'git switch -c experiment', "Switched to a new branch 'experiment'", 'local'],
    [44.6, 'git commit -am "Try ivory background"', '[experiment …] Try ivory background', 'local'],
    [47.2, 'git switch main', "Switched to branch 'main'", 'local'],
    [48.6, 'git commit -am "Try cream background"', '[main …] Try cream background', 'local'],
    [54.1, 'git merge experiment', 'CONFLICT (content): Merge conflict in styles.css', 'local'],
    [58, 'git add styles.css', '', 'local'],
    [60, 'git merge --continue', '[main …] Merge branch experiment', 'local'],
    [64, 'git remote add origin <repository-url>', '', 'local'],
    [66, 'git push -u origin main', 'Writing objects: 100%\nmain -> main', 'local → origin'],
    [72, 'git clone <repository-url>', 'Cloning into git-demo…\nColleague later commits N and pushes.', 'colleague'],
    [74.2, 'git fetch origin', 'From origin\nM..N  main -> origin/main  (symbolic tips)', 'local'],
    [77, 'git pull --ff-only origin main', 'Fast-forward\nmain now names N', 'local'],
    [79, 'git commit -am "Private draft"', '[main …] Private draft', 'local'],
    [80, 'git reset --soft HEAD~1', '', 'local · mistaken ref movement'],
    [82, 'git reflog', 'N HEAD@{0}: reset: moving to HEAD~1\nP HEAD@{1}: commit: Private draft', 'local · symbolic tips'],
    [84, 'git branch rescue \'HEAD@{1}\'', '', 'local'],
    [86, 'git log --graph --decorate --all --oneline', 'P (rescue) · N (HEAD -> main, origin/main)\nC (experiment) · earlier ancestry retained', 'local · synthesis']
  ];
  stage.innerHTML = `<div class="cinema-orientation"><span id="cinema-chapter"></span><strong id="cinema-thesis"></strong><svg class="cinema-portrait" id="cinema-portrait" viewBox="-5 0 105 105" aria-hidden="true"><use href="#developer-body"/></svg></div><div class="cinema-viewport" id="cinema-viewport"></div><div class="cinema-state" id="cinema-state" aria-live="off"></div>`;
  stage.classList.add('cinema-stage');
  const terminal = document.createElement('div');
  terminal.className = 'cinema-terminal';
  terminal.innerHTML = `<div class="cinema-terminal-heading"><span id="cinema-machine">local</span><span>Illustrative session · no commands executed</span></div><pre><code><span class="cinema-prompt">$ </span><span id="cinema-command"></span><span class="cinema-cursor" id="cinema-cursor" aria-hidden="true">▌</span></code></pre><pre class="cinema-output"><code id="cinema-output"></code></pre>`;
  stage.after(terminal);
  $('film-controls').hidden = false;
  let elements = {},
    layout, nodePoints, elapsed = 0,
    anchorElapsed = 0;
  let anchorTime = 0,
    rate = 1,
    running = false,
    raf = null;
  let chapterIndex = -1;
  const text = (id, value) => {
    const el = elements[id] || $(id);
    if (el.textContent !== value) el.textContent = value;
  };
  const attr = (id, name, value) => elements[id].setAttribute(name, String(value));
  const opacity = (id, value) => {
    elements[id].style.opacity = clamp(value);
  };
  const move = (id, x, y, scale = 1) => attr(id, 'transform', `translate(${x.toFixed(2)} ${y.toFixed(2)}) scale(${scale.toFixed(4)})`);
  const svgText = (x, y, value, cls = 'world-label', extra = '') => `<text x="${x}" y="${y}" class="${cls}" ${extra}>${esc(value)}</text>`;

  function panel(id, x, y, w, h, title, tone, body = '') {
    return `<g id="${id}" transform="translate(${x} ${y})" class="world-panel ${tone}"><rect class="world-shadow" x="5" y="9" width="${w}" height="${h}" rx="20"/><rect class="world-surface" width="${w}" height="${h}" rx="20"/>${svgText(20,32,title,'world-zone')}${body}</g>`;
  }

  function documentCard(id, x, y, name, color) {
    return `<g id="${id}" transform="translate(${x} ${y})"><path class="document-shadow" d="M4 9H79L98 29V118H4Z"/><path class="document-paper" d="M0 0H73L92 20V110H0Z"/><path d="M73 0V20H92" fill="none" stroke="${color}" stroke-width="2"/><rect x="11" y="18" width="22" height="7" rx="3" fill="${color}"/><g id="${id}-lines" stroke="${color}" stroke-width="4" stroke-linecap="round"><path d="M12 40h58M12 52h42M12 64h54M12 76h30"/></g><g id="${id}-edit" fill="${color}"><rect x="11" y="44" width="68" height="6" rx="3"/><rect x="11" y="58" width="49" height="6" rx="3"/><rect x="11" y="72" width="61" height="6" rx="3"/></g>${svgText(2,134,name,'world-file-name')}<circle id="${id}-status" cx="79" cy="96" r="7" fill="${color}"/></g>`;
  }

  function buildWorld() {
    const mobile = narrow.matches;
    layout = mobile ? {
      w: 600,
      h: 1160,
      work: [30, 310, 245, 230],
      index: [315, 310, 245, 230],
      graph: [30, 565, 530, 280],
      db: [30, 905, 530, 180],
      remote: [275, 28, 285, 175],
      person: [40, 125],
      meta: [35, 215],
      port: [295, 1115]
    } : {
      w: 1280,
      h: 735,
      work: [150, 320, 280, 230],
      index: [465, 320, 245, 230],
      graph: [750, 230, 500, 280],
      db: [750, 545, 500, 160],
      remote: [920, 24, 330, 180],
      person: [35, 315],
      meta: [155, 590],
      port: [590, 690]
    };
    const L = layout,
      [wx, wy, ww, wh] = L.work,
      [ix, iy, iw, ih] = L.index,
      [gx, gy, gw, gh] = L.graph,
      [dx, dy, dw, dh] = L.db,
      [rx, ry, rw, rh] = L.remote;
    nodePoints = {
      A: [35, 135],
      B: [105, 135],
      C: [200, 98],
      D: [200, 170],
      M: [290, 135],
      N: [365, 135],
      P: [440, 135]
    };
    const nodeBorn = {
      A: 35.2,
      B: 40.5,
      C: 46,
      D: 50,
      M: 61,
      N: 76,
      P: 79.7
    };
    const edges = [
      ['B', 'A'],
      ['C', 'B'],
      ['D', 'B'],
      ['M', 'D'],
      ['M', 'C'],
      ['N', 'M'],
      ['P', 'N']
    ];
    const edgeMarkup = edges.map(([child, parent]) => {
      const a = nodePoints[child],
        b = nodePoints[parent],
        vx = b[0] - a[0],
        vy = b[1] - a[1],
        length = Math.hypot(vx, vy),
        ux = vx / length,
        uy = vy / length;
      return `<path id="edge-${child}-${parent}" class="commit-edge" pathLength="1" d="M${a[0]+ux*17} ${a[1]+uy*17}L${b[0]-ux*21} ${b[1]-uy*21}" marker-end="url(#world-parent)"/>`;
    }).join('');
    const nodes = Object.entries(nodePoints).map(([name, [x, y]]) => `<g id="commit-${name}" data-born="${nodeBorn[name]}"><circle id="halo-${name}" class="commit-halo" cx="${x}" cy="${y}" r="29"/><g id="node-${name}"><circle class="commit-node" cx="${x}" cy="${y}" r="17"/>${svgText(x,y+6,name,'commit-letter','text-anchor="middle"')}</g></g>`).join('');
    const pointer = (id, label, cls, width = 88) => `<g id="${id}" class="ref-pointer ${cls}"><path id="${id}-line" class="ref-line" marker-end="url(#world-ref)"/><rect width="${width}" height="28" rx="7"/>${svgText(width/2,20,label,'ref-label','text-anchor="middle"')}</g>`;
    const graphBody = `<g id="graph-grid"><path d="M20 100H480M20 200H480" class="world-gridline"/></g><g id="merge-base-ring"><circle cx="105" cy="135" r="32" class="merge-base-ring"/>${svgText(105,191,'base B','base-label','text-anchor="middle"')}</g>${edgeMarkup}${nodes}<path id="reflog-trail" d="M365 163Q430 220 440 140" class="reflog-trail" marker-end="url(#world-ref)"/>${pointer('main-ref','main','main-pointer')}${pointer('experiment-ref','experiment','experiment-pointer',132)}${pointer('tracking-ref','origin/main','tracking-pointer',135)}${pointer('rescue-ref','rescue','rescue-pointer',95)}<g id="head-badge"><path d="M0 13H-7" class="head-link" marker-end="url(#world-ref)"/><rect width="62" height="27" rx="13"/>${svgText(31,19,'HEAD','head-label','text-anchor="middle"')}</g>${svgText(20,264,'parent arrows: child → parent','world-micro')}<g id="reflog-label">${svgText(255,40,'reflog · previous tip P','reflog-label')}</g>`;
    const workingBody = `<path class="desk-front" d="M-8 152H${ww+8}V174H-8Z"/>${documentCard('file-html',18,55,'index.html','#bd730f')}${documentCard('file-css',ww-146,69,'styles.css','#9f6597')}${documentCard('file-readme',ww-107,85,'README.md','#427baa')}<g id="status-scan"><path d="M9 54V161H${ww-9}V54"/><circle id="status-spark" r="5"/></g>`;
    const indexBody = `<path class="index-tray" d="M15 81H${iw-15}L${iw-30} 161H30Z"/><g id="staged-sheet"><rect x="42" y="58" width="135" height="81" rx="7"/><path d="M56 79h99M56 94h73M56 109h84"/>${svgText(50,133,'content v1','index-version','id="index-version"')}</g><g id="staged-extra"><rect x="63" y="64" width="129" height="74" rx="7"/><rect x="77" y="74" width="129" height="74" rx="7"/></g>${svgText(20,ih-15,'proposed snapshot','world-micro')}`;
    const dbBody = `<g id="blobs"><g transform="translate(25 57)"><rect class="blob-tile" width="92" height="58" rx="10"/>${svgText(46,23,'BLOB','object-kind','text-anchor="middle"')}${svgText(46,44,'a41e7c…','object-id','text-anchor="middle"')}</g><g transform="translate(36 124)">${svgText(0,0,'+ reused content','world-micro')}</g></g><g id="tree-object"><rect class="tree-tile" x="${dw*.41}" y="57" width="100" height="58" rx="10"/>${svgText(dw*.41+50,80,'TREE','object-kind','text-anchor="middle"')}${svgText(dw*.41+50,101,'9bd133…','object-id','text-anchor="middle"')}</g><g id="commit-object"><rect class="commit-tile" x="${dw-119}" y="57" width="100" height="58" rx="10"/>${svgText(dw-69,80,'COMMIT','object-kind','text-anchor="middle"')}${svgText(dw-69,101,'→ root tree','object-id','text-anchor="middle"')}</g><g id="object-links"><path d="M${dw-124} 85H${dw*.41+107}" class="object-links" marker-end="url(#world-parent)"/><path d="M${dw*.41-7} 85H127" class="object-links" marker-end="url(#world-parent)"/></g><g id="object-preview">${svgText(dw*.41,143,'tree + commit: next construction','world-micro')}</g>`;
    const remoteBody = `<g id="remote-M"><circle cx="85" cy="84" r="16" class="remote-node"/>${svgText(85,90,'M','commit-letter','text-anchor="middle"')}</g><g id="remote-N"><path d="M${rw-95} 84H106" class="remote-edge" marker-end="url(#world-parent)"/><circle cx="${rw-76}" cy="84" r="16" class="remote-node"/>${svgText(rw-76,90,'N','commit-letter','text-anchor="middle"')}</g><g id="remote-pointer"><rect x="-44" y="0" width="88" height="27" rx="6"/>${svgText(0,19,'main','ref-label','text-anchor="middle"')}<path d="M0 0V-10" class="ref-line" marker-end="url(#world-ref)"/></g>${svgText(20,rh-14,'separate refs · separate objects','world-micro')}`;
    const person = `<g id="developer" transform="translate(${L.person.join(' ')})"><ellipse cx="45" cy="178" rx="48" ry="10" class="person-shadow"/><g id="developer-body"><path d="M10 172V116Q45 90 80 116V172" fill="#416c82"/><path d="M35 87V108Q45 119 57 108V87" fill="#d8aa89"/><g id="developer-head"><ellipse cx="46" cy="57" rx="29" ry="36" fill="#edc7a3"/><path d="M17 54Q9 12 45 13 84 12 76 54L64 32 18 48" fill="#334854"/><g id="developer-eyes"><circle cx="35" cy="60" r="2.5"/><circle cx="57" cy="60" r="2.5"/></g><path id="developer-brows" d="M27 48L40 48M51 48L64 48" class="person-face"/><path id="developer-mouth" class="person-face" d="M35 78Q46 80 59 77"/></g><path id="developer-arm" d="M15 126Q-5 152 17 157" fill="none" stroke="#416c82" stroke-width="13" stroke-linecap="round"/><rect x="16" y="130" width="72" height="45" rx="5" fill="#f5efe3" stroke="#62808c" stroke-width="2"/><path d="m43 151 8-6 8 6-8 6z" fill="#db8256"/></g></g>`;
    const fx = wx + 65,
      fy = wy + 105,
      tx = ix + 110,
      ty = iy + 93;
    const networkStart = [dx + dw - 45, dy + 60],
      networkEnd = [rx + rw - 42, ry + rh - 30];
    L.network = [networkStart, [L.w - 5, dy],
      [L.w - 4, ry + rh + 70], networkEnd
    ];
    L.copy = [
      [fx, fy],
      [fx + 100, fy - 80],
      [tx - 80, ty - 70],
      [tx, ty]
    ];
    L.blob = [
      [tx, ty],
      [tx + 90, ty + 160],
      [dx - 60, dy + 70],
      [dx + 71, dy + 80]
    ];
    L.commitFlight = [
      [dx + dw - 69, dy + 86],
      [dx + dw - 50, dy - 55],
      [gx + 35, gy + 190],
      [gx + 35, gy + 135]
    ];
    const pathD = points => `M${points[0]}C${points[1]} ${points[2]} ${points[3]}`;
    L.svgPath = pathD;
    const packets = Array.from({length: 7}, (_, i) => `<rect id="packet-${i}" x="-6" y="-6" width="12" height="12" rx="3"/>`).join('');
    $('cinema-viewport').innerHTML = `<svg class="git-world" id="git-world" viewBox="0 0 ${L.w} ${L.h}" role="img" aria-labelledby="world-title world-description"><title id="world-title">Git as one persistent environment</title><desc id="world-description">A developer, working files, the index, an object database, parent-linked commits and a separate remote. The terminal below shows the active command.</desc><defs><linearGradient id="world-wash" x2="1" y2="1"><stop stop-color="#fff0d3"/><stop offset=".45" stop-color="#e8f4f3"/><stop offset="1" stop-color="#ece6fa"/></linearGradient><radialGradient id="world-light"><stop stop-color="#d8c6ff" stop-opacity=".55"/><stop offset="1" stop-color="#d8c6ff" stop-opacity="0"/></radialGradient><pattern id="world-grid" width="35" height="35" patternUnits="userSpaceOnUse"><path d="M35 0H0V35" fill="none" stroke="#9eaaba" stroke-opacity=".13"/></pattern><marker id="world-parent" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0L10 5 0 10Z" fill="#6f55a8"/></marker><marker id="world-ref" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="4" markerHeight="4" orient="auto"><path d="M0 0L10 5 0 10Z" fill="#496274"/></marker></defs><rect width="${L.w}" height="${L.h}" fill="url(#world-wash)"/><g id="ambient"><ellipse cx="${L.w*.65}" cy="${L.h*.4}" rx="350" ry="260" fill="url(#world-light)"/></g><g id="world-camera"><rect x="-600" y="-600" width="2400" height="2400" fill="url(#world-grid)"/><ellipse cx="${wx+ww/2}" cy="${wy+wh+18}" rx="${ww*.67}" ry="25" class="ground-shadow"/><path id="network-channel" d="${pathD(L.network)}" class="network-channel"/><path id="capture-channel" d="${pathD(L.copy)}" class="capture-channel"/><path id="blob-channel" d="${pathD(L.blob)}" class="blob-channel"/>${panel('working-world',wx,wy,ww,wh,'WORKING TREE','working-tone',workingBody)}${panel('index-world',ix,iy,iw,ih,'INDEX','index-tone',indexBody)}${panel('graph-world',gx,gy,gw,gh,'LOCAL HISTORY','graph-tone',graphBody)}${panel('database-world',dx,dy,dw,dh,'OBJECT DATABASE','database-tone',dbBody)}<g id="metadata" transform="translate(${L.meta.join(' ')})"><rect width="${mobile?510:425}" height="93" rx="12"/>${svgText(17,29,'.git/   HEAD → main (unborn)','metadata-text')}${svgText(17,55,'config · objects/ · refs/','metadata-text')}${svgText(17,78,'index created when needed · conceptual layout','world-micro')}</g>${panel('remote-world',rx,ry,rw,rh,'GITHUB / REMOTE ORIGIN','remote-tone',remoteBody)}${person}<g id="colleague"><path d="M${rx-104} ${ry+48}h75v55h-75zM${rx-112} ${ry+112}h92" fill="#eef1f8" stroke="#60799c" stroke-width="4"/>${svgText(rx-65,ry+140,'clone','world-file-name','text-anchor="middle"')}</g><g id="capture-ghost"><rect x="-36" y="-25" width="72" height="50" rx="7"/><path d="M-24-10h45M-24 2h33M-24 14h40"/></g><g id="object-flight"><rect x="-17" y="-17" width="34" height="34" rx="8"/></g><g id="network-packets">${packets}</g><g id="command-pulse"><circle r="9"/><circle r="17" fill="none" stroke-width="2"/></g><g id="fast-forward" transform="translate(${mobile?75:290} ${mobile?535:190})"><rect width="340" height="150" rx="16"/>${svgText(18,30,'ALTERNATIVE · FAST-FORWARD','world-micro')}<path d="M275 72H68" class="commit-edge" marker-end="url(#world-parent)"/><circle cx="60" cy="72" r="16" class="commit-node"/><circle cx="280" cy="72" r="16" class="commit-node"/>${svgText(60,78,'B','commit-letter','text-anchor="middle"')}${svgText(280,78,'C','commit-letter','text-anchor="middle"')}<g id="ff-ref"><rect x="-40" y="0" width="80" height="27" rx="6"/>${svgText(0,20,'main','ref-label','text-anchor="middle"')}</g>${svgText(18,140,'only the reference moves','world-micro')}</g></g></svg><div class="cinema-conflict" id="cinema-conflict" hidden><div class="conflict-versions"><span>main · cream</span><span>experiment · ivory</span></div><pre><code>&lt;&lt;&lt;&lt;&lt;&lt;&lt; HEAD\nbackground: cream;\n=======\nbackground: ivory;\n&gt;&gt;&gt;&gt;&gt;&gt;&gt; experiment</code></pre><div id="conflict-resolution">Choose the intended content, then stage it.</div></div>`;
    elements = Object.fromEntries([...stage.querySelectorAll('[id]')].map(el => [el.id, el]));
    Object.assign(elements, Object.fromEntries([...terminal.querySelectorAll('[id]')].map(el => [el.id, el])));
    renderAtTime(elapsed);
  }

  function bezier(points, p) {
    const q = 1 - p;
    return [0, 1].map(k => q * q * q * points[0][k] + 3 * q * q * p * points[1][k] + 3 * q * p * p * points[2][k] + p * p * p * points[3][k]);
  }

  function follow(id, points, p) {
    const [x, y] = bezier(points, p);
    move(id, x, y);
  }

  function refPosition(name, point) {
    const [x, y] = point;
    if (name === 'experiment') return [Math.max(18, x - 62), y - 49];
    if (name === 'tracking') return [Math.min(350, x - 70), 235];
    if (name === 'rescue') return [Math.min(400, x - 47), y - 60];
    return [Math.max(18, Math.min(338, x - 43)), y + 40];
  }

  function drawRef(id, name, point, width, alpha) {
    const [x, y] = refPosition(name, point);
    move(id, x, y);
    opacity(id, alpha);
    const below = name === 'main' || name === 'tracking';
    const path = name === 'tracking' ?
      `M10 0Q-60 -70 ${point[0]-x-20} ${point[1]-y}` :
      `M${width/2} ${below?0:28}L${point[0]-x} ${point[1]-y+(below?20:-20)}`;
    attr(`${id}-line`, 'd', path);
    return [x + width + 8, y];
  }

  function movingTip(t, steps) {
    let previous = steps[0][1];
    for (let i = 1; i < steps.length; i++) {
      const [at, target] = steps[i];
      if (t < at) return nodePoints[previous];
      if (t < at + .85) {
        const p = progress(t, at, .85);
        return nodePoints[previous].map((v, k) => mix(v, nodePoints[target][k], p));
      }
      previous = target;
    }
    return nodePoints[previous];
  }

  function currentTip(t, steps) {
    let tip = steps[0][1];
    for (const [at, id] of steps)
      if (t >= at) tip = id;
    return tip;
  }
  const mainSteps = [
    [0, 'A'],
    [40.5, 'B'],
    [50, 'D'],
    [61, 'M'],
    [78.4, 'N'],
    [79.7, 'P'],
    [81.3, 'N']
  ];
  const experimentSteps = [
    [0, 'B'],
    [46, 'C']
  ];

  function camera(t) {
    const mobile = narrow.matches;
    const frames = mobile ? [
      [0, 280, 400, 1.15],
      [6, 285, 450, 1.1],
      [18, 300, 425, 1.1],
      [25, 295, 870, 1.25],
      [32, 300, 810, 1.1],
      [42, 300, 720, 1.2],
      [52, 300, 730, 1.15],
      [64, 310, 558, 1],
      [72, 310, 558, 1],
      [80, 300, 730, 1.17],
      [86, 300, 580, 1]
    ] : [
      [0, 325, 360, 1.45],
      [6, 445, 470, 1.3],
      [18, 445, 405, 1.3],
      [25, 883, 593, 1.55],
      [32, 895, 450, 1.3],
      [42, 906, 365, 1.38],
      [52, 805, 365, 1.22],
      [64, 905, 330, 1.08],
      [72, 905, 330, 1.08],
      [80, 921, 360, 1.38],
      [86, 640, 367.5, 1]
    ];
    let pose = frames[0].slice(1);
    for (let i = 1; i < frames.length; i++) {
      const [at, ...target] = frames[i];
      if (t < at) break;
      const p = progress(t, at, 1.6);
      pose = pose.map((v, k) => mix(v, target[k], p));
      if (t < at + 1.6) break;
    }
    if (reduced.matches) pose = mobile ? [300, 580, 1] : [640, 367.5, 1];
    const [cx, cy, s] = pose;
    move('world-camera', layout.w / 2 - cx * s, layout.h / 2 - cy * s, s);
  }

  function expression(t) {
    // Curves interpolate by the same time used for the command and graph, so
    // seeking cannot leave a worried face attached to a completed recovery.
    const worry = interval(t, 54.7, 59) + interval(t, 81.2, 84.7);
    const delight = interval(t, 35, 37) + interval(t, 61, 64) + progress(t, 84.7);
    const puzzlement = interval(t, 42.5, 45);
    const mouthY = 80 - 12 * worry + 9 * delight;
    attr('developer-mouth', 'd', `M35 78Q46 ${mouthY.toFixed(2)} 59 77`);
    attr('developer-brows', 'd', `M27 ${48-4*puzzlement}L40 ${48+4*worry}M51 ${48+4*worry}L64 ${48-3*puzzlement}`);
    const angle = reduced.matches ? 0 : -4 * worry + 3 * puzzlement;
    attr('developer-head', 'transform', `rotate(${angle} 46 80)`);
    const blink = reduced.matches ? 1 : 1 - .9 * interval(t % 5.7, 4.8, 5.05);
    attr('developer-eyes', 'transform', `translate(0 ${60*(1-blink)}) scale(1 ${blink})`);
    attr('developer-arm', 'd', `M15 126Q${-5-6*delight} ${152-15*delight} ${17+8*delight} ${157-18*delight}`);
    elements.developer.dataset.expression = worry > .5 ? 'concerned' : delight > .5 ? 'relieved' : puzzlement > .5 ? 'puzzled' : 'focused';
  }

  function renderTerminal(t) {
    const record = commands.findLast(c => t >= c[0]) || commands[0];
    const [at, command, output, machine] = record, local = t - at;
    const typingTime = Math.min(1.15, .25 + command.length / 55);
    const count = reduced.matches ? command.length : Math.floor(command.length * clamp(local / typingTime));
    text('cinema-command', command.slice(0, count));
    text('cinema-machine', machine);
    const reveal = reduced.matches ? 1 : progress(local, typingTime + .15, .4);
    const lines = output.split('\n');
    text('cinema-output', reveal > 0 ? lines.slice(0, reduced.matches ? lines.length : Math.min(lines.length, 1 + Math.floor(Math.max(0, local - typingTime - .15) / .35))).join('\n') : '');
    elements['cinema-output'].style.opacity = reveal;
    elements['cinema-output'].style.transform = `translateY(${reduced.matches?0:(1-reveal)*5}px)`;
    elements['cinema-cursor'].style.opacity = reduced.matches ? 0 : .35 + .65 * Math.abs(Math.cos(t * 3.1));
    terminal.dataset.command = command;
  }

  function renderAtTime(t) {
    if (!layout) return;
    const chapter = chapters.findLastIndex(c => t >= c[0]);
    if (chapter !== chapterIndex) {
      chapterIndex = chapter;
      const [, n, title, thesis, caption] = chapters[chapter];
      text('cinema-chapter', `${n}  ${title}`);
      text('cinema-thesis', thesis);
      $('film-caption').textContent = caption;
      text('world-description', caption);
    }
    camera(t);
    expression(t);
    // A small reaction cut-in keeps the same live SVG face readable when the
    // virtual camera has left the workbench. It disappears in the wide shots.
    const portraitAlpha = narrow.matches ? interval(t, 26.5, 64) + interval(t, 81.5, 86) : interval(t, 26.5, 86);
    opacity('cinema-portrait', portraitAlpha);
    opacity('developer', 1 - portraitAlpha);
    renderTerminal(t);
    const [wx, wy] = layout.work, [ix, iy] = layout.index, [gx, gy] = layout.graph, [dx, dy, dw] = layout.db, [rx, ry, rw] = layout.remote;
    const edit1 = progress(t, 1.4, 1.7),
      edit2 = progress(t, 36.5, .7);
    opacity('file-html-edit', edit1);
    opacity('file-html-lines', 1 - edit1 * .9);
    attr('file-html-edit', 'transform', `translate(0 ${-2*edit2}) scale(${1+edit2*.06} 1)`);
    opacity('file-css-edit', progress(t, 44, .7));
    opacity('file-readme-edit', progress(t, 79, .3));
    attr('file-html-status', 'fill', t < 21 ? '#bb760f' : t < 35.2 ? '#168f91' : t < 36.5 ? '#3e9566' : t < 38.2 ? '#bb760f' : t < 40.5 ? '#168f91' : '#3e9566');
    opacity('index-world', .18 + .82 * progress(t, 17.4));
    opacity('database-world', progress(t, 7.4));
    opacity('graph-world', progress(t, 8));
    opacity('metadata', interval(t, 6.8, 32));
    opacity('status-scan', interval(t, 12.8, 17.5));
    move('status-spark', 10 + (layout.work[2] - 20) * clamp((t - 13) / 3.5), 162);
    const staged = progress(t, 21.2);
    opacity('staged-sheet', staged);
    opacity('staged-extra', progress(t, 24.5) * .65);
    text('index-version', t >= 79.7 ? 'content v3' : t >= 38.2 ? 'content v2' : 'content v1');
    move('staged-sheet', 0, 10 * (1 - staged));
    opacity('capture-channel', interval(t, 19.5, 24.8) + interval(t, 37, 39));
    opacity('blob-channel', interval(t, 21, 29.5));
    const copyStart = t >= 37 ? 37.4 : 19.8,
      copyEnd = copyStart + 1.5;
    const copy = clamp((t - copyStart) / (copyEnd - copyStart));
    follow('capture-ghost', layout.copy, ease(copy));
    opacity('capture-ghost', reduced.matches ? 0 : interval(t, copyStart, copyEnd + .5));
    follow('object-flight', t < 32 ? layout.blob : layout.commitFlight, progress(t, t < 32 ? 21.1 : 34, 1.2));
    opacity('object-flight', reduced.matches ? 0 : interval(t, 21.1, 22.7) + interval(t, 34, 35.4));
    opacity('blobs', progress(t, 21.5));
    opacity('tree-object', .18 + .82 * progress(t, 34));
    opacity('commit-object', .18 + .82 * progress(t, 34.8));
    opacity('object-links', progress(t, 34.4));
    opacity('object-preview', 1 - progress(t, 34));
    const birth = {
      A: 35.2,
      B: 40.5,
      C: 46,
      D: 50,
      M: 61,
      N: 76,
      P: 79.7
    };
    for (const [name, at] of Object.entries(birth)) {
      let alpha = progress(t, at, .65);
      if (name === 'P') alpha *= 1 - .73 * interval(t, 81.3, 84.8);
      opacity(`commit-${name}`, alpha);
      const [x, y] = nodePoints[name], scale = reduced.matches ? 1 : .4 + .6 * progress(t, at, .65);
      attr(`node-${name}`, 'transform', `translate(${x*(1-scale)} ${y*(1-scale)}) scale(${scale})`);
      const glow = interval(t, at, at + 1.6) + (name === 'B' ? interval(t, 54, 60) : 0) + (name === 'P' ? interval(t, 83, 85.7) : 0);
      opacity(`halo-${name}`, glow * .65);
    }
    for (const [child, parent] of [
        ['B', 'A'],
        ['C', 'B'],
        ['D', 'B'],
        ['M', 'D'],
        ['M', 'C'],
        ['N', 'M'],
        ['P', 'N']
      ]) {
      const amount = progress(t, birth[child] + (child === 'M' && parent === 'C' ? .4 : .1), .65);
      attr(`edge-${child}-${parent}`, 'stroke-dasharray', '1');
      attr(`edge-${child}-${parent}`, 'stroke-dashoffset', 1 - amount);
      opacity(`edge-${child}-${parent}`, amount * (child === 'P' ? 1 - .73 * interval(t, 81.3, 84.8) : 1));
    }
    const mainPoint = movingTip(t, mainSteps),
      experimentPoint = movingTip(t, experimentSteps);
    const headMain = drawRef('main-ref', 'main', mainPoint, 88, progress(t, 35.2));
    const headExperiment = drawRef('experiment-ref', 'experiment', experimentPoint, 132, progress(t, 42.8));
    drawRef('tracking-ref', 'tracking', movingTip(t, [
      [0, 'M'],
      [76.2, 'N']
    ]), 135, progress(t, 70));
    drawRef('rescue-ref', 'rescue', nodePoints.P, 95, progress(t, 84.8));
    const onExperiment = progress(t, 43, .75) * (1 - progress(t, 47.8, .75));
    const headPoint = headMain.map((v, k) => mix(v, headExperiment[k], onExperiment));
    move('head-badge', ...headPoint);
    opacity('head-badge', progress(t, 35.2));
    opacity('merge-base-ring', interval(t, 54, 61.5));
    opacity('fast-forward', interval(t, 52, 54.2));
    move('ff-ref', mix(60, 280, progress(t, 52.45, 1)), 103);
    opacity('reflog-trail', interval(t, 82.3, 86));
    opacity('reflog-label', interval(t, 82.3, 86));
    const conflictAlpha = interval(t, 54.9, 60.5);
    elements['cinema-conflict'].hidden = conflictAlpha === 0;
    elements['cinema-conflict'].style.opacity = conflictAlpha;
    const resolved = progress(t, 58.1, .6);
    elements['cinema-conflict'].classList.toggle('resolved', resolved > .5);
    elements['cinema-conflict'].style.setProperty('--resolved', resolved);
    text('conflict-resolution', resolved > .5 ? 'Resolved: background: ivory; → git add styles.css' : 'Choose the intended content, then stage it.');
    opacity('remote-world', progress(t, 64));
    opacity('remote-M', progress(t, 70));
    opacity('remote-N', progress(t, 73.6));
    move('remote-pointer', mix(85, rw - 76, progress(t, 73.6, .8)), 111);
    // The remote ref is deliberately absent before the objects arrive.
    opacity('remote-pointer', progress(t, 70));
    opacity('colleague', interval(t, 72, 79));
    opacity('network-channel', progress(t, 64.4) * .8);
    const sending = interval(t, 66.8, 70),
      fetching = interval(t, 74.6, 76.5),
      handshake = interval(t, 77.2, 78);
    for (let i = 0; i < 7; i++) {
      const phase = sending > .01 ? (t - 66.8) / 2.5 - i * .095 : fetching > .01 ? 1 - ((t - 74.6) / 1.4 - i * .1) : 1 - ((t - 77.2) / .7 - i * .14);
      follow(`packet-${i}`, layout.network, clamp(phase));
      opacity(`packet-${i}`, !reduced.matches && phase > 0 && phase < 1 ? Math.max(sending, fetching, handshake) * Math.sin(phase * Math.PI) : 0);
      attr(`packet-${i}`, 'fill', sending > .01 ? '#db745b' : handshake > .01 ? '#9faeb8' : '#367da1');
    }
    // Execution pulse: terminal → source, then the content copy follows its own
    // semantic route. These are transient signals, not moving working files.
    const pulseAt = t < 18 ? 12.8 : t < 32 ? 19.2 : t < 64 ? 33.2 : 66.5;
    const pulseEnd = t < 18 ? [wx + 60, wy + 105] : t < 32 ? [wx + 60, wy + 105] : t < 64 ? [ix + 110, iy + 100] : [dx + dw - 65, dy + 85];
    const pp = progress(t, pulseAt, .7),
      start = layout.port;
    move('command-pulse', mix(start[0], pulseEnd[0], pp), mix(start[1], pulseEnd[1], pp));
    opacity('command-pulse', reduced.matches ? 0 : interval(t, pulseAt, pulseAt + 1));
    // Ambient light and a final guided highlight use this clock too: pause freezes
    // everything, including decorative movement. Reduced motion removes drift.
    move('ambient', reduced.matches ? 0 : Math.sin(t * .13) * 28, reduced.matches ? 0 : Math.cos(t * .1) * 20);
    const highlight = t >= 86 ? Math.min(3, Math.floor(t - 86)) : -1;
    ['working-world', 'index-world', 'database-world', 'remote-world'].forEach((id, i) => elements[id].classList.toggle('world-highlight', i === highlight));
    const mainName = currentTip(t, mainSteps),
      version = t >= 79.7 ? 'v3' : t >= 38.2 ? 'v2' : 'v1';
    const states = t < 6 ? 'Working files · no repository' : t < 18 ? 'Working files · untracked | HEAD → unborn main' : t < 35.2 ? `Files stay · index ${t>=21.2?'captures v1':'not yet staged'} | no commit` : `HEAD → ${onExperiment>.5?'experiment':'main'} → ${onExperiment>.5?(t>=46?'C':'B'):mainName}  ·  index ${version}${t>=70?`  ·  origin/main → ${t>=76.2?'N':'M'}`:''}${t>=84.8?'  ·  rescue → P':''}`;
    text('cinema-state', states);
    stage.dataset.elapsed = t.toFixed(3);
    stage.dataset.chapter = chapter;
    stage.dataset.main = t < 35.2 ? 'unborn' : mainName;
    stage.dataset.tracking = t < 70 ? 'none' : t < 76.2 ? 'M' : 'N';
    stage.dataset.head = onExperiment > .5 ? 'experiment' : 'main';
    $('film-position').value = t;
    $('film-position').setAttribute('aria-valuetext', `${stamp(t)} of 1:30; ${chapters[chapter][2]}`);
    $('film-time').textContent = `${stamp(t)} / 1:30`;
    $('film-play').disabled = running;
    $('film-pause').disabled = !running;
  }

  function advance(now) {
    elapsed = Math.min(duration, anchorElapsed + (now - anchorTime) * rate / 1000);
  }

  function tick(now) {
    if (!running) return;
    advance(now);
    if (elapsed >= duration) {
      running = false;
      raf = null;
    }
    renderAtTime(elapsed);
    if (running) raf = requestAnimationFrame(tick);
  }

  function pause() {
    if (running) advance(performance.now());
    running = false;
    cancelAnimationFrame(raf);
    raf = null;
    renderAtTime(elapsed);
  }

  function play() {
    if (running) return;
    if (elapsed >= duration) elapsed = 0;
    running = true;
    anchorElapsed = elapsed;
    anchorTime = performance.now();
    renderAtTime(elapsed);
    raf = requestAnimationFrame(tick);
  }
  $('film-play').addEventListener('click', play);
  $('film-pause').addEventListener('click', pause);
  $('film-replay').addEventListener('click', () => {
    pause();
    elapsed = 0;
    play();
  });
  $('film-position').addEventListener('input', event => {
    // A seek is a new clock anchor. Playing stays playing; paused stays paused.
    elapsed = Number(event.target.value);
    anchorElapsed = elapsed;
    anchorTime = performance.now();
    if (elapsed >= duration) {
      running = false;
      cancelAnimationFrame(raf);
      raf = null;
    }
    renderAtTime(elapsed);
  });
  $('film-speed').addEventListener('input', event => {
    const now = performance.now();
    if (running) advance(now);
    rate = Number(event.target.value);
    anchorElapsed = elapsed;
    anchorTime = now;
    $('film-rate').textContent = `${rate.toFixed(2).replace(/0$/,'')}×`;
    renderAtTime(elapsed);
  });

  function motionMode() {
    if (reduced.matches) {
      pause();
      $('film-mode').textContent = 'Reduced motion: automatic playback is off. Use the timeline to inspect states, or Play for a static-state presentation.';
    } else $('film-mode').textContent = 'Starts automatically on page load. Scrubbing preserves play/pause; all motion and typing follow one clock.';
    renderAtTime(elapsed);
  }
  // No viewport observer: autoplay must begin on landing, even below the fold.
  // A hidden browser tab suspends presentation time, without surprising restarts.
  let resumeAfterHidden = false;
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      resumeAfterHidden = running;
      pause();
    } else if (resumeAfterHidden) {
      resumeAfterHidden = false;
      play();
    }
  });
  reduced.addEventListener('change', motionMode);
  narrow.addEventListener('change', () => {
    chapterIndex = -1;
    buildWorld();
  });
  buildWorld();
  motionMode();
  if (!reduced.matches) play();
})();
