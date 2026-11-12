/* Independent, bounded teaching models. Nothing here executes Git or writes files. */
'use strict';
(() => {
  const $ = id => document.getElementById(id);
  const escape = text => String(text).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;');
  const motion = matchMedia('(prefers-reduced-motion: reduce)');
  document.querySelectorAll('.js-controls').forEach(el => {
    el.hidden = false;
  });

  // Relationship map: reads/writes are explicit, rather than color-only decoration.
  const map = {
    add: {
      states: ['reads', 'writes', 'writes', 'idle'],
      roles: ['Reads selected content', 'Updates selected entries', 'Writes or reuses blobs; refs stay', 'Not contacted'],
      text: 'Working tree → blobs + index. git add records selected content without moving or deleting the working file. It creates no commit.'
    },
    commit: {
      states: ['idle', 'reads', 'both', 'idle'],
      roles: ['Unstaged files not included', 'Reads proposed snapshot', 'Reads parent; writes tree/commit; advances branch', 'Not contacted'],
      text: 'Index → tree + commit → current ref. Ordinary commit records the staged snapshot and leaves unstaged edits alone. It does not upload anything.'
    },
    fetch: {
      states: ['idle', 'idle', 'writes', 'reads'],
      roles: ['Working files stay', 'Staged snapshot stays', 'Receives objects; updates tracking refs', 'Reads selected advertised history'],
      text: 'Remote → local objects + remote-tracking refs (usual refspec). origin/main may advance while main, index and working files stay put. Integration is separate.'
    },
    push: {
      states: ['idle', 'idle', 'both', 'both'],
      roles: ['Uncommitted files not sent', 'Staged-only changes not sent', 'Reads objects/refs; may update tracking ref and upstream config', 'Negotiates objects; accepts or rejects ref proposal'],
      text: 'Local reachable objects → remote; request an update to its ref. Server policy, permissions and ancestry checks can refuse. push -u also records upstream configuration on success.'
    },
    unstage: {
      states: ['idle', 'writes', 'reads', 'idle'],
      roles: ['Working bytes preserved', 'Restores selected paths from HEAD', 'Reads HEAD tree; refs stay', 'Not contacted'],
      text: 'HEAD tree → selected index entries. git restore --staged file removes that staged difference while leaving the working edit available.'
    },
    restore: {
      states: ['writes', 'reads', 'reads', 'idle'],
      roles: ['Overwrites selected working content', 'Reads indexed object ID', 'Reads blob; no ref movement', 'Not contacted'],
      text: 'Index → working file. Plain git restore file discards unstaged edits for that path. No new commit, no branch movement; inspect diff before replacing content.'
    },
    switch: {
      states: ['writes', 'writes', 'both', 'idle'],
      roles: ['Updates for destination; preserves compatible edits', 'Updates for destination', 'Reads target snapshot; changes HEAD', 'Not contacted'],
      text: 'Destination branch → HEAD + checkout state. git switch branch normally refuses to overwrite incompatible local edits. Branch switching itself makes no commit.'
    },
    reset: {
      states: ['idle', 'writes', 'both', 'idle'],
      roles: ['Working files kept', 'Replaced by target tree', 'Reads target; repositions current branch/HEAD', 'Not contacted'],
      text: 'git reset --mixed C moves the current tip to C and resets the index to C’s tree. Working bytes remain. Path-based reset is different: it does not move HEAD.'
    },
    pull: {
      states: ['writes', 'writes', 'both', 'reads'],
      roles: ['Updated only if integration succeeds', 'Updated only if integration succeeds', 'Fetches objects/tracking; fast-forwards current branch if possible', 'Fetches selected remote history'],
      text: 'git pull --ff-only: fetch, then an ancestry check. On success it advances the current branch and checkout; on divergence it refuses integration, though fetched information remains.'
    }
  };

  const mapRoutes = {
    add: ['Working content', 'Local blobs + index'],
    commit: ['Index', 'Tree + commit', 'Current branch'],
    fetch: ['Remote', 'Local objects + origin/main'],
    push: ['Local objects + tip', 'Remote ref proposal'],
    unstage: ['HEAD tree', 'Selected index entries'],
    restore: ['Indexed content', 'Working file'],
    switch: ['Destination branch', 'HEAD + index + files'],
    reset: ['Target commit', 'Current ref + index'],
    pull: ['Remote', 'Fetch results', 'Fast-forward check', 'Checkout']
  };

  function mapSelect(key) {
    const value = map[key];
    $('map-flow').innerHTML = mapRoutes[key].map((name, i) => `${i ? '<b aria-hidden="true">→</b>' : ''}<span>${escape(name)}</span>`).join('');
    ['working', 'index', 'local', 'remote'].forEach((name, i) => {
      const el = $(`map-${name}`);
      el.dataset.state = value.states[i];
      el.querySelector('.map-role').textContent = value.roles[i];
    });
    $('map-description').textContent = value.text;
    document.querySelectorAll('[data-map]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.map === key)));
  }
  document.querySelectorAll('[data-map]').forEach(b => b.addEventListener('click', () => mapSelect(b.dataset.map)));
  mapSelect('add');

  // A compact catalog supplements the static chapters with comparable effects.
  // Each record: category, form, reads, changes, graph, network, rewrite, risk, example, mistake, related.
  const catalog = [
    ['inspect', 'git status', 'HEAD tree, index, working tree', 'Report; may refresh index bookkeeping', 'None', 'No', 'No', 'None for ordinary status', 'git status --short', 'Clean does not mean pushed.', 'diff, log'],
    ['inspect', 'git log', 'Parent graph from selected tips', 'Report only', 'Traverses reachable ancestry', 'No', 'No', 'None', 'git log --graph --decorate --all --oneline', 'It is not a reflog of local actions.', 'show, rev-list, reflog'],
    ['inspect', 'git show', 'Specified object and comparison context', 'Report only', 'None', 'No', 'No', 'None', 'git show HEAD:index.html', 'A commit display is not a complete directory dump.', 'diff, cat-file'],
    ['inspect', 'git diff', 'Index and working files', 'Report only', 'None', 'No', 'No', 'None', 'git diff -- index.html', 'Untracked content is not automatically shown.', 'status, add -p'],
    ['inspect', 'git diff --staged', 'HEAD tree and index', 'Report only', 'None', 'No', 'No', 'None', 'git diff --staged', 'This excludes edits made after the last add.', 'commit, restore --staged'],
    ['inspect', 'git diff HEAD / A B', 'HEAD and working tree, or two specified trees', 'Report only', 'None', 'No', 'No', 'None', 'git diff main..experiment', 'Two dots here compare endpoint trees, not a log range.', 'diff --staged, log'],
    ['stage', 'git add <path>', 'Selected working content and index', 'Writes/reuses blobs; updates index entries', 'No new commit or moved ref', 'No', 'No', 'Working content stays; previous staged selection can be replaced', 'git add index.html', 'Add again after another edit if you want the new version staged.', 'diff --staged, commit'],
    ['stage', 'git add .', 'Working content under current directory', 'Index additions, changes and deletions in scope', 'No new commit', 'No', 'No', 'Can stage unintended files; inspect before commit', 'git add .', 'Scope depends on current directory, not always repo root.', 'status, add -p'],
    ['stage', 'git rm / git mv', 'Tracked paths and index', 'Working paths + index (rm --cached keeps working file)', 'No commit until you commit', 'No', 'No', 'Removal can lose work if safeguards overridden', 'git mv draft.html index.html', 'A rename is not stored as a special rename object.', 'diff --staged, restore'],
    ['commit', 'git commit / -m', 'Resolved index and current parent', 'Trees, commit, current ref and reflog', 'Adds a commit; root has no parent', 'No', 'No: extends history', 'Ordinary commit preserves working files', 'git commit -m "Add navigation"', 'Unstaged and untracked edits are not included by default.', 'add, status, log'],
    ['commit', 'git commit --amend', 'Index and previous tip’s parents/metadata', 'Replacement commit and current ref', 'Replaces reachable tip; old object may remain', 'No', 'Yes, if replacing an existing tip', 'Published identity changes; files normally preserved', 'git commit --amend -m "Clarify navigation"', 'Staged edits are included even for an intended message-only change.', 'rebase, reflog'],
    ['branch', 'git branch', 'Refs; HEAD for branch creation', 'Lists refs, or creates named ref', 'New name at an existing commit', 'No', 'No', 'Ordinary creation preserves work', 'git branch experiment', 'Creating a branch does not switch to it.', 'switch, show-ref'],
    ['branch', 'git switch / switch -c', 'Destination ref/tree, index, local changes', 'HEAD, index and working tree; -c creates a ref', 'Chooses a branch; no new commit', 'No', 'No', 'Normally refuses destructive overwrites; discard options change this', 'git switch -c experiment', 'A branch is not a copied directory.', 'branch, checkout'],
    ['branch', 'git switch --detach', 'Specified commit tree', 'HEAD directly names commit; updates checkout', 'New commits have no advancing branch name', 'No', 'No', 'Save work first; name new detached commits to retain them', 'git switch --detach HEAD~1', 'Detached HEAD is not a damaged repository.', 'switch -c, reflog'],
    ['branch', 'git checkout <branch>', 'Destination branch and local checkout state', 'HEAD, index, working tree', 'Selects branch', 'No', 'No', 'Normal safeguards; force can discard edits', 'git checkout experiment', 'Checkout also has separate path forms; read the arguments.', 'switch, restore'],
    ['integrate', 'git merge', 'Current tip, other tip, merge base(s)', 'Current branch, index, working tree; possibly new objects', 'Fast-forward or new multi-parent commit', 'No', 'No: ordinarily preserves old ancestry', 'Conflicts possible; preserve unrelated local work first', 'git merge experiment', 'No network fetch is implicit in merge.', 'merge-base, pull'],
    ['integrate', 'git rebase', 'Selected local commits and new base', 'Reconstructed commits, branch, index, working tree', 'New parent chain, generally new IDs', 'No', 'Yes for commits actually reconstructed', 'Conflicts and shared-history coordination', 'git rebase main', 'Existing commit objects are not physically moved.', 'merge, reflog'],
    ['integrate', 'git cherry-pick', 'Selected change and current tree', 'Usually new commit; index, working tree and current ref', 'New descendant here, without importing source ancestry', 'No', 'No: normally adds history', 'Can conflict; save unrelated work first', 'git cherry-pick <commit-id>', 'Prerequisite commits are not automatically imported.', 'rebase, merge'],
    ['synchronize', 'git clone', 'Source repository and advertised refs', 'New local repo, config, tracking refs and usually checkout', 'Obtains available history', 'Yes for network URLs', 'No', 'Normally refuses a nonempty target', 'git clone https://github.com/USER/PROJECT.git', 'Clone is richer than downloading a ZIP.', 'init, fetch, remote'],
    ['synchronize', 'git remote / remote -v', 'Local remote configuration', 'Report only', 'None', 'No', 'No', 'None', 'git remote -v', 'Origin is not a synonym for GitHub.', 'remote add, fetch'],
    ['synchronize', 'git remote add', 'Supplied name and URL', 'Local remote configuration', 'None', 'No, without fetch option', 'No', 'None to working files', 'git remote add origin <url>', 'This does not push anything.', 'clone, push -u'],
    ['synchronize', 'git fetch', 'Remote refs/objects and local refspec', 'Local objects, tracking refs, FETCH_HEAD; possibly tags', 'Learns remote ancestry; current branch normally stays', 'Yes for network remotes', 'No rewrite of your current branch; tracking can reflect remote rewrites', 'Ordinary fetch keeps index and working tree', 'git fetch origin', 'origin/main is a local ref, not a live server read.', 'pull, merge'],
    ['synchronize', 'git pull', 'Remote state, current tip and integration policy', 'Fetch results then branch, index and working tree if integrated', 'Fast-forward, merge, rebase, or refusal depending on policy', 'Yes for network remotes', 'Possible with rebase policy', 'Integration may conflict; preserve local edits first', 'git pull --ff-only origin main', 'Pull integrates into the currently checked-out branch.', 'fetch, merge, rebase'],
    ['synchronize', 'git push / push -u', 'Local objects/refs and remote advertised tips', 'Remote objects/refs if accepted; local tracking/upstream as applicable', 'Proposes remote ref advancement', 'Yes for network remotes', 'Normal branch push requires fast-forward', 'No local file deletion; remote policy may reject', 'git push -u origin main', 'Uncommitted content is not sent.', 'fetch, remote'],
    ['synchronize', 'git push --force-with-lease', 'Local proposal and expected remote ref value', 'Remote ref if lease/policy allow; transfers needed objects', 'Can replace published branch ancestry', 'Yes', 'Yes, if non-fast-forward', 'Can displace shared history even with a matching lease', 'git push --force-with-lease origin feature', 'Background fetch can weaken shorthand lease assumptions.', 'fetch, rebase, push'],
    ['undo', 'git restore <file>', 'Index by default, or chosen source tree', 'Selected working-tree content', 'No ref movement', 'No', 'No', 'Yes: overwrites unstaged edits', 'git restore index.html', 'Default source is index, not necessarily HEAD.', 'diff, restore --staged'],
    ['undo', 'git restore --staged', 'HEAD tree by default', 'Selected index entries', 'No ref movement', 'No', 'No', 'Staged version replaced; working content remains', 'git restore --staged index.html', 'Unstaging does not discard the visible working edit.', 'reset HEAD -- file, add'],
    ['undo', 'git checkout -- <file>', 'Index entries', 'Selected working files', 'No ref movement', 'No', 'No', 'Yes: overwrites unstaged edits', 'git checkout -- index.html', 'Adding a source revision changes which states are replaced.', 'restore'],
    ['undo', 'git reset --soft', 'Target commit', 'Current branch/HEAD; reflog and ORIG_HEAD', 'Repositions tip; old objects remain initially', 'No', 'Repositions branch history', 'Index and working files kept; shared ancestry can change', 'git reset --soft HEAD~1', 'Index may still differ from working files.', 'commit, reflog'],
    ['undo', 'git reset --mixed', 'Target commit and tree', 'Current branch/HEAD and index', 'Repositions tip', 'No', 'Repositions branch history', 'Index replaced; working files kept', 'git reset --mixed HEAD~1', 'Path-form reset does not move HEAD.', 'restore --staged, reflog'],
    ['undo', 'git reset --hard', 'Target commit/tree', 'Current branch/HEAD, index, working tree', 'Repositions tip', 'No', 'Repositions branch history', 'YES: tracked edits and obstructing untracked paths', 'git reset --hard <verified-target>', 'Reflog cannot promise to recover overwritten uncommitted bytes.', 'diff, stash, reflog'],
    ['undo', 'git reset HEAD -- <file>', 'HEAD tree', 'Selected index entries only', 'No ref movement', 'No', 'No', 'Replaces staged selection; files stay', 'git reset HEAD -- index.html', 'This is not the commit-and-mode form of reset.', 'restore --staged'],
    ['undo', 'git revert', 'Selected commit change and current tree', 'Inverse change, normally a new commit, checkout and ref', 'Extends history; keeps original commit', 'No', 'No', 'Can conflict; needs a deliberate resolution', 'git revert <commit-id>', 'Reverting B after C does not necessarily restore the entire A tree.', 'cherry-pick, reset'],
    ['undo', 'git clean', 'Untracked paths and ignore rules', 'Deletes selected untracked working files', 'No graph change', 'No', 'No', 'YES; preview -nd before -fd; -x includes ignored files', 'git clean -nd', 'Reflog is not a backup of untracked files.', 'status, .gitignore'],
    ['recover', 'git stash / stash push', 'HEAD, index and working content', 'Special commits/ref; normally cleans saved working changes', 'Adds stash structure, not a branch commit', 'No', 'No ordinary branch rewrite', 'Default excludes untracked; -u includes them', 'git stash push -u -m "WIP"', 'It is not a remote backup.', 'stash list, apply, pop'],
    ['recover', 'git stash list', 'Stash reflog', 'Report only', 'None', 'No', 'No', 'None', 'git stash list', 'Entries are local and can expire or be dropped.', 'reflog, stash show'],
    ['recover', 'git stash apply / pop', 'Saved stash and current files/index', 'Applies content; pop drops entry after success', 'No ordinary branch commit', 'No', 'No', 'Can conflict; --index attempts staging restoration', 'git stash apply \'stash@{0}\'', 'Pop retains the entry when application fails with conflicts.', 'stash push, drop'],
    ['recover', 'git stash drop', 'Selected stash entry', 'Removes entry’s normal reflog handle', 'Objects may become eligible for later pruning', 'No', 'No branch rewrite', 'Yes: can remove recovery handle', 'git stash drop \'stash@{0}\'', 'Objects remaining temporarily is not a backup guarantee.', 'stash apply, reflog'],
    ['recover', 'git reflog', 'Local logged reference updates', 'Report only', 'Finds past positions beyond current branch ancestry', 'No', 'No', 'None for show/list form', 'git reflog', 'Not shared by clone; not every working edit is recorded.', 'branch rescue, log'],
    ['advanced', 'git init', 'Directory, config and templates', 'Repository infrastructure and initial HEAD', 'No commit until created', 'No', 'No', 'Ordinary init preserves project files', 'git init -b main', 'Index and branch tip may not exist immediately.', 'clone, add'],
    ['advanced', 'git tag / tag -a', 'Target object, name and optional metadata', 'Tag ref; annotated form also creates a tag object', 'Adds a name, not a parent edge', 'No; tag push is separate', 'No ordinary ancestry rewrite', 'Normal creation keeps working files', 'git tag -a v1.0 -m "First release"', 'Tags do not advance with ordinary commits.', 'show, push origin v1.0'],
    ['advanced', 'git hash-object', 'Typed header and stored content bytes', 'With -w stores object; otherwise reports ID', 'No refs or commits unless hashing that type deliberately', 'No', 'No', 'None for blob calculation/storage', 'git hash-object -w hello.txt', 'The object hash is not just the raw-file hash.', 'cat-file, add'],
    ['advanced', 'git cat-file / ls-tree', 'Available object or tree', 'Report only', 'None', 'No', 'No', 'None', 'git cat-file -p HEAD', 'A blob contains no intrinsic filename.', 'show, hash-object'],
    ['advanced', 'git write-tree', 'Resolved index', 'Writes/reuses tree objects', 'No commit or branch update', 'No', 'No', 'None to working files', 'git write-tree', 'An unresolved index cannot be written as a normal tree.', 'commit-tree, ls-tree'],
    ['advanced', 'git commit-tree', 'Tree ID, parent options, metadata and message', 'Writes commit object; returns ID', 'New node; no branch automatically moves', 'No', 'No by itself', 'Unreferenced object needs a name to retain it', 'printf "Demo\\n" | git commit-tree <tree-id>', 'Creating an object is separate from updating a ref.', 'write-tree, update-ref'],
    ['advanced', 'git rev-parse / show-ref / rev-list', 'Names, refs or parent graph', 'Report only in these forms', 'Resolve names or enumerate reachable commits', 'No', 'No', 'None', 'git rev-list --parents HEAD', 'A revision name and a literal object ID are not the same representation.', 'log, reflog, merge-base'],
    ['advanced', 'git update-ref', 'New ID and optional expected old ID', 'Named ref, with applicable reflog update', 'Can reposition a branch without checkout changes', 'No', 'Possible', 'Can remove normal reachability; use expected-old-ID checks', 'git update-ref refs/heads/demo <new-id> <old-id>', 'It does not update working tree or index for you.', 'branch, reset'],
    ['advanced', 'git merge-base', 'Two or more commit histories', 'Report only', 'Finds best common ancestors', 'No', 'No', 'None', 'git merge-base --all main feature', 'Best common ancestor need not be unique.', 'merge, rev-list'],
    ['advanced', 'git count-objects -v / gc', 'Object storage, refs, retention roots and settings', 'Count reports only; gc maintains storage and may prune', 'Reachable history kept; eligible unreachable objects may disappear', 'No', 'No reachable ancestry rewrite', 'GC can end recovery opportunities; avoid aggressive pruning', 'git count-objects -v', 'Packed deltas are not the commit parent graph.', 'reflog, fsck'],
    ['advanced', 'git blame / grep / shortlog', 'Selected content and history', 'Report only', 'No change', 'No', 'No', 'None', 'git blame -L 1,20 index.html', 'Attribution is context, not a complete measure of contribution.', 'log, show'],
    ['advanced', 'git bisect', 'Good/bad labels and candidate graph', 'Selects checkouts and bisect state', 'Searches history; does not rewrite commits', 'No', 'No', 'Checkout changes; save unrelated work first', 'git bisect start', 'A reliable good/bad test matters more than the label.', 'bisect reset, log'],
    ['advanced', 'git worktree', 'Shared objects/refs and chosen branch', 'Additional working tree, index and per-worktree HEAD', 'May create branch; shares objects', 'No', 'No', 'Ordinary add preserves existing checkout', 'git worktree add ../review -b review', 'Two working trees are not two independent object databases.', 'switch, worktree list'],
    ['advanced', 'git sparse-checkout', 'Sparse rules, index and selected tree', 'Working-tree population and sparse index/rules', 'History unchanged', 'No in normal full clone', 'No', 'Save local edits before changing sparse layout', 'git sparse-checkout set src docs', 'Sparse checkout is not the same as partial clone.', 'clone --filter, worktree']
  ];

  function commandRender() {
    const item = catalog[Number($('command-choice').value)];
    if (!item) return;
    const [, form, reads, changes, graph, net, rewrite, risk, example, mistake, related] = item;
    const fields = [
      ['What it reads', reads],
      ['What it changes', changes],
      ['Graph effect', graph],
      ['Remote network activity?', net],
      ['Rewrites history?', rewrite],
      ['Potential data loss?', risk]
    ];
    $('command-details').innerHTML = `<h3 class="explorer-title">${escape(form)}</h3><dl class="mechanics">${fields.map(([a,b])=>`<div><dt>${a}</dt><dd>${escape(b)}</dd></div>`).join('')}</dl><pre><code>${escape(example)}</code></pre><p><strong>Watch for:</strong> ${escape(mistake)}</p><p class="small"><strong>Related:</strong> ${escape(related)}. Network labels assume required objects are already local and no custom hooks initiate network activity.</p>`;
  }

  function categoryRender() {
    const category = $('command-category').value;
    $('command-choice').innerHTML = catalog.map((c, i) => c[0] === category ? `<option value="${i}">${escape(c[1])}</option>` : '').join('');
    commandRender();
  }
  $('command-category').addEventListener('change', categoryRender);
  $('command-choice').addEventListener('change', commandRender);
  categoryRender();

  // Three maps represent complete snapshots in this bounded two-file model.
  let serial = 1,
    head, index, working;

  function statusReset() {
    serial = 1;
    head = {
      'index.html': 'v1'
    };
    index = {
      ...head
    };
    working = {
      ...head
    };
  }

  function statusRender(message) {
    const paths = [...new Set([...Object.keys(head), ...Object.keys(index), ...Object.keys(working)])];
    let lines = [];
    $('status-files').innerHTML = paths.map(path => {
      const h = head[path],
        i = index[path],
        w = working[path];
      if (!i && w) lines.push(`?? ${path}`);
      else {
        const x = i !== h ? (h ? 'M' : 'A') : ' ',
          y = w !== i ? 'M' : ' ';
        if (x + y !== '  ') lines.push(`${x}${y} ${path}`);
      }
      return `<tr><td>${escape(path)}</td><td>${h||'—'}</td><td>${i||'—'}</td><td>${w||'—'}</td></tr>`;
    }).join('');
    $('status-output').textContent = '$ git status --short\n' + (lines.join('\n') || '(clean)');
    $('status-explanation').textContent = message;
    document.querySelector('[data-status-action="create"]').disabled = Boolean(working['notes.txt']);
  }
  document.querySelectorAll('[data-status-action]').forEach(button => button.addEventListener('click', () => {
    let message;
    switch (button.dataset.statusAction) {
      case 'edit':
      case 'edit-again':
        working['index.html'] = `v${++serial}`;
        message = 'Edited working index.html only. The index and HEAD keep their previous versions.';
        break;
      case 'create':
        working['notes.txt'] = 'note1';
        message = 'Created an untracked file. It is absent from the index and HEAD.';
        break;
      case 'add':
        index = {
          ...working
        };
        message = 'Staged both current files. Working files stayed in place; HEAD did not change.';
        break;
      case 'commit':
        if (JSON.stringify(head) === JSON.stringify(index)) {
          message = 'Nothing staged to commit. Unstaged edits are not included automatically.';
        } else {
          head = {
            ...index
          };
          message = 'Recorded the indexed snapshot. Any newer working version remains an unstaged edit.';
        }
        break;
      case 'restore':
        working['index.html'] = index['index.html'];
        message = 'Replaced working index.html with its indexed version; unstaged edits to it were discarded in this model.';
        break;
      case 'unstage':
        index = {
          ...head
        };
        message = 'Restored the index from HEAD. Working files remain; a newly unstaged file becomes untracked.';
        break;
      default:
        statusReset();
        message = 'Model reset: all three states agree at v1.';
    }
    statusRender(message);
  }));
  statusReset();
  statusRender('All three states agree. Edit, add, then edit again to see two different changes for one path.');

  // Graph renderer is shared by the comparison and graph lab. Parent edges are
  // always child → parent; mobile uses a vertical layout instead of tiny labels.
  let graphSerial = 0;

  function graphSVG(nodes, refs, current, container, reachable = null) {
    const levels = new Map();
    nodes.forEach(n => levels.set(n.id, n.parents.length ? 1 + Math.max(...n.parents.map(p => levels.get(p))) : 0));
    const maxLevel = Math.max(...levels.values());
    const occupied = nodes.map(n => `${levels.get(n.id)}:${n.lane}`);
    const vertical = matchMedia('(max-width: 700px)').matches || maxLevel > 5 || new Set(occupied).size < nodes.length;
    const width = vertical ? 320 : Math.max(460, 90 + maxLevel * 112),
      height = vertical ? 100 + (nodes.length - 1) * 88 : 255;
    const points = new Map(nodes.map(n => [n.id, vertical ? [75 + n.lane * 165, 38 + nodes.indexOf(n) * 88] : [40 + levels.get(n.id) * 112, 75 + n.lane * 125]]));
    const marker = `lab-arrow-${graphSerial++}`;
    const edges = nodes.flatMap(n => n.parents.map(p => {
      const [x, y] = points.get(n.id), [px, py] = points.get(p), d = Math.hypot(px - x, py - y), ux = (px - x) / d, uy = (py - y) / d;
      // Route long links around retained nodes, so an edge cannot look as if
      // it passes through a commit that is not actually one of its parents.
      const obstructed = nodes.some(other => {
        if (other.id === n.id || other.id === p) return false;
        const [ox, oy] = points.get(other.id);
        const projection = ((ox - x) * ux + (oy - y) * uy) / d;
        const distance = Math.abs((ox - x) * uy - (oy - y) * ux);
        return projection > 0 && projection < 1 && distance < 26;
      });
      let path = `M${x+ux*20} ${y+uy*20}L${px-ux*24} ${py-uy*24}`;
      if (obstructed) {
        const bend = n.lane === 0 ? -56 : 56;
        path = vertical ?
          `M${x} ${y-20}C${x+bend} ${y-55} ${px+bend} ${py+55} ${px} ${py+24}` :
          `M${x-20} ${y}C${x-50} ${y+bend} ${px+50} ${py+bend} ${px+24} ${py}`;
      }
      return `<path class="edge ${reachable&&!reachable.has(n.id)?'old':''}" d="${path}" marker-end="url(#${marker})"/>`;
    })).join('');
    const circles = nodes.map(n => {
      const [x, y] = points.get(n.id), names = Object.entries(refs).filter(([, id]) => id === n.id).map(([name]) => name);
      const unreachable = reachable && !reachable.has(n.id);
      return `<g class="${unreachable?'old':''}"><circle class="node ${n.id.includes('′')?'new-node':''}" cx="${x}" cy="${y}" r="20"/><text text-anchor="middle" x="${x}" y="${y+5}">${escape(n.id)}</text>${names.map((name,i)=>`<text class="ref" text-anchor="middle" x="${x}" y="${y+38+i*17}">${escape(name)}</text>`).join('')}</g>`;
    }).join('');
    container.innerHTML = `<svg class="dag" viewBox="0 0 ${width} ${height}" role="img" aria-label="${escape(nodes.map(n=>`${n.id} has ${n.parents.length?'parents '+n.parents.join(', '):'no parent'}`).join('; '))}"><defs><marker id="${marker}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0L10 5 0 10z" fill="#788575"/></marker></defs>${edges}${circles}</svg><div class="graph-key"><span>HEAD → ${escape(current)}</span><span>Arrows → parents</span>${reachable?'<span>Dashed / faded = retained, unreachable from branch tips</span>':''}</div>`;
  }
  const startNodes = [{
    id: 'A',
    parents: [],
    lane: 0
  }, {
    id: 'B',
    parents: ['A'],
    lane: 0
  }, {
    id: 'C',
    parents: ['B'],
    lane: 0
  }, {
    id: 'D',
    parents: ['B'],
    lane: 1
  }, {
    id: 'E',
    parents: ['D'],
    lane: 1
  }];
  let integration = 'before';

  function integrationRender() {
    let nodes = startNodes.map(n => ({
        ...n
      })),
      refs = {
        main: 'C',
        feature: 'E'
      },
      desc = 'Main names C; feature names E. Both lines descend from B. No integration yet.';
    if (integration === 'merge') {
      nodes.push({
        id: 'M',
        parents: ['E', 'C'],
        lane: 1
      });
      refs.feature = 'M';
      desc = 'Merge on feature creates M with parents E and C. D and E retain their identities and remain reachable; the fork is visible in ancestry.';
    }
    if (integration === 'rebase') {
      nodes = [...nodes.slice(0, 3), {
        id: 'D′',
        parents: ['C'],
        lane: 1
      }, {
        id: 'E′',
        parents: ['D′'],
        lane: 1
      }];
      refs.feature = 'E′';
      desc = 'Rebase on feature reconstructs D′ and E′ on C. The original D and E are no longer in this branch ancestry, though retained objects/reflogs may still contain them.';
    }
    graphSVG(nodes, refs, 'feature', $('integration-graph'));
    $('integration-description').textContent = desc;
    document.querySelectorAll('[data-integrate]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.integrate === integration)));
  }
  document.querySelectorAll('[data-integrate]').forEach(b => b.addEventListener('click', () => {
    integration = b.dataset.integrate;
    integrationRender();
  }));
  integrationRender();
  let graphNodes, refs, current, nextID;

  function graphReset() {
    graphNodes = [{
      id: 'A',
      parents: [],
      lane: 0
    }, {
      id: 'B',
      parents: ['A'],
      lane: 0
    }];
    refs = {
      main: 'B'
    };
    current = 'main';
    nextID = 2;
  }

  function ancestors(tip) {
    const result = new Set(),
      queue = [tip];
    while (queue.length) {
      const id = queue.pop();
      if (result.has(id)) continue;
      result.add(id);
      queue.push(...graphNodes.find(n => n.id === id).parents);
    }
    return result;
  }

  function graphRender(message) {
    const reached = new Set(Object.values(refs).flatMap(tip => [...ancestors(tip)]));
    graphSVG(graphNodes, refs, current, $('graph-canvas'), reached);
    if (message) $('graph-description').textContent = message;
    document.querySelector('[data-graph-action="branch"]').disabled = Boolean(refs.feature);
    document.querySelector('[data-graph-action="switch"]').disabled = !refs.feature;
    document.querySelector('[data-graph-action="merge"]').disabled = !refs.feature || graphNodes.length >= 10;
    document.querySelector('[data-graph-action="commit"]').disabled = graphNodes.length >= 10;
    document.querySelector('[data-graph-action="reset"]').disabled = !graphNodes.find(n => n.id === refs[current]).parents.length;
  }

  function graphAdd(parents) {
    const id = String.fromCharCode(65 + nextID++);
    graphNodes.push({
      id,
      parents,
      lane: current === 'main' ? 0 : 1
    });
    refs[current] = id;
    return id;
  }
  document.querySelectorAll('[data-graph-action]').forEach(button => button.addEventListener('click', () => {
    let message;
    const tip = refs[current],
      other = current === 'main' ? 'feature' : 'main';
    switch (button.dataset.graphAction) {
      case 'branch':
        refs.feature = tip;
        message = `Created feature at ${tip}. HEAD still names ${current}; no objects were copied.`;
        break;
      case 'commit': {
        const id = graphAdd([tip]);
        message = `Created ${id} with parent ${tip}; only ${current} advanced.`;
        break;
      }
      case 'switch':
        current = other;
        message = `HEAD now names ${current}, at ${refs[current]}. The graph did not change.`;
        break;
      case 'merge': {
        const otherTip = refs[other];
        if (ancestors(tip).has(otherTip)) message = 'Already up to date: the other tip is already an ancestor (or the same commit).';
        else if (ancestors(otherTip).has(tip)) {
          refs[current] = otherTip;
          message = `Fast-forward: moved ${current} to ${otherTip}; no new commit.`;
        } else {
          const id = graphAdd([tip, otherTip]);
          message = `Three-way merge model: ${id} has first parent ${tip} and second parent ${otherTip}. File conflicts are outside this model.`;
        }
        break;
      }
      case 'reset':
        refs[current] = graphNodes.find(n => n.id === tip).parents[0];
        message = `Moved ${current} from ${tip} to its first parent. ${tip} stays visible as a retained object; other refs may still reach it.`;
        break;
      default:
        graphReset();
        message = 'Restarted at HEAD → main → B → A. Create feature to give B another name.';
    }
    graphRender(message);
  }));
  graphReset();
  graphRender();
  let resizeFrame;
  window.addEventListener('resize', () => {
    cancelAnimationFrame(resizeFrame);
    resizeFrame = requestAnimationFrame(() => {
      graphRender();
      integrationRender();
    });
  });

  // Object identity cascade: one short RAF clock; reduced motion shows final state.
  let hashFrame = null,
    hashStart = 0;
  const oldLabels = ['return 1;', 'blob α', 'tree β', 'tree γ', 'commit δ'],
    newLabels = ['return 2;', 'blob α′', 'tree β′', 'tree γ′', 'commit δ′'];
  const steps = [...$('hash-chain').children];

  function hashPaint(count) {
    steps.forEach((step, i) => {
      step.classList.toggle('active', i < count);
      step.querySelector('code').textContent = (i < count ? newLabels : oldLabels)[i];
    });
  }

  function hashTick(now) {
    const count = Math.min(5, 1 + Math.floor((now - hashStart) / 650));
    hashPaint(count);
    if (count < 5) hashFrame = requestAnimationFrame(hashTick);
    else {
      hashFrame = null;
      $('hash-edit').disabled = false;
      $('hash-description').textContent = 'All dependent IDs changed in the new snapshot. Unchanged sibling objects can still be reused. These IDs are symbolic.';
    }
  }
  $('hash-edit').addEventListener('click', () => {
    cancelAnimationFrame(hashFrame);
    $('hash-description').textContent = 'An edit changes stored content; recording the new snapshot propagates that identity through each dependent object.';
    if (motion.matches) {
      hashPaint(5);
      return;
    }
    $('hash-edit').disabled = true;
    hashStart = performance.now();
    hashFrame = requestAnimationFrame(hashTick);
  });
  $('hash-reset').addEventListener('click', () => {
    cancelAnimationFrame(hashFrame);
    hashFrame = null;
    hashPaint(0);
    $('hash-edit').disabled = false;
    $('hash-description').textContent = 'Symbolic IDs, not real hashes. Editing a working file alone does not write all these objects; staging and committing construct the new snapshot.';
  });
  motion.addEventListener('change', () => {
    if (motion.matches && hashFrame) {
      cancelAnimationFrame(hashFrame);
      hashFrame = null;
      hashPaint(5);
      $('hash-edit').disabled = false;
    }
  });

  const synthesis = {
    remote: 'A separate repository with its own objects and refs. Push proposes an update there; fetch obtains its objects and reference information. Hosting is optional.',
    tracking: 'origin/main is a local observation of the remote branch, normally refreshed by fetch. It can be stale. Your own main can differ from it.',
    head: 'HEAD normally symbolically names a branch such as main; that branch names C. In detached state HEAD directly names a commit.',
    branch: 'Experiment is another ref into the same object history. It names D without copying the project. A commit made while on experiment advances this ref.',
    commit: 'A commit names a root tree, its parent commit(s), and metadata. Parent links form the commit DAG; snapshots are reached through trees.',
    tree: 'A tree associates path components and modes with object IDs. Trees store filenames and connect a directory hierarchy to content.',
    blob: 'A blob stores content, not a filename. Equal stored bytes can reuse a blob across names or snapshots, within the same object format.',
    working: 'The files you edit. They can differ from both the index and HEAD. Git status and diff compare these states.',
    index: 'The proposed next snapshot: paths, modes and object IDs, plus bookkeeping. Add changes it; ordinary commit records it. Conflicts can give a path multiple stages.'
  };

  function synthSelect(button) {
    $('synth-info').textContent = synthesis[button.dataset.synth];
    document.querySelectorAll('[data-synth]').forEach(b => {
      b.classList.toggle('selected', b === button);
      b.setAttribute('aria-pressed', String(b === button));
    });
  }
  document.querySelectorAll('[data-synth]').forEach(button => {
    button.setAttribute('aria-pressed', 'false');
    button.addEventListener('click', () => synthSelect(button));
    button.addEventListener('focus', () => synthSelect(button));
    button.addEventListener('pointerenter', event => {
      if (event.pointerType === 'mouse') synthSelect(button);
    });
  });
})();
