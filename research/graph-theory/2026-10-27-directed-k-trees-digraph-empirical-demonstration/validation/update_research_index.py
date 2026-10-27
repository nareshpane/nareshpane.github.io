"""Add only the explicitly requested Graph Theory section to the repository index."""
from pathlib import Path
root=Path(__file__).resolve().parents[4]
target=root/'research.html'
raw=target.read_bytes()
needle=b'      </ul>\r\n    </section>'
if needle not in raw:
    needle=b'      </ul>\n    </section>'
assert raw.count(needle)==1, 'Expected one final research list'
assert b'2026-10-27-directed-k-trees-digraph-empirical-demonstration.html' not in raw, 'Already integrated'
block='''      </ul>

      <details class="graph-theory-section" style="margin-top: 28px; border-top: 1px solid var(--line); padding-top: 18px;">
        <summary style="cursor: pointer; font-family: Georgia, serif; font-size: 1.8rem; color: var(--text);">Graph Theory</summary>
        <ul class="research-list">
          <li>
            <a class="project-title" href="research/graph-theory/2026-10-27-directed-k-trees-digraph-empirical-demonstration.html">
              Directed k-Trees in Digraphs: Mathematical Foundations and Empirical Demonstration
            </a>
            <div class="project-desc">
              An interactive exploration of directed k-trees in digraphs, combining original mathematical research, rigorous derivations, graph-theoretic visualizations, and computational experiments to examine and validate key theoretical results.
            </div>
          </li>
        </ul>
      </details>
    </section>'''
newline=b'\r\n' if b'\r\n' in raw else b'\n'
target.write_bytes(raw.replace(needle,block.replace('\n',newline.decode()).encode('utf-8')))
print('Added Graph Theory section to', target)
