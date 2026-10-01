#!/usr/bin/env python3
"""Check the guide's mechanics using only throwaway repositories under /tmp.

Run: python3 research/git-commands/verify_git.py
No Git mutation is ever performed in the website repository.
"""
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile


def main():
    with tempfile.TemporaryDirectory(prefix='git-guide-check-', dir='/tmp') as temp:
        root = Path(temp).resolve()
        env = dict(os.environ, GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL='/dev/null',
                   GIT_AUTHOR_NAME='Demo', GIT_AUTHOR_EMAIL='demo@example.invalid',
                   GIT_COMMITTER_NAME='Demo', GIT_COMMITTER_EMAIL='demo@example.invalid',
                   GIT_AUTHOR_DATE='2026-09-27T12:00:00+00:00',
                   GIT_COMMITTER_DATE='2026-09-27T12:00:00+00:00')
        # Inherited Git environment must never redirect a test into a real repo.
        for key in list(env):
            if key.startswith('GIT_') and key not in {
                'GIT_CONFIG_NOSYSTEM', 'GIT_CONFIG_GLOBAL', 'GIT_AUTHOR_NAME',
                'GIT_AUTHOR_EMAIL', 'GIT_COMMITTER_NAME', 'GIT_COMMITTER_EMAIL',
                'GIT_AUTHOR_DATE', 'GIT_COMMITTER_DATE',
            }:
                del env[key]

        def git(repo, *args, data=None, success=True):
            assert Path(repo).resolve().is_relative_to(root), 'Temporary repos only'
            result = subprocess.run(['git', '-C', str(repo), *args], input=data,
                                    text=True, capture_output=True, env=env)
            if success and result.returncode:
                raise AssertionError(f'{args}: {result.stderr}')
            return result.stdout.rstrip('\n') if success else result

        def init(name, fmt='sha1'):
            repo = root / name
            repo.mkdir()
            git(repo, 'init', '-b', 'main', f'--object-format={fmt}')
            return repo

        def commit(repo, name, content):
            (repo / name).write_text(content)
            git(repo, 'add', name)
            git(repo, 'commit', '-m', content.strip())
            return git(repo, 'rev-parse', 'HEAD')

        repo = init('snapshots')
        assert git(repo, 'symbolic-ref', 'HEAD') == 'refs/heads/main'
        assert git(repo, 'rev-parse', '--verify', 'HEAD', success=False).returncode
        path = repo / 'index.html'
        path.write_text('v1\n')
        assert git(repo, 'status', '--short') == '?? index.html'
        git(repo, 'add', 'index.html')
        a = git(repo, 'write-tree')
        git(repo, 'commit', '-m', 'A')
        root_commit = git(repo, 'rev-parse', 'HEAD')
        assert git(repo, 'rev-list', '--parents', '-n', '1', 'HEAD') == root_commit
        path.write_text('v2\n')
        git(repo, 'add', 'index.html')
        path.write_text('v3\n')
        assert git(repo, 'status', '--short') == 'MM index.html'
        assert git(repo, 'show', ':index.html') == 'v2'
        git(repo, 'commit', '-m', 'B')
        b = git(repo, 'rev-parse', 'HEAD')
        assert git(repo, 'show', 'HEAD:index.html') == 'v2'
        assert path.read_text() == 'v3\n'
        git(repo, 'restore', 'index.html')
        assert path.read_text() == 'v2\n'
        path.write_text('v4\n')
        git(repo, 'add', 'index.html')
        git(repo, 'restore', '--staged', 'index.html')
        assert git(repo, 'show', ':index.html') == 'v2'
        assert path.read_text() == 'v4\n'
        git(repo, 'reset', '--soft', root_commit)
        assert git(repo, 'show', ':index.html') == 'v2'
        assert path.read_text() == 'v4\n'
        git(repo, 'reset', '--mixed', root_commit)
        assert git(repo, 'show', ':index.html') == 'v1'
        assert path.read_text() == 'v4\n'
        git(repo, 'reset', '--hard', b)
        assert path.read_text() == 'v2\n'
        git(repo, 'reset', '--hard', root_commit)
        assert git(repo, 'rev-parse', 'HEAD@{1}') == b
        git(repo, 'branch', 'rescue', 'HEAD@{1}')
        assert git(repo, 'rev-parse', 'rescue') == b
        print('PASS: unborn HEAD, index snapshots, restore, reset modes, reflog rescue')

        for fmt, hash_fn in [('sha1', hashlib.sha1), ('sha256', hashlib.sha256)]:
            obj_repo = init('objects-' + fmt, fmt)
            oid = git(obj_repo, 'hash-object', '-w', '--stdin', data='hello\n')
            assert oid == hash_fn(b'blob 6\0hello\n').hexdigest()
            assert git(obj_repo, 'cat-file', '-t', oid) == 'blob'
            assert git(obj_repo, 'cat-file', '-p', oid) == 'hello'
            (obj_repo / 'hello.txt').write_text('hello\n')
            (obj_repo / 'same.txt').write_text('hello\n')
            git(obj_repo, 'add', '.')
            tree = git(obj_repo, 'write-tree')
            listing = git(obj_repo, 'ls-tree', tree)
            assert listing.count(oid) == 2
            c1 = git(obj_repo, 'commit-tree', tree, data='first\n')
            c2 = git(obj_repo, 'commit-tree', tree, data='different message\n')
            assert c1 != c2
            assert git(obj_repo, 'rev-parse', c1 + '^{tree}') == tree
            git(obj_repo, 'update-ref', 'refs/heads/main', c1)
            assert git(obj_repo, 'rev-parse', 'HEAD') == c1
        print('PASS: SHA-1/SHA-256 typed hashes, blob reuse, tree/commit/ref plumbing')

        graph = init('graph')
        a = commit(graph, 'base.txt', 'base\n')
        git(graph, 'switch', '-c', 'feature')
        d = commit(graph, 'feature.txt', 'feature\n')
        git(graph, 'switch', 'main')
        c = commit(graph, 'main.txt', 'main\n')
        assert git(graph, 'merge-base', 'main', 'feature') == a
        git(graph, 'merge', 'feature', '-m', 'M')
        merge = git(graph, 'rev-parse', 'HEAD')
        assert git(graph, 'rev-parse', 'HEAD^1') == c
        assert git(graph, 'rev-parse', 'HEAD^2') == d
        assert git(graph, 'rev-parse', 'HEAD~2') == a
        git(graph, 'switch', 'feature')
        git(graph, 'rebase', c)
        d_prime = git(graph, 'rev-parse', 'HEAD')
        assert d_prime != d
        assert git(graph, 'rev-parse', 'HEAD^') == c
        git(graph, 'switch', '-c', 'pick', c)
        git(graph, 'cherry-pick', d)
        assert git(graph, 'rev-parse', 'HEAD') != d
        assert git(graph, 'rev-parse', 'HEAD^') == c
        git(graph, 'tag', 'light')
        git(graph, 'tag', '-a', 'annotated', '-m', 'release')
        assert git(graph, 'cat-file', '-t', 'light') == 'commit'
        assert git(graph, 'cat-file', '-t', 'annotated') == 'tag'
        print('PASS: merge bases/parents, revision syntax, rebase/cherry-pick IDs, tags')

        path = graph / 'main.txt'
        path.write_text('edited\n')
        (graph / 'notes.txt').write_text('untracked\n')
        git(graph, 'stash', 'push', '-u', '-m', 'test')
        assert not git(graph, 'status', '--short')
        assert len(git(graph, 'rev-list', '--parents', '-n', '1', 'stash@{0}').split()) == 4
        git(graph, 'stash', 'apply')
        assert (graph / 'notes.txt').exists()
        assert git(graph, 'stash', 'list')
        print('PASS: stash commits and untracked inclusion; apply retains entry')

        remote = root / 'remote.git'
        remote.mkdir()
        git(remote, 'init', '--bare', '-b', 'main')
        local = init('local')
        commit(local, 'index.html', 'one\n')
        git(local, 'remote', 'add', 'origin', str(remote))
        git(local, 'push', '-u', 'origin', 'main')
        peer = root / 'peer'
        # clone command's working directory is still inside the guarded temp root.
        git(root, 'clone', str(remote), str(peer))
        remote_tip = commit(peer, 'peer.txt', 'peer\n')
        git(peer, 'push', 'origin', 'main')
        old_head = git(local, 'rev-parse', 'HEAD')
        git(local, 'fetch', 'origin')
        assert git(local, 'rev-parse', 'HEAD') == old_head
        assert git(local, 'rev-parse', 'origin/main') == remote_tip
        assert not (local / 'peer.txt').exists()
        git(local, 'merge', '--ff-only', 'origin/main')
        assert (local / 'peer.txt').exists()
        commit(local, 'local.txt', 'local\n')
        commit(peer, 'other.txt', 'other\n')
        git(peer, 'push', 'origin', 'main')
        assert git(local, 'push', 'origin', 'main', success=False).returncode
        print('PASS: clone/fetch separation, fast-forward integration, divergent push rejection')
    print('All mechanics checks passed; temporary repositories removed.')


if __name__ == '__main__':
    main()
