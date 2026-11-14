"""Local QA preview without stale development caches. Serves the repository read-only."""
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class NoCache(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cache-Control','no-store')
        super().end_headers()

    def send_head(self):
        if 'If-Modified-Since' in self.headers:
            del self.headers['If-Modified-Since']
        return super().send_head()


if __name__=='__main__':
    repository=Path(__file__).resolve().parents[4]
    print('http://127.0.0.1:8766/research/machine-learning/2026-11-13-pytorch-demonstration.html',flush=True)
    ThreadingHTTPServer(('127.0.0.1',8766),partial(NoCache,directory=str(repository))).serve_forever()
