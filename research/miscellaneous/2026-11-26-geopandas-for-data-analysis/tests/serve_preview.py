"""Serve only the research HTML and its public assets on loopback."""
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
SLUG = ROOT.name
PREFIX = '/research/miscellaneous/'


class Preview(SimpleHTTPRequestHandler):
    def relative(self):
        path = unquote(urlsplit(self.path).path)
        if not path.startswith(PREFIX):
            return None
        path = path.removeprefix(PREFIX)
        file = (ROOT.parent / path).resolve()
        if file == ROOT.with_suffix('.html'):
            return file
        if not file.is_relative_to(ROOT) or not file.is_file():
            return None
        parts = file.relative_to(ROOT).parts
        if any(part.startswith('.') or part in {'raw', '__pycache__'} for part in parts):
            return None
        return file

    def send_head(self):
        self.asset = self.relative()
        if self.asset is None:
            self.send_error(404, 'Preview serves only the requested page and public project assets')
            return None
        return super().send_head()

    def translate_path(self, _):
        return str(self.asset)


if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    with ThreadingHTTPServer(('127.0.0.1', port), partial(Preview, directory=ROOT)) as server:
        print(f'http://127.0.0.1:{port}{PREFIX}{SLUG}.html', flush=True)
        server.serve_forever()

