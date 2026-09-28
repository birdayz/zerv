"""Extract downloaded arXiv HTML for source inspection; no production dependency.
Run: tools/py docs/research/2026-09-28-kv-tier-papers/extract_html.py FILE.html ...
"""
from html.parser import HTMLParser
from pathlib import Path
import sys

class Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.skip = 0
    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style'): self.skip += 1
        if tag in ('p', 'section', 'h1', 'h2', 'h3', 'h4', 'li', 'tr'): self.parts.append('\n')
    def handle_endtag(self, tag):
        if tag in ('script', 'style'): self.skip -= 1
        if tag in ('p', 'section', 'h1', 'h2', 'h3', 'h4', 'li', 'tr'): self.parts.append('\n')
    def handle_data(self, data):
        if not self.skip: self.parts.append(data)

for filename in sys.argv[1:]:
    path = Path(filename)
    parser = Text()
    parser.feed(path.read_text())
    lines = [' '.join(s.split()) for s in ''.join(parser.parts).splitlines()]
    path.with_suffix('.txt').write_text('\n'.join(s for s in lines if s) + '\n')
