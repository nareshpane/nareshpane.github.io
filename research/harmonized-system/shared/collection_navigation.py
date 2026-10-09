"""Maintain the collection's static top navigation; never touch research content.

Run this file from any directory to update the seven collection page headers.
The teaching-page renderer also uses render_top_nav to retain this convention.
"""
from pathlib import Path
from html import escape
import re

COLLECTION=Path(__file__).resolve().parents[1]
NAV_ITEMS=[
    ('harmonized-system-index.html','Collection Home'),
    ('harmonized-system-canada.html','Page 1: HS Classification'),
    ('section-338-hs4-hs6-exposure-canada.html','Page 2: Section 338 Exposure'),
    ('canada-and-provinces-trade-by-hs.html','Page 3: Provincial Exports'),
    ('alberta-trade-by-hs.html','Page 4: Alberta Export Atlas'),
    ('machine-learning-trade-sector-prediction.html','Page 5: Machine Learning'),
    ('details-of-machine-learning-models.html','Page 6: Model Details'),
]
STYLESHEET='<link rel="stylesheet" href="shared/collection-navigation.css">'
NAV_PATTERN=r'<nav\b[^>]*class="[^"]*collection-nav[^"]*"[^>]*>.*?</nav>'

def render_top_nav(current_page):
    links=['<a href="'+href+'"'+(' aria-current="page"' if href==current_page else '')+'>'+escape(label)+'</a>' for href,label in NAV_ITEMS]
    return '<nav class="collection-nav collection-top-nav" aria-label="Collection navigation">'+''.join(links)+'</nav>'

def update_pages():
    for name,_ in NAV_ITEMS:
        page=COLLECTION/name
        assert page.is_file(),f'Missing collection page: {name}'
        # Preserve original line endings and all bytes outside the header changes.
        original=page.read_bytes().decode('utf-8')
        updated,count=re.subn(NAV_PATTERN,lambda _:render_top_nav(name),original,count=1,flags=re.S)
        assert count==1,f'Missing top collection navigation: {name}'
        if STYLESHEET not in updated:
            assert '</head>' in updated
            updated=updated.replace('</head>',STYLESHEET+'</head>',1)
        page.write_bytes(updated.encode('utf-8'))
        print('Updated top navigation:',name)

if __name__=='__main__':update_pages()
