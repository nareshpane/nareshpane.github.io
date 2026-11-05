"""Rebuild the monograph's original small editorial SVG illustrations.

Python standard library only. Run from any directory. Artwork uses a transparent
canvas, a common 180 x 128 coordinate system, and consistent 1.7-unit outlines.
"""
from pathlib import Path
from html import escape
import xml.etree.ElementTree as ET

OUT = Path(__file__).resolve().parents[1] / 'svg' / 'illustrations'
INK = '#485d60'
GREEN = '#d9eddb'
BLUE = '#dce8f5'
ROSE = '#f3dadd'
LAVENDER = '#e6dff2'
PEACH = '#f6e4d5'
IVORY = '#fffaf2'
GRAY = '#e9ece9'
ACCENT = '#72a88b'


def path(d, fill='none', stroke=INK, extra=''):
    return f'<path d="{d}" fill="{fill}" stroke="{stroke}" {extra}/>'


def rect(x, y, w, h, fill=IVORY, r=3, stroke=INK):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}"/>'


def circle(x, y, r, fill=IVORY, stroke=INK):
    return f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}"/>'


def line(x1, y1, x2, y2, stroke=INK, extra=''):
    return f'<path d="M{x1} {y1}H{x2}" stroke="{stroke}" fill="none" {extra}/>' if y1 == y2 else path(f'M{x1} {y1}L{x2} {y2}', stroke=stroke, extra=extra)


def text(x, y, value, size=7, fill=INK, extra=''):
    return f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" stroke="none" {extra}>{escape(value)}</text>'


def cross(x, y, color=ACCENT, size=5):
    return path(f'M{x-size} {y}h{size*2}M{x} {y-size}v{size*2}', stroke=color, extra='stroke-width="3"')


def arrow(x1, y1, x2, y2, color=INK):
    # All connecting arrows point horizontally to keep small cards readable.
    return line(x1, y1, x2, y2, color) + path(f'M{x2-4} {y2-3}l4 3-4 3', stroke=color)


def blob(color=GREEN):
    return path('M29 35C42 11 93 9 120 22S162 53 155 82 115 115 79 111 22 106 18 80 19 52 29 35Z', color, 'none')


def ground():
    return line(17, 112, 163, 112, '#c5ceca')


def plant(x, y, color=GREEN):
    return path(f'M{x} {y}v-23m0 13q-15-3-10-11 12 0 10 11m0-7q12-4 12-11-13-1-12 11', color) + path(f'M{x-7} {y-1}h14l-2 12h-10Z', BLUE)


def head(x, y, skin=PEACH, hair=INK):
    return circle(x, y, 8, skin) + path(f'M{x-8} {y-1}q-2-13 8-11 11-2 9 10l-5-3-8 2Z', hair, hair) + path(f'M{x+1} {y+1}h1m-4 4q3 2 5-1', extra='stroke-width="1.1"')


def table(x, y, w=60, h=48, band=None, select=GREEN):
    s = rect(x, y, w, h, IVORY, 3)
    if band == 'row':
        s += rect(x+1, y+h/4+1, w-2, h/4-1, select, 0, 'none')
    if band == 'column':
        s += rect(x+1, y+1, w/3-1, h-2, select, 0, 'none')
    s += rect(x+1, y+1, w-2, h/4-1, BLUE, 0, 'none')
    for i in range(1, 4):
        s += line(x, y+h*i/4, x+w, y+h*i/4, '#a7b5b1')
    for i in range(1, 3):
        s += line(x+w*i/3, y, x+w*i/3, y+h, '#a7b5b1')
    return s


def cylinder(x, y, w=52, h=60, fill=BLUE):
    rx, ry = w/2, 7
    return path(f'M{x} {y+ry}v{h-14}a{rx} {ry} 0 0 0 {w} 0V{y+ry}', fill) + f'<ellipse cx="{x+rx}" cy="{y+ry}" rx="{rx}" ry="{ry}" fill="{fill}" stroke="{INK}"/>' + path(f'M{x} {y+h/2}a{rx} {ry} 0 0 0 {w} 0')


def screen(x, y, w=46, h=35, fill=IVORY):
    return rect(x,y,w,h,fill) + line(x+w/2,y+h,x+w/2,y+h+9) + line(x+w/2-9,y+h+9,x+w/2+9,y+h+9)


def server(x, y, w=40):
    s = ''
    for i in range(3):
        s += rect(x,y+i*16,w,13,IVORY,2) + rect(x+4,y+i*16+4,w-17,5,GREEN,0,'none') + circle(x+w-7,y+i*16+6.5,1.4,ACCENT,ACCENT)
    return s


def key(x, y, color=ACCENT):
    return circle(x,y,5,IVORY,color) + path(f'M{x+5} {y}h12m-4 0v4m-4-4v3',stroke=color,extra='stroke-width="2.2"')


def save(name, title, body, desc):
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 180 128" role="img" aria-labelledby="title desc">
<title id="title">{escape(title)}</title><desc id="desc">{escape(desc)}</desc>
<g fill="none" stroke="{INK}" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" font-family="Arial, sans-serif">{body}</g></svg>'''
    ET.fromstring(svg)
    assert len(svg.encode()) < 10000
    (OUT / f'{name}.svg').write_text(svg, encoding='utf-8')


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    # Clinical vignettes: the record moves through care, then into analysis.
    s = blob() + ground() + head(76,37)
    s += path('M62 52Q76 45 90 52l8 36H56Z', IVORY)
    s += path('M72 49l4 19 6-19', GREEN) + path('M65 54v18q11 12 20 0V55')
    s += circle(84,73,3,BLUE) + rect(94,56,27,35,PEACH) + line(100,65,115,65) + line(100,71,111,71)
    s += path('M63 66l-8 18 17 7m17-28 9 17-9 6', PEACH)
    s += rect(45,91,86,7,BLUE) + line(52,98,52,111) + line(125,98,125,111) + cross(39,42)
    save('physician','Physician consultation',s,'A clinician with a stethoscope records an encounter on a clipboard.')

    s = blob() + ground() + rect(24,24,58,68,IVORY)
    for y in [41,61,81]:
        s += line(25,y+9,81,y+9)
        for x in [31,47,63]:
            s += rect(x,y-6,10,14,GREEN,1) + rect(x+2,y-10,6,4,BLUE,1)
    s += head(116,40) + path('M103 54q12-7 25 0l7 31H97Z',IVORY) + path('M112 51l5 18 5-18',GREEN)
    s += path('M105 65l-12 8 11 6m23-14 13 9-12 6',PEACH)
    s += rect(69,82,84,25,BLUE) + cross(110,95,ACCENT,6) + rect(134,63,17,17,IVORY) + cross(142.5,71.5,ACCENT,3)
    save('pharmacy','Pharmacy dispensing',s,'Medicine shelves, a pharmacist, and a dispensing counter represent a pharmacy transaction.')

    s = blob(BLUE) + ground() + rect(22,32,40,61,IVORY) + cross(42,46,'#b8757b',7)
    s += path('M23 63h38m-19 0v30',stroke='#a7b5b1') + head(113,40)
    s += path('M101 53q12-5 24 0l5 32H97Z',BLUE) + path('M101 82l-5 25m23-25 8 25',stroke=INK,extra='stroke-width="6"')
    s += path('M102 62 89 76 74 72m49-10 9 17-13 6',PEACH)
    s += rect(56,80,47,10,IVORY) + line(62,91,62,106) + line(97,91,97,106) + circle(62,108,3,GRAY) + circle(97,108,3,GRAY)
    s += path('M61 78q9-13 19-2l17 2',ROSE) + circle(65,71,5,PEACH)
    save('emergency','Emergency encounter',s,'An emergency clinician stands beside a patient trolley and a marked emergency entrance.')

    s = blob() + ground() + rect(27,25,49,31,IVORY) + line(51,26,51,55) + rect(121,69,27,37,IVORY)
    s += line(125,81,143,81) + line(125,93,143,93) + plant(134,57)
    s += path('M36 87h81v12H36Z',BLUE) + path('M36 69v37m81-37v37') + line(41,99,41,110) + line(112,99,112,110)
    s += rect(42,70,22,12,IVORY,5) + circle(59,70,7,PEACH) + path('M66 73q18-9 29 3l15 10H64Z',GREEN)
    s += path('M90 78v9M27 69V34m-5 5h10m-10 0v15h10V39',stroke='#7c9993')
    save('hospital','Hospital inpatient stay',s,'A patient rests in a hospital bed beside an intravenous stand and bedside cabinet.')

    s = blob(BLUE) + ground() + server(70,30,48) + table(22,35,32,29,'row',PEACH) + table(24,76,32,29,'column',GREEN)
    s += arrow(56,49,66,49) + arrow(58,91,66,91) + screen(131,68,30,27)
    s += path('M136 88v-6m7 6V76m7 12v-9',stroke=ACCENT,extra='stroke-width="4"') + arrow(120,70,128,70)
    s += line(77,83,77,102) + line(77,102,117,102) + line(117,102,117,83)
    save('warehouse','Health data warehouse',s,'Source tables feed a central server stack that supplies an analytical dashboard.')

    s = blob(LAVENDER) + ground() + screen(86,28,67,48)
    s += path('M94 63l11-11 10 4 12-15 17 5',stroke=ACCENT,extra='stroke-width="2.5"') + path('M97 67v-8m12 8v-5m12 5V54m12 13V52',stroke='#a696bb',extra='stroke-width="4"')
    s += head(55,46) + path('M43 60q14-7 25 0l8 30H39Z',ROSE) + path('M67 63l16 14 16-7m-54-3-6 18 23 5',PEACH)
    s += rect(73,86,84,6,IVORY) + line(79,93,79,111) + line(149,93,149,111)
    s += path('M47 91v18h13m6-18 1 18h12',stroke=INK,extra='stroke-width="5"') + plant(25,93)
    save('analyst','Analyst interpreting evidence',s,'An analyst examines a trend and bar chart on a desktop display.')

    # Foundational concepts are intentionally simpler than clinical vignettes.
    s = blob(PEACH) + ground() + table(35,28,110,68,'row',GREEN)
    s += path('M24 54h9m-4-4 4 4-4 4',stroke=ACCENT,extra='stroke-width="2.3"')
    for x in [49,85,122]:
        s += circle(x,53.5,2.4,ACCENT,ACCENT)
    save('row','One row, one record',s,'A green horizontal band selects one complete record across three fields.')

    s = blob(BLUE) + ground() + table(35,28,110,68,'column',LAVENDER)
    s += path('M53 15v10m-4-4 4 4 4-4',stroke='#8c78a9',extra='stroke-width="2.3"')
    for y in [54,70,87]:
        s += circle(53,y,2.4,'#a696bb','#a696bb')
    save('column','One column, one attribute',s,'A lavender vertical band selects the same attribute across every record.')

    s = blob() + ground() + table(35,28,110,68)
    s += text(53,40,'ID',8,extra='text-anchor="middle"') + text(90,40,'DAY',8,extra='text-anchor="middle"') + text(127,40,'SITE',8,extra='text-anchor="middle"')
    for y in [56,73,90]:
        s += line(44,y,62,y,'#7c9993') + line(81,y,99,y,'#7c9993') + line(119,y,135,y,'#7c9993')
    save('relation','A relation organizes records',s,'A complete table has named columns and multiple records.')

    s = blob(LAVENDER) + ground() + table(23,27,51,38) + table(106,27,51,38) + table(65,80,51,29)
    s += path('M74 46h16V94H65M106 46H90m0 20v14',stroke='#8c78a9') + circle(90,66,3,LAVENDER)
    save('schema','A schema connects table definitions',s,'Three related table structures share a defined set of connections.')

    s = blob(BLUE) + ground() + cylinder(42,27,68,72,BLUE) + table(85,56,61,46,'row',GREEN)
    save('database','A database stores organized data',s,'A storage cylinder and a structured table represent a managed relational database.')

    s = blob() + ground() + cylinder(78,32,54,69,GREEN) + table(15,28,39,31,'row',PEACH) + table(15,75,39,31,'column',BLUE)
    s += arrow(57,44,74,44) + arrow(57,91,74,91) + table(137,47,31,39,'row',LAVENDER) + arrow(134,66,137,66)
    save('integrated-warehouse','An integrated warehouse',s,'Two differently organized source tables feed a common store and a consistent analytical table.')

    # Key roles: unique row, cross-table link, and external business identity.
    s = blob(PEACH) + ground() + table(35,39,110,62,'column',ROSE)
    s += key(61,24,color='#a66f69') + text(54,52,'PK',8,extra='text-anchor="middle"')
    for i,y in enumerate([65,81,96],1):
        s += text(54,y,str(i),8,extra='text-anchor="middle"')
    save('primary-key','Primary key identifies one row',s,'A key marks the identifier column; each record has a different value.')

    s = blob(BLUE) + ground() + table(17,29,62,53,'column',ROSE) + table(105,56,62,53,'column',GREEN)
    s += text(27,40,'PK',7) + text(115,67,'FK',7) + path('M105 70H92V95H28V82',stroke=ACCENT,extra='stroke-width="2"')
    s += path('M24 86L28 82L32 86',stroke=ACCENT) + circle(28,95,2.5,GREEN)
    save('foreign-key','Foreign key links related records',s,'The foreign key in a second table points to the first table’s primary key.')

    s = blob(LAVENDER) + ground() + rect(21,34,93,59,IVORY,5) + circle(39,51,7,PEACH)
    s += path('M28 73q0-12 11-12t11 12',BLUE) + text(57,51,'BK',9) + text(28,84,'CLIENT_SYN_001',7.6)
    s += arrow(118,65,132,65) + rect(137,46,29,39,ROSE) + text(151.5,59,'PK',8,extra='text-anchor="middle"') + text(151.5,75,'1',10,extra='text-anchor="middle"')
    save('business-key','Business identity and warehouse identity',s,'A synthetic business identifier CLIENT_SYN_001 maps to an independent numeric warehouse primary key, 1.')

    # Architecture stages share the same canvas and visual vocabulary.
    s = blob(BLUE) + ground() + server(21,40,37) + server(123,40,37)
    s += line(61,66,120,66,ACCENT,extra='stroke-dasharray="3 4"') + path('M75 46l15-6 15 6v16q-1 17-15 24-14-7-15-24Z',GREEN)
    s += rect(83,58,14,13,IVORY) + path('M86 58v-5a4 4 0 0 1 8 0v5') + circle(90,64,1.2,INK,INK)
    save('secure-transmission','Secure data transmission',s,'An encrypted link connects two server stacks; a shield and padlock indicate protected transport.')

    s = blob(PEACH) + ground() + table(22,28,43,32,'row',BLUE) + table(22,72,43,32,'column',LAVENDER)
    s += arrow(69,45,89,45) + arrow(69,87,89,87) + rect(95,32,61,69,IVORY)
    s += path('M102 44h47M102 59h47M102 74h47M102 89h47',stroke='#a7b5b1')
    s += path('M139 52l3 3 6-7m-9 19 3 3 6-7m-9 19 3 3 6-7',stroke=ACCENT)
    save('staging','Staging and validation',s,'Incoming extracts enter a staging area, where individual batches receive validation checks.')

    s = blob() + ground() + table(18,28,39,30,'row',PEACH) + table(18,74,39,30,'column',BLUE)
    s += path('M60 43h11l18 22-18 24H60',stroke='#7c9993') + path('M79 54l11-11 21 22-21 22-11-11 11-11Z',LAVENDER)
    s += arrow(113,65,124,65) + table(130,40,36,53,'row',GREEN) + path('M138 80l5 5 13-15',stroke=ACCENT,extra='stroke-width="2.5"')
    save('integration','Integration and harmonization',s,'Two source structures converge through a transformation step into one validated structure.')

    s = blob(LAVENDER) + ground() + cylinder(22,39,46,59,BLUE)
    s += path('M70 69h17V33h25M87 69h25M87 69v36h25',stroke='#8c78a9')
    for y,fill in [(20,GREEN),(56,PEACH),(92,BLUE)]:
        s += rect(115,y,45,25,fill) + path(f'M123 {y+17}v-5m10 5V{y+7}m10 10v-8m9 8V{y+5}',stroke='#7c9993',extra='stroke-width="3"')
    save('marts','Purpose-specific data marts',s,'A shared warehouse supplies three distinct subject-oriented analytical views.')

    s = blob(PEACH) + ground() + table(19,29,53,73,'row',GREEN)
    s += text(28,55,'A',8) + text(28,73,'A',8) + text(28,92,'B',8)
    s += path('M75 48h12l12 18-12 21H75',stroke=ACCENT) + arrow(100,66,111,66)
    s += table(116,42,48,48,'row',GREEN) + text(124,66,'A',8) + text(124,83,'B',8)
    s += path('M143 61h10m-5-4v8M143 79h10m-5-4v8',stroke=ACCENT)
    save('aggregate','Aggregation reduces rows to groups',s,'Three source records converge into two group totals; repeated A records become a single group.')

    s = blob(BLUE) + ground() + table(24,28,103,73,'row',GREEN)
    s += rect(128,28,28,73,LAVENDER,3) + line(128,46.25,156,46.25) + line(128,64.5,156,64.5) + line(128,82.75,156,82.75)
    s += path('M18 47h-7v35h7',stroke='#8c78a9',extra='stroke-width="2.2"')
    for y in [56,74,93]:
        s += line(107,y,122,y,ACCENT) + text(142,y+3,str([56,74,93].index(y)+1),9,extra='text-anchor="middle"')
    save('analytic-window','A window calculation preserves rows',s,'A bracket selects a window of records, while an additional lavender result column keeps every original record.')

    s = blob() + ground() + rect(75,21,31,18,PEACH)
    s += text(90.5,33,'50',9,extra='text-anchor="middle"') + path('M90 39v12H42v12m48-12h48v12')
    s += rect(23,64,39,18,IVORY) + rect(119,64,39,18,GREEN)
    s += text(42.5,76,'10 · 30',8,extra='text-anchor="middle"') + text(138.5,76,'60 · 90',8,extra='text-anchor="middle"')
    s += path('M138 83v13h-16',stroke=ACCENT,extra='stroke-width="2.4"') + rect(78,88,43,20,BLUE) + text(99.5,101,'ROW',8,extra='text-anchor="middle"')
    s += path('M97 43l36 18m-5-1 5 1-1-5',stroke=ACCENT,extra='stroke-width="2.4"')
    save('index','An index narrows a lookup',s,'A branching search structure follows the selected right branch to locate a record without scanning every row.')

    s = blob(BLUE) + ground() + rect(16,39,43,54,IVORY)
    s += text(23,53,'SQL',9) + line(23,63,50,63,'#a7b5b1') + line(23,70,45,70,'#a7b5b1') + line(23,77,49,77,'#a7b5b1')
    s += arrow(63,65,74,65) + cylinder(79,37,34,58,GREEN) + arrow(117,65,128,65) + table(133,41,32,48,'row',PEACH)
    save('query','A query returns a result set',s,'A SQL worksheet is processed by the database, which returns a tabular result.')

    s = blob(LAVENDER) + ground() + table(30,29,91,73,'column',GREEN)
    for i,y in enumerate([57,75,94],1):
        s += text(45,y,str(i),9,extra='text-anchor="middle"')
    s += path('M128 43l21-7 15 7v22q0 18-18 30-18-12-18-30Z',IVORY)
    s += path('M136 64l8 8 13-17',stroke=ACCENT,extra='stroke-width="3"')
    save('unique','Uniqueness guards against duplicates',s,'Different identifier values in the selected column are protected by a check-mark shield.')


if __name__ == '__main__':
    build()
    files = sorted(OUT.glob('*.svg'))
    print(f'Rebuilt and parsed {len(files)} original SVG illustrations in {OUT}')
    print(f'Largest SVG: {max(p.stat().st_size for p in files):,} bytes')
