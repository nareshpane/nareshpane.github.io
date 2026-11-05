"""Build original dimension, Oracle lifecycle, and ETL teaching mini-scenes.

Uses the established 180 x 128 editorial family. Standard library only; run this
file directly. It writes only the 26 named assets below, leaving earlier SVGs
unchanged. No raster inputs, geographic boundary data, or external SVG resources.
"""
from pathlib import Path
from html import escape
import xml.etree.ElementTree as ET

from build_illustrations import (
    INK, GREEN, BLUE, ROSE, LAVENDER, PEACH, IVORY, GRAY, ACCENT,
    path, rect, circle, line, text, arrow, blob, ground, head,
    table, cylinder, screen, server, key, cross, save, OUT,
)

CREATED = []


def write(name, title, scene, description):
    save(name, title, scene, description)
    CREATED.append(name)


def check(x, y, color=ACCENT):
    return path(f'M{x} {y}l4 4 9-11', stroke=color, extra='stroke-width="2.5"')


def paper(x, y, w=37, h=48, fill=IVORY, label=None):
    s = rect(x, y, w, h, fill)
    if label:
        s += text(x+6, y+13, label, 8)
    for i, fraction in enumerate([.72, .53, .65]):
        s += line(x+6, y+23+i*7, x+w*fraction, y+23+i*7, '#9eada8')
    return s


def gear(x, y, color=LAVENDER):
    # Simplified control wheel, with four visible pairs of teeth.
    return path(f'M{x-5} {y-16}h10v5l5 3 5-3 5 9-5 3v6l5 3-5 9-5-3-5 3v5h-10v-5l-5-3-5 3-5-9 5-3v-6l-5-3 5-9 5 3 5-3Z', color) + circle(x,y,6,IVORY)


def build():
    OUT.mkdir(parents=True, exist_ok=True)

    s = blob(BLUE) + ground() + rect(29,29,92,72,IVORY,5)
    s += rect(30,30,90,17,BLUE,4,'none') + line(29,47,121,47)
    s += path('M47 23v12m56-12v12', stroke=INK, extra='stroke-width="3"')
    for y in [59,75,91]:
        for x in [46,67,88,109]:
            s += rect(x-5,y-5,10,10,GREEN if (x,y)==(67,75) else GRAY,1,'none')
    s += rect(127,46,34,38,PEACH) + text(144,61,'2026',8,extra='text-anchor="middle"') + text(144,75,'Q4',9,extra='text-anchor="middle"')
    s += line(123,66,127,66,ACCENT)
    write('dimension-date','Date dimension',s,'A calendar highlights one day and connects it to a year and quarter, illustrating attributes in a date dimension.')

    s = blob(GREEN) + ground() + rect(61,21,58,22,BLUE) + text(90,35,'PROVINCE',7,extra='text-anchor="middle"')
    s += path('M90 43v12H46v9m44-9h44v9', stroke='#7c9993')
    s += rect(18,65,57,20,IVORY) + text(46.5,78,'DISTRICT',7,extra='text-anchor="middle"')
    s += rect(106,65,57,20,IVORY) + text(134.5,78,'DISTRICT',7,extra='text-anchor="middle"')
    s += path('M46 85v12H31m15 0h28m60-12v12h-16m16 0h16',stroke='#7c9993')
    for x in [23,66,110,145]:
        s += rect(x,98,16,11,PEACH,2)
    write('dimension-geography','Geographic hierarchy',s,'An abstract province branches into districts and municipalities; the diagram represents hierarchy without depicting a real national boundary.')

    s = blob(PEACH) + ground() + rect(28,36,59,67,IVORY,5)
    s += rect(38,24,39,13,BLUE,3) + rect(35,57,45,24,GREEN,2,'none') + cross(57.5,69,ACCENT,6)
    s += rect(106,37,48,62,IVORY) + text(113,51,'PRODUCT',7)
    s += text(113,65,'Ingredient',6.8) + text(113,78,'Strength',6.8) + text(113,91,'Form',6.8)
    s += line(89,69,104,69,ACCENT)
    write('dimension-product','Product dimension',s,'A medicine bottle links to a catalogue card containing ingredient, strength, and dosage-form attributes.')

    s = blob(LAVENDER) + ground() + rect(23,27,69,76,IVORY,5)
    s += head(57,47) + path('M39 80q0-19 18-19t18 19',GREEN) + text(57,95,'CLIENT',7,extra='text-anchor="middle"')
    s += rect(104,35,53,60,IVORY,5) + head(130,52) + path('M115 81q0-17 15-17t15 17',BLUE) + cross(130,76,ACCENT,4)
    s += text(130,91,'PRESCRIBER',6,extra='text-anchor="middle"')
    write('dimension-people','People dimensions',s,'Two separate identity cards distinguish a client from a clinical prescriber, with descriptive attributes attached to each identity.')

    s = blob(BLUE) + ground() + rect(39,34,80,76,IVORY,3) + rect(29,32,100,10,GREEN)
    s += cross(79,23,ACCENT,7) + rect(68,85,22,25,BLUE)
    for x in [49,94]:
        for y in [51,69,88]:
            s += rect(x,y,14,10,PEACH,1)
    s += path('M138 54c-11 0-14 13-7 21l7 10 7-10c7-8 4-21-7-21Z',ROSE) + circle(138,66,4,IVORY)
    write('dimension-facility','Facility dimension',s,'A health facility with a clinical cross and location marker represents a stable facility identity and its descriptive geography.')

    s = blob(ROSE) + ground() + paper(24,28,63,75,IVORY,'DIAGNOSIS')
    s += path('M33 63h10l5-12 8 25 7-15h14',stroke='#a66f69')
    s += rect(107,37,49,24,BLUE) + text(131.5,52,'CODE',8,extra='text-anchor="middle"')
    s += rect(107,78,49,24,GREEN) + text(131.5,93,'GROUP',8,extra='text-anchor="middle"')
    s += path('M89 64h9V49h9m-9 15v26h9',stroke='#7c9993')
    write('dimension-diagnosis','Diagnosis classification',s,'A clinical diagnosis record connects to a coded term and a broader classification group, separating clinical meaning from an event occurrence.')

    # Query lifecycle: submitted text, checking, plan choice, execution, results.
    s = blob(BLUE) + ground() + screen(20,33,63,45)
    s += text(27,48,'SELECT',8) + line(27,58,72,58,'#9eada8') + line(27,65,60,65,'#9eada8')
    s += arrow(87,56,111,56,ACCENT) + server(119,34,40)
    s += path('M29 97h44l8 7H21Z',PEACH)
    write('query-submit','Submit a SQL statement',s,'A SQL worksheet sends statement text to the database server over a client connection.')

    s = blob(PEACH) + ground() + paper(23,29,58,72,IVORY,'SELECT')
    s += line(29,47,74,47,'#b8757b') + circle(111,54,20,BLUE) + path('M126 70l19 23',stroke=INK,extra='stroke-width="5"')
    s += path('M101 53l6 6 14-16',stroke=ACCENT,extra='stroke-width="3"')
    s += text(111,96,'syntax · names',6.7,extra='text-anchor="middle"')
    write('query-parse','Parse and validate SQL',s,'A magnifying glass checks statement syntax and object names; the check symbol represents successful validation.')

    s = blob(LAVENDER) + ground() + rect(22,29,42,22,IVORY) + text(43,43,'SQL',9,extra='text-anchor="middle"')
    s += path('M64 40h21v-9h18m-18 9v27h18m-18-27v65h18',stroke='#9eada8')
    for y,fill,label in [(20,GRAY,'A'),(56,GREEN,'B'),(94,GRAY,'C')]:
        s += rect(105,y,48,20,fill) + text(114,y+13,label,8) + line(128,y+7,145,y+7,'#9eada8') + line(128,y+13,140,y+13,'#9eada8')
    s += check(136,83)
    write('query-optimize','Choose an execution plan',s,'Several candidate plans branch from one SQL statement; a green middle plan is selected using optimizer estimates rather than a universal rule.')

    s = blob(GREEN) + ground() + table(21,33,53,68,'row',BLUE)
    s += arrow(78,67,94,67,ACCENT) + gear(115,67,PEACH) + path('M115 45v-9h20v9m0 47v13h-20V93',stroke='#9eada8')
    s += rect(141,51,24,32,IVORY) + path('M147 59h12m-12 8h8m-8 8h12',stroke='#9eada8')
    write('query-execute','Execute the chosen plan',s,'Rows enter an execution operator, which reads and processes data according to the selected plan.')

    s = blob(BLUE) + ground() + cylinder(23,34,42,64,GREEN)
    s += arrow(69,63,95,63,ACCENT) + screen(104,32,58,54)
    s += table(109,38,48,40,'row',PEACH) + circle(83,50,2.5,BLUE) + circle(88,76,2.5,BLUE)
    s += text(83,96,'row batches',7,extra='text-anchor="middle"')
    write('query-fetch','Fetch result rows',s,'The database returns batches of result rows to a client display, which shows the resulting table.')

    # Transaction lifecycle separates recoverability from later file writing.
    s = blob(PEACH) + ground() + paper(20,30,52,71,IVORY,'UPDATE')
    s += arrow(76,64,95,64,ACCENT) + table(101,32,59,70,'row',GREEN)
    s += path('M143 25l10-10 7 7-10 10-10 3Z',ROSE) + path('M153 15l7 7')
    write('txn-dml','Change rows with DML',s,'An UPDATE statement changes a selected table row; the pencil indicates a transaction modification.')

    s = blob(LAVENDER) + ground() + table(65,42,49,48,'row',GREEN)
    s += paper(14,28,37,68,BLUE,'UNDO') + paper(130,28,37,68,PEACH,'REDO')
    s += path('M65 58H53m5-4-5 4 5 4',stroke='#8c78a9') + arrow(115,74,128,74,'#b68c66')
    s += text(32.5,107,'prior values',6.6,extra='text-anchor="middle"') + text(148.5,107,'change record',6.6,extra='text-anchor="middle"')
    write('txn-undo-redo','Record undo and redo',s,'A changed row is accompanied by undo information for prior values and redo information describing the change; the two records serve different recovery purposes.')

    s = blob(BLUE) + ground() + rect(24,25,132,78,IVORY,6) + text(33,40,'BUFFER CACHE',8)
    for i,(x,y) in enumerate([(35,49),(74,49),(113,49),(35,75),(74,75),(113,75)]):
        s += rect(x,y,30,21,GREEN if i in [1,4] else BLUE,2)
    s += circle(100,51,3,ROSE,'#a66f69') + circle(100,77,3,ROSE,'#a66f69')
    s += text(90,115,'modified blocks in memory',7,extra='text-anchor="middle"')
    write('txn-cache','Modified database blocks in memory',s,'The buffer cache holds several data blocks; two green blocks with rose markers indicate modified buffers awaiting later database-file writes.')

    s = blob(GREEN) + ground() + paper(19,31,50,68,IVORY,'COMMIT')
    s += arrow(73,64,89,64,ACCENT) + rect(95,33,64,66,PEACH,4)
    s += text(127,47,'REDO LOG',8,extra='text-anchor="middle"') + path('M104 58h46m-46 9h46m-46 9h31',stroke='#9eada8') + check(126,87)
    s += text(90,114,'acknowledge after redo is durable',6.9,extra='text-anchor="middle"')
    write('txn-commit','Commit and durable redo',s,'A commit request leads to durable redo and acknowledgement under normal synchronous commit semantics; it does not imply that every changed data block has already reached a data file.')

    s = blob(PEACH) + ground() + rect(17,35,46,60,IVORY) + text(40,49,'CACHE',7,extra='text-anchor="middle"')
    s += rect(24,58,32,12,GREEN,1) + rect(24,76,32,12,GREEN,1)
    s += arrow(66,66,81,66,ACCENT) + circle(99,66,15,LAVENDER) + text(99,69,'DBW',8,extra='text-anchor="middle"')
    s += arrow(117,66,128,66,ACCENT) + cylinder(134,38,31,60,BLUE)
    write('txn-writer','Database writer writes modified buffers',s,'The database writer moves modified buffers from the buffer cache into database files independently of foreground commit acknowledgement. Eligible modified buffers may be written before or after a transaction commits; redo must satisfy write-ahead requirements.')

    # ETL mini-scenes describe a governed batch pipeline, not a universal order.
    s = blob(BLUE) + ground() + server(20,32,43)
    s += arrow(67,63,92,63,ACCENT) + paper(99,25,47,61,IVORY,'BATCH') + paper(111,43,47,61,PEACH,'COPY')
    s += text(43,101,'source',7,extra='text-anchor="middle"')
    write('etl-extract','Extract source records',s,'A source server supplies copied batch records while the original operational source remains distinct.')

    s = blob(PEACH) + ground() + table(16,31,55,67,'row',BLUE)
    s += path('M74 64h14V43h16m-16 21v26h16',stroke='#9eada8') + rect(108,27,53,31,GREEN) + check(126,43)
    s += rect(108,76,53,31,ROSE) + path('M128 84l12 12m0-12-12 12',stroke='#a66f69')
    s += text(134,70,'check / quarantine',6.5,extra='text-anchor="middle"')
    write('etl-validate','Validate and quarantine',s,'Incoming records split into accepted records and quarantined failures after structural and domain checks.')

    s = blob(LAVENDER) + ground() + table(20,26,52,76,'row',ROSE)
    for y,label in [(57,'A'),(76,'A'),(95,'B')]:
        s += text(28,y,label,8)
    s += arrow(77,65,99,65,ACCENT) + table(106,39,54,53,'row',GREEN)
    s += text(115,66,'A',8) + text(115,84,'B',8) + path('M43 65h22',stroke='#a66f69',extra='stroke-width="2"')
    write('etl-deduplicate','Resolve repeated records',s,'Two records with the same designated event identity are resolved into one retained record; a distinct B record remains separate.')

    s = blob(GREEN) + ground() + paper(17,30,44,68,PEACH,'SOURCE')
    s += arrow(65,65,76,65,ACCENT) + gear(97,65,LAVENDER) + arrow(119,65,129,65,ACCENT)
    s += paper(134,30,33,68,BLUE,'STD') + text(39,113,'codes and types',6.5,extra='text-anchor="middle"')
    write('etl-transform','Standardize and transform',s,'Source representations pass through a transformation wheel to standardized codes, types, and governed identifiers.')

    s = blob(BLUE) + ground() + table(18,34,52,64,'row',GREEN)
    s += arrow(75,65,101,65,ACCENT) + cylinder(107,26,55,79,BLUE)
    s += path('M124 71l9 9 15-19',stroke=ACCENT,extra='stroke-width="3"') + text(44,111,'validated batch',6.5,extra='text-anchor="middle"')
    write('etl-load','Load warehouse structures',s,'A validated batch is loaded into warehouse storage after the required checks and transformations.')

    s = blob(PEACH) + ground() + rect(20,30,58,69,IVORY) + rect(102,30,58,69,IVORY)
    s += text(49,43,'SOURCE',8,extra='text-anchor="middle"') + text(131,43,'TARGET',8,extra='text-anchor="middle"')
    for y,label in [(59,'Rows'),(74,'Totals'),(89,'Rejects')]:
        s += text(27,y,label,7) + line(53,y-2,70,y-2,'#9eada8') + text(109,y,label,7) + line(135,y-2,152,y-2,'#9eada8')
    s += path('M83 60h14m-14 8h14',stroke=ACCENT,extra='stroke-width="2.5"') + check(83,91)
    write('etl-reconcile','Reconcile batch results',s,'Source and target row counts, totals, and rejected-record counts are compared to explain the outcome of a warehouse batch.')

    s = blob(BLUE) + ground() + rect(22,24,136,80,IVORY,6) + text(90,39,'INSTANCE MEMORY',8,extra='text-anchor="middle"')
    for x,y,w,h,fill,label in [(32,48,66,45,BLUE,'BUFFER'),(106,48,41,20,PEACH,'REDO'),(106,75,41,18,LAVENDER,'SHARED')]:
        s += rect(x,y,w,h,fill) + text(x+w/2,y+h/2+3,label,6.8,extra='text-anchor="middle"')
    for x in [40,58,76]:
        s += rect(x,67,12,17,GREEN,1)
    write('instance-memory','Oracle instance memory',s,'A shared memory enclosure distinguishes buffer-cache blocks, the redo log buffer, and shared statement structures; this is a conceptual view rather than an exhaustive memory map.')

    s = blob(LAVENDER) + ground() + rect(22,23,136,82,IVORY,6) + text(90,38,'INSTANCE PROCESSES',8,extra='text-anchor="middle"')
    for x,label,fill in [(48,'DBW',BLUE),(90,'LGWR',PEACH),(132,'CKPT',GREEN)]:
        s += circle(x,66,16,fill) + text(x,69,label,7,extra='text-anchor="middle"') + line(x,83,x,96,'#9eada8')
    s += path('M48 96h84',stroke='#9eada8')
    write('instance-processes','Oracle instance background processes',s,'Database writer, log writer, and checkpoint process symbols identify distinct coordinated background responsibilities within an Oracle instance.')

    s = blob(GREEN) + ground()
    for x,fill,label in [(20,BLUE,'DATA'),(72,PEACH,'REDO'),(124,LAVENDER,'CONTROL')]:
        s += cylinder(x,32,38,66,fill) + text(x+19,111,label,7,extra='text-anchor="middle"')
    write('database-files','Physical Oracle database files',s,'Separate persistent file groups represent data files, online redo log files, and control files; memory and processes are not depicted as database files.')

    # Opening vignette uses a double-size canvas for a richer compact scene.
    # Every shape is original vector geometry; no external references or logos.
    s = path('M29 58C54 23 106 25 143 33S212 10 266 30 342 62 341 117 325 224 273 229 216 208 168 224 80 242 43 211 10 94 29 58Z',PEACH,'none')
    s += path('M106 120C125 89 193 110 222 135S263 181 242 209 146 234 113 207 92 148 106 120Z',BLUE,'none')
    s += line(18,233,342,233,'#c5ceca')
    # A clinical identity record provides an intuitive source-system starting point.
    s += rect(20,66,73,111,IVORY,7) + rect(37,57,39,16,GREEN,4)
    s += head(56,95) + path('M38 130q0-19 18-19t18 19',BLUE)
    s += cross(78,80,ACCENT,5) + text(30,146,'CARE RECORD',7.8)
    s += line(30,157,81,157,'#9eada8') + line(30,166,66,166,'#9eada8')
    s += arrow(98,105,120,105,ACCENT)
    # Central store and a pair of actual key-role table structures.
    s += cylinder(126,47,66,80,GREEN) + text(159,91,'DATABASE',8,extra='text-anchor="middle"')
    s += path('M159 128v10h-19v10m19-10h61v10',stroke='#9eada8')
    for x,w,label,fill,role in [(109,59,'CLIENT',ROSE,'PK'),(190,62,'DISPENSE',GREEN,'FK')]:
        s += rect(x,151,w,61,IVORY,4)
        s += rect(x+1,152,w-2,15,BLUE,3,'none') + rect(x+1,168,20,43,fill,0,'none')
        s += text(x+w/2,163,label,7,extra='text-anchor="middle"')
        s += line(x,168,x+w,168,'#9eada8') + line(x,190,x+w,190,'#9eada8') + line(x+22,168,x+22,212,'#9eada8')
        s += text(x+11,179,role,7,extra='text-anchor="middle"') + text(x+11,202,'1',8,extra='text-anchor="middle"')
        s += line(x+29,180,x+w-7,180,'#9eada8') + line(x+29,201,x+w-7,201,'#9eada8')
    # The lower bracket points from the foreign-key value to its parent key.
    s += path('M201 214v10h-81v-10m-4 4 4-4 4 4',stroke=ACCENT,extra='stroke-width="2.2"')
    # A readable worksheet bridges storage to an analytical result.
    s += arrow(197,84,217,84,ACCENT) + rect(222,41,115,84,IVORY,6)
    s += rect(223,42,113,14,LAVENDER,5,'none') + text(231,52,'SQL WORKSHEET',7.6)
    s += text(232,73,'SELECT ...',10) + text(232,90,'FROM ...',10)
    s += line(232,105,322,105,'#9eada8') + line(232,114,296,114,'#9eada8')
    s += path('M304 128v17m-4-4 4 4 4-4',stroke=ACCENT,extra='stroke-width="2.2"')
    s += screen(269,150,70,67,IVORY) + line(278,204,331,204,'#9eada8')
    for x,y,w,h,fill in [(280,184,9,20,GREEN),(297,174,9,30,BLUE),(314,163,9,41,ROSE)]:
        s += rect(x,y,w,h,fill,1,'none')
    s += path('M281 176l20-12 19-7',stroke=ACCENT,extra='stroke-width="2.2"')
    title = 'From healthcare records to relational data and analytical SQL'
    desc = ('A clinical care record flows into a central database. Two small relational tables show a rose primary-key column and a green foreign-key column connected from child to parent. A SQL worksheet with SELECT and FROM text leads to a compact result chart.')
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 360 256" role="img" aria-labelledby="title desc"><title id="title">{escape(title)}</title><desc id="desc">{escape(desc)}</desc><g fill="none" stroke="{INK}" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" font-family="Arial, sans-serif">{s}</g></svg>'
    ET.fromstring(svg)
    (OUT / 'opening-ecosystem.svg').write_text(svg, encoding='utf-8')
    CREATED.append('opening-ecosystem')


if __name__ == '__main__':
    build()
    assert len(CREATED) == len(set(CREATED)) == 26
    for name in CREATED:
        node = ET.parse(OUT / f'{name}.svg').getroot()
        assert node.attrib['viewBox'] == ('0 0 360 256' if name == 'opening-ecosystem' else '0 0 180 128')
        assert len(node.findall('{http://www.w3.org/2000/svg}title')) == 1
        assert len(node.findall('{http://www.w3.org/2000/svg}desc')) == 1
    print(f'Created and XML-validated {len(CREATED)} original teaching SVG illustrations:')
    print('\n'.join(f'{name}.svg' for name in CREATED))
