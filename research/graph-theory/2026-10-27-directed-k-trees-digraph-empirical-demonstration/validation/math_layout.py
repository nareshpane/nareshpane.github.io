"""Break TeX at semantic boundaries into inline MathJax units that flow naturally.

MathJax 3 SVG output does not line-break an indivisible SVG. Keep matrices,
scripts, groups and endpoint pairs intact; allow browser wrapping between
relations, set conditions, sums and listed terms. No content is scaled or clipped.
"""
import re,html

def flow_math(tex):
    tex=re.sub(r'\\(?:begin|end)\{(?:aligned|gathered)\}','',tex)
    units=[];chunk='';i=0;depth=0;parens=0
    def flush():
        nonlocal chunk
        if chunk.strip():units.append(('math',chunk.strip()))
        chunk=''
    while i<len(tex):
        # A matrix/cases environment is a single indivisible mathematical unit.
        env=re.match(r'\\begin\{(pmatrix|bmatrix|matrix|cases)\}',tex[i:])
        if env:
            end=r'\end{'+env[1]+'}'
            j=tex.index(end,i)+len(end)
            chunk+=tex[i:j];i=j;continue
        if tex.startswith(r'\text{',i):
            flush();j=i+6;level=1;k=j
            while level:
                if tex[k]=='{':level+=1
                elif tex[k]=='}':level-=1
                k+=1
            units.append(('prose',tex[j:k-1]));i=k;continue
        if depth==0 and tex.startswith('\\\\',i):
            flush();units.append(('break',''));i+=2;continue
        space=re.match(r'\\(?:qquad|quad|,|;|!| )',tex[i:])
        if depth==0 and space:
            flush();i+=len(space[0]);continue
        if tex[i]=='\\':
            command=re.match(r'\\(?:[A-Za-z]+|.)',tex[i:])[0]
            chunk+=command;i+=len(command)
            if depth==0 and command in (r'\mid',r'\oplus',r'\cup',r'\cap',r'\cdots'):
                flush()
            continue
        c=tex[i];i+=1
        if c=='&' and depth==0:continue
        chunk+=c
        if c=='{':depth+=1
        elif c=='}':depth-=1
        elif c=='(':parens+=1
        elif c==')':parens-=1
        if depth==0 and parens==0:
            if c in '=*,' or (c==')' and i<len(tex) and tex[i]=='('):flush()
    flush()
    rendered=[]
    for kind,value in units:
        if kind=='break':rendered.append('<wbr>')
        elif kind=='prose':rendered.append(html.escape(value))
        else:rendered.append(r'\('+value+r'\)')
    return ' '.join(rendered)

def fit_math(page):
    head,page=page.split('<body>',1)
    protected=re.findall(r'\\begin\{(?:pmatrix|bmatrix|matrix|cases)\}.*?\\end\{(?:pmatrix|bmatrix|matrix|cases)\}',page,flags=re.S)
    # Remove the old fragment scrollers. The math itself will supply breakpoints.
    page=re.sub(r'<span class="inline-math-scroll">(.*?)</span>',r'\1',page,flags=re.S)
    def display(m):
        return '<div class="math-flow">'+flow_math(m[1])+'</div>'
    page=re.sub(r'\\\[(.*?)\\\]',display,page,flags=re.S)
    def inline(m):
        tex=m[1]
        if len(tex)>35:
            return '<span class="math-flow-inline">'+flow_math(tex)+'</span>'
        return m[0]
    # Don't reprocess the new smaller fragments.
    parts=re.split(r'(<div class="math-flow">.*?</div>)',page,flags=re.S)
    for i,part in enumerate(parts):
        if not part.startswith('<div class="math-flow">'):
            parts[i]=re.sub(r'\\\((.*?)\\\)',inline,part,flags=re.S)
    page=''.join(parts)
    # Allow all non-numeric tables to stack into labelled records on narrow screens.
    def label_table(m):
        block=m[0]
        if re.search(r'class="(?:matrix|entry-grid)',block):return block
        head=re.search(r'<thead>(.*?)</thead>',block,flags=re.S)
        if not head:return block
        labels=re.findall(r'<th[^>]*>(.*?)</th>',head[1],flags=re.S)
        def row(r):
            index=0
            def cell(c):
                nonlocal index
                label=labels[index] if index<len(labels) else ''
                index+=1
                return '<td data-label="'+html.escape(re.sub('<[^>]*>','',label),quote=True)+'">'+c[1]+'</td>'
            return re.sub(r'<td>(.*?)</td>',cell,r[0],flags=re.S)
        return re.sub(r'<tr>(.*?)</tr>',row,block,flags=re.S)
    result=head+'<body>'+re.sub(r'<table\b.*?</table>',label_table,page,flags=re.S)
    assert all(block in result for block in protected),'A matrix/cases block changed'
    return result
