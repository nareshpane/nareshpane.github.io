"""Build the editable master and a distinct lightweight web GLB with bpy 4.5.

Each building is ONE object. Twelve shared families contain joined wall, roof,
trim and window geometry. Vegetation/lights use shared meshes; markings batch.
Run: blender -b -t 6 --python-exit-code 1 --python scripts/build_city.py
"""
import hashlib
import json
import math
import random
import sys
from pathlib import Path
import time
import bpy
from mathutils import Vector

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
CITY=json.loads((ROOT/'data/city.json').read_text())
ANALYSIS=json.loads((ROOT/'data/analysis.json').read_text())

class MeshBuilder:
    def __init__(self):
        self.vertices=[]; self.faces=[]; self.materials=[]
    def face(self,vertices,mat=0):
        i=len(self.vertices)
        self.vertices.extend(vertices)
        self.faces.append(tuple(range(i,i+len(vertices))))
        self.materials.append(mat)
    def box(self,p,s,mat=0):
        x,y,z=p; a,b,c=[v/2 for v in s]
        v=[(x-a,y-b,z-c),(x+a,y-b,z-c),(x+a,y+b,z-c),(x-a,y+b,z-c),
           (x-a,y-b,z+c),(x+a,y-b,z+c),(x+a,y+b,z+c),(x-a,y+b,z+c)]
        i=len(self.vertices); self.vertices.extend(v)
        for f in [(0,3,2,1),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7),(4,5,6,7)]:
            self.faces.append(tuple(i+j for j in f)); self.materials.append(mat)
    def rod(self,a,b,r,mat=0,n=8):
        a,b=Vector(a),Vector(b); direction=(b-a).normalized()
        t=direction.cross(Vector((0,0,1)))
        if t.length<.01: t=direction.cross(Vector((0,1,0)))
        t.normalize(); u=direction.cross(t)
        ring=[r*(math.cos(i*2*math.pi/n)*t+math.sin(i*2*math.pi/n)*u) for i in range(n)]
        i=len(self.vertices)
        self.vertices.extend(tuple(p+v) for p in [a,b] for v in ring)
        self.faces.append(tuple(i+j for j in reversed(range(n)))); self.materials.append(mat)
        self.faces.append(tuple(i+n+j for j in range(n))); self.materials.append(mat)
        for j in range(n):
            self.faces.append((i+j,i+(j+1)%n,i+(j+1)%n+n,i+j+n)); self.materials.append(mat)
    def crown(self,p,r,mat=0):
        x,y,z=p
        v=[(x+r,y,z),(x,y+r,z),(x-r,y,z),(x,y-r,z),(x,y,z+r*1.25),(x,y,z-r*.9)]
        i=len(self.vertices); self.vertices.extend(v)
        for j in range(4):
            self.faces.extend([(i+j,i+(j+1)%4,i+4),(i+(j+1)%4,i+j,i+5)])
            self.materials.extend([mat,mat])
    def mesh(self,name,palette):
        m=bpy.data.meshes.new(name)
        m.from_pydata(self.vertices,[],self.faces); m.update()
        for material in palette: m.materials.append(material)
        for p,index in zip(m.polygons,self.materials): p.material_index=index
        return m

def material(name,color,metallic=0,emission=0):
    m=bpy.data.materials.new(name); m.diffuse_color=(*color,1); m.use_nodes=True
    p=m.node_tree.nodes.get('Principled BSDF')
    p.inputs['Base Color'].default_value=(*color,1)
    p.inputs['Roughness'].default_value=.62
    p.inputs['Metallic'].default_value=metallic
    p.inputs['Emission Color'].default_value=(*color,1)
    p.inputs['Emission Strength'].default_value=emission
    return m

def collection(name):
    c=bpy.data.collections.new(name); bpy.context.scene.collection.children.link(c); return c

def object_from(name,mesh,c,location=(0,0,0),scale=(1,1,1)):
    o=bpy.data.objects.new(name,mesh); c.objects.link(o); o.location=location; o.scale=scale; return o

def orient(o,target):
    o.rotation_euler=(Vector(target)-o.location).to_track_quat('-Z','Y').to_euler()

def archetype(index,palette,web=False):
    """Unit footprint/height mesh; windows are faces, trim is batched geometry."""
    m=MeshBuilder()
    if index in [10,11]:
        for z,w,h in [(.3,1,.6),(.725,.76,.25),(.925,.56,.15)]:m.box((0,0,z),(w,w,h),0)
    else:m.box((0,0,.5),(1,1,1),0)
    m.box((0,0,.025),(1.06,1.06,.05),1)
    top=.6 if index in [10,11] else 1.08
    m.box((0,0,1.01),(top,top,.025),1)
    if index in [0,1,3]:
        m.face([(-.56,-.56,1.03),(0,-.56,1.22),(.56,-.56,1.03)],3)
        m.face([(.56,.56,1.03),(0,.56,1.22),(-.56,.56,1.03)],3)
        m.face([(-.56,-.56,1.03),(-.56,.56,1.03),(0,.56,1.22),(0,-.56,1.22)],3)
        m.face([(0,-.56,1.22),(0,.56,1.22),(.56,.56,1.03),(.56,-.56,1.03)],3)
    else:
        m.box((0,0,1.035),(top*.86,top*.86,.03),3)
        m.box((.12,.12,1.075),(.2,.23,.06),0)
        if not web:
            for j in [-1,0,1]: m.box((-.22,j*.24,1.083),(.3,.18,.016),2)
    floors=[3,3,1,4,4,5,5,7,7,10,10,12][index]
    for floor in range(floors):
        z=(floor+.55)/floors if index!=2 else .72
        dz=(.29 if index not in [4,9,11] else .36)/floors if index!=2 else .12
        half=(.5 if z<.6 else .38 if z<.85 else .28) if index in [10,11] else .5
        positions=[-.32,.0,.32] if index not in [1,6,9] else [-.26,.26]
        for k0 in positions:
            k=k0*half*2
            dw=(.09 if index in [1,3,10] else .13 if index in [4,9,11] else .105)*half*2
            for sign in [-1,1]:
                # Two facade orientations, explicit outward winding.
                y=sign*(half+.002)
                v=[(k-dw,y,z-dz),(k+dw,y,z-dz),(k+dw,y,z+dz),(k-dw,y,z+dz)]
                m.face(v if sign<0 else list(reversed(v)),2)
                x=sign*(half+.002)
                v=[(x,k-dw,z-dz),(x,k+dw,z-dz),(x,k+dw,z+dz),(x,k-dw,z+dz)]
                m.face(v if sign>0 else list(reversed(v)),2)
                if not web:
                    m.box((k,y,z-dz-.006),(dw*2.2,.025,.012),1)
                    m.box((x,k,z-dz-.006),(.025,dw*2.2,.012),1)
        if not web and floor>0 and floor%2==0:
            m.box((0,0,floor/floors),(half*2+.02,half*2+.02,.009),1)
        if index in [5,7,8] and floor>0 and not web:
            for xx in [-.28,.28]:
                m.box((xx,-.54,z-dz-.016),(.28,.18,.018),1)
                m.box((xx,-.62,z-dz+.025),(.28,.012,.075),3)
    door=2.5/[10,10,10,16,16,20,20,28,28,40,40,55][index]
    m.box((0,-.509,door/2),(.19,.018,door),2)
    if index in [2,4,7]:
        for x in [-.3,.3]:m.box((x,-.508,door*.55),(.34,.015,door*.92),2)
        m.box((0,-.56,door*1.05),(.96,.2,.025),3)
    if index in [4,6,10]:
        for x in [-.44,-.14,.14,.44]:m.box((x,-.52,.48),(.037,.04,.91),1)
        m.box((0,-.52,.96),(1.08,.12,.045),1)
    if index==1:
        for k in range(3):m.box((0,-.57-k*.035,.009*(3-k)),(.28,.1,.018*(3-k)),1)
    if index==2:
        for x in [-.31,0,.31]:m.box((x,0,1.055),(.20,.7,.06),3)
    return m.mesh(f'Archetype_{index}_'+('web' if web else 'master'),palette)

def main():
    start=time.perf_counter()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene=bpy.context.scene
    from world_assets import tree_mesh, person_mesh, animate_people, TREE_NAMES, FAMILY_NAMES
    bpy.context.preferences.filepaths.save_version=0
    palette=[material('Facade • limestone',(.74,.66,.51)),material('Ivory trim',(.9,.85,.72)),
             material('Blue glazing',(.055,.23,.29),.28),material('Slate roofs',(.13,.24,.27))]
    hoods={k:material(k,c) for k,c in [
        ('Old Quay',(.72,.36,.24)),('Garden Quarter',(.37,.54,.43)),
        ('Civic Terrace',(.73,.64,.42)),('Innovation District',(.26,.46,.53))]}
    ground=material('Sage landscape',(.27,.39,.29))
    stone=material('Warm pavement',(.7,.67,.56))
    asphalt=material('Blue charcoal streets',(.072,.106,.13))
    water=material('River teal',(.065,.32,.39),.22)
    white=palette[1]; dark=palette[3]
    gold=material('Marking gold',(.98,.7,.21))
    line_mat=material('Transit coral',(.95,.26,.12),emission=.25)
    graph_mat=material('Graph cyan',(.12,.8,.84),emission=.7)
    route_mat=material('Route gold', (1,.55,.035),emission=1)
    wood=material('Bark',(.27,.15,.06))
    leaf=material('Leaf canopy',(.24,.45,.22))
    land=collection('01 Landscape and parks')
    roads=collection('02 Roads • graph edge IDs')
    details=collection('03 Batched sidewalks and markings')
    buildings=collection('04 Buildings • one object each')
    vegetation=collection('05 Shared tree instances')
    furniture=collection('06 Shared street furniture')
    transit=collection('07 Existing transit and stations')
    overlay=collection('08 Graph overlay • toggle render visibility')
    route_collection=collection('09 Shortest route')
    proposal=collection('10 Proposed transit')
    studio=collection('11 Cameras and lighting')
    people=collection('12 Pedestrians • shared clothing and walking poses')
    m=MeshBuilder(); m.box((0,0,-2),(392,338,4)); object_from('City foundation',m.mesh('Base',[dark]),land)
    m=MeshBuilder()
    for x in [-106,106]: m.box((x,0,.15),(178,332,.3))
    object_from('Two riverbanks',m.mesh('Riverbank mesh',[ground]),land)
    m=MeshBuilder(); m.box((0,0,-.05),(34,332,.4))
    object_from('River • fictional coordinates',m.mesh('Water mesh',[water]),land)
    m=MeshBuilder()
    for x in [-18,18]: m.box((x,0,.25),(2,332,.9))
    object_from('River retaining walls',m.mesh('Quays',[stone]),land)
    paving=MeshBuilder(); markings=MeshBuilder(); rails=MeshBuilder()
    for b in CITY['blocks']:
        paving.box((b['x'],b['y'],.4),(40,40,.5))
        if b['park']:
            m=MeshBuilder(); m.box((b['x'],b['y'],.72),(36,36,.15))
            object_from('Park_'+b['id'],m.mesh('Park turf',[ground]),land)
    object_from('All block sidewalks',paving.mesh('Joined sidewalk slabs',[stone]),details)
    for e in CITY['roads']:
        a,b=[CITY['nodes'][i] for i in [e['u'],e['v']]]
        dx,dy=b['x']-a['x'],b['y']-a['y']
        # Stop at the junction boundary; a single shared intersection batch fills
        # the 8 m square. Coplanar overlapping road boxes cause render artifacts.
        length=e['length_m'];angle=math.atan2(dy,dx)
        m=MeshBuilder(); m.box((0,0,.33),(length-8 if e['kind']!='footway' else length,e['width_m'],.22))
        if e['id'] in ['33-40','13-22']:
            m=MeshBuilder();w=e['width_m']/2
            ramp=[(-length/2,.445),(-length/2+8,.82),(length/2-8,.82),(length/2,.445)]
            for (x0,z0),(x1,z1) in zip(ramp,ramp[1:]):
                m.face([(x0,-w,z0),(x1,-w,z1),(x1,w,z1),(x0,w,z0)])
        o=object_from('Road_'+e['id'],m.mesh('RoadMesh_'+e['id'],[asphalt]),roads,
                      ((a['x']+b['x'])/2,(a['y']+b['y'])/2,0))
        o['graph_edge']=e['id']; o['seconds']=e['seconds']; o['length_m']=e['length_m']
        o.rotation_euler.z=angle;o['road_class']=e['road_class'];o['kind']=e['kind']
        if e['kind']=='footway':
            # Park paths sit above turf/sidewalk; outer paths sit above grass.
            continue
        edge_markings=MeshBuilder() if e['kind']=='bridge' else markings
        for j in range(1,8):
            t=j/8
            edge_markings.box((a['x']+dx*t,a['y']+dy*t,.452),(2.7 if dx else .2,2.7 if dy else .2,.018),0)
        if e['kind']=='bridge':
            o=object_from('BridgeMarkings_'+e['id'],edge_markings.mesh('Bridge markings',[gold]),roads)
            o['graph_edge']=e['id']
            m=MeshBuilder()
            for yoff in [-4.6,4.6]:
                m.box((0,yoff,.9),(48,.35,1.25))
                for xx in range(-20,21,8): m.box((xx,yoff,-1),(1,1,3))
            o=object_from('BridgeRail_'+e['id'],m.mesh('Bridge infrastructure',[white]),roads,
                          ((a['x']+b['x'])/2,a['y'],0)); o['graph_edge']=e['id']
    junctions=MeshBuilder()
    for n in CITY['nodes']:
        junctions.box((n['x'],n['y'],.33),(3 if n['id']>=56 else 8,3 if n['id']>=56 else 8,.22))
        if n['id']>=56:continue
        for j in range(5):
            markings.box((n['x']-2.5+j*1.25,n['y']+5,.455),(.65,1.8,.02),1)
    object_from('All intersection surfaces • no overlapping faces',junctions.mesh('Joined junctions',[asphalt]),details)
    object_from('All lane dashes and crossings',markings.mesh('Batched markings',[gold,white]),details)
    master_meshes=[archetype(i,palette) for i in range(12)]
    colours=[(.37,.47,.37),(.5,.25,.19),(.33,.32,.28),(.53,.25,.17),(.66,.51,.34),(.65,.71,.61),(.73,.70,.59),(.56,.38,.29),(.48,.67,.64),(.48,.55,.60),(.55,.57,.5),(.35,.48,.52)]
    facades={}
    for family,colour in enumerate(colours):
        for hood,mat in hoods.items():
            facades[family,hood]=material(f'{FAMILY_NAMES[family]} • {hood}',tuple(.85*c+.15*mat.diffuse_color[k] for k,c in enumerate(colour)))
    for b in CITY['buildings']:
        o=object_from(b['id']+' • '+b['neighbourhood'],master_meshes[b['family']],buildings,
                      (b['x'],b['y'],.65),(b['width'],b['depth'],b['height']))
        o['role']='building'; o['building_id']=b['id']; o['node_id']=b['node']; o['archetype']=b['archetype']
        o['height_parameter_m']=b['height']; o['neighbourhood']=b['neighbourhood']
        o['family']=b['family'];o['architectural_family']=FAMILY_NAMES[b['family']]
        o.material_slots[0].link='OBJECT'; o.material_slots[0].material=facades[b['family'],b['neighbourhood']]
    tree_palette=[wood,leaf,material('Foliage • light',(.31,.49,.19)),material('Foliage • shade',(.14,.32,.12))]
    trees=[tree_mesh(i,tree_palette) for i in range(7)]
    rng=random.Random(CITY['seed']+810)
    for i,b in enumerate(CITY['blocks']):
        points=[(-17,-16),(17,16),(-17,16),(17,-16)]
        if b['park']: points += [(-12,-5),(-11,5),(-6,12),(3,12),(12,7),(12,-5),(7,-12),(-3,-12),(-12,11)]
        for j,(dx,dy) in enumerate(points):
            family=(i*3+j)%7;s=rng.uniform(.76,1.03)
            tx,ty=b['x']+dx,b['y']+dy
            # Keep trunks clear of the actual diagonal park path.
            if b['park'] and abs((dx+dy) if b['id']=='0-4' else (dx-dy))<5:
                ty+=-6 if dy>=0 else 6
            o=object_from(f'Tree_{i:02}_{j:02}',trees[family],vegetation,(tx,ty,.8 if b['park'] else .65),(s*rng.uniform(.86,1.08),s,s*rng.uniform(.9,1.1)))
            o.rotation_euler.z=rng.uniform(0,math.tau);o['role']='tree';o['tree_family']=family
    skins=[(.55,.30,.18),(.78,.53,.36),(.32,.17,.11),(.64,.4,.25),(.83,.65,.47),(.43,.25,.16)]
    clothes=[(.12,.27,.38),(.64,.24,.12),(.29,.4,.21),(.72,.54,.24),(.38,.22,.36),(.2,.39,.43)]
    person_palettes=[]
    for k in range(6):
        person_palettes.append([material(f'Skin {k}',skins[k]),material(f'Clothing {k}',clothes[k]),
            material(f'Trousers {k}',(.08+.02*k,.1+.015*k,.14+.01*k)),material(f'Shoes {k}',(.035,.035,.028)),
            material(f'Hair {k}',(.07+.02*k,.045+.012*k,.025)),material(f'Accessory {k}',(.22,.15,.09))])
    persons=[person_mesh(k,person_palettes[k]) for k in range(6)]
    # Twelve walkers occupy clear sidewalk strips, including the hero corridor.
    walk_blocks=[b for b in CITY['blocks'] if not b['park'] and b['id'].split('-')[0] in ['5','6']]
    walk_blocks.append(next(b for b in CITY['blocks'] if b['id']=='4-0'))
    for i,b in enumerate(walk_blocks):
        x=b['x']+(18.1 if b['id'].startswith('5-') else -18.1)
        o=object_from(f'Pedestrian_{i:02} • walking',persons[i%6],people,(x,b['y'],.67),(.96,.96,.96))
        o['role']='pedestrian';o['person_family']=i%6;o['animated_pedestrian']=True;o['walk_direction']=1 if i%2 else -1
    for i in range(36):
        b=CITY['blocks'][i];side=-1 if i%2 else 1
        o=object_from(f'Pedestrian_{i+12:02} • standing',persons[(i+2)%6],people,
                      (b['x']+side*(17.3 if b['park'] else 18.1),b['y']+(4 if i%3 else -5),.81 if b['park'] else .67),
                      (1,1,rng.uniform(.91,1.06)))
        o.rotation_euler.z=rng.uniform(-math.pi,math.pi)
        o['role']='pedestrian';o['person_family']=(i+2)%6;o['animated_pedestrian']=False
    animate_people(people)
    scene.frame_set(1)
    m=MeshBuilder(); m.rod((0,0,0),(0,0,5),.12,0)
    m.box((.65,0,5),(1.5,.2,.2),0); m.box((1.2,0,4.9),(.65,.5,.18),1)
    lamp_mesh=m.mesh('Shared streetlight assembly',[dark,gold])
    for n in CITY['nodes']:
        object_from('Lamp_'+str(n['id']),lamp_mesh,furniture,(n['x']+5.5,n['y']-5.5,.5))
    m=MeshBuilder()
    for e in CITY['transit']:
        a,b=[CITY['nodes'][i] for i in [e['u'],e['v']]]
        m.rod((a['x']+2,a['y'],1.2),(b['x']+2,b['y'],1.2),.65)
    object_from('Existing transit • corridor offset 2 m for visibility',m.mesh('Transit route',[line_mat]),transit)
    m=MeshBuilder()
    for i in CITY['stations']:
        n=CITY['nodes'][i]
        m.box((n['x']+6,n['y'],.9),(3,7,.7),0)
        m.box((n['x']+6,n['y'],3.4),(3.5,7.5,.28),1)
        for dy in [-3,3]:m.rod((n['x']+6,n['y']+dy,1),(n['x']+6,n['y']+dy,3.4),.12,0)
    object_from('Four station shelters • batched',m.mesh('Station geometry',[white,line_mat]),transit)
    m=MeshBuilder()
    for e in CITY['roads']:
        a,b=[CITY['nodes'][i] for i in [e['u'],e['v']]]
        m.rod((a['x'],a['y'],2),(b['x'],b['y'],2),.6)
    object_from('Graph edges • exact road centre lines',m.mesh('Graph edge batch',[graph_mat]),overlay)
    m=MeshBuilder(); m.rod((0,0,0),(0,0,1),1)
    node_mesh=m.mesh('Shared graph node glyph',[graph_mat])
    for n in CITY['nodes']:
        o=object_from('Node_'+str(n['id']),node_mesh,overlay,(n['x'],n['y'],2),(2,2,2))
        o['node_id']=n['id']; o['role']='graph_node'
    for key,col,mat in [('route',route_collection,route_mat),('closure_route',route_collection,line_mat)]:
        m=MeshBuilder()
        path=ANALYSIS[key]['nodes']
        for u,v in zip(path,path[1:]):
            a,b=CITY['nodes'][u],CITY['nodes'][v]
            m.rod((a['x'],a['y'],3),(b['x'],b['y'],3),1.15)
        object_from(key,m.mesh(key,[mat]),col)
    m=MeshBuilder()
    a,b=[CITY['nodes'][i] for i in [33,38]]
    m.rod((a['x'],a['y'],5),(b['x'],b['y'],5),1)
    object_from('New transit connection • schematic corridor',m.mesh('Proposal',[route_mat]),proposal)
    for c in [overlay,route_collection,proposal]:c.hide_render=True;c.hide_viewport=True
    # A large studio floor is deliberately excluded from the web model.
    m=MeshBuilder(); m.box((0,0,-4.2),(2000,2000,.2))
    o=object_from('Studio floor • excluded from GLB',m.mesh('Floor',[material('Studio cream',(.69,.73,.69))]),studio)
    camera_data=bpy.data.cameras.new('Isometric camera'); camera_data.type='ORTHO'; camera_data.ortho_scale=575;camera_data.clip_end=5000
    cam=bpy.data.objects.new('Camera_Hero',camera_data);studio.objects.link(cam);cam.location=(480,-480,490);orient(cam,(0,0,10));scene.camera=cam
    detail_data=camera_data.copy();detail_data.name='Detail camera';detail_data.ortho_scale=170
    detail=bpy.data.objects.new('Camera_Detail',detail_data);studio.objects.link(detail);detail.location=(220,-160,180);orient(detail,(55,30,12))
    analysis_data=camera_data.copy();analysis_data.name='Analysis camera';analysis_data.ortho_scale=720
    analysis_camera=bpy.data.objects.new('Camera_Analysis',analysis_data);studio.objects.link(analysis_camera)
    analysis_camera.location=(220,-300,780);orient(analysis_camera,(0,0,0))
    for name,pos,power,size,color in [('Warm softbox',(-170,-220,360),900000,200,(1,.82,.64)),('Sky fill',(230,120,320),700000,220,(.63,.8,1))]:
        d=bpy.data.lights.new(name,'AREA');d.energy=power;d.shape='DISK';d.size=size;d.color=color
        o=bpy.data.objects.new(name,d);studio.objects.link(o);o.location=pos;orient(o,(0,0,0))
    d=bpy.data.lights.new('Afternoon sun','SUN');d.energy=2.1;d.angle=.18
    o=bpy.data.objects.new('Afternoon sun',d);studio.objects.link(o);o.rotation_euler=(.4,-.5,-.4)
    scene.world=bpy.data.worlds.new('Studio atmosphere');scene.world.use_nodes=True
    scene.world.node_tree.nodes['Background'].inputs[0].default_value=(.56,.67,.8,1)
    scene.world.node_tree.nodes['Background'].inputs[1].default_value=.35
    scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=32;scene.cycles.use_denoising=True;scene.cycles.seed=CITY['seed'];scene.cycles.max_bounces=4
    scene.render.resolution_x=1440;scene.render.resolution_y=1000;scene.render.resolution_percentage=100
    scene.render.image_settings.file_format='PNG';scene.view_settings.view_transform='AgX'
    scene.unit_settings.system='METRIC';scene['seed']=CITY['seed'];scene['city_sha256']=hashlib.sha256((ROOT/'data/city.json').read_bytes()).hexdigest()
    scene['coordinate_system']='Synthetic local metric Cartesian grid. No EPSG code; Z is height.'
    scene['building_editing']='Each B### is one object. Meshes are shared across archetypes; make single-user to edit topology independently.'
    scene['world_revision']=2;scene['tree_archetypes']=7;scene['architectural_families']=12
    # Reuse the successful cinematic lighting in Eevee for current-world stills.
    from render_cinematic import configure_lighting
    configure_lighting(scene,16)
    scene.render.resolution_x=1440;scene.render.resolution_y=1000
    bpy.context.view_layer.update()
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type=='VIEW_3D':area.spaces.active.region_3d.view_perspective='CAMERA'
    bpy.ops.wm.save_as_mainfile(filepath=str(ROOT/'blender/city_master.blend'),compress=True)
    master_seconds=time.perf_counter()-start
    # Export geometry only. Shared meshes stay shared; master stays on disk unchanged.
    web_meshes=[archetype(i,palette,web=True) for i in range(12)]
    for o in buildings.objects:o.data=web_meshes[o['family']]
    web_trees=[tree_mesh(i,tree_palette,web=True) for i in range(7)]
    for o in vegetation.objects:o.data=web_trees[o['tree_family']]
    web_people=[person_mesh(i,person_palettes[i],web=True) for i in range(6)]
    for o in people.objects:
        o.data=web_people[o['person_family']];o.animation_data_clear();o['animated_pedestrian']=False;o['web_pose']='static LOD'
    bpy.ops.object.select_all(action='DESELECT')
    for c in [land,roads,details,buildings,vegetation,furniture,transit,people]:
        for o in c.objects:o.select_set(True)
    bpy.ops.export_scene.gltf(filepath=str(ROOT/'models/city_web.glb'),export_format='GLB',
        use_selection=True,export_apply=False,export_extras=True,export_cameras=False,export_lights=False,export_materials='EXPORT',export_animations=False)
    # Graph-only secondary GLB: no buildings or studio bounds.
    overlay.hide_viewport=False;overlay.hide_render=False
    bpy.ops.object.select_all(action='DESELECT')
    for o in overlay.objects:o.select_set(True)
    bpy.ops.export_scene.gltf(filepath=str(ROOT/'models/graph_web.glb'),export_format='GLB',
        use_selection=True,export_extras=True,export_cameras=False,export_lights=False)
    (ROOT/'data/build_runtime.json').write_text(json.dumps(dict(blender_version=bpy.app.version_string,
        seed=CITY['seed'],master_generation_seconds=round(master_seconds,3),
        generation_and_export_seconds=round(time.perf_counter()-start,3)),indent=2)+'\n')
    print('CITY_BUILD_SUCCESS',flush=True)

if __name__=='__main__':main()
