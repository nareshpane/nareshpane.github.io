"""Original procedural tree/human geometry; shared meshes, no external assets.

Trees are branching trunks and joined irregular foliage lobes, never individual
leaf objects. Humans are proportioned, clothed volumes with articulated limbs;
only twelve master instances get four-phase walking shape keys. Web people are
static shared pose meshes, without morph targets or animation payloads.
"""
import math
import random
import bpy
from mathutils import Vector

TREE_NAMES=['Mature maple','Columnar street tree','Ornamental pear','Young lime',
            'Spreading park oak','Light birch','Small flowering tree']
FAMILY_NAMES=['Gabled cottage','Narrow townhouse','Brick workshop','Brick walk-up',
              'Masonry shopfront','Garden apartments','Civic colonnade','Mixed-use block',
              'Waterfront terraces','Concrete and glass','Restrained Art Deco','Setback office tower']


def taper(m,a,b,r0,r1,mat=0,n=7):
    a,b=Vector(a),Vector(b);d=(b-a).normalized()
    u=d.cross(Vector((0,0,1)))
    if u.length<.01:u=d.cross(Vector((0,1,0)))
    u.normalize();v=d.cross(u);base=len(m.vertices)
    for p,r in [(a,r0),(b,r1)]:
        m.vertices.extend(tuple(p+r*(math.cos(2*math.pi*j/n)*u+math.sin(2*math.pi*j/n)*v)) for j in range(n))
    m.faces.extend([tuple(base+j for j in reversed(range(n))),tuple(base+n+j for j in range(n))]);m.materials.extend([mat,mat])
    for j in range(n):m.faces.append((base+j,base+(j+1)%n,base+(j+1)%n+n,base+j+n));m.materials.append(mat)


def lobe(m,p,scale,mat=0,n=10,rings=6,seed=0,roughness=0):
    """Closed UV ellipsoid, tiny deterministic radial irregularity, smooth normals."""
    rng=random.Random(seed);base=len(m.vertices)
    m.vertices.append((p[0],p[1],p[2]+scale[2]))
    for k in range(1,rings):
        phi=math.pi*k/rings
        for j in range(n):
            theta=2*math.pi*j/n;w=1+rng.uniform(-roughness,roughness)
            m.vertices.append((p[0]+scale[0]*math.sin(phi)*math.cos(theta)*w,
                               p[1]+scale[1]*math.sin(phi)*math.sin(theta)*w,
                               p[2]+scale[2]*math.cos(phi)*w))
    bottom=len(m.vertices);m.vertices.append((p[0],p[1],p[2]-scale[2]))
    for j in range(n):
        m.faces.append((base,base+1+j,base+1+(j+1)%n));m.materials.append(mat)
    for k in range(rings-2):
        for j in range(n):
            a=base+1+k*n+j;b=base+1+k*n+(j+1)%n
            m.faces.append((a,a+n,b+n,b));m.materials.append(mat)
    for j in range(n):
        a=base+1+(rings-2)*n+j;b=base+1+(rings-2)*n+(j+1)%n
        m.faces.append((a,bottom,b));m.materials.append(mat)


def tree_mesh(index,palette,web=False):
    from build_city import MeshBuilder
    rng=random.Random(260905+index*91);m=MeshBuilder()
    heights=[8.0,8.6,5.3,4.6,8.9,7.4,4.9];spreads=[2.9,1.7,2.0,1.25,3.8,2.45,2.1]
    h,r=heights[index],spreads[index];lean=(rng.uniform(-.42,.42),rng.uniform(-.3,.3))
    trunk=[(0,0,0),(lean[0]*.4,lean[1]*.4,h*.25),(lean[0],lean[1],h*.52),(lean[0]*1.2,lean[1]*1.4,h*.80)]
    radius=.26 if index not in [3,6] else .14
    radii=[radius,radius*.72,radius*.42,radius*.12]
    for k,(a,b) in enumerate(zip(trunk,trunk[1:])):taper(m,a,b,radii[k],radii[k+1],0)
    count=8 if index in [3,6] else 11
    for j in range(count):
        angle=j*2.399+rng.uniform(-.3,.3);radius=r*rng.uniform(.45,.86)
        z=h*(.52+.28*j/max(1,count-1))+rng.uniform(-.4,.4)
        end=(lean[0]+math.cos(angle)*radius,lean[1]+math.sin(angle)*radius,z)
        start=(lean[0]*.7,lean[1]*.7,h*(.34+.28*j/count))
        elbow=tuple(start[k]*.45+end[k]*.55 for k in range(3))
        taper(m,start,elbow,.075,.045,0,n=6);taper(m,elbow,end,.045,.018,0,n=5)
        if not web:
            twig=(end[0]+.45*math.cos(angle+.6),end[1]+.45*math.sin(angle+.6),end[2]+.45)
            taper(m,elbow,twig,.035,.01,0,n=5)
        size=rng.uniform(.76,1.13)
        lobe(m,end,(r*.53*size,r*.48*size,h*.19*size),1+j%3,
             n=8 if web else 12,rings=4 if web else 7,seed=index*50+j,roughness=.12)
    lobe(m,(lean[0]*1.2,lean[1]*1.4,h*.86),(r*.63,r*.56,h*.23),2,
         n=8 if web else 12,rings=4 if web else 7,seed=index,roughness=.1)
    mesh=m.mesh(f'Tree family {index} • {TREE_NAMES[index]} • '+('web' if web else 'master'),palette)
    for poly in mesh.polygons:poly.use_smooth=True
    return mesh


def human_builder(kind,phase=0,web=False,walking=False):
    from build_city import MeshBuilder
    m=MeshBuilder();n=8 if web else 10;rings=4 if web else 6
    swing=math.sin(phase*2*math.pi) if walking else [0,.22,-.12,.08,-.17,.14][kind]
    bounce=.022*(1-math.cos(phase*4*math.pi)) if walking else 0
    # Facing local -Y. Heads, pelvis and a shaped torso give a human silhouette.
    lobe(m,(0,0,1.17+bounce),(.24,.14,.33),1,n,rings)
    lobe(m,(0,0,.91+bounce),(.19,.125,.16),2,n,rings)
    taper(m,(0,0,1.43+bounce),(0,0,1.53+bounce),.072,.070,0,n)
    lobe(m,(0,-.008,1.64+bounce),(.105,.10,.135),0,n,rings)
    lobe(m,(0,.004,1.725+bounce),(.109,.102,.069),4,n,rings)
    lobe(m,(0,-.105,1.63+bounce),(.025,.035,.028),0,6,4)
    for side in [-1,1]:
        stride=side*swing*.24
        lift=max(0,-side*math.cos(phase*2*math.pi))*.075 if walking else 0
        hip=(side*.105,0,.92+bounce)
        knee=(side*.112,-stride*.62-.025,.52+lift*.5)
        ankle=(side*.115,-stride,.09+lift)
        taper(m,hip,knee,.092,.069,2,n);taper(m,knee,ankle,.068,.045,2,n)
        lobe(m,(ankle[0],ankle[1]-.045,ankle[2]-.04),(.075,.14,.055),3,n,rings)
        shoulder=(side*.20,0,1.39+bounce)
        elbow=(side*.28,stride*.65,1.14+bounce)
        hand=(side*.29,stride*1.03-.025,.99+bounce)
        taper(m,shoulder,elbow,.087,.060,1,n);taper(m,elbow,hand,.060,.039,1 if kind!=2 else 0,n)
        lobe(m,hand,(.047,.042,.065),0,n,rings)
    if kind in [1,4]:
        # Knee-length coat / skirt, still joined to the same object.
        taper(m,(0,0,.67+bounce),(0,0,1.12+bounce),.23,.18,1 if kind==1 else 2,10)
    if kind in [0,3,5]:
        lobe(m,(0,.175,1.23+bounce),(.16,.085,.23),5,n,rings)
    if kind==2:
        lobe(m,(.31,0,.91+bounce),(.11,.07,.14),5,n,rings)
    return m


def person_mesh(kind,palette,web=False):
    m=human_builder(kind,web=web)
    mesh=m.mesh(f'Pedestrian clothing family {kind} • '+('web' if web else 'master'),palette)
    for face in mesh.polygons:face.use_smooth=True
    return mesh


def animate_people(collection,fps=24,duration=24):
    """Twelve lightweight mesh morph walkers; no leaf/limb object explosion."""
    for obj in collection.objects:
        if not obj.get('animated_pedestrian'):continue
        kind=obj['person_family'];obj.data=obj.data.copy()
        base=human_builder(kind,phase=0,walking=True)
        for v,co in zip(obj.data.vertices,base.vertices):v.co=co
        obj.shape_key_add(name='Basis')
        for k in range(1,4):
            key=obj.shape_key_add(name=f'Gait quarter {k}')
            pose=human_builder(kind,phase=k/4,walking=True)
            assert len(key.data)==len(pose.vertices)
            for v,co in zip(key.data,pose.vertices):v.co=co
            for frame in [1,1+fps*.3125,1+fps*.625,1+fps*.9375,1+fps*1.25]:
                quarter=round((frame-1)/(fps*.3125))
                key.value=1 if quarter==k else 0
                key.keyframe_insert('value',frame=frame)
        for curve in obj.data.shape_keys.animation_data.action.fcurves:
            for key in curve.keyframe_points:key.interpolation='LINEAR'
            curve.modifiers.new('CYCLES')
        x,y,z=obj.location;direction=obj.get('walk_direction',1)
        # A clear sidewalk corridor, with a held turn at each end. The root
        # returns to its first location/orientation for a clean hero loop.
        for t,offset,angle in [(0,-7,math.pi),(0.43,7,math.pi),(.5,7,0),(.93,-7,0),(1,-7,math.pi)]:
            obj.location=(x,y+offset*direction,z)
            obj.rotation_euler.z=angle if direction==1 else angle+math.pi
            obj.keyframe_insert('location',frame=1+t*(duration*fps-1))
            obj.keyframe_insert('rotation_euler',frame=1+t*(duration*fps-1))
        for curve in obj.animation_data.action.fcurves:
            for key in curve.keyframe_points:
                key.interpolation='LINEAR' if curve.data_path=='location' else 'BEZIER'
                key.handle_left_type=key.handle_right_type='AUTO_CLAMPED'
