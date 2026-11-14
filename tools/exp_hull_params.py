"""Experiment: measure convex-hull triangle counts and geometric deviation
for different planar-dissolve angle limits.  Blender headless.

Usage: blender -b --factory-startup --python tools/exp_hull_params.py -- <urdf> <outdir>
"""
import bpy, bmesh, json, math, os, sys, time
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
URDF = argv[0] if len(argv) > 0 else "linkerhand_l20.urdf"
OUT = argv[1] if len(argv) > 1 else "/tmp/exp"
os.makedirs(OUT, exist_ok=True)

ROOT = os.path.dirname(os.path.abspath(URDF))
mapping = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "mapping.json")))
rows = [r for r in mapping if r["visual"]]

ANGLES = [0.0, 1.0, 2.0, 5.0, 10.0, 20.0]


def wipe():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def load_welded(path):
    wipe()
    bpy.ops.wm.stl_import(filepath=path, global_scale=1.0)
    objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    if len(objs) > 1:
        bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active
    me = obj.data
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=1e-7)
    bm.to_mesh(me)
    bm.free()
    return obj


def hull_from_object(obj):
    me = obj.data
    bm = bmesh.new()
    bm.from_mesh(me)
    res = bmesh.ops.convex_hull(bm, input=bm.verts[:], use_existing_faces=False)
    # delete interior / unused geometry
    kill = set(res.get("geom_interior", [])) | set(res.get("geom_unused", []))
    if kill:
        bmesh.ops.delete(bm, geom=[g for g in kill if isinstance(g, (bmesh.types.BMVert, bmesh.types.BMEdge, bmesh.types.BMFace))], context="VERTS")
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    bm.to_mesh(me)
    bm.free()
    me.update()
    return obj


def signed_volume(bm):
    v = 0.0
    for f in bm.faces:
        vs = [l.vert.co for l in f.loops]
        for i in range(1, len(vs) - 1):
            a, b, c = vs[0], vs[i], vs[i + 1]
            v += a.dot(b.cross(c)) / 6.0
    return v


def dissolve(obj, angle_deg):
    me = obj.data
    bm = bmesh.new()
    bm.from_mesh(me)
    if angle_deg > 0:
        bmesh.ops.dissolve_limit(bm, angle_limit=math.radians(angle_deg),
                                 verts=bm.verts[:], edges=bm.edges[:],
                                 delimit={"NORMAL"})
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    bm.to_mesh(me)
    bm.free()
    me.update()
    return obj


def tri_count(obj):
    bm = bmesh.new(); bm.from_mesh(obj.data)
    bmesh.ops.triangulate(bm, faces=bm.faces[:], quad_method="BEAUTY", ngon_method="BEAUTY")
    n = len(bm.faces)
    bm.free()
    return n


def export(obj, path):
    for o in bpy.context.scene.objects:
        o.select_set(o == obj)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.wm.stl_export(filepath=path, ascii_format=False, apply_modifiers=True,
                          use_scene_unit=False, global_scale=1.0)


report = []
for r in rows:
    link = r["link"]
    vis = os.path.join(ROOT, r["visual"]["filename"])
    t0 = time.time()
    # original vertex cloud (welded) – reference for containment checks
    obj = load_welded(vis)
    n_src_v = len(obj.data.vertices)
    obj = hull_from_object(obj)
    hull_tris = tri_count(obj)
    export(obj, os.path.join(OUT, f"{link}__hull_exact.stl"))
    ent = {"link": link, "src_verts": n_src_v, "hull_tris": hull_tris, "variants": []}
    for a in ANGLES:
        if a == 0.0:
            continue
        ob = load_welded(vis)
        ob = hull_from_object(ob)
        ob = dissolve(ob, a)
        # count faces both as ngons and triangulated
        ng = len(ob.data.polygons)
        tr = tri_count(ob)
        export(ob, os.path.join(OUT, f"{link}__a{a:g}.stl"))
        ent["variants"].append({"angle": a, "ngons": ng, "tris": tr})
    ent["secs"] = round(time.time() - t0, 1)
    report.append(ent)
    print("EXP", json.dumps(ent))

json.dump(report, open(os.path.join(OUT, "exp_report.json"), "w"), indent=2)
print("DONE")
