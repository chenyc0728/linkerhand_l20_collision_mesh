"""Experiment v2: voxel-thinned point cloud -> convex hull -> exact containment correction.
Blender headless.
blender -b --factory-startup --python tools/exp_v2.py -- <urdf> <outdir> <grid_mm_csv>
"""
import bpy, bmesh, json, math, os, sys, time
import numpy as np

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
URDF = argv[0]
OUT = argv[1]
GRIDS = [float(x) for x in argv[2].split(",")] if len(argv) > 2 else [1.0]
os.makedirs(OUT, exist_ok=True)
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.abspath(URDF))
rows = [r for r in json.load(open(os.path.join(HERE, "mapping.json"))) if r["visual"]]


def load_points(path):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.wm.stl_import(filepath=path, global_scale=1.0)
    objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    if len(objs) > 1:
        bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active
    me = obj.data
    bm = bmesh.new(); bm.from_mesh(me)
    bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=1e-7)
    pts = np.array([v.co[:] for v in bm.verts], dtype=np.float64)
    bm.free()
    return pts


def voxel_thin(p, g, center):
    """keep the point farthest from center inside each g-sized voxel"""
    ijk = np.floor((p - p.min(0)) / g).astype(np.int64)
    order = np.lexsort((ijk[:, 2], ijk[:, 1], ijk[:, 0]))
    k = ijk[order]; ps = p[order]
    d = np.linalg.norm(ps - center, axis=1)
    start = np.ones(len(k), bool); start[1:] = np.any(np.diff(k, axis=0) != 0, axis=1)
    grp = np.cumsum(start) - 1
    maxd = np.full(grp.max() + 1, -1.0); np.maximum.at(maxd, grp, d)
    cand = d >= maxd[grp] - 1e-18
    seen = set(); keep = []
    for i in np.nonzero(cand)[0]:
        gi = grp[i]
        if gi not in seen:
            seen.add(gi); keep.append(i)
    return ps[np.array(keep)]


def new_bm(pts):
    bm = bmesh.new()
    for p in pts:
        bm.verts.new(p)
    bm.verts.ensure_lookup_table()
    return bm


def do_hull(pts):
    bm = new_bm(pts)
    res = bmesh.ops.convex_hull(bm, input=bm.verts[:], use_existing_faces=False)
    interior = res.get("geom_interior", [])
    if interior:
        bmesh.ops.delete(bm, geom=[g for g in interior if isinstance(g, bmesh.types.BMVert)], context="VERTS")
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    return bm


def bm_arrays(bm):
    bm.verts.ensure_lookup_table()
    V = np.array([v.co[:] for v in bm.verts], dtype=np.float64)
    F = [tuple(v.index for v in f.verts) for f in bm.faces]
    return V, F


def face_planes(V, F):
    N = np.zeros((len(F), 3)); D = np.zeros(len(F))
    for i, f in enumerate(F):
        P = V[list(f)]
        n = np.zeros(3)                      # Newell
        for a in range(len(P)):
            b = (a + 1) % len(P)
            n[0] += (P[a][1] - P[b][1]) * (P[a][2] + P[b][2])
            n[1] += (P[a][2] - P[b][2]) * (P[a][0] + P[b][0])
            n[2] += (P[a][0] - P[b][0]) * (P[a][1] + P[b][1])
        ln = np.linalg.norm(n)
        if ln == 0:
            continue
        n = n / ln
        N[i] = n
        D[i] = n @ P.mean(0)
    return N, D


def max_violation(V, F, P, chunk=4096):
    """signed max distance of source points P outside polytope (V,F); per facet assignment"""
    N, D = face_planes(V, F)
    best = np.full(len(P), -1e30); arg = np.zeros(len(P), np.int64)
    for s in range(0, len(P), chunk):
        blk = P[s:s + chunk]
        d = blk @ N.T - D[None, :]
        i = d.argmax(1)
        best[s:s + chunk] = d[np.arange(len(i)), i]
        arg[s:s + chunk] = i
    return best, arg, N, D


def correct_containment(V, F, P, target=1e-7, maxit=25, log=print):
    """grow the convex polytope outward (per-facet plane push) until it contains all P"""
    it = 0; hist = []
    while it < maxit:
        best, arg, N, D = max_violation(V, F, P)
        mx = best.max()
        hist.append(float(mx))
        if mx <= target:
            break
        need = np.zeros(len(F))
        sel = best > target
        np.maximum.at(need, arg[sel], best[sel])
        disp = np.zeros_like(V)
        cnt = np.zeros(len(V))
        V = V.copy()
        for fi, f in enumerate(F):
            if need[fi] <= 0:
                continue
            for vi in f:
                disp[vi] = disp[vi] + need[fi] * N[fi]   # accumulate
        V = V + disp
        bm = do_hull(V)
        V, F = bm_arrays(bm); bm.free()
        it += 1
    return V, F, it, hist


results = []
for g_mm in GRIDS:
    g = g_mm * 1e-3
    for r in rows:
        link = r["link"]
        src = os.path.join(ROOT, r["visual"]["filename"])
        t0 = time.time()
        P = load_points(src)
        c = (P.max(0) + P.min(0)) / 2
        Q = P if g <= 0 else voxel_thin(P, g, c)
        bm = do_hull(Q)
        V0, F0 = bm_arrays(bm); bm.free()
        n_hull0 = len(F0)
        V, F, it, hist = correct_containment(V0, F0, P)
        bm = new_bm(V); bm.verts.ensure_lookup_table()
        for f in F:
            try:
                bm.faces.new([bm.verts[i] for i in f])
            except ValueError:
                pass
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
        me = bpy.data.meshes.new("h"); bm.to_mesh(me); bm.free()
        obj = bpy.data.objects.new("h", me); bpy.context.scene.objects.link(obj) if hasattr(bpy.context.scene.objects, "link") else bpy.context.collection.objects.link(obj)
        # triangulate copy count
        bm2 = bmesh.new(); bm2.from_mesh(me)
        bmesh.ops.triangulate(bm2, faces=bm2.faces[:])
        n_tri = len(bm2.faces); bm2.free()
        path = os.path.join(OUT, f"{link}__g{g_mm:g}.stl")
        for o in bpy.context.scene.objects:
            o.select_set(o == obj)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.wm.stl_export(filepath=path, ascii_format=False, apply_modifiers=True, global_scale=1.0)
        ent = dict(link=link, grid=g_mm, src_verts=len(P), thin_verts=len(Q),
                   hull_faces=n_hull0, final_faces=len(F), tris=n_tri, iters=it,
                   hist=[f"{x*1e6:.2f}" for x in hist], secs=round(time.time() - t0, 1))
        results.append(ent)
        print("V2", json.dumps(ent), flush=True)
json.dump(results, open(os.path.join(OUT, "v2_report.json"), "w"), indent=2)
print("DONE")
