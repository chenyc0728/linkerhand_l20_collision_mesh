"""Approximation error of the generated collision hulls w.r.t. the EXACT convex
hull of the corresponding visual mesh (reference computed with scipy/Qhull).

distance(Q -> conv(P))  : how far the collision surface bulges OUTSIDE the ideal hull
distance(conv(P) -> Q)  : should be ~0 (the collision hull must contain the visual)
Also reports the convexity of the visual shape: V_conv(P) / V_bbox.
"""
import json, sys, os, numpy as np, trimesh
from scipy.spatial import ConvexHull

def surf_samples(mesh, n, seed=3):
    pts, _ = trimesh.sample.sample_surface(mesh, n, seed=seed)
    return pts

def one_way(a_pts, b_mesh):
    d = trimesh.proximity.closest_point(b_mesh, a_pts)[1]
    return d

rows = [r for r in json.load(open("tools/mapping.json")) if r["visual"]]
grids = [(g, f"/tmp/col_g{g}") for g in (0, 0.5, 1.0, 1.5, 2.5)]
store = {}
for g, d in grids:
    out = []
    for r in rows:
        link = r["link"]
        vis = trimesh.load(r["visual"]["filename"], force="mesh", process=True)
        P = np.unique(np.round(vis.vertices, 9), axis=0)
        ch = ConvexHull(P)
        ref = trimesh.Trimesh(vertices=P, faces=ch.simplices, process=True)
        Q = trimesh.load(os.path.join(d, os.path.basename(r["collision"]["filename"])),
                         force="mesh", process=True)
        sq = surf_samples(Q, 6000)
        sref = surf_samples(ref, 6000)
        d_out = one_way(sq, ref)      # Q surface -> exact hull: bulge
        d_in = one_way(sref, Q)       # exact hull -> Q: should be 0
        bbox_v = np.prod(P.max(0) - P.min(0))
        out.append(dict(link=link, tris=int(len(Q.faces)),
                        bulge_mean_mm=float(d_out.mean()*1e3),
                        bulge_p95_mm=float(np.percentile(d_out, 95)*1e3),
                        bulge_max_mm=float(d_out.max()*1e3),
                        inset_max_mm=float(d_in.max()*1e3),
                        conv_ratio=float(ch.volume / bbox_v),
                        vol_ratio_q_vs_hull=float(Q.volume / ch.volume)))
    store[g] = out
    print(f"\n===== grid {g} mm =====", flush=True)
    print(f"{'link':26s} {'tris':>5s} {'bulge_mean':>10s} {'bulge_p95':>9s} {'bulge_max':>9s} {'inset_max':>9s} {'conv':>5s} {'volQ/volHull':>12s}")
    for r in out:
        print(f"{r['link']:26s} {r['tris']:5d} {r['bulge_mean_mm']:10.4f} {r['bulge_p95_mm']:9.4f} "
              f"{r['bulge_max_mm']:9.4f} {r['inset_max_mm']:9.4f} {r['conv_ratio']:5.2f} {r['vol_ratio_q_vs_hull']:12.3f}")
json.dump({str(k): v for k, v in store.items()}, open("logs/deviation_vs_exact_hull.json", "w"), indent=2)
