"""Analyze v2 outputs vs exact convex hull of the source point cloud (chunked)."""
import json, glob, sys, numpy as np, trimesh
from scipy.spatial import ConvexHull
OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/exp2"
rows = [r for r in json.load(open("tools/mapping.json")) if r["visual"]]
rng = np.random.default_rng(7)
U = rng.normal(size=(20000, 3)); U /= np.linalg.norm(U, axis=1, keepdims=True)

def support(V, U, chunk=2000):
    out = np.empty(len(U))
    for s in range(0, len(U), chunk):
        out[s:s+chunk] = (V @ U[s:s+chunk].T).max(0)
    return out

print(f"{'link':24s} {'tris':>5s} {'wt':>3s} {'wind':>4s} {'bodies':>6s} {'vol_mm3':>8s} "
      f"{'bulge_um':>8s} {'viol_um':>7s} {'volratio':>8s}", flush=True)
for r in rows:
    link = r["link"]
    src = trimesh.load(r["visual"]["filename"], force="mesh", process=True)
    P = np.unique(np.round(src.vertices, 9), axis=0)
    f = glob.glob(f"{OUT}/{link}__*.stl")[0]
    m = trimesh.load(f, force="mesh", process=True)
    V = np.unique(np.round(m.vertices, 9), axis=0)
    hP = support(P, U); hV = support(V, U)
    bulge = (hV - hP).max() * 1e6
    viol = (hP - hV).max() * 1e6
    volP = ConvexHull(P).volume
    print(f"{link:24s} {len(m.faces):5d} {str(m.is_watertight)[0]:>3s} {str(m.is_winding_consistent)[0]:>4s} "
          f"{m.body_count:6d} {m.volume*1e9:8.1f} {bulge:8.1f} {viol:7.1f} {m.volume/volP:8.3f}", flush=True)
