"""Reference convex hulls from scipy/Qhull for the cross-check (graspqp env)."""
import json, os, sys, numpy as np, trimesh
from scipy.spatial import ConvexHull
GRID = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
OUT = sys.argv[2] if len(sys.argv) > 2 else "/tmp/hullref"
os.makedirs(OUT, exist_ok=True)
rows = [r for r in json.load(open("tools/mapping.json")) if r["visual"]]
def voxel_thin(p, g, center):
    if g <= 0: return p
    ijk = np.floor((p - p.min(0)) / g).astype(np.int64)
    order = np.lexsort((ijk[:,2], ijk[:,1], ijk[:,0])); key, pts = ijk[order], p[order]
    start = np.ones(len(key), bool); start[1:] = np.any(np.diff(key, axis=0) != 0, axis=1)
    grp = np.cumsum(start)-1
    dist = np.linalg.norm(pts - center, axis=1)
    mx = np.full(grp.max()+1, -1.0); np.maximum.at(mx, grp, dist)
    cand = dist >= mx[grp]-1e-18
    seen=set(); keep=[]
    for i in np.nonzero(cand)[0]:
        if grp[i] not in seen: seen.add(grp[i]); keep.append(i)
    return pts[np.asarray(keep)]
for r in rows:
    m = trimesh.load(r["visual"]["filename"], force="mesh", process=True)
    P = np.unique(np.round(m.vertices, 9), axis=0)
    Q = voxel_thin(P, GRID*1e-3, (P.max(0)+P.min(0))/2)
    h = ConvexHull(Q)
    np.savez(os.path.join(OUT, f"{r['link']}__g{GRID}.npz"), P=Q, V=Q[h.vertices],
             F=h.simplices, vol=h.volume)
    print(r["link"], len(Q), len(h.vertices), len(h.simplices), f"{h.volume:.6e}")
