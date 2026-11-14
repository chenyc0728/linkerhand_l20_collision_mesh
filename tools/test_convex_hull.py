"""Cross-check tools/convex_hull_np.py against scipy Qhull.

Runs inside Blender (numpy only); reference hulls are produced by the graspqp
env beforehand (tools/dump_scipy_hull.py) -> /tmp/hullref/<link>__g<grid>.npz
    blender -b --factory-startup --python tools/test_convex_hull.py -- /tmp/hullref 1.0
"""
import json, os, sys, glob
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from convex_hull_np import convex_hull, hull_is_convex, facets_planes

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
REF = argv[0] if argv else "/tmp/hullref"

rows = [r for r in json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "mapping.json"))) if r["visual"]]
bad = 0
print(f"{'link':26s} {'npts':>6s} {'Vnp':>5s} {'Vref':>5s} {'Fnp':>5s} {'Fref':>5s} {'vol_np':>11s} {'vol_ref':>11s} {'dvol':>8s} {'convex':>6s} {'worst_um':>8s}")
for r in rows:
    link = r["link"]
    f = [p for p in glob.glob(os.path.join(REF, f"{link}__*.npz"))]
    if not f:
        print(f"{link}: no reference"); continue
    z = np.load(sorted(f)[0])
    P, refV, refF, refVol = z["P"], z["V"], z["F"], float(z["vol"])
    try:
        res = convex_hull(P)
    except Exception as e:
        print(f"{link}: HULL FAILED {e}"); bad += 1; continue
    V, F = res.coords, res.facets
    # volume
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    vol = float(np.abs(np.einsum("ij,ij->i", a, np.cross(b, c)).sum()) / 6.0)
    # convex hull contains all input points?
    n, d, good = facets_planes(V, F)
    worst_pt = max(float((P[s:s+2048] @ n.T - d[None, :]).max()) for s in range(0, len(P), 2048))
    ok, worst = hull_is_convex(V, F)
    flag = "" if (ok and worst_pt <= 1e-9) else "  <-- BAD"
    if flag: bad += 1
    print(f"{link:26s} {len(P):6d} {len(V):5d} {len(refV):5d} {len(F):5d} {len(refF):5d} "
          f"{vol:11.6e} {refVol:11.6e} {(vol-refVol)/refVol*100:7.4f}% {str(ok):>6s} {worst_pt*1e6:8.4f}{flag}")
print("\nRESULT:", "ALL HULLS MATCH" if bad == 0 else f"{bad} PROBLEMS")
