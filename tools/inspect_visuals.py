import json, os, numpy as np, trimesh
rows = [r for r in json.load(open("tools/mapping.json")) if r["visual"]]
print(f"{'link':28s} {'faces':>7s} {'verts':>7s} {'watert':>6s} {'wind':>5s} {'vol':>12s} {'bodies':>6s}  bbox_extent")
out = {}
for r in rows:
    m = trimesh.load(r["visual"]["filename"], force="mesh", process=True)
    v = m.vertices; f = m.faces
    ext = m.bounds[1] - m.bounds[0]
    out[r["link"]] = dict(faces=int(len(f)), verts=int(len(v)),
                          watertight=bool(m.is_watertight), winding=bool(m.is_winding_consistent),
                          volume=float(m.volume), bodies=int(m.body_count),
                          extent=[float(x) for x in ext],
                          bmin=[float(x) for x in m.bounds[0]], bmax=[float(x) for x in m.bounds[1]])
    print(f"{r['link']:28s} {len(f):7d} {len(v):7d} {str(m.is_watertight):>6s} {str(m.is_winding_consistent):>5s} {m.volume:12.6f} {m.body_count:6d}  {np.round(ext,4)}")
json.dump(out, open("logs/visual_stats.json","w"), indent=2)
