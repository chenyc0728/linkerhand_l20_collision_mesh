#!/usr/bin/env python3
"""Hull fidelity review for the generated collision meshes (independent of Blender).

For every link it answers "is this hull still the part, or has it wrapped a concavity
into a slab?" with numbers instead of eyeballing:

  * how much of the *visual* surface lies on the collision surface
    (sample the visual surface, distance to the collision mesh):
        frac_0.2mm  fraction of the visual surface within 0.2 mm of the hull
        p50/p95/max distance visual surface -> hull
  * hull volume / bbox volume  (a slab or a straight column wrapping a curved finger
    shows up as a value far from the typical 0.45-0.7)
  * where the hull spans a concavity: the max distance found above, in mm, plus the
    direction in which the hull sticks out beyond the visual bbox (should be 0).

Usage:
    /path/to/graspqp/python tools/hull_fidelity.py --urdf linkerhand_l20.urdf \
        --out-json logs/hull_fidelity.json [--samples 4000]
"""

import argparse
import json
import os
import xml.etree.ElementTree as ET

import numpy as np
import trimesh
from trimesh.proximity import closest_point


def parse_urdf(urdf_path):
    root = ET.parse(urdf_path).getroot()
    rows = []
    for link in root.findall("link"):
        vis = link.find("visual/geometry/mesh")
        col = link.find("collision/geometry/mesh")
        if vis is None or col is None:
            continue
        rows.append((link.get("name"), vis.get("filename"), col.get("filename")))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", default="linkerhand_l20.urdf")
    ap.add_argument("--out-json", default="logs/hull_fidelity.json")
    ap.add_argument("--samples", type=int, default=4000)
    args = ap.parse_args()
    root = os.path.dirname(os.path.abspath(args.urdf))

    print(f"{'link':26s} {'hullF':>5s} {'vol/bbox':>8s} {'frac<0.2mm':>10s} "
          f"{'p50_um':>7s} {'p95_um':>7s} {'max_mm':>7s}  notes")
    records, flags = [], []
    for link, vis_rel, col_rel in parse_urdf(args.urdf):
        vis = trimesh.load(os.path.join(root, vis_rel), force="mesh", process=True)
        col = trimesh.load(os.path.join(root, col_rel), force="mesh", process=True)
        pts, _ = trimesh.sample.sample_surface(vis, args.samples, seed=7)
        _, dist, _ = closest_point(col, pts)
        dist_mm = np.asarray(dist) * 1e3
        frac = float((dist_mm < 0.2).mean())
        vb = vis.bounds
        cb = col.bounds
        vol_bbox = float(np.prod(cb[1] - cb[0]))
        ratio = float(col.volume / vol_bbox)
        excess = float(np.max(np.maximum(vb[0] - cb[0], cb[1] - vb[1])) * 1e3)
        diag_mm = float(np.linalg.norm(vb[1] - vb[0])) * 1e3
        note = []
        if frac < 0.05:
            note.append("almost no visual surface touches the hull (possible wrap)")
        if dist_mm.max() > 0.15 * diag_mm:
            note.append(f"hull spans a deep concavity (gap {dist_mm.max():.1f} mm "
                        f"= {dist_mm.max() / diag_mm * 1e2:.0f}% of the bbox diagonal)")
        if len(col.faces) > 5000:
            note.append(f"{len(col.faces)} faces (above the 3-5k recommendation)")
        rec = {
            "link": link,
            "collision_faces": int(len(col.faces)),
            "volume_mm3": round(float(col.volume) * 1e9, 2),
            "vol_over_bbox": round(ratio, 4),
            "frac_visual_on_hull_0p2mm": round(frac, 4),
            "dist_p50_um": round(float(np.percentile(dist_mm, 50)) * 1e3, 2),
            "dist_p95_um": round(float(np.percentile(dist_mm, 95)) * 1e3, 2),
            "dist_max_mm": round(float(dist_mm.max()), 3),
            "bbox_excess_mm": round(excess, 6),
            "gap_over_diag": round(float(dist_mm.max()) / diag_mm, 4),
            "notes": note,
        }
        records.append(rec)
        if note:
            flags.append((link, "; ".join(note), rec["dist_max_mm"], rec["frac_visual_on_hull_0p2mm"]))
        print(f"{link:26s} {rec['collision_faces']:5d} {ratio:8.3f} {frac:10.3f} "
              f"{rec['dist_p50_um']:7.1f} {rec['dist_p95_um']:7.1f} {rec['dist_max_mm']:7.3f}  "
              f"{'; '.join(note)}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, "w") as fh:
        json.dump({"records": records,
                   "flags": [{"link": a, "note": b, "dist_max_mm": c, "frac": d}
                             for a, b, c, d in flags]}, fh, indent=2)
    print(f"\n{len(flags)} link(s) flagged for human review"
          + (": " + ", ".join(f[0] for f in flags) if flags else ""))
    print(f"json -> {args.out_json}")


if __name__ == "__main__":
    main()
