#!/usr/bin/env python3
"""Independent validation of the generated collision meshes (trimesh / numpy).

Deliberately independent from the Blender build script: it re-parses the URDF,
loads the *visual* and the *collision* mesh of every link from disk and checks

  watertight (2-manifold, no boundary edges)      mesh.is_watertight
  winding consistent / outward                    is_winding_consistent + volume > 0
  connected components (bodies)                   mesh.body_count
  containment of the visual mesh                  every visual vertex inside the
                                                  collision hull (plane test)
  how far the collision surface sits from the      closest distance visual->collision
  visual surface (collision must be OUTSIDE)
  face count vs visual                            reduction factor
  bbox                                            collision bbox vs visual bbox

Usage:
    /path/to/graspqp/python tools/validate_collisions.py --urdf linkerhand_l20.urdf \
        [--out-json logs/validation.json] [--md   markdown table only]

No optional dependency beyond trimesh/numpy is used (trimesh is already in graspqp).
"""

import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import trimesh


def parse_urdf(urdf_path):
    root = ET.parse(urdf_path).getroot()
    rows = []
    for link in root.findall("link"):
        entry = {"link": link.get("name"), "visual": None, "collision": None}
        for tag in ("visual", "collision"):
            node = link.find(tag)
            if node is None:
                continue
            mesh = node.find("geometry/mesh")
            if mesh is None:
                continue
            origin = node.find("origin")
            entry[tag] = {
                "filename": mesh.get("filename"),
                "scale": np.array([float(x) for x in (mesh.get("scale") or "1 1 1").split()]),
                "xyz": np.array([float(x) for x in ((origin.get("xyz") if origin is not None else None) or "0 0 0").split()]),
                "rpy": np.array([float(x) for x in ((origin.get("rpy") if origin is not None else None) or "0 0 0").split()]),
            }
        rows.append(entry)
    return rows


def load(path, scale):
    m = trimesh.load(path, force="mesh", process=True)
    if not np.allclose(scale, 1.0):
        m.apply_scale(scale)
    return m


def point_outside_distance(collision, points):
    """Signed distance of `points` w.r.t. the *convex* collision mesh (positive = outside).

    Uses the plane-test on the triangular faces (exact for convex solids, which every
    collision mesh here is), so a positive value = the visual surface pokes out.
    """
    tri = collision.triangles            # (F,3,3)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    ln = np.linalg.norm(n, axis=1)
    ok = ln > 0
    n = n[ok] / ln[ok, None]
    d = (n * tri[ok][:, 0]).sum(1)
    best = np.full(len(points), -np.inf)
    for s in range(0, len(points), 4096):
        blk = points[s:s + 4096]
        proj = blk @ n.T - d[None, :]
        best[s:s + 4096] = proj.max(1)
    return best


def surface_offset(collision, points, sample=4000, seed=0):
    """Unsigned distance from visual surface points to the collision surface."""
    if len(points) > sample:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(points), sample, replace=False)
        points = points[idx]
    # distance to the collision surface, sampled densely on its triangles
    pts, _ = trimesh.sample.sample_surface(
        collision, max(20000, 8 * len(collision.faces)), seed=seed)
    from scipy.spatial import cKDTree
    dist, _ = cKDTree(pts).query(points)
    return dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", default="linkerhand_l20.urdf")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--tol-contain-um", type=float, default=1.0,
                    help="allowed visual penetration into the collision mesh [um]")
    ap.add_argument("--md", action="store_true", help="print a markdown table")
    ap.add_argument("--collision-root", default=None,
                    help="load collision meshes from this dir instead of the URDF path")
    args = ap.parse_args()

    urdf_path = os.path.abspath(args.urdf)
    project = os.path.dirname(urdf_path)
    rows = parse_urdf(urdf_path)

    table, failures = [], []
    if args.md:
        print("| link | visual faces | collision faces | reduction | watertight | winding | bodies | "
              "volume mm3 | max penetration um | offset mean mm | offset max mm | "
              "bbox excess mm | bbox deficit mm |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|", flush=True)
    if not args.md:
        print(f"{'link':26s} {'visF':>7s} {'colF':>5s} {'red':>6s} {'water':>5s} {'wind':>5s} "
          f"{'bodies':>6s} {'vol_mm3':>10s} {'pen_um':>8s} {'off_mean':>8s} {'off_max':>8s} "
              f"{'bbox_exc_mm':>11s} {'bbox_def_mm':>11s}", flush=True)
    for r in rows:
        link = r["link"]
        if not r["visual"] or not r["collision"]:
            continue
        vis_path = os.path.join(project, r["visual"]["filename"])
        col_path = os.path.join(project, r["collision"]["filename"])
        if args.collision_root:
            col_path = os.path.join(args.collision_root, os.path.basename(r["collision"]["filename"]))
        missing = [p for p in (vis_path, col_path) if not os.path.exists(p)]
        if missing:
            failures.append((link, f"missing file(s): {missing}"))
            continue
        vis = load(vis_path, r["visual"]["scale"])
        col = load(col_path, r["collision"]["scale"])

        vp = np.unique(np.round(vis.vertices, 9), axis=0)
        pen = point_outside_distance(col, vp)
        pen_max_um = float(pen.max() * 1e6)
        off = surface_offset(col, vp, seed=1)
        vb, cb = vis.bounds, col.bounds
        # per-axis: collision box grows (excess) or shrinks (deficit) w.r.t. the visual box
        exc = np.maximum(vb[0] - cb[0], cb[1] - vb[1]) * 1e3   # >0 : collision reaches further out
        dfc = np.maximum(cb[0] - vb[0], vb[1] - cb[1]) * 1e3   # >0 : collision stays inside
        rec = {
            "link": link,
            "visual_file": r["visual"]["filename"],
            "collision_file": r["collision"]["filename"],
            "visual_faces": int(len(vis.faces)),
            "collision_faces": int(len(col.faces)),
            "reduction": float(len(vis.faces) / max(len(col.faces), 1)),
            "watertight": bool(col.is_watertight),
            "winding_consistent": bool(col.is_winding_consistent),
            "bodies": int(col.body_count),
            "volume_m3": float(col.volume),
            "euler_number": int(col.euler_number),
            "penetration_um": pen_max_um,
            "offset_mean_mm": float(off.mean() * 1e3),
            "offset_p95_mm": float(np.percentile(off, 95) * 1e3),
            "offset_max_mm": float(off.max() * 1e3),
            "bbox_excess_mm": [round(float(x), 4) for x in exc],
            "bbox_deficit_mm": [round(float(x), 4) for x in dfc],
        }
        bad = []
        if not rec["watertight"]:
            bad.append("not watertight")
        if not rec["winding_consistent"]:
            bad.append("winding inconsistent")
        if rec["volume_m3"] <= 0:
            bad.append("volume <= 0")
        if rec["bodies"] != 1:
            bad.append(f"bodies={rec['bodies']}")
        if pen_max_um > args.tol_contain_um:
            bad.append(f"visual pokes out {pen_max_um:.2f} um")
        if np.any(dfc > 1e-6):
            bad.append("collision bbox smaller than visual bbox")
        rec["problems"] = bad
        if bad:
            failures.append((link, "; ".join(bad)))
        table.append(rec)
        if args.md:
            print(f"| `{link}` | {rec['visual_faces']} | {rec['collision_faces']} | "
                  f"{rec['reduction']:.1f}x | {'Y' if rec['watertight'] else 'N'} | "
                  f"{'Y' if rec['winding_consistent'] else 'N'} | {rec['bodies']} | "
                  f"{rec['volume_m3']*1e9:.1f} | {pen_max_um:.3f} | {rec['offset_mean_mm']:.4f} | "
                  f"{rec['offset_max_mm']:.4f} | {'/'.join(f'{x:.3f}' for x in exc)} | "
                  f"{'/'.join(f'{x:.3f}' for x in dfc)} |", flush=True)
        if not args.md:
            print(f"{link:26s} {len(vis.faces):7d} {len(col.faces):5d} {rec['reduction']:6.1f} "
                  f"{str(rec['watertight'])[0]:>5s} {str(rec['winding_consistent'])[0]:>5s} "
                  f"{rec['bodies']:6d} {rec['volume_m3']*1e9:10.3f} {pen_max_um:8.3f} "
                  f"{rec['offset_mean_mm']:8.4f} {rec['offset_max_mm']:8.4f} "
                  f"{np.array2string(exc, precision=3, floatmode='fixed'):>11s} "
                  f"{np.array2string(dfc, precision=3, floatmode='fixed'):>11s}", flush=True)

    tot_v = sum(t["visual_faces"] for t in table)
    tot_c = sum(t["collision_faces"] for t in table)
    print(f"\nTOTAL visual faces {tot_v}, collision faces {tot_c}, reduction {tot_v/max(tot_c,1):.1f}x")
    print(f"max offset (visual surface -> collision surface): "
          f"{max((t['offset_max_mm'] for t in table), default=0):.4f} mm")
    print(f"links checked {len(table)}, failing {len(failures)}")
    for link, msg in failures:
        print(f"  FAIL {link}: {msg}")
    print("RESULT:", "ALL CHECKS PASSED" if not failures else "PROBLEMS FOUND")

    if args.out_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
        json.dump({"urdf": urdf_path, "tol_contain_um": args.tol_contain_um,
                   "rows": table, "failures": [[l, m] for l, m in failures],
                   "totals": {"visual_faces": tot_v, "collision_faces": tot_c,
                              "reduction": tot_v / max(tot_c, 1)}},
                  open(args.out_json, "w"), indent=2)
        print("json ->", args.out_json)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
