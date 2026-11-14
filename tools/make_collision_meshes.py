#!/usr/bin/env python3
"""Generate true collision-convex-hull meshes for the linkerhand_l20 URDF.

Headless usage (everything runs inside Blender, no GUI, no extra packages):

    /home/chenyc/Blender/blender-5.1.2-linux-x64/blender --background --factory-startup \
        --python tools/make_collision_meshes.py -- \
            --urdf linkerhand_l20.urdf \
            --grid-mm 1.0 \
            --log logs/collision_build.json \
            --backup-dir backup/visual_collisions

Pipeline per URDF link (prints + logs every step):

  1. read the URDF, take the <visual> mesh of the link and the exact
     <collision><geometry><mesh filename=...> it must produce (same path, same
     filename, same format, same units/axes -- nothing in the URDF is changed);
  2. import the visual mesh (STL is a triangle soup: weld coincident vertices) ->
     point cloud of the real surface, including all wrinkles, screw bosses, tubes;
  3. voxel-thin the cloud on a --grid-mm grid, keeping per voxel the point farthest
     from the cloud centroid.  This only *seeds* the hull (it makes the first hull
     cheap); it does not decide the final shape;
  4. exact float64 convex hull of the cloud (tools/convex_hull_np.py -- Blender's own
     bmesh.ops.convex_hull returns non-convex facets for clouds this large, so it is
     not used).  Every hull vertex is therefore a real visual-mesh vertex;
  5. containment loop: measure the exact violation of every visual vertex against the
     hull facet planes and re-hull with the offending vertices added, until the hull
     contains the complete visual surface.  The fixpoint is exactly the convex hull of
     the visual mesh, so the result contains the visual by construction, has no
     invented geometry and bulges only where a convex body must bulge over a concave
     feature (screw holes, tubes, supports and wrinkles are simply covered);
  6. cleanup: weld/dissolve degenerates, consistent outward winding (signed volume
     > 0), verify 2-manifold and one connected component;
  7. export to the URDF collision path (binary STL, metres, no axis conversion);
  8. re-import the written file and verify it from disk (watertight / winding /
     volume / components / convexity / containment of the visual vertices / bbox /
     tri count).
"""

import argparse
import json
import os
import shutil
import sys
import time
import xml.etree.ElementTree as ET

import bpy
import bmesh
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convex_hull_np import (convex_hull, facets_planes, hull_is_convex,  # noqa: E402
                            inflate_facets, HullError)


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #


def log(msg):
    print(msg, flush=True)


def parse_urdf(urdf_path):
    """Ordered mapping of link -> the visual/collision mesh the URDF references."""
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
                "scale": [float(x) for x in (mesh.get("scale") or "1 1 1").split()],
                "xyz": [float(x) for x in ((origin.get("xyz") if origin is not None else None) or "0 0 0").split()],
                "rpy": [float(x) for x in ((origin.get("rpy") if origin is not None else None) or "0 0 0").split()],
            }
        rows.append(entry)
    return rows


def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def import_geometry(path):
    """Import a mesh file, join everything, weld, return (points float64, n_faces)."""
    ext = os.path.splitext(path)[1].lower()
    reset_scene()
    if ext == ".stl":
        bpy.ops.wm.stl_import(filepath=path, global_scale=1.0, use_scene_unit=False)
    elif ext == ".obj":
        bpy.ops.wm.obj_import(filepath=path, forward_axis="Y", up_axis="Z", global_scale=1.0)
    elif ext == ".ply":
        bpy.ops.wm.ply_import(filepath=path, global_scale=1.0)
    else:
        raise RuntimeError(f"unsupported mesh format: {ext}")

    objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not objs:
        raise RuntimeError(f"nothing imported from {path}")
    for o in bpy.context.scene.objects:
        o.select_set(o in objs)
    bpy.context.view_layer.objects.active = objs[0]
    if len(objs) > 1:
        bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active

    bm = bmesh.new()
    bm.from_mesh(obj.data)
    n_faces = len(bm.faces)
    bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=1e-7)      # 0.1 um
    pts = np.array([v.co[:] for v in bm.verts], dtype=np.float64)
    bm.free()
    return pts, n_faces


def voxel_thin(points, grid, center):
    """Indices of one point per `grid` voxel: the one farthest from `center`."""
    if grid <= 0:
        return np.arange(len(points))
    ijk = np.floor((points - points.min(0)) / grid).astype(np.int64)
    order = np.lexsort((ijk[:, 2], ijk[:, 1], ijk[:, 0]))
    key, pts = ijk[order], points[order]
    start = np.ones(len(key), bool)
    start[1:] = np.any(np.diff(key, axis=0) != 0, axis=1)
    group = np.cumsum(start) - 1
    dist = np.linalg.norm(pts - center, axis=1)
    maxdist = np.full(group.max() + 1, -1.0)
    np.maximum.at(maxdist, group, dist)
    cand = dist >= maxdist[group] - 1e-18
    seen, keep = set(), []
    order_idx = order[np.nonzero(cand)[0]]
    for i, g in zip(order_idx, group[np.nonzero(cand)[0]]):
        if g not in seen:
            seen.add(g)
            keep.append(int(i))
    return np.asarray(sorted(keep), dtype=np.int64)


def max_violation(V, F, points, inside=None, chunk=4096):
    """Per point: how far it sticks out of the convex polytope (V,F), and its facet."""
    N, D, good = facets_planes(V, F, inside=inside)
    best = np.full(len(points), -1e30)
    arg = np.zeros(len(points), np.int64)
    for s in range(0, len(points), chunk):
        blk = points[s:s + chunk]
        proj = blk @ N.T - D[None, :]
        if not good.all():
            proj[:, ~good] = -np.inf
        i = proj.argmax(1)
        best[s:s + chunk] = proj[np.arange(len(i)), i]
        arg[s:s + chunk] = i
    return best, arg, N, D


def degenerate_facets(V, F, min_height=1e-9):
    """Hull vertices taking part in a (nearly) zero-area facet.

    These appear where the visual mesh samples a straight CAD edge with collinear
    vertices; the facets are micrometre sized and their area is < 1e-16 m^2, so they
    are reported (see `mesh_stats`) but not modified -- removing the vertex would
    dent the hull, which the containment check refuses.
    """
    V = np.asarray(V, dtype=np.float64)
    tips = set()
    for f in np.asarray(F):
        a, b, c = V[f[0]], V[f[1]], V[f[2]]
        longest = max(np.linalg.norm(b - a), np.linalg.norm(c - b), np.linalg.norm(a - c))
        area = 0.5 * float(np.linalg.norm(np.cross(b - a, c - a)))
        if longest <= 0 or 2.0 * area / longest > min_height:
            continue
        # the "tip" is the vertex whose corner angle is closest to 180 degrees
        ang = []
        for i, (p, q, r) in enumerate(((a, b, c), (b, c, a), (c, a, b))):
            u, w = q - p, r - p
            nu, nw = np.linalg.norm(u), np.linalg.norm(w)
            if nu == 0 or nw == 0:
                ang.append((0.0, f[i]))
                continue
            ang.append((float(u @ w / (nu * nw)), f[i]))
        tips.add(min(ang)[1])
    return sorted(tips)


def hull_containing(points, seed_idx, max_iter=15, tol=1e-9):
    """Convex hull of the surface that provably contains the complete surface.

    `seed_idx` selects the points used for the first hull (the voxel-thinned cloud) --
    it is an accelerator only.  The loop then re-hulls with the *actual* surface
    points that still stick out, so the hull keeps only real surface points as
    vertices (no invented, inflated geometry) and converges to the exact convex hull of
    the visual mesh: every added point is a visual vertex, and the loop stops when the
    newest hull already contains every visual vertex.

    Returns (V, F, iterations, violation trace, point indices used).
    """
    idx = np.unique(np.asarray(seed_idx, dtype=np.int64))
    trace = []
    V = F = None
    used = idx
    for it in range(max_iter + 1):
        res = convex_hull(points[idx])
        V, F, used = res.coords, res.facets, idx
        best, _, _, _ = max_violation(V, F, points)
        trace.append(float(best.max()))
        if best.max() <= tol:
            break
        if it == max_iter:
            return V, F, max_iter, trace, idx
        viol = np.nonzero(best > tol)[0]
        idx = np.unique(np.concatenate([idx, viol]))

    return V, F, it, trace, idx


def signed_volume(bm):
    vol = 0.0
    for f in bm.faces:
        vs = [l.vert.co[:] for l in f.loops]
        for i in range(1, len(vs) - 1):
            a = np.asarray(vs[0], dtype=np.float64)
            b = np.asarray(vs[i], dtype=np.float64)
            c = np.asarray(vs[i + 1], dtype=np.float64)
            vol += float(np.dot(a, np.cross(b, c))) / 6.0
    return vol


def mesh_stats(bm):
    bm.verts.ensure_lookup_table()
    bm.verts.index_update()
    bm.faces.ensure_lookup_table()
    bm.faces.index_update()
    hist = {}
    for e in bm.edges:
        k = len(e.link_faces)
        hist[k] = hist.get(k, 0) + 1
    seen, comps = set(), 0
    for f in bm.faces:
        if f.index in seen:
            continue
        comps += 1
        stack = [f]
        seen.add(f.index)
        while stack:
            cf = stack.pop()
            for e in cf.edges:
                for nf in e.link_faces:
                    if nf.index not in seen:
                        seen.add(nf.index)
                        stack.append(nf)
    return {
        "verts": int(len(bm.verts)),
        "faces": int(len(bm.faces)),
        "boundary_edges": int(hist.get(1, 0)),
        "nonmanifold_edges": int(sum(v for k, v in hist.items() if k > 2)),
        "components": int(comps),
    }


def build_bmesh(V, F):
    """Closed, outward-oriented, welded triangle mesh from arrays."""
    bm = bmesh.new()
    verts = [bm.verts.new(tuple(map(float, p))) for p in V]
    bm.verts.ensure_lookup_table()
    for tri in F:
        idx = list(dict.fromkeys(int(i) for i in tri))
        if len(idx) != 3:
            continue
        try:
            bm.faces.new([verts[i] for i in idx])
        except ValueError:                       # duplicate face
            pass
    bm.faces.ensure_lookup_table()
    bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=1e-9)
    bmesh.ops.dissolve_degenerate(bm, dist=1e-12, edges=bm.edges[:])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    if signed_volume(bm) < 0:
        bmesh.ops.reverse_faces(bm, faces=bm.faces[:])
    return bm


def bm_to_arrays(bm):
    bm.verts.ensure_lookup_table()
    bm.verts.index_update()
    bm.faces.ensure_lookup_table()
    V = np.array([v.co[:] for v in bm.verts], dtype=np.float64)
    F = np.array([[v.index for v in f.verts] for f in bm.faces], dtype=np.int64)
    return V, F


def export_mesh(bm, path, ascii_format=False):
    reset_scene()
    me = bpy.data.meshes.new("collision")
    bm.to_mesh(me)
    obj = bpy.data.objects.new("collision", me)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    ext = os.path.splitext(path)[1].lower()
    if ext == ".stl":
        bpy.ops.wm.stl_export(filepath=path, ascii_format=ascii_format,
                              export_selected_objects=True, global_scale=1.0,
                              use_scene_unit=False, apply_modifiers=True)
    elif ext == ".obj":
        bpy.ops.wm.obj_export(filepath=path, export_selected_objects=True,
                              forward_axis="Y", up_axis="Z", global_scale=1.0,
                              apply_modifiers=True)
    elif ext == ".ply":
        bpy.ops.wm.ply_export(filepath=path, export_selected_objects=True,
                              global_scale=1.0, apply_modifiers=True)
    else:
        raise RuntimeError(f"unsupported export format: {ext}")


def verify_exported(path, src_points, src_faces, margin_um):
    """Re-import the written file and check it purely from disk."""
    pts, _ = import_geometry(path)
    obj = [o for o in bpy.context.scene.objects if o.type == "MESH"][0]
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    st = mesh_stats(bm)
    vol = signed_volume(bm)
    V, F = bm_to_arrays(bm)
    bm.free()

    best, _, _, _ = max_violation(V, F, src_points)
    vb = np.array([src_points.min(0), src_points.max(0)])
    cb = np.array([pts.min(0), pts.max(0)])
    excess = np.maximum(vb[0] - cb[0], cb[1] - vb[1])       # collision reaches further out
    deficit = np.maximum(cb[0] - vb[0], vb[1] - cb[1])      # collision stays inside
    ok_convex, worst_convex = hull_is_convex(V, F, tol=1e-9)
    return {
        "tris": int(st["faces"]),
        "verts": int(st["verts"]),
        "watertight": bool(st["boundary_edges"] == 0 and st["nonmanifold_edges"] == 0),
        "boundary_edges": st["boundary_edges"],
        "nonmanifold_edges": st["nonmanifold_edges"],
        "components": st["components"],
        "convex": bool(ok_convex),
        "convex_violation_um": float(worst_convex * 1e6),
        "volume_m3": float(vol),
        "volume_positive": bool(vol > 0),
        "max_penetration_um": float(best.max() * 1e6),
        "contains_visual": bool(best.max() <= margin_um * 1e-6),
        "bbox_excess_mm": [round(float(x) * 1e3, 4) for x in excess],
        "bbox_deficit_mm": [round(float(x) * 1e3, 4) for x in deficit],
        "bbox_contains_visual": bool(np.all(deficit <= 1e-9)),
        "visual_faces_before": int(src_faces),
    }


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", required=True)
    ap.add_argument("--out-dir", default=None,
                    help="write collision meshes here (default: the URDF's own path)")
    ap.add_argument("--grid-mm", type=float, default=1.0,
                    help="seed grid for the first hull (0 = use every vertex from the start)")
    ap.add_argument("--margin-um", type=float, default=1.0,
                    help="containment tolerance used for reporting/verification")
    ap.add_argument("--log", default=None, help="JSON report path")
    ap.add_argument("--only", default=None, help="comma separated link names")
    ap.add_argument("--backup-dir", default=None,
                    help="copy the current collision meshes here before overwriting")
    ap.add_argument("--mode", choices=("exact", "coarse"), default="exact",
                    help="exact = convex hull of the visual mesh (no bulge); "
                         "coarse = seed hull with its facets pushed out to the true "
                         "support (fewer faces, small bulge)")
    ap.add_argument("--offset-um", default="auto",
                    help="coarse mode only: extra clearance added to every facet plane "
                         "on top of the measured thinning cut ('auto' = cut + 50 um)")
    ap.add_argument("--ascii", action="store_true", help="write ASCII STL")
    ap.add_argument("--dry-run", action="store_true", help="compute only, write nothing")
    args = ap.parse_args(argv)

    urdf_abs = os.path.abspath(args.urdf)
    project = os.path.dirname(urdf_abs)
    rows = parse_urdf(urdf_abs)
    if args.only:
        want = {s.strip() for s in args.only.split(",") if s.strip()}
        rows = [r for r in rows if r["link"] in want]

    log(f"[info] blender    {bpy.app.version_string}")
    log(f"[info] urdf       {urdf_abs}")
    log(f"[info] grid       {args.grid_mm} mm ({'no thinning' if args.grid_mm <= 0 else 'voxel-thinned'})")
    log(f"[info] margin     {args.margin_um} um")
    log(f"[info] mode       {args.mode}")
    log(f"[info] out-dir    {args.out_dir or '(URDF collision paths)'}")
    if args.backup_dir and not args.dry_run:
        os.makedirs(args.backup_dir, exist_ok=True)

    records, problems = [], []
    t0 = time.time()
    for r in rows:
        link = r["link"]
        if not r["visual"] or not r["collision"]:
            records.append({"link": link, "skipped": "no visual or collision mesh"})
            continue
        vis_rel, col_rel = r["visual"]["filename"], r["collision"]["filename"]
        vis_path = os.path.join(project, vis_rel)
        col_path = (os.path.join(args.out_dir, os.path.basename(col_rel))
                    if args.out_dir else os.path.join(project, col_rel))
        if not os.path.exists(vis_path):
            problems.append((link, f"missing visual mesh {vis_path}"))
            continue

        t1 = time.time()
        src_points, src_faces = import_geometry(vis_path)
        centre = (src_points.max(0) + src_points.min(0)) / 2.0
        seed = voxel_thin(src_points, args.grid_mm * 1e-3, centre)
        try:
            if args.mode == "coarse":
                res0 = convex_hull(src_points[seed])
                best0, _, _, _ = max_violation(res0.coords, res0.facets, src_points)
                cut = float(best0.max())
                off = (cut + 50e-6) if args.offset_um == "auto" else float(args.offset_um) * 1e-6
                log(f"[info] {link}: thinning cut {cut * 1e6:.1f} um -> facet offset {off * 1e6:.1f} um")
                V, F, iters, trace = inflate_facets(res0.coords, res0.facets, src_points,
                                                    offset=off, max_move=5.0 * off)
                used = seed
                iters = [iters, round(cut * 1e6, 1)]
            else:
                V, F, iters, trace, used = hull_containing(src_points, seed)
        except HullError as exc:                      # pragma: no cover - safety net
            problems.append((link, f"convex hull failed: {exc}"))
            log(f"[fail] {link}: convex hull failed ({exc})")
            continue
        hull_faces = len(F)
        detail = iters if isinstance(iters, list) else None
        iters = iters[0] if isinstance(iters, list) else iters
        bm = build_bmesh(V, F)
        st = mesh_stats(bm)
        vol = signed_volume(bm)

        # last-resort repair if the weld/dissolve left the mesh open
        if st["boundary_edges"] or st["nonmanifold_edges"] or st["components"] != 1:
            log(f"[warn] {link}: mesh not clean after cleanup {st} -> re-hulling")
            V2, _ = bm_to_arrays(bm)
            bm.free()
            hull = convex_hull(V2)
            bm = build_bmesh(hull.coords, hull.facets)
            st = mesh_stats(bm)
            vol = signed_volume(bm)

        final_tris = st["faces"]
        if not args.dry_run:
            if args.backup_dir and os.path.exists(col_path):
                shutil.copy2(col_path, os.path.join(args.backup_dir, os.path.basename(col_path)))
            export_mesh(bm, col_path, ascii_format=args.ascii)
        bm.free()

        rec = {
            "link": link,
            "visual_file": vis_rel,
            "collision_file": os.path.relpath(col_path, project),
            "visual_faces": int(src_faces),
            "visual_verts_welded": int(len(src_points)),
            "hull_input_verts": int(len(used)),
            "degenerate_hull_facets": len(degenerate_facets(V, F)),
            "coarse_info": detail,
            "hull_facets": int(hull_faces),
            "collision_tris": int(final_tris),
            "reduction": round(src_faces / max(final_tris, 1), 2),
            "inflation_iters": int(iters),
            "violation_trace_um": [round(x * 1e6, 3) for x in trace],
            "volume_m3": float(vol),
            "stats": st,
            "secs": round(time.time() - t1, 2),
        }
        if not args.dry_run:
            rec["verify"] = verify_exported(col_path, src_points, src_faces, args.margin_um)
            v = rec["verify"]
            bad = []
            if not v["watertight"]:
                bad.append(f"not watertight (boundary={v['boundary_edges']}, "
                           f"nonmanifold={v['nonmanifold_edges']})")
            if v["components"] != 1:
                bad.append(f"components={v['components']}")
            if not v["volume_positive"]:
                bad.append("volume<=0")
            if not v["contains_visual"]:
                bad.append(f"visual pokes out {v['max_penetration_um']:.2f} um")
            if not v["convex"]:
                bad.append(f"not convex ({v['convex_violation_um']:.2f} um)")
            if bad:
                problems.append((link, "; ".join(bad)))
                rec["problems"] = bad
        records.append(rec)
        v = rec.get("verify", {})
        log(f"[ok] {link:24s} tris {final_tris:5d} (vis {src_faces:6d}, {rec['reduction']:6.1f}x) "
            f"iters {iters:2d} pen {v.get('max_penetration_um', float('nan')):7.3f}um "
            f"wt {v.get('watertight')} comps {v.get('components')} "
            f"bbox+ {v.get('bbox_excess_mm')} ({rec['secs']}s)")

    tot_v = sum(r.get("visual_faces", 0) for r in records)
    tot_c = sum(r.get("collision_tris", 0) for r in records)
    summary = {
        "blender": bpy.app.version_string,
        "urdf": urdf_abs,
        "grid_mm": args.grid_mm,
        "margin_um": args.margin_um,
        "out_dir": args.out_dir,
        "records": records,
        "problems": [{"link": l, "problem": p} for l, p in problems],
        "totals": {"visual_faces": tot_v, "collision_tris": tot_c,
                   "reduction": round(tot_v / max(tot_c, 1), 2)},
        "total_secs": round(time.time() - t0, 2),
    }
    log(f"[info] total faces {tot_v} -> {tot_c} ({summary['totals']['reduction']}x), "
        f"{len(problems)} problem(s), {summary['total_secs']}s")
    if args.log:
        os.makedirs(os.path.dirname(os.path.abspath(args.log)), exist_ok=True)
        with open(args.log, "w") as fh:
            json.dump(summary, fh, indent=2)
        log(f"[info] report -> {args.log}")
    for link, p in problems:
        log(f"[problem] {link}: {p}")
    log("DONE")


if __name__ == "__main__":
    main()
