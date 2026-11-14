#!/usr/bin/env python3
"""Exact convex hull (QuickHull) for 3D point clouds in pure numpy.

Why this exists
---------------
Blender's ``bmesh.ops.convex_hull`` is an epsilon-based, single-precision routine.  On
the point clouds we get from the visual meshes (1e4..2e4 points) it silently returns
*self-inconsistent* polytopes: 10-20 % of the returned facets are not supporting
planes of the returned vertex set, i.e. the "hull" is locally concave by up to several
millimetres (measured: 454 of 3036 facets on hand_base_link).  That is fatal for a
collision mesh, so the hull is computed here in float64 with a plain QuickHull.
numpy is bundled with Blender, so this stays inside the headless Blender process.

Algorithm (standard QuickHull)
------------------------------
* start from a tetrahedron spanned by 4 extremal points,
* hand every remaining point to the facet it is farthest outside of,
* repeatedly take the facet with the farthest outside point, delete every facet that
  point can see (its conflict cone), stitch the freed horizon edges to that point and
  re-distribute the orphaned outside points,
* every facet is a supporting plane by construction; the caller re-verifies this with
  :func:`hull_is_convex`.

API
---
    res = convex_hull(points)                 # res.coords, res.facets, res.verts_idx
    ok, worst = hull_is_convex(res.coords, res.facets)
"""

import heapq

import numpy as np


class HullError(RuntimeError):
    pass


class ConvexHullResult:
    """Hull vertex coordinates, their indices in the input array, and the facets.

    `verts_idx` indexes the array that was passed to :func:`convex_hull` (not the
    caller's cloud), `facets` indexes `coords`.
    """

    __slots__ = ("coords", "facets", "verts_idx")

    def __init__(self, coords, facets, verts_idx):
        self.coords = coords
        self.facets = facets
        self.verts_idx = verts_idx

    def __repr__(self):
        return f"<ConvexHull {len(self.coords)} verts, {len(self.facets)} facets>"


def convex_hull(points, tol=None):
    pts = np.ascontiguousarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise HullError("points must be (N,3)")
    if len(pts) < 4:
        raise HullError("need at least 4 points")

    uniq, inv = np.unique(np.round(pts, 12), axis=0, return_inverse=True)
    inv = inv.ravel()
    if len(uniq) < 4:
        raise HullError("degenerate point set (<4 unique points)")
    first_orig = np.full(len(uniq), -1, dtype=np.int64)
    first_orig[inv[::-1]] = np.arange(len(pts), dtype=np.int64)[::-1]

    ext = uniq.max(0) - uniq.min(0)
    scale = float(np.linalg.norm(ext))
    if tol is None:
        tol = max(scale, 1.0) * 1e-12

    # ---------------- initial tetrahedron -----------------------------------
    i0 = int(np.argmin(uniq[:, 0]))
    d = np.linalg.norm(uniq - uniq[i0], axis=1)
    i1 = int(np.argmax(d))
    if d[i1] <= tol:
        raise HullError("all points coincide")
    axis = uniq[i1] - uniq[i0]
    axis /= np.linalg.norm(axis)
    w = uniq - uniq[i0]
    perp = w - np.outer(w @ axis, axis)
    dp = np.linalg.norm(perp, axis=1)
    i2 = int(np.argmax(dp))
    if dp[i2] <= tol:
        raise HullError("all points collinear")
    nrm = np.cross(uniq[i1] - uniq[i0], uniq[i2] - uniq[i0])
    nrm /= np.linalg.norm(nrm)
    hgt = w @ nrm
    i3 = int(np.argmax(np.abs(hgt)))
    if abs(hgt[i3]) <= tol:
        raise HullError("all points coplanar")
    inside = uniq[[i0, i1, i2, i3]].mean(0)

    def make_face(tri, out=None):
        a, b, c = uniq[tri[0]], uniq[tri[1]], uniq[tri[2]]
        n = np.cross(b - a, c - a)
        ln = np.linalg.norm(n)
        if ln == 0.0:
            return None
        n = n / ln
        dd = float(n @ a)
        if n @ inside > dd:
            n, dd = -n, -dd
            tri = (tri[0], tri[2], tri[1])
        return {"v": tri, "n": n, "d": dd, "out": out if out is not None else []}

    faces = {}
    edge_map = {}
    next_id = [0]

    def add_face(fdict):
        if fdict is None:
            return None
        fid = next_id[0]
        next_id[0] += 1
        a, b, c = fdict["v"]
        for u, v in ((a, b), (b, c), (c, a)):
            edge_map.setdefault((min(u, v), max(u, v)), set()).add(fid)
        faces[fid] = fdict
        return fid

    def remove_face(fid):
        f = faces.pop(fid)
        a, b, c = f["v"]
        for u, v in ((a, b), (b, c), (c, a)):
            s = edge_map.get((min(u, v), max(u, v)))
            if s is not None:
                s.discard(fid)
                if not s:
                    del edge_map[(min(u, v), max(u, v))]

    base = [(i0, i1, i2), (i0, i1, i3), (i1, i2, i3), (i2, i0, i3)]
    for tri in base:
        f = make_face(tri)
        if f is None:
            raise HullError("degenerate starting tetrahedron")
        add_face(f)

    # ---------------- distribute the remaining points -----------------------
    rest = np.setdiff1d(np.arange(len(uniq), dtype=np.int64),
                        np.array([i0, i1, i2, i3], dtype=np.int64))
    bucket = {}
    for fid in list(faces):
        f = faces[fid]
        if len(rest):
            dist = uniq[rest] @ f["n"] - f["d"]
            sel = rest[dist > tol]
            f["out"] = sel.tolist()
        else:
            f["out"] = []
        if f["out"]:
            dd = uniq[np.asarray(f["out"])] @ f["n"] - f["d"]
            bucket[fid] = -float(dd.max())

    heap = [(v, fid) for fid, v in bucket.items()]
    heapq.heapify(heap)

    # ---------------- main loop ---------------------------------------------
    guard = 0
    while heap:
        guard += 1
        if guard > 4_000_000:
            raise HullError("hull did not converge")
        negfar, fid = heapq.heappop(heap)
        f = faces.get(fid)
        if f is None or not f["out"]:
            continue
        cand = np.asarray(f["out"], dtype=np.int64)
        dist = uniq[cand] @ f["n"] - f["d"]
        k = int(np.argmax(dist))
        if dist[k] <= tol:
            continue
        pi = int(cand[k])
        p = uniq[pi]

        # conflict cone (BFS over shared edges)
        visible = {fid}
        stack = [fid]
        while stack:
            cf = faces[stack.pop()]
            a, b, c = cf["v"]
            for u, v in ((a, b), (b, c), (c, a)):
                for nb in tuple(edge_map.get((min(u, v), max(u, v)), ())):
                    if nb in visible:
                        continue
                    fnb = faces[nb]
                    if p @ fnb["n"] - fnb["d"] > tol:
                        visible.add(nb)
                        stack.append(nb)

        # horizon: edges of the visible faces whose neighbour is not visible
        horizon = []
        for vfid in visible:
            a, b, c = faces[vfid]["v"]
            for u, v in ((a, b), (b, c), (c, a)):
                nbrs = edge_map.get((min(u, v), max(u, v)), ())
                if any(x not in visible for x in nbrs):
                    horizon.append((u, v))
        if not horizon:
            raise HullError("empty horizon (numerical failure)")

        orphans = []
        for vfid in visible:
            orphans.extend(faces[vfid]["out"])
            remove_face(vfid)

        new_ids = []
        for (u, v) in horizon:
            if u == pi or v == pi:
                continue
            nf = make_face((u, v, pi))
            if nf is None:
                continue
            nid = add_face(nf)
            if nid is not None:
                new_ids.append(nid)

        if not new_ids:
            raise HullError("no new facets (numerical failure)")

        cand = np.unique(np.asarray(orphans, dtype=np.int64)) if orphans else np.empty(0, np.int64)
        if len(cand):
            cand = cand[cand != pi]
        if len(cand):
            best_f = np.full(len(cand), -1, dtype=np.int64)
            best_d = np.full(len(cand), tol)
            for nid in new_ids:
                nf = faces[nid]
                dd = uniq[cand] @ nf["n"] - nf["d"]
                upd = dd > best_d
                best_d[upd] = dd[upd]
                best_f[upd] = nid
            for nid in new_ids:
                sel = cand[best_f == nid]
                if len(sel):
                    faces[nid]["out"] = sel.tolist()
                    heap.append((-(float(best_d[best_f == nid].max())), nid))
        heapq.heapify(heap)

    if len(faces) < 4:
        raise HullError("hull degenerate")

    vids = sorted({i for f in faces.values() for i in f["v"]})
    remap = {old: new for new, old in enumerate(vids)}
    facets = np.array([[remap[i] for i in faces[fid]["v"]] for fid in sorted(faces)],
                      dtype=np.int64)
    coords = uniq[np.asarray(vids, dtype=np.int64)]
    verts_idx = first_orig[np.asarray(vids, dtype=np.int64)]
    return ConvexHullResult(coords, facets, verts_idx)


# --------------------------------------------------------------------------- #
# verification / plane helpers
# --------------------------------------------------------------------------- #


def facets_planes(V, F, inside=None):
    """Outward facet planes (n . x <= d) for triangles V[F], orientation-checked."""
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    ln = np.linalg.norm(n, axis=1)
    good = ln > 0
    n[good] /= ln[good, None]
    d = (n * V[F[:, 0]]).sum(1)
    c = V.mean(0) if inside is None else np.asarray(inside, dtype=float)
    flip = (n @ c) > d
    n[flip] *= -1.0
    d[flip] *= -1.0
    return n, d, good


def hull_is_convex(V, F, tol=1e-9, chunk=1024):
    """(ok, worst) - worst = max over vertices/planes of the plane violation."""
    n, d, good = facets_planes(V, F)
    V = np.asarray(V, dtype=np.float64)
    worst = -np.inf
    for s in range(0, len(V), chunk):
        proj = V[s:s + chunk] @ n.T - d[None, :]
        if not good.all():
            proj[:, ~good] = -np.inf
        worst = max(worst, float(proj.max()))
    return worst <= tol, worst


def _incident(n_verts, F):
    inc = [[] for _ in range(n_verts)]
    for fi, f in enumerate(F):
        for vi in f:
            inc[int(vi)].append(fi)
    return inc


def inflate_facets(V, F, points, offset=0.0, max_iter=25, sv_tol=0.05, max_move=None):
    """Grow the polytope by moving its facet planes out, keeping the facet structure.

    Each vertex is re-solved from *its own* pushed incident planes (minimum-norm
    correction through a truncated SVD, so that nearly coplanar facet triples cannot
    throw the vertex away).  Every step is verified against `points` and halved when it
    does not reduce the violation, so the procedure cannot diverge; it stops when the
    cloud is inside (violation <= 0).

    Returns (V, F, iterations, violation trace).
    """
    V = np.array(V, dtype=np.float64, copy=True)
    F = np.asarray(F, dtype=np.int64)
    inc = _incident(len(V), F)
    P = np.asarray(points, dtype=np.float64)
    trace = []

    def violation(Vc):
        n, d, good = facets_planes(Vc, F)
        worst = -np.inf
        for s in range(0, len(P), 4096):
            proj = P[s:s + 4096] @ n.T - d[None, :]
            if not good.all():
                proj[:, ~good] = -np.inf
            worst = max(worst, float(proj.max()))
        return worst

    scale = float(np.linalg.norm(V.max(0) - V.min(0)))
    if max_move is None:
        max_move = 0.1 * scale
    damp = 1.0
    for it in range(max_iter + 1):
        N, D, good = facets_planes(V, F)
        viol = violation(V)
        trace.append(viol)
        if viol <= 0.0:
            return V, F, it, trace
        if it == max_iter:
            break
        # per facet: how far it has to go (its worst offender, plus the base offset)
        best = np.full(len(P), -np.inf)
        arg = np.zeros(len(P), np.int64)
        for s in range(0, len(P), 4096):
            proj = P[s:s + 4096] @ N.T - D[None, :]
            if not good.all():
                proj[:, ~good] = -np.inf
            i = proj.argmax(1)
            best[s:s + 4096] = proj[np.arange(len(i)), i]
            arg[s:s + 4096] = i
        need = np.full(len(F), offset * damp)
        sel = best > 0
        np.maximum.at(need, arg[sel], best[sel] * damp + 1e-9)
        D2 = D + need
        newV = V.copy()
        for vi, facs in enumerate(inc):
            if len(facs) < 3:
                continue
            A = N[facs]
            res = D2[facs] - A @ V[vi]
            if res.max() <= 0:
                continue
            try:
                u, sv, vt = np.linalg.svd(A, full_matrices=False)
            except np.linalg.LinAlgError:
                continue
            keep = sv > sv_tol * max(sv[0], 1e-30)
            if not keep.any():
                continue
            delta = vt[keep].T @ ((u[:, keep].T @ res) / sv[keep])
            ln = float(np.linalg.norm(delta))
            if ln > max_move:
                delta *= max_move / ln
            newV[vi] = V[vi] + delta
        cand_viol = violation(newV)
        if cand_viol < viol:
            V = newV
            damp = min(1.0, damp * 1.5)
        else:
            damp *= 0.4
            if damp < 1e-4:
                break
    return V, F, max_iter, trace
