#!/usr/bin/env python3
"""Render headless preview images (Workbench) of visual vs collision meshes.

    blender -b --factory-startup --python tools/render_preview.py -- \
        --urdf linkerhand_l20.urdf --collision-root /tmp/col_g1.5 \
        --out-dir /tmp/preview --links hand_base_link,index_distal
"""
import argparse, json, math, os, sys
import bpy
from mathutils import Vector, Matrix

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--urdf", required=True)
ap.add_argument("--collision-root", default=None)
ap.add_argument("--out-dir", required=True)
ap.add_argument("--links", default=None)
ap.add_argument("--res", type=int, default=520)
args = ap.parse_args(argv)

import xml.etree.ElementTree as ET
project = os.path.dirname(os.path.abspath(args.urdf))
rows = []
for link in ET.parse(args.urdf).getroot().findall("link"):
    v = link.find("visual/geometry/mesh")
    c = link.find("collision/geometry/mesh")
    if v is None or c is None:
        continue
    rows.append((link.get("name"), v.get("filename"), c.get("filename")))
if args.links:
    want = set(args.links.split(","))
    rows = [r for r in rows if r[0] in want]

os.makedirs(args.out_dir, exist_ok=True)
VIEWS = {"iso": (1, -1, 0.7), "front": (0, -1, 0), "side": (-1, 0, 0), "top": (0, 0, 1)}


def clear():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def import_stl(path, color, name):
    bpy.ops.wm.stl_import(filepath=path, global_scale=1.0)
    objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    if len(objs) > 1:
        bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active
    obj.name = name
    obj.color = color
    return obj


def frame(objs, view_dir, res):
    pts = []
    for o in objs:
        for v in o.data.vertices:
            pts.append(o.matrix_world @ v.co)
    mn = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    mx = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    centre = (mn + mx) / 2
    extent = max((mx - mn).x, (mx - mn).y, (mx - mn).z)
    cam_data = bpy.data.cameras.new("cam")
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = extent * 1.25
    cam = bpy.data.objects.new("cam", cam_data)
    bpy.context.collection.objects.link(cam)
    d = Vector(view_dir).normalized()
    cam.location = centre + d * (extent * 3)
    # look at centre
    direction = (centre - cam.location).normalized()
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.camera = cam
    sc = bpy.context.scene
    sc.render.engine = "BLENDER_WORKBENCH"
    sc.render.resolution_x = res
    sc.render.resolution_y = res
    sc.render.film_transparent = False
    sh = sc.display.shading
    sh.light = "STUDIO"
    sh.color_type = "OBJECT"
    sh.show_cavity = True
    sh.show_object_outline = True
    sh.background_type = "VIEWPORT"
    sh.background_color = (0.10, 0.10, 0.12)
    sc.world = bpy.data.worlds.new("w")
    sc.world.color = (0.15, 0.15, 0.17)


for link, vis_rel, col_rel in rows:
    vis_path = os.path.join(project, vis_rel)
    col_path = os.path.join(args.collision_root, os.path.basename(col_rel)) \
        if args.collision_root else os.path.join(project, col_rel)
    for view, d in VIEWS.items():
        for kind, path, colour in (("visual", vis_path, (0.7, 0.7, 0.72, 1)),
                                   ("collision", col_path, (0.9, 0.55, 0.15, 1))):
            clear()
            obj = import_stl(path, colour, kind)
            frame([obj], d, args.res)
            out = os.path.join(args.out_dir, f"{link}__{view}__{kind}.png")
            bpy.context.scene.render.filepath = out
            bpy.ops.render.render(write_still=True)
            print("RENDER", out, flush=True)
print("DONE")
