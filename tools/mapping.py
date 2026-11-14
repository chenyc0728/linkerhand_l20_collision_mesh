"""Parse linkerhand URDF -> link/visual/collision mapping table (JSON)."""
import json, os, sys
import xml.etree.ElementTree as ET

URDF = sys.argv[1] if len(sys.argv) > 1 else "linkerhand_l20.urdf"
root_dir = os.path.dirname(os.path.abspath(URDF))
tree = ET.parse(URDF)
root = tree.getroot()

rows = []
for link in root.findall("link"):
    name = link.get("name")
    def geom(tag):
        dst = link.find(tag)
        if dst is None:
            return None
        m = dst.find("geometry/mesh")
        if m is None:
            return None
        o = dst.find("origin")
        r = {"filename": m.get("filename"),
             "scale": m.get("scale") or "1 1 1",
             "xyz": (o.get("xyz") if o is not None else "0 0 0"),
             "rpy": (o.get("rpy") if o is not None else "0 0 0")}
        return r
    rows.append({"link": name,
                 "visual": geom("visual"),
                 "collision": geom("collision")})

for r in rows:
    print(json.dumps(r))
json.dump(rows, open("tools/mapping.json", "w"), indent=2)

# sanity: missing files, unreferenced files
ref_v = set(); ref_c = set()
for r in rows:
    if r["visual"]: ref_v.add(r["visual"]["filename"])
    if r["collision"]: ref_c.add(r["collision"]["filename"])
print("\n-- missing referenced files --")
for f in sorted(ref_v | ref_c):
    if not os.path.exists(os.path.join(root_dir, f)):
        print("MISSING:", f)
print("\n-- referenced but absent from listing --")
allf = set(os.path.join(*p.split("/")[1:]) for p in [])
vis_disk = set("meshes/" + x for x in os.listdir(os.path.join(root_dir, "meshes")) if os.path.isfile(os.path.join(root_dir, "meshes", x)))
col_disk = set("meshes/collisions/" + x for x in os.listdir(os.path.join(root_dir, "meshes", "collisions")))
print("unreferenced visual  :", sorted(vis_disk - ref_v))
print("unreferenced collision:", sorted(col_disk - ref_c))
print("collision refs w/o file:", sorted(ref_c - col_disk))
