"""Output #2 (3D): build a 3D scene from the floorplan3d JSON.
Writes .glb (Blender / three.js / Unity / Unreal / Windows 3D Viewer),
.obj and a preview PNG. Works fully offline."""
import os
import numpy as np

FLOOR_RGBA = {"living": (230, 200, 150), "living_kitchen": (230, 195, 140), "kitchen": (220, 180, 140),
              "bedroom": (190, 210, 235), "master_bedroom": (180, 200, 235), "bathroom": (190, 230, 230),
              "balcony": (190, 220, 170), "hallway": (215, 200, 235), "storage": (215, 215, 180)}


def _box(trimesh, cx, cy, cz, sx, sy, sz, color):
    m = trimesh.creation.box(extents=[max(sx, 1e-3), max(sy, 1e-3), max(sz, 1e-3)])
    m.apply_translation([cx, cy, cz])
    _paint(trimesh, m, color)
    return m


def _paint(trimesh, m, color):
    """Assign a glTF PBR material (works in Blender, three.js, Unity, Windows 3D Viewer)."""
    from trimesh.visual.material import PBRMaterial
    c = list(color) + [255] * (4 - len(color))
    mat = PBRMaterial(baseColorFactor=[int(v) for v in c], metallicFactor=0.0, roughnessFactor=0.9,
                      alphaMode="BLEND" if c[3] < 255 else "OPAQUE", doubleSided=True)
    m.visual = trimesh.visual.TextureVisuals(material=mat)


def wall_pieces(wall, openings):
    """Split a wall into solid boxes around its openings. Returns list of
    (along_start, along_end, z0, z1) in metres along the wall."""
    L = wall["length_m"]; Hh = wall["height_m"]
    ops = sorted([o for o in openings if o.get("wall_id") == wall["id"] and "offset_from_wall_start_m" in o],
                 key=lambda o: o["offset_from_wall_start_m"])
    pieces = []; cur = 0.0
    for o in ops:
        a = max(0.0, o["offset_from_wall_start_m"]); b = min(L, a + o["width_m"])
        if b <= a:
            continue
        if a > cur:
            pieces.append((cur, a, 0.0, Hh))
        z0 = o["sill_height_m"]; z1 = o["sill_height_m"] + o["height_m"]
        if z0 > 0:
            pieces.append((a, b, 0.0, z0))
        if z1 < Hh:
            pieces.append((a, b, z1, Hh))
        cur = max(cur, b)
    if cur < L:
        pieces.append((cur, L, 0.0, Hh))
    return pieces


def build_meshes(data):
    import trimesh
    from shapely.geometry import Polygon
    from shapely.ops import triangulate
    meshes = []
    ops = data["openings"]
    for w in data["walls"]:
        (x1, y1), (x2, y2) = w["start"], w["end"]
        t = w["thickness_m"]
        col = (245, 245, 240, 255) if w["type"] == "interior" else (225, 225, 220, 255)
        for a, b, z0, z1 in wall_pieces(w, ops):
            if w["orientation"] == "horizontal":
                cx = x1 + (a + b) / 2; meshes.append(("wall_" + w["id"], _box(trimesh, cx, (z0 + z1) / 2, y1, b - a, z1 - z0, t, col)))
            else:
                cz = y1 + (a + b) / 2; meshes.append(("wall_" + w["id"], _box(trimesh, x1, (z0 + z1) / 2, cz, t, z1 - z0, b - a, col)))
    for o in ops:
        if o.get("wall_id") is None:
            continue
        w = next(w for w in data["walls"] if w["id"] == o["wall_id"])
        cx, cz = o["center"]
        if w["orientation"] == "horizontal":
            cz = w["start"][1]
        else:
            cx = w["start"][0]
        if o["type"] == "window":
            th = 0.02; col = (120, 180, 230, 120)
            y = o["sill_height_m"] + o["height_m"] / 2
            if w["orientation"] == "horizontal":
                m = _box(trimesh, cx, y, cz, o["width_m"], o["height_m"], th, col)
            else:
                m = _box(trimesh, cx, y, cz, th, o["height_m"], o["width_m"], col)
            meshes.append(("window_" + o["id"], m))
        else:  # door leaf, shown open at 90 degrees is ambiguous -> thin frame on the head
            pass
    # balcony railings (1.0 m) on balcony edges that are not building walls
    for r in data["rooms"]:
        if r["type"] != "balcony":
            continue
        P = r["polygon"]
        for i in range(len(P)):
            (ax, ay), (bx, by) = P[i - 1], P[i]
            L = float(np.hypot(bx - ax, by - ay))
            if L < 0.2:
                continue
            mx, my = (ax + bx) / 2, (ay + by) / 2
            near_wall = False
            for w in data["walls"]:
                (x1, y1), (x2, y2) = w["start"], w["end"]
                d = abs(mx - x1) if w["orientation"] == "vertical" else abs(my - y1)
                inside = (min(y1, y2) - 0.1 <= my <= max(y1, y2) + 0.1) if w["orientation"] == "vertical" else (min(x1, x2) - 0.1 <= mx <= max(x1, x2) + 0.1)
                if d < w["thickness_m"] / 2 + 0.25 and inside:
                    near_wall = True; break
            if near_wall:
                continue
            ang = np.arctan2(by - ay, bx - ax)
            m = trimesh.creation.box(extents=[L, 1.0, 0.05])
            m.apply_transform(trimesh.transformations.rotation_matrix(-ang, [0, 1, 0]))
            m.apply_translation([mx, 0.5, my])
            _paint(trimesh, m, (200, 205, 210, 255))
            meshes.append(("railing_" + r["id"], m))
    for r in data["rooms"]:
        poly = Polygon(r["polygon"]).buffer(0)
        if poly.is_empty:
            continue
        tris = [tr for tr in triangulate(poly) if poly.buffer(1e-6).contains(tr.centroid)]
        if not tris:
            continue
        V = []; F = []
        for tr in tris:
            c = list(tr.exterior.coords)[:3]
            base = len(V)
            V += [[p[0], 0.0, p[1]] for p in c]
            F.append([base, base + 2, base + 1])  # face up (+Y)
        m = trimesh.Trimesh(vertices=np.array(V), faces=np.array(F), process=True)
        c = FLOOR_RGBA.get(r["type"], (210, 210, 210))
        _paint(trimesh, m, list(c) + [255])
        meshes.append(("floor_" + r["id"] + "_" + r["type"], m))
    return meshes


def build_scene(data, out_dir, stem):
    import trimesh
    meshes = build_meshes(data)
    scene = trimesh.Scene()
    for i, (name, m) in enumerate(meshes):
        nm = f"{name}_{i}"
        scene.add_geometry(m, node_name=nm, geom_name=nm)
    out = {}
    p = os.path.join(out_dir, f"{stem}_3d.glb"); scene.export(p); out["glb"] = p
    p2 = os.path.join(out_dir, f"{stem}_3d.obj")
    from trimesh.exchange.obj import export_obj
    from trimesh.resolvers import FilePathResolver
    txt = export_obj(scene, mtl_name=f"{stem}_3d.mtl", resolver=FilePathResolver(out_dir), header="floorplan3d")
    with open(p2, "w", encoding="utf-8") as fh:
        fh.write(txt)
    out["obj"] = p2
    out["viewer_html"] = write_viewer(data, p, os.path.join(out_dir, f"{stem}_3d_viewer.html"))
    try:
        out["preview"] = render_preview(meshes, os.path.join(out_dir, f"{stem}_3d_preview.png"))
    except Exception as e:  # preview is optional
        print("preview skipped:", e)
    return out


def render_preview(meshes, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig = plt.figure(figsize=(12, 8), dpi=120)
    fig.subplots_adjust(0, 0, 1, 1)
    ax = fig.add_subplot(111, projection="3d")
    allv = []
    for name, m in meshes:
        tri = m.vertices[m.faces]
        tri = tri[:, :, [0, 2, 1]]  # (x, z, y) so that up is matplotlib z
        tri[:, :, 1] *= -1           # plan y down -> flip for a natural view
        col = np.array(m.visual.material.baseColorFactor, float) / 255.0
        fc = np.tile(col, (len(m.faces), 1))
        if name.startswith("wall_"):
            # simple shading by normal
            n = m.face_normals[:, [0, 2, 1]]
            shade = 0.55 + 0.45 * np.clip(np.abs(n @ np.array([0.4, -0.6, 0.7])), 0, 1)
            fc = np.c_[fc[:, :3] * shade[:, None], fc[:, 3]]
        pc = Poly3DCollection(tri, facecolors=fc, edgecolors=(0, 0, 0, 0.08), linewidths=0.2)
        ax.add_collection3d(pc); allv.append(tri.reshape(-1, 3))
    V = np.concatenate(allv)
    mn, mx = V.min(0), V.max(0)
    ax.set_xlim(mn[0], mx[0]); ax.set_ylim(mn[1], mx[1]); ax.set_zlim(0, max(mx[2], 0.1))
    ax.set_box_aspect(tuple(np.maximum(mx - mn, 0.1))); ax.view_init(elev=40, azim=-62); ax.set_axis_off()
    ax.set_title("3D model preview (walls with door/window openings, floors coloured by room type)", fontsize=10)
    plt.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path


VIEWER_TMPL = """<!doctype html><html><head><meta charset="utf-8"><title>__TITLE__ - 3D</title>
<style>html,body{margin:0;height:100%;background:#eef1f4;font-family:Segoe UI,Arial,sans-serif;overflow:hidden}
#hud{position:absolute;left:12px;top:10px;background:#fffd;padding:8px 12px;border-radius:8px;font-size:13px;line-height:1.5;max-width:340px;box-shadow:0 1px 4px #0002}
#hud b{font-size:14px}</style></head><body><div id="hud"></div>
<script>__THREE__</script><script>__GLTF__</script><script>__ORBIT__</script>
<script>
const DATA=__DATA__; const GLB="__GLB__";
const hud=document.getElementById('hud'); const b=DATA.building;
hud.innerHTML='<b>'+DATA.source_image+'</b><br>Footprint '+b.width_ft_in+' x '+b.depth_ft_in+' ('+b.width_m.toFixed(2)+' x '+b.depth_m.toFixed(2)+' m)<br>'+
 b.room_count+' rooms, '+b.wall_count+' walls, '+b.door_count+' doors, '+b.window_count+' windows<br>'+
 DATA.rooms.map(r=>'&#9632; '+r.name+' &ndash; '+r.area_ft2.toFixed(0)+' ft&sup2; ('+r.area_m2.toFixed(1)+' m&sup2;)').join('<br>')+
 '<br><i>Drag to orbit, scroll to zoom, right-drag to pan</i>';
const renderer=new THREE.WebGLRenderer({antialias:true}); renderer.setPixelRatio(devicePixelRatio);
renderer.outputEncoding=THREE.sRGBEncoding; renderer.setSize(innerWidth,innerHeight); document.body.appendChild(renderer.domElement);
const scene=new THREE.Scene(); scene.background=new THREE.Color(0xeef1f4);
const cam=new THREE.PerspectiveCamera(45,innerWidth/innerHeight,0.05,500);
scene.add(new THREE.HemisphereLight(0xffffff,0x667788,0.55)); const sun=new THREE.DirectionalLight(0xffffff,0.55); sun.position.set(5,12,8); scene.add(sun);
const ctl=new THREE.OrbitControls(cam,renderer.domElement);
const bin=Uint8Array.from(atob(GLB),c=>c.charCodeAt(0)).buffer;
new THREE.GLTFLoader().parse(bin,'',g=>{ const edges=[]; g.scene.traverse(o=>{ if(o.isMesh){ o.material.side=THREE.DoubleSide;
   if(o.material.transparent){o.material.opacity=0.45;o.material.depthWrite=false;}
   else { edges.push(o); } } });
 edges.forEach(o=>{ const l=new THREE.LineSegments(new THREE.EdgesGeometry(o.geometry,25),new THREE.LineBasicMaterial({color:0x555555})); o.add(l); });
 const grid=new THREE.GridHelper(200,200,0xc8ccd2,0xdde0e4); grid.position.y=-0.01; scene.add(grid);
 scene.add(g.scene); const bb=new THREE.Box3().setFromObject(g.scene); const c=bb.getCenter(new THREE.Vector3()); const s=bb.getSize(new THREE.Vector3()).length();
 ctl.target.copy(c); cam.position.set(c.x+s*0.35,c.y+s*0.75,c.z+s*0.75); ctl.update(); });
addEventListener('resize',()=>{cam.aspect=innerWidth/innerHeight;cam.updateProjectionMatrix();renderer.setSize(innerWidth,innerHeight)});
(function loop(){requestAnimationFrame(loop);renderer.render(scene,cam)})();
</script></body></html>"""


def write_viewer(data, glb_path, out_path):
    """Self-contained offline HTML viewer (three.js r147, MIT) - just double-click it."""
    import base64, json as _json
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "three")
    rd = lambda f: open(os.path.join(d, f), encoding="utf-8").read().replace("</script>", "<\\/script>")
    glb = base64.b64encode(open(glb_path, "rb").read()).decode()
    html = (VIEWER_TMPL.replace("__THREE__", rd("three.min.js")).replace("__GLTF__", rd("GLTFLoader.js"))
            .replace("__ORBIT__", rd("OrbitControls.js")).replace("__TITLE__", data["source_image"])
            .replace("__DATA__", _json.dumps(data).replace("</", "<\\/")).replace("__GLB__", glb))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path
