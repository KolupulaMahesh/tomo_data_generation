"""
Read-only export of a Blender scene's LOOK - every object's mesh, material, texture, light and camera, the bone-driven
meshes as USD skeletons - to USD with Blender's own exporter (2026-10-29: the generic renderer). The motion is NOT taken
from here: the renderer poses every object and bone each frame from our rig (rig.json + rig.py). Each prim carries its
Blender name (author_blender_name), so the renderer maps the rig's objects to prims without guessing how the exporter
renamed them. The .blend is NOT saved or changed.

    blender -b <scene>.blend --python blender_scripts/export_scene_usd.py -- blender_export/<scene>

-> <out_dir>/scene/scene.usdc + <out_dir>/scene/textures/, and <out_dir>/scene/export.json:
   objects    {Blender object: prim}, materials {Blender material: prim}
   skeletons  [{prim, armature, joints (USD), bones (the Blender bone of each joint)}]
   emissive   materials whose surface is an image (alone or through Emission): Cycles shows the image self-lit, the
              exporter writes nothing for it -> the renderer rebuilds them (the image is copied into textures/)
   camera_hidden  meshes Blender renders without camera visibility (Object > Visibility > Camera off)
   geometry_changes  meshes whose shape changes over time other than by bones (modifiers, shape keys): the static
              export shows frame 0's shape for them, so the renderer refuses such a scene
   unsupported_animation  what moves in Blender but our rig does not read (drivers, NLA strips, animated materials /
              lights / cameras / ...): make_motion.py refuses such a scene
"""
import json
import shutil
import sys
import time
from pathlib import Path

import bpy
import numpy as np
from pxr import Tf, Usd, UsdSkel

OUT = Path(sys.argv[sys.argv.index("--") + 1]) / "scene"
FRAME = 0                                          # the rest state; every frame is posed by the renderer
CHECK_FRAMES = 5                                   # frames compared to find shapes that change over time
SHAPE_TOL = 1e-4                                   # m


def export(path):
    bpy.ops.wm.usd_export(filepath=str(path), check_existing=False, selected_objects_only=False,
                          export_animation=False, export_hair=False, export_uvmaps=True, export_mesh_colors=True,
                          export_normals=True, export_materials=True, export_armatures=True,
                          only_deform_bones=False, export_shapekeys=True, use_instancing=False,
                          evaluation_mode="RENDER", generate_preview_surface=True, convert_orientation=False,
                          export_textures_mode="NEW", overwrite_textures=True, relative_paths=True,
                          xform_op_mode="TRS", root_prim_path="/root", export_custom_properties=True,
                          author_blender_name=True, convert_world_material=True, export_meshes=True,
                          export_lights=True, export_cameras=True, export_curves=True, export_points=True,
                          export_volumes=True, triangulate_meshes=False, merge_parent_xform=False,
                          convert_scene_units="METERS")


def read_prims(path, objects_by_name):
    """Blender name -> prim, from the names the exporter wrote; each skeleton's joints -> Blender bones."""
    st = Usd.Stage.Open(str(path))
    objects, materials, skeletons = {}, {}, []
    for p in st.Traverse():
        a = p.GetAttribute("userProperties:blender:object_name")
        if a and a.Get():
            objects[a.Get()] = str(p.GetPath())
        d = p.GetAttribute("userProperties:blender:data_name")
        if p.GetTypeName() == "Material" and d and d.Get():
            materials[d.Get()] = str(p.GetPath())
    for p in st.Traverse():
        if not p.IsA(UsdSkel.Skeleton):
            continue
        arm = p.GetParent().GetAttribute("userProperties:blender:object_name").Get()
        by_joint = {}
        for b in objects_by_name[arm].data.bones:          # the exporter's joint path: valid identifiers, root first
            chain, x = [], b
            while x:
                chain.append(Tf.MakeValidIdentifier(x.name))
                x = x.parent
            by_joint["/".join(reversed(chain))] = b.name
        joints = [str(j) for j in UsdSkel.Skeleton(p).GetJointsAttr().Get()]
        missing = [j for j in joints if j not in by_joint]
        if missing:
            raise RuntimeError(f"{p.GetPath()}: joints with no Blender bone {missing[:5]}")
        skeletons.append({"prim": str(p.GetPath()), "armature": arm, "joints": joints,
                          "bones": [by_joint[j] for j in joints]})
    return objects, materials, skeletons


def _source(sock):
    while sock is not None and sock.is_linked:
        node = sock.links[0].from_node
        if node.type != "REROUTE":
            return node
        sock = node.inputs[0]
    return None


def emissive_images(materials):
    """Materials whose surface is an image texture (Cycles: shown self-lit) or Emission fed by one."""
    out = []
    for m in bpy.data.materials:
        if m.name not in materials or not (m.use_nodes and m.node_tree):
            continue
        o = next((n for n in m.node_tree.nodes if n.type == "OUTPUT_MATERIAL" and n.is_active_output), None)
        node = _source(o.inputs["Surface"]) if o else None
        img, strength = None, 1.0
        if node is not None and node.type == "TEX_IMAGE":
            img = node.image
        elif node is not None and node.type == "EMISSION":
            tex = _source(node.inputs["Color"])
            img = tex.image if tex is not None and tex.type == "TEX_IMAGE" else None
            s = node.inputs["Strength"]
            strength = 1.0 if s.is_linked else float(s.default_value)
        if img is None:
            continue
        dst = OUT / "textures" / (bpy.path.basename(img.filepath) or f"{img.name}.png")
        if img.packed_file:
            dst.write_bytes(img.packed_file.data)
        else:
            shutil.copyfile(bpy.path.abspath(img.filepath, library=img.library), dst)
        out.append({"material": m.name, "prim": materials[m.name], "image": f"textures/{dst.name}",
                    "strength": strength})
    return out


def geometry_changes(scene):
    """Meshes with a modifier other than Armature, or shape keys: their shape (bones off) at CHECK_FRAMES frames."""
    cands = [o for o in scene.objects if o.type == "MESH" and not o.hide_render
             and (any(m.type != "ARMATURE" and m.show_render for m in o.modifiers) or o.data.shape_keys)]
    muted = [m for o in cands for m in o.modifiers if m.type == "ARMATURE" and m.show_viewport]
    for m in muted:
        m.show_viewport = False
    ref, worst = {}, {}
    for f in np.linspace(scene.frame_start, scene.frame_end, CHECK_FRAMES).round().astype(int):
        scene.frame_set(int(f))
        dg = bpy.context.evaluated_depsgraph_get()
        for o in cands:
            me = o.evaluated_get(dg).data
            co = np.empty(len(me.vertices) * 3, np.float32)
            me.vertices.foreach_get("co", co)
            r = ref.setdefault(o.name, co)
            worst[o.name] = max(worst.get(o.name, 0.0), float(np.abs(co - r).max()) if co.shape == r.shape
                                else float("inf"))
    for m in muted:
        m.show_viewport = True
    scene.frame_set(FRAME)
    return len(cands), [{"object": k, "max_m": v} for k, v in sorted(worst.items(), key=lambda x: -x[1])
                        if v > SHAPE_TOL]


def unsupported_animation(scene):
    """The rig reads each object's action only: drivers, NLA strips and animated data-blocks would be missed."""
    out = []
    for o in scene.objects:
        ad = o.animation_data
        if ad and len(ad.drivers):
            out.append(f"{o.name}: {len(ad.drivers)} drivers")
        if ad and any(not t.mute and len(t.strips) for t in ad.nla_tracks):
            out.append(f"{o.name}: NLA strips")
    for coll in (bpy.data.materials, bpy.data.lights, bpy.data.cameras, bpy.data.worlds, bpy.data.node_groups,
                 bpy.data.shape_keys, bpy.data.meshes, bpy.data.armatures):
        for x in coll:
            ad = getattr(x, "animation_data", None)
            if ad and (ad.action or len(ad.drivers) or len(ad.nla_tracks)):
                out.append(f"{type(x).__name__} {x.name}: animated")
    return out


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    by_name = {}
    for o in scene.objects:
        if o.name in by_name:
            raise RuntimeError(f"two objects named {o.name} (one from a library): rename one")
        by_name[o.name] = o
    scene.frame_set(FRAME)
    path = OUT / "scene.usdc"
    export(path)
    objects, materials, skeletons = read_prims(path, by_name)
    n_checked, changes = geometry_changes(scene)
    info = {"blend": bpy.data.filepath, "blender": bpy.app.version_string, "frame": FRAME, "usd": path.name,
            "fps": scene.render.fps / scene.render.fps_base, "frame_start": scene.frame_start,
            "frame_end": scene.frame_end, "objects_in_scene": len(scene.objects),
            "objects": objects, "materials": materials, "skeletons": skeletons,
            "emissive": emissive_images(materials),
            "camera_hidden": [o.name for o in scene.objects
                              if o.type == "MESH" and not o.hide_render and not o.visible_camera and o.name in objects],
            "holdout_or_shadow_catcher": [o.name for o in scene.objects if o.type == "MESH" and o.name in objects
                                          and (o.is_holdout or o.is_shadow_catcher)],
            "geometry_checked": n_checked, "geometry_changes": changes,
            "unsupported_animation": unsupported_animation(scene), "seconds": round(time.time() - t0, 1)}
    (OUT / "export.json").write_text(json.dumps(info, indent=1))
    print(f"SCENE EXPORT done -> {path}: {len(objects)} objects, {len(skeletons)} skeletons, "
          f"{len(info['emissive'])} self-lit image materials, {n_checked} meshes checked for shape changes, "
          f"{len(changes)} change ({info['seconds']} s)", flush=True)


main()
