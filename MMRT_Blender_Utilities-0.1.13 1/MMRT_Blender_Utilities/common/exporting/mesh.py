"""Mesh preparation and STL/OBJ writing helpers."""

import re
import struct

import bpy
import mathutils

from ..config import SceneRecognitionConfig
from ..robot_description.storage import MESH_SOURCE_MATRIX_PROP


def write_stl(filepath, obj):
    write_stl_objects(filepath, [obj], use_source_transforms=False)


def _source_matrix(obj):
    values = obj.get(MESH_SOURCE_MATRIX_PROP)
    if values is not None and len(values) == 16:
        return mathutils.Matrix(tuple(tuple(values[row * 4:(row + 1) * 4]) for row in range(4)))
    return mathutils.Matrix.Identity(4)


def write_stl_objects(filepath, objects, use_source_transforms=True):
    """Write one STL from one or more archived mesh objects."""
    mesh_objects = [obj for obj in objects if obj and obj.type == SceneRecognitionConfig.MESH_TYPE]
    triangle_count = 0
    for obj in mesh_objects:
        obj.data.calc_loop_triangles()
        triangle_count += len(obj.data.loop_triangles)

    with open(filepath, "wb") as output:
        header = b"Blender STL Export"
        output.write(header + b" " * (80 - len(header)))
        output.write(struct.pack("<I", triangle_count))

        for obj in mesh_objects:
            mesh = obj.data
            transform = _source_matrix(obj) if use_source_transforms else mathutils.Matrix.Identity(4)
            for triangle in mesh.loop_triangles:
                vertices = [transform @ mesh.vertices[index].co for index in triangle.vertices]
                normal = (vertices[1] - vertices[0]).cross(vertices[2] - vertices[0])
                if normal.length_squared:
                    normal.normalize()
                output.write(struct.pack("<3f", normal.x, normal.y, normal.z))
                for vertex in vertices:
                    output.write(struct.pack("<3f", vertex.x, vertex.y, vertex.z))
                output.write(struct.pack("<H", 0))


def write_obj_objects(filepath, objects, use_source_transforms=True):
    """Preserve each archived hull as an OBJ object, without welding vertices.

    Robot bundles pass False: archived mesh coordinates are already in the
    source asset frame. Bone poses and Blender's OBJ import axis rotation must
    not be baked into geometry that the URDF will position again.
    """
    vertex_offset = 1
    with open(filepath, "w", encoding="utf-8", newline="\n") as output:
        output.write("# MMRT collision hulls: one object per archived mesh\n")
        for index, obj in enumerate(objects):
            if not obj or obj.type != SceneRecognitionConfig.MESH_TYPE:
                continue
            mesh = obj.data
            mesh.calc_loop_triangles()
            if not mesh.loop_triangles:
                raise ValueError(f"Collision mesh '{obj.name}' has no faces.")
            name = re.sub(r"[^A-Za-z0-9_.-]+", "_", obj.name)
            output.write(f"o hull_{index}_{name}\n")
            transform = _source_matrix(obj) if use_source_transforms else mathutils.Matrix.Identity(4)
            for vertex in mesh.vertices:
                x, y, z = transform @ vertex.co
                output.write(f"v {x:.17g} {y:.17g} {z:.17g}\n")
            for triangle in mesh.loop_triangles:
                a, b, c = (vertex_offset + i for i in triangle.vertices)
                output.write(f"f {a} {b} {c}\n")
            vertex_offset += len(mesh.vertices)


def create_export_object(source, apply_modifiers=True):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = source.evaluated_get(depsgraph) if apply_modifiers else source
    mesh = bpy.data.meshes.new_from_object(
        evaluated,
        preserve_all_data_layers=True,
        depsgraph=depsgraph,
    )
    temporary = bpy.data.objects.new(source.name + "_tmp", mesh)
    # The active collection may be a library override (for example, the
    # ``box_jig`` collection in a linked environment).  Temporary export
    # objects are local data and must not be linked into that collection,
    # otherwise Blender cannot unlink them during cleanup.  The scene root is
    # always a local collection suitable for these short-lived objects.
    bpy.context.scene.collection.objects.link(temporary)
    temporary.matrix_world = source.matrix_world.copy()
    return temporary


def bake_world_transform(obj):
    obj.data.transform(obj.matrix_world)
    obj.matrix_world.identity()


def bake_custom_transform(obj, reference):
    reference_inverse = reference.matrix_world.inverted()
    obj.data.transform(reference_inverse @ obj.matrix_world)
    obj.matrix_world = reference.matrix_world.copy()
