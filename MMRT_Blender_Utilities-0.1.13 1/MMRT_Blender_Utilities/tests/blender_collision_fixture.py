"""Export touching authored hulls using real Blender meshes (called by pytest)."""

from pathlib import Path
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import bpy
from mathutils import Matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from MMRT_Blender_Utilities.common.exporting.robot import export_robot_bundle


output_dir = Path(sys.argv[sys.argv.index("--") + 1])
collection = bpy.data.collections.new("collision_sources")
vertices = [(x, y, z) for x in [-.5, .5] for y in [-.5, .5] for z in [-.5, .5]]
faces = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1),
         (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
for i in range(2):
    mesh = bpy.data.meshes.new(f"hull_{i}")
    mesh.from_pydata([(x + i, y + i, z) for x, y, z in vertices], [], faces)
    obj = bpy.data.objects.new(f"authored hull {i}", mesh)
    collection.objects.link(obj)
    obj["mmrt_relative_mesh_path"] = "source.obj"
    obj["mmrt_mesh_export_source"] = True
    # Archived OBJ import rotation and current scene pose must not enter URDF geometry.
    obj["mmrt_source_matrix"] = [v for row in Matrix.Rotation(1.57079632679, 4, "X") for v in row]
    obj.location = (100, 200, 300)

urdf = bpy.data.texts.new("robot_URDF")
urdf.write('''<robot name="touching"><link name="base">
  <inertial><mass value="1"/><inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/></inertial>
  <visual><geometry><mesh filename="source.obj"/></geometry></visual>
  <collision name="custom"><geometry><mesh filename="source.obj"/></geometry></collision>
</link></robot>''')
srdf = bpy.data.texts.new("robot_SRDF")
srdf.write('<robot name="touching"/>')
armature = bpy.data.objects.new("robot", bpy.data.armatures.new("robot"))
robot = SimpleNamespace(obj=armature, urdf_text=urdf, srdf_text=srdf, meshes_collection=collection)
context = SimpleNamespace(scene=SimpleNamespace(mmrt_recorder=SimpleNamespace(robots=[robot])))
export_robot_bundle(context, str(output_dir), "robot")
root = ET.parse(output_dir / "robot.urdf")
visual = root.find("./link/visual/geometry/mesh").get("filename")
collision = root.find("./link/collision/geometry/mesh").get("filename")
assert visual.endswith(".stl") and collision.endswith(".obj")
assert sum(line.startswith("o ") for line in (output_dir / collision).read_text().splitlines()) == 2
print("Blender collision OBJ export passed")
