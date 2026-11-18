"""Run with pytest and the MMRT backend dependencies installed."""

from pathlib import Path
import copy
import shutil
import struct
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "processes" / "mmrt"))
from common.collision_meshes import prepare_collision_urdf


def write_pieces(path, ascii_stl=False):
    vertices = np.array([
        [-.5, -.5, -.5], [.5, -.5, -.5], [.5, .5, -.5], [-.5, .5, -.5],
        [-.5, -.5, .5], [.5, -.5, .5], [.5, .5, .5], [-.5, .5, .5],
    ])
    faces = np.array([
        [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
    ])
    triangles = np.concatenate([(vertices + [x, 0, 0])[faces] for x in [-2, 2]])
    if ascii_stl:
        text = "solid pieces\n"
        for triangle in triangles:
            text += "facet normal 0 0 0\nouter loop\n"
            text += "".join(f"vertex {x} {y} {z}\n" for x, y, z in triangle)
            text += "endloop\nendfacet\n"
        path.write_text(text + "endsolid pieces\n")
    else:
        with path.open("wb") as output:
            output.write(b"solid binary header".ljust(80, b" "))
            output.write(struct.pack("<I", len(triangles)))
            for triangle in triangles:
                output.write(struct.pack("<12fH", 0, 0, 0, *triangle.flatten(), 0))


def write_robot(tmp_path):
    path = tmp_path / "robot.urdf"
    path.write_text('''<robot name="pieces"><link name="base">
      <inertial><mass value="1"/><inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/></inertial>
      <visual><geometry><mesh filename="pieces.stl"/></geometry></visual>
      <collision name="custom"><origin xyz="1 2 3" rpy="0 0 1.5707963267948966"/>
        <geometry><mesh filename="pieces.stl" scale="0.1 0.2 0.3"/></geometry>
      </collision>
    </link></robot>''')
    return path


@pytest.mark.parametrize("ascii_stl", [False, True])
def test_preserves_pieces_transforms_visuals_and_source(tmp_path, ascii_stl):
    write_pieces(tmp_path / "pieces.stl", ascii_stl)
    source = write_robot(tmp_path)
    original = source.read_bytes()
    with prepare_collision_urdf(source) as prepared:
        root = ET.parse(prepared)
        collisions = root.findall("./link/collision")
        assert len(collisions) == 2
        collision = collisions[0]
        mesh = collision.find("geometry/mesh")
        obj = Path(mesh.get("filename"))
        assert obj.read_text().startswith("o hull_0\n")
        assert mesh.get("scale") == "0.1 0.2 0.3"
        assert collision.find("origin").get("xyz") == "1 2 3"
        assert collision.find("origin").get("rpy") == "0 0 1.5707963267948966"
        assert Path(root.find("./link/visual/geometry/mesh").get("filename")) == tmp_path / "pieces.stl"
    assert not Path(prepared).exists()
    assert not obj.exists()
    assert source.read_bytes() == original


def test_primitives_missing_collisions_and_explicit_concave_are_untouched(tmp_path):
    source = tmp_path / "simple.urdf"
    source.write_text('''<robot name="simple"><link name="base">
      <collision><geometry><box size="1 2 3"/></geometry></collision>
      <collision concave="yes"><geometry><mesh filename="static.stl"/></geometry></collision>
    </link><link name="no_collision"/></robot>''')
    with prepare_collision_urdf(source) as prepared:
        assert Path(prepared) == source


def test_cleanup_on_load_failure_and_reuse_shared_mesh(tmp_path):
    write_pieces(tmp_path / "pieces.stl")
    source = write_robot(tmp_path)
    tree = ET.parse(source)
    link = tree.getroot().find("link")
    link.append(copy.deepcopy(link.find("collision")))
    tree.write(source)
    with pytest.raises(RuntimeError, match="load failed"):
        with prepare_collision_urdf(source) as prepared:
            meshes = ET.parse(prepared).findall("./link/collision/geometry/mesh")
            assert len(meshes) == 4
            assert meshes[0].get("filename") == meshes[2].get("filename")
            assert meshes[1].get("filename") == meshes[3].get("filename")
            directory = Path(prepared).parent
            raise RuntimeError("load failed")
    assert not directory.exists()


def test_single_component_stl_and_existing_obj_are_untouched(tmp_path):
    mesh = tmp_path / "pieces.stl"
    write_pieces(mesh)
    data = mesh.read_bytes()
    mesh.write_bytes(data[:80] + struct.pack("<I", 12) + data[84:84 + 12 * 50])
    source = write_robot(tmp_path)
    with prepare_collision_urdf(source) as prepared:
        assert Path(prepared) == source
    tree = ET.parse(source)
    tree.find("./link/collision/geometry/mesh").set("filename", "custom.obj")
    tree.write(source)
    (tmp_path / "custom.obj").write_text("o hull\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    with prepare_collision_urdf(source) as prepared:
        assert Path(prepared) == source


def test_missing_custom_mesh_fails_instead_of_using_visuals(tmp_path):
    source = write_robot(tmp_path)
    with pytest.raises(FileNotFoundError):
        with prepare_collision_urdf(source):
            pytest.fail("Missing collision geometry must not be silently replaced")


def test_pybullet_gap_and_contact_with_origin_rotation_and_scale(tmp_path):
    p = pytest.importorskip("pybullet")
    from common.simulation import Simulation

    write_pieces(tmp_path / "pieces.stl")
    source = write_robot(tmp_path)
    simulation = Simulation(gui=False)
    try:
        probe = p.createMultiBody(
            baseMass=0, baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_SPHERE, radius=.01),
            basePosition=[1, 2, 3],
        )
        original = p.loadURDF(str(source), useFixedBase=True)
        assert p.getClosestPoints(original, probe, distance=0), "Original hull fills the gap"
        p.removeBody(original)
        robot = simulation.load_robot(str(source))
        assert not p.getClosestPoints(robot, probe, distance=0), "Custom pieces leave the gap open"
        p.resetBasePositionAndOrientation(probe, [1, 1.8, 3], [0, 0, 0, 1])
        assert p.getClosestPoints(robot, probe, distance=0), "Transformed custom piece still collides"
    finally:
        simulation.disconnect()


def test_obj_touching_hulls_keep_authored_boundaries(tmp_path):
    p = pytest.importorskip("pybullet")
    from common.simulation import Simulation

    # Two cubes share an edge. Welding equal positions (as in STL recovery)
    # would join them, filling the empty space alongside that edge.
    vertices = np.array([[x, y, z] for x in [-.5, .5] for y in [-.5, .5] for z in [-.5, .5]])
    faces = [[1, 2, 4, 3], [5, 7, 8, 6], [1, 5, 6, 2],
             [3, 4, 8, 7], [1, 3, 7, 5], [2, 6, 8, 4]]
    obj = tmp_path / "touching.obj"
    with obj.open("w") as output:
        for i, offset in enumerate([[0, 0, 0], [1, 1, 0]]):
            output.write(f"o authored_hull_{i}\n")
            for x, y, z in vertices + offset:
                output.write(f"v {x} {y} {z}\n")
            for face in faces:
                # Negative vertex indices and texture/normal suffixes are legal OBJ.
                output.write("f " + " ".join(f"{v - 9}/1/1" for v in face) + "\n")
    source = tmp_path / "robot.urdf"
    source.write_text('''<robot name="touching"><link name="base">
      <collision><geometry><mesh filename="touching.obj"/></geometry></collision>
    </link></robot>''')
    original = source.read_bytes()
    sim = Simulation(gui=False)
    try:
        robot = sim.load_robot(str(source))
        assert len(p.getCollisionShapeData(robot, -1)) == 2
        probe = p.createMultiBody(
            baseMass=0, baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_SPHERE, radius=.01),
            basePosition=[.8, .2, 0],
        )
        assert not p.getClosestPoints(robot, probe, 0)
        for position in [[0, 0, 0], [1, 1, 0]]:
            p.resetBasePositionAndOrientation(probe, position, [0, 0, 0, 1])
            assert p.getClosestPoints(robot, probe, 0)
        assert source.read_bytes() == original
    finally:
        sim.disconnect()


def test_blender_export_to_pybullet_preserves_touching_hulls(tmp_path):
    p = pytest.importorskip("pybullet")
    from common.simulation import Simulation

    blender = shutil.which("blender")
    if not blender:
        pytest.skip("Blender is needed for the exporter integration test")
    script = Path(__file__).with_name("blender_collision_fixture.py")
    result = subprocess.run(
        [blender, "--background", "--factory-startup", "--disable-autoexec",
         "--python-exit-code", "1", "--python", str(script), "--", str(tmp_path)],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    sim = Simulation(gui=False)
    try:
        robot = sim.load_robot(str(tmp_path / "robot.urdf"))
        assert len(p.getCollisionShapeData(robot, -1)) == 2
        probe = p.createMultiBody(
            baseMass=0, baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_SPHERE, radius=.01),
            basePosition=[.8, .2, 0],
        )
        assert not p.getClosestPoints(robot, probe, 0)
        for position in [[0, 0, 0], [1, 1, 0]]:
            p.resetBasePositionAndOrientation(probe, position, [0, 0, 0, 1])
            assert p.getClosestPoints(robot, probe, 0)
    finally:
        sim.disconnect()
