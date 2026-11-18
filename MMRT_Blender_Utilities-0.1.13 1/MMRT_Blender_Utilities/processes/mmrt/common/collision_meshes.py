"""Preserve authored OBJ hulls and recover disconnected legacy STL pieces."""

from contextlib import contextmanager
from copy import deepcopy
import logging
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
import xml.etree.ElementTree as ET

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


logger = logging.getLogger(__name__)
_STL_TRIANGLE = np.dtype([
    ("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2"),
])


def _read_stl(path):
    """Read binary (including headers starting with 'solid') or ASCII STL."""
    data = path.read_bytes()
    if len(data) >= 84:
        count = struct.unpack_from("<I", data, 80)[0]
        if len(data) == 84 + count * _STL_TRIANGLE.itemsize:
            triangles = np.frombuffer(data, dtype=_STL_TRIANGLE, offset=84)["vertices"]
            if not count or not np.isfinite(triangles).all():
                raise ValueError(f"Empty or non-finite collision STL: {path}")
            return triangles
    try:
        vertices = [
            [float(value) for value in fields[1:]]
            for line in data.decode("ascii").splitlines()
            if (fields := line.split()) and fields[0].lower() == "vertex"
        ]
        triangles = np.asarray(vertices, dtype=np.float64)
        if (triangles.ndim != 2 or triangles.shape[1] != 3
                or len(triangles) % 3 or not np.isfinite(triangles).all()):
            raise ValueError("Invalid triangle vertices")
        return triangles.reshape(-1, 3, 3)
    except (UnicodeError, ValueError) as exc:
        raise ValueError(f"Invalid collision STL: {path}") from exc


def _write_compound_obj(source, destination):
    """Give each disconnected STL component its own OBJ object for Bullet.

    Bullet creates one hull per OBJ object. This preserves an existing convex
    decomposition; it does not decompose a connected concave component. Exact
    coordinate matching avoids welding nearby but intentionally separate parts.
    """
    triangles = _read_stl(source)
    vertices, indices = np.unique(triangles.reshape(-1, 3), axis=0, return_inverse=True)
    faces = indices.reshape(-1, 3)
    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    graph = coo_matrix(
        (np.ones(len(edges), dtype=np.uint8), (edges[:, 0], edges[:, 1])),
        shape=(len(vertices), len(vertices)),
    )
    count, labels = connected_components(graph, directed=False)
    if count <= 1:
        return count
    with destination.open("w", encoding="ascii") as output:
        for x, y, z in vertices:
            output.write(f"v {x:.17g} {y:.17g} {z:.17g}\n")
        face_labels = labels[faces[:, 0]]
        for component in range(count):
            output.write(f"o collision_piece_{component}\n")
            for a, b, c in faces[face_labels == component] + 1:
                output.write(f"f {a} {b} {c}\n")
    return count


def _asset_path(filename, base):
    if filename.startswith("file://"):
        return Path(filename[7:])
    if "://" in filename:
        # Package URIs are resolved by the pipeline's XACRO resolver / Bullet.
        return None
    return (base / filename).resolve()


def _split_obj_hulls(source, directory, prefix):
    """Write one collision OBJ per nonempty object/group, never welding hulls.

    Only positions and face vertex indices affect collision geometry. Resolve
    negative indices at the face declaration, and discard texture/normal data.
    Explicit collision elements make the hull boundaries independent of how a
    Bullet build handles multi-object OBJ files (and visible in shape queries).
    """
    vertices, groups, faces = [], [], []
    with source.open(encoding="utf-8-sig") as input_file:
        for line_number, line in enumerate(input_file, 1):
            fields = line.partition("#")[0].split()
            if not fields:
                continue
            try:
                if fields[0] == "v":
                    vertex = tuple(float(value) for value in fields[1:4])
                    if len(vertex) != 3 or not np.isfinite(vertex).all():
                        raise ValueError("Invalid vertex")
                    vertices.append(vertex)
                elif fields[0] in {"o", "g"}:
                    if faces:
                        groups.append(faces)
                        faces = []
                elif fields[0] == "f":
                    face = []
                    for token in fields[1:]:
                        index = int(token.split("/")[0])
                        if index == 0:
                            raise ValueError("OBJ indices start at 1")
                        index = index - 1 if index > 0 else len(vertices) + index
                        if not 0 <= index < len(vertices):
                            raise ValueError("Vertex index out of range")
                        face.append(index)
                    if len(face) < 3:
                        raise ValueError("Face has fewer than three vertices")
                    faces.append(face)
            except ValueError as exc:
                raise ValueError(f"Invalid collision OBJ {source}:{line_number}: {exc}") from exc
    if faces:
        groups.append(faces)
    if not groups:
        raise ValueError(f"Collision OBJ has no faces: {source}")
    if len(groups) == 1:
        return [source]

    paths = []
    for index, faces in enumerate(groups):
        path = directory / f"{prefix}_hull_{index}.obj"
        used = sorted({vertex for face in faces for vertex in face})
        remap = {vertex: i + 1 for i, vertex in enumerate(used)}
        with path.open("w", encoding="ascii") as output:
            output.write(f"o hull_{index}\n")
            for vertex in used:
                x, y, z = vertices[vertex]
                output.write(f"v {x:.17g} {y:.17g} {z:.17g}\n")
            for face in faces:
                output.write("f " + " ".join(str(remap[v]) for v in face) + "\n")
        paths.append(path)
    return paths


@contextmanager
def prepare_collision_urdf(urdf_path):
    """Load each authored OBJ hull as its own URDF collision element.

    Preserve origins, scales, and collision attributes on every piece. Legacy
    STL components are recovered where possible; touching hulls whose boundaries
    were lost in STL need re-exporting from Blender as OBJ. Explicit concave
    declarations and visual geometry retain their original behavior.
    """
    source = Path(urdf_path).resolve()
    tree = ET.parse(source)
    with TemporaryDirectory(prefix="mmrt_collision_") as directory:
        converted = {}
        changed = False
        collisions = [(link, collision) for link in tree.findall("./link")
                      for collision in link.findall("collision")]
        for link, collision in collisions:
            if "concave" in collision.attrib:
                continue
            mesh = collision.find("geometry/mesh")
            if mesh is None:
                continue
            filename = mesh.get("filename", "")
            path = _asset_path(filename, source.parent)
            if path is None or path.suffix.lower() not in {".stl", ".obj"}:
                continue
            if path not in converted:
                prefix = f"collision_{len(converted)}"
                obj_path = path
                if path.suffix.lower() == ".stl":
                    obj_path = Path(directory) / f"{prefix}.obj"
                    count = _write_compound_obj(path, obj_path)
                    if count <= 1:
                        converted[path] = [path]
                        continue
                    logger.warning(
                        "Recovered %d disconnected pieces from legacy STL %s; "
                        "re-export as OBJ to preserve touching hull boundaries.", count, path.name,
                    )
                converted[path] = _split_obj_hulls(obj_path, Path(directory), prefix)
                logger.info("Using %d custom collision hull(s) from %s", len(converted[path]), path.name)
            pieces = converted[path]
            if len(pieces) > 1:
                position = list(link).index(collision)
                link.remove(collision)
                for index, piece in enumerate(pieces):
                    child = deepcopy(collision)
                    child.find("geometry/mesh").set("filename", piece.as_posix())
                    if child.get("name"):
                        child.set("name", f"{child.get('name')}_hull_{index}")
                    link.insert(position + index, child)
                changed = True
        if not changed:
            yield str(source)
            return

        # The temporary URDF lives outside the recording folder. Rebase all
        # local assets, including visuals and textures, before handing it off.
        for element in tree.findall(".//mesh") + tree.findall(".//texture"):
            filename = element.get("filename")
            if filename:
                path = _asset_path(filename, source.parent)
                if path is not None:
                    element.set("filename", path.as_posix())
        prepared = Path(directory) / source.name
        tree.write(prepared, encoding="utf-8", xml_declaration=True)
        yield str(prepared)
