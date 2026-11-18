"""Portable robot-description and source-mesh bundle export."""

import hashlib
import os
import re
import shutil
import xml.etree.ElementTree as ET

from ..config import SceneRecognitionConfig
from ..robot_description.storage import (
    MESH_EXPORT_SOURCE_PROP,
    MESH_RELATIVE_PATH_PROP,
    MESH_SOURCE_PATH_PROP,
    get_text_mesh_path_base,
)
from ..robot_description.xml import resolve_mesh_path
from ...processes.mmrt.definitions import GeneratedAppDirectories, MDataLabel, OutputSuffixes, RobotDescriptionKind
from .mesh import write_obj_objects, write_stl_objects


def _portable_key(path):
    return os.path.normcase(os.path.normpath(str(path).replace("\\", os.sep)))


def _safe_stem(path):
    path = str(path).replace("\\", "/")
    stem = os.path.splitext(path.rsplit("/", 1)[-1])[0]
    return re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._") or "mesh"


def _portable_basename(path):
    return str(path).replace("\\", "/").rsplit("/", 1)[-1]


def _collection_mesh_indexes(collection):
    by_relative = {}
    by_source = {}
    by_basename = {}
    if not collection:
        return by_relative, by_source, by_basename

    for obj in collection.all_objects:
        if obj.type != SceneRecognitionConfig.MESH_TYPE or not obj.get(MESH_EXPORT_SOURCE_PROP, True):
            continue
        relative_path = str(obj.get(MESH_RELATIVE_PATH_PROP, "")).strip()
        source_path = str(obj.get(MESH_SOURCE_PATH_PROP, "")).strip()
        if relative_path:
            by_relative.setdefault(_portable_key(relative_path), []).append(obj)
        if source_path:
            source_key = _portable_key(source_path)
            by_source.setdefault(source_key, []).append(obj)
            basename = os.path.normcase(_portable_basename(source_path))
            by_basename.setdefault(basename, []).append(obj)
    return by_relative, by_source, by_basename


def _resolve_text_mesh_path(mesh_path, text_block, output_dir):
    base_dir = get_text_mesh_path_base(text_block) or output_dir
    description_path = os.path.join(base_dir, f"description.{text_block.get('robot_description_kind', RobotDescriptionKind.URDF)}")
    return resolve_mesh_path(mesh_path, description_path, ())


def _asset_filename(identity, original_path, extension):
    digest = hashlib.sha1(identity.encode("utf-8")).hexdigest()[:10]
    return f"{_safe_stem(original_path)}_{digest}{extension.lower()}"


def _atomic_write_xml(root, filepath):
    if hasattr(ET, "indent"):
        ET.indent(root, space="  ")
    temporary_path = filepath + ".tmp"
    ET.ElementTree(root).write(
        temporary_path,
        encoding="utf-8",
        xml_declaration=True,
    )
    os.replace(temporary_path, filepath)


def _copy_asset(source_path, destination_path):
    temporary_path = destination_path + ".tmp"
    shutil.copy2(source_path, temporary_path)
    os.replace(temporary_path, destination_path)


def export_robot_bundle(context, output_dir, base_filename):
    """Export the Recorder robot's XML and archived source geometry.

    Relative paths in the returned metadata are rooted at ``output_dir``.
    Recordings without robot entries remain valid and return ``None``.
    """
    robots = [item for item in context.scene.mmrt_recorder.robots if item.obj]
    if not robots:
        return None
    if any(item.obj.type != SceneRecognitionConfig.ROBOT_TYPE for item in robots):
        raise ValueError("Every Recorder robot entry must reference an armature.")

    if any(not item.urdf_text or not item.srdf_text for item in robots):
        raise ValueError("Every Recorder robot needs a URDF and SRDF Text datablock.")

    first = robots[0]
    urdf_xml = first.urdf_text.as_string()
    srdf_xml = first.srdf_text.as_string()
    if any(
        item.urdf_text.as_string() != urdf_xml
        or item.srdf_text.as_string() != srdf_xml
        for item in robots[1:]
    ):
        raise ValueError("All Recorder robots must share the same URDF/SRDF pair.")

    try:
        roots = [
            (RobotDescriptionKind.URDF, ET.fromstring(urdf_xml), first.urdf_text),
            (RobotDescriptionKind.SRDF, ET.fromstring(srdf_xml), first.srdf_text),
        ]
    except ET.ParseError as exc:
        raise ValueError(f"Stored robot description is invalid XML: {exc}") from exc

    mesh_collection = next(
        (item.meshes_collection for item in robots if item.meshes_collection),
        None,
    )
    by_relative, by_source, by_basename = _collection_mesh_indexes(mesh_collection)

    output_dir = os.path.abspath(output_dir)
    urdf_path = os.path.join(output_dir, base_filename + OutputSuffixes.URDF)
    srdf_path = os.path.join(output_dir, base_filename + OutputSuffixes.SRDF)
    description_dir = os.path.dirname(urdf_path)
    os.makedirs(description_dir, exist_ok=True)
    bundle_digest = hashlib.sha1(str(base_filename).encode("utf-8")).hexdigest()[:8]
    bundle_mesh_dir = f"{_safe_stem(base_filename)}_{bundle_digest}"
    mesh_dir = os.path.join(
        output_dir,
        GeneratedAppDirectories.MESH_DIR,
        GeneratedAppDirectories.ROBOT_MESH_DIR,
        bundle_mesh_dir,
    )
    os.makedirs(mesh_dir, exist_ok=True)
    exported_assets = {}

    for _kind, root, text_block in roots:
        collision_meshes = set(root.findall(".//collision/geometry/mesh"))
        for mesh in root.findall(".//mesh"):
            is_collision = mesh in collision_meshes
            original_path = str(mesh.get("filename", "")).strip()
            if not original_path:
                continue

            resolved_path = _resolve_text_mesh_path(original_path, text_block, output_dir)
            source_key = _portable_key(resolved_path) if resolved_path else ""
            relative_key = _portable_key(original_path)
            mesh_objects = by_relative.get(relative_key) or by_source.get(source_key)
            if not mesh_objects:
                basename_matches = by_basename.get(
                    os.path.normcase(_portable_basename(original_path)),
                    [],
                )
                unique_sources = {
                    _portable_key(obj.get(MESH_SOURCE_PATH_PROP, ""))
                    for obj in basename_matches
                }
                if len(unique_sources) == 1:
                    mesh_objects = basename_matches

            identity = source_key or relative_key
            # Visual and collision references can share one source file but
            # need different output formats. Never reuse the visual STL here.
            asset_key = (identity, is_collision)
            if asset_key in exported_assets:
                mesh.set("filename", exported_assets[asset_key])
                continue

            if mesh_objects:
                extension = ".obj" if is_collision else ".stl"
                asset_name = _asset_filename(identity, original_path, extension)
                destination = os.path.join(mesh_dir, asset_name)
                writer = write_obj_objects if is_collision else write_stl_objects
                writer(destination, mesh_objects, use_source_transforms=False)
            elif resolved_path and os.path.isfile(resolved_path):
                extension = os.path.splitext(resolved_path)[1] or ".mesh"
                asset_name = _asset_filename(identity, original_path, extension)
                destination = os.path.join(mesh_dir, asset_name)
                _copy_asset(resolved_path, destination)
            else:
                raise FileNotFoundError(
                    f"Robot mesh '{original_path}' is neither in the Original Meshes "
                    "collection nor resolvable from its stored path."
                )

            relative_asset_path = os.path.relpath(destination, description_dir).replace("\\", "/")
            exported_assets[asset_key] = relative_asset_path
            mesh.set("filename", relative_asset_path)

    _atomic_write_xml(roots[0][1], urdf_path)
    _atomic_write_xml(roots[1][1], srdf_path)

    return {
        MDataLabel.ROBOT_NAME_KEY: first.obj.name,
        MDataLabel.URDF_FILE_KEY: os.path.relpath(urdf_path, description_dir).replace("\\", "/"),
        MDataLabel.SRDF_FILE_KEY: os.path.relpath(srdf_path, description_dir).replace("\\", "/"),
        "mesh_dir": os.path.relpath(mesh_dir, description_dir).replace("\\", "/"),
    }
