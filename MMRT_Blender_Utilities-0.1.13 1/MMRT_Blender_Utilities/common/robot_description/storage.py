import os
import time
import xml.etree.ElementTree as ET
import bpy
from ...processes.mmrt.definitions import RobotDescriptionKind


LEGACY_TEXT_NAMES = {
    RobotDescriptionKind.URDF: "Robot_Description_URDF",
    RobotDescriptionKind.SRDF: "Robot_Description_SRDF",
}
ROBOT_MESH_COLLECTION_PROP = "mmrt_robot_mesh_collection"
ROBOT_URDF_TEXT_PROP = "mmrt_robot_urdf_text"
ROBOT_SRDF_TEXT_PROP = "mmrt_robot_srdf_text"
MESH_SOURCE_PATH_PROP = "mmrt_source_mesh_path"
MESH_RELATIVE_PATH_PROP = "mmrt_relative_mesh_path"
MESH_SOURCE_MATRIX_PROP = "mmrt_source_matrix"
MESH_EXPORT_SOURCE_PROP = "mmrt_mesh_export_source"


def store_robot_description(
    kind,
    xml_text,
    *,
    text_name,
    source_path="",
    mesh_path_base="",
):
    """Store robot XML in the current .blend as a Text data-block."""
    if kind not in LEGACY_TEXT_NAMES:
        raise ValueError(f"Unsupported robot description kind: {kind}")
    if not text_name:
        raise ValueError("A robot-specific Text data-block name is required.")

    text_block = bpy.data.texts.get(text_name) or bpy.data.texts.new(text_name)
    text_block.clear()
    text_block.write(xml_text)
    if ROBOT_MESH_COLLECTION_PROP in text_block:
        del text_block[ROBOT_MESH_COLLECTION_PROP]
    text_block["robot_description_kind"] = kind
    text_block["source_path"] = source_path
    text_block["mesh_path_base"] = mesh_path_base
    text_block["stored_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    return text_block


def _update_text_name_fallbacks(kind, old_name, new_name):
    fallback_prop = (
        ROBOT_URDF_TEXT_PROP if kind == RobotDescriptionKind.URDF else ROBOT_SRDF_TEXT_PROP
    )
    for obj in bpy.data.objects:
        if obj.get(fallback_prop, "") == old_name:
            obj[fallback_prop] = new_name


def migrate_legacy_description(kind, preferred_replacement=None):
    """Move a legacy global Text block to robot-specific storage without data loss."""
    legacy_name = LEGACY_TEXT_NAMES.get(kind)
    legacy = bpy.data.texts.get(legacy_name) if legacy_name else None
    if legacy is None:
        return preferred_replacement

    legacy_xml = legacy.as_string()
    replacement = None
    if (
        preferred_replacement is not None
        and preferred_replacement != legacy
        and preferred_replacement.as_string() == legacy_xml
    ):
        replacement = preferred_replacement
    if replacement is None:
        suffix = f"_{kind.upper()}"
        replacement = next(
            (
                text
                for text in bpy.data.texts
                if text != legacy
                and text.as_string() == legacy_xml
                and (
                    text.get("robot_description_kind", "") == kind
                    or text.name.upper().endswith(suffix)
                )
            ),
            None,
        )

    if replacement is not None:
        legacy.user_remap(replacement)
        _update_text_name_fallbacks(kind, legacy_name, replacement.name)
        bpy.data.texts.remove(legacy)
        return replacement

    try:
        robot_name = str(ET.fromstring(legacy_xml).get("name", "Robot")).strip()
    except ET.ParseError:
        robot_name = "Robot"
    robot_name = robot_name or "Robot"
    robot_object_name = (
        robot_name if robot_name.endswith("_Object") else f"{robot_name}_Object"
    )
    legacy.name = f"{robot_object_name}_{kind.upper()}"
    legacy["robot_description_kind"] = kind
    _update_text_name_fallbacks(kind, legacy_name, legacy.name)
    return legacy


def get_text_mesh_path_base(text_block):
    """Return the absolute base directory for relative mesh paths in a Text block."""
    if not text_block:
        return ""

    path_base = str(text_block.get("mesh_path_base", "")).strip()
    if path_base:
        return os.path.normpath(bpy.path.abspath(path_base))

    source_path = str(text_block.get("source_path", "")).strip()
    if source_path:
        return os.path.dirname(os.path.normpath(bpy.path.abspath(source_path)))

    if bpy.data.filepath:
        return os.path.dirname(bpy.data.filepath)
    return ""
