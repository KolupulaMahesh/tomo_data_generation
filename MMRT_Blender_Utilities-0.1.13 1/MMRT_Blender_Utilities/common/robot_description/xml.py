import os
import re
import tempfile
import copy
import warnings
import xml.etree.ElementTree as ET
from io import StringIO

import bpy
import mathutils
from ...processes.mmrt.definitions import GeneratedAppDirectories, MDataLabel, OutputSuffixes, RobotDescriptionKind, RobotJointTypes

from .storage import store_robot_description


def resolve_mesh_path(path, urdf_filepath, search_paths):
    """Resolve package, absolute, Blender-relative, or XML-relative mesh paths."""
    if not path:
        return None

    path = os.path.expandvars(os.path.expanduser(str(path).strip()))
    if path.startswith("file://"):
        path = path[len("file://"):]
        if os.name == "nt" and path.startswith("/") and len(path) > 2 and path[2] == ":":
            path = path[1:]

    if path.startswith("package://"):
        rel = path[len("package://"):].replace("\\", "/")
        if "/" not in rel:
            return None
        pkg, subpath = rel.split("/", 1)
        for base in search_paths:
            if os.path.basename(base) != pkg:
                continue
            candidates = [
                os.path.join(base, subpath),
                os.path.join(base, GeneratedAppDirectories.MESH_DIR, subpath),
                os.path.join(base, "resources", subpath),
                os.path.join(base, RobotDescriptionKind.URDF, subpath),
                os.path.join(base, "visuals", subpath),
                os.path.join(base, "src", GeneratedAppDirectories.MESH_DIR, subpath),
            ]
            for c in candidates:
                if os.path.exists(c):
                    return os.path.abspath(os.path.normpath(c))
        return None

    if path.startswith("//"):
        blender_path = bpy.path.abspath(path)
        if os.path.exists(blender_path):
            return os.path.abspath(os.path.normpath(blender_path))

    native_path = path.replace("/", os.sep).replace("\\", os.sep)
    if os.path.isabs(native_path) and os.path.exists(native_path):
        return os.path.abspath(os.path.normpath(native_path))

    description_dir = os.path.dirname(os.path.abspath(urdf_filepath))
    candidates = [os.path.join(description_dir, native_path)]
    candidates.extend(os.path.join(base, native_path) for base in search_paths)
    for candidate in candidates:
        if os.path.exists(candidate):
            return os.path.abspath(os.path.normpath(candidate))
    return None


def parse_urdf(filepath):
    tree = ET.parse(filepath)
    root = tree.getroot()
    data = {'joints': [], 'links': [], MDataLabel.ROBOT_NAME_KEY: root.get('name', 'Robot')}

    for j in root.findall('joint'):
        data['joints'].append({
            'name': j.get('name'),
            'type': j.get('type', RobotJointTypes.FIXED),
            'parent': j.find('parent').get('link'),
            'child': j.find('child').get('link'),
            'origin': j.find('origin').attrib if j.find('origin') is not None else {},
            'axis': parse_axis_vector(j),
            'limit': {**(j.find('limit').attrib if j.find('limit') is not None else {})},
            'mimic': j.find('mimic').attrib if j.find('mimic') is not None else None,
        })

    for l in root.findall('link'):
        viz = l.find('visual')
        mesh = viz.find('geometry').find('mesh') if viz is not None else None
        data['links'].append({
            'name': l.get('name'),
            'visual': {'filename': mesh.get('filename')} if mesh is not None else None
        })

    return data


def is_xacro_file(path):
    return path.endswith(".xacro") or "<xacro:" in open(path, encoding="utf-8").read()


def extract_properties(root_element):
    props = {}
    for prop in root_element.findall(".//xacro:property", namespaces={"xacro": "http://www.ros.org/wiki/xacro"}):
        name = prop.get("name")
        value = prop.get("value")
        if name and value:
            props[name] = value
    return props


def substitute_properties_in_element(element, properties):
    for attr in element.attrib:
        val = element.attrib[attr]
        if isinstance(val, str):
            for key, value in properties.items():
                val = re.sub(rf"\$\{{{key}\}}", value, val)
            element.attrib[attr] = val
    for child in element:
        substitute_properties_in_element(child, properties)


def recursively_expand_xacro(filepath, package_dir_map, properties=None):
    """
    Recursively expands <xacro:include> and substitutes <xacro:property> references.
    Returns the combined XML root.
    """
    if properties is None:
        properties = {}

    def resolve_include_path(include_filename):
        if include_filename.startswith("$(find "):
            end_idx = include_filename.find(")")
            pkg = include_filename[7:end_idx]
            rel_path = include_filename[end_idx+1:].lstrip("/")

            if pkg not in package_dir_map:
                raise FileNotFoundError(f"Package '{pkg}' not found in package_dir_map")

            return os.path.join(package_dir_map[pkg], rel_path)
        else:
            include_filename = include_filename.replace("/", os.sep).replace("\\", os.sep)
            if os.path.isabs(include_filename):
                return include_filename
            return os.path.join(os.path.dirname(filepath), include_filename)

    tree = ET.parse(filepath)
    root = tree.getroot()

    new_props = extract_properties(root)
    properties.update(new_props)

    substitute_properties_in_element(root, properties)

    # Resolve before merging so paths in an included Xacro remain relative to
    # the file that declared them, rather than the top-level description.
    resolve_mesh_paths_in_element(root, filepath, list(package_dir_map.values()))

    merged_root = ET.Element(root.tag, root.attrib)

    for elem in root:
        if elem.tag.endswith("include") and "filename" in elem.attrib:
            include_path = resolve_include_path(elem.attrib["filename"])
            included_tree = recursively_expand_xacro(include_path, package_dir_map, properties)
            for child in included_tree:
                merged_root.append(child)
        else:
            merged_root.append(elem)

    return merged_root


def save_element_to_temp_file(element: ET.Element):
    temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=OutputSuffixes.URDF, delete=False)
    tree = ET.ElementTree(element)
    tree.write(temp_file, encoding="unicode", xml_declaration=True)
    temp_file.close()
    return temp_file.name


def element_to_xml_string(element: ET.Element):
    stream = StringIO()
    tree = ET.ElementTree(element)
    tree.write(stream, encoding="unicode", xml_declaration=True)
    return stream.getvalue()


def resolve_mesh_paths_in_element(root_element, urdf_filepath, search_paths):
    for mesh in root_element.findall(".//mesh"):
        original = mesh.get("filename")
        if not original:
            continue
        resolved = resolve_mesh_path(original, urdf_filepath, search_paths)
        if resolved:
            mesh.set("filename", resolved)


def _default_relative_mesh_base(source_filepath):
    if bpy.data.filepath:
        return os.path.dirname(os.path.abspath(bpy.data.filepath))
    return os.path.dirname(os.path.abspath(source_filepath))


def _portable_blender_path(path):
    path = os.path.abspath(os.path.normpath(path))
    if not bpy.data.filepath:
        return path
    try:
        return bpy.path.relpath(path)
    except ValueError:
        return path


def make_mesh_paths_relative(
    root_element,
    source_filepath,
    search_paths=(),
    relative_base=None,
):
    """Rewrite resolvable mesh filenames relative to the blend by default.

    Absolute paths remain only when the source and destination are on different
    Windows drives and no relative representation is possible.
    """
    relative_base = os.path.abspath(
        relative_base or _default_relative_mesh_base(source_filepath)
    )
    for mesh in root_element.findall(".//mesh"):
        original = mesh.get("filename")
        if not original:
            continue
        resolved = resolve_mesh_path(original, source_filepath, list(search_paths))
        if not resolved:
            continue
        try:
            portable_path = os.path.relpath(resolved, relative_base)
        except ValueError:
            portable_path = resolved
            warnings.warn(
                f"Mesh '{resolved}' is on a different filesystem root; storing its absolute path.",
                RuntimeWarning,
            )
        mesh.set("filename", portable_path.replace("\\", "/"))
    return relative_base


def _get_package_dirs(source_filepath):
    folder_root = os.path.dirname(source_filepath)
    project_root = os.path.abspath(os.path.join(folder_root, "..", ".."))
    if not os.path.isdir(project_root):
        return {}

    return {
        d: os.path.join(project_root, d)
        for d in os.listdir(project_root)
        if os.path.isdir(os.path.join(project_root, d))
    }


def read_robot_xml(source_filepath, resolve_mesh_paths=False):
    source_filepath = bpy.path.abspath(source_filepath)
    if not os.path.exists(source_filepath):
        raise FileNotFoundError(f"File not found at {source_filepath}")

    package_dirs = _get_package_dirs(source_filepath)
    if is_xacro_file(source_filepath):
        root = recursively_expand_xacro(source_filepath, package_dirs)
    else:
        root = ET.parse(source_filepath).getroot()

    if resolve_mesh_paths:
        resolve_mesh_paths_in_element(root, source_filepath, list(package_dirs.values()))

    return root, package_dirs


def store_robot_xml(
    kind,
    root_element,
    source_filepath="",
    relative_mesh_paths=True,
    search_paths=(),
    text_name=None,
):
    source_filepath = bpy.path.abspath(source_filepath) if source_filepath else ""
    stored_root = copy.deepcopy(root_element)
    mesh_path_base = ""
    if relative_mesh_paths and source_filepath:
        mesh_path_base = make_mesh_paths_relative(
            stored_root,
            source_filepath,
            search_paths=search_paths,
        )

    portable_source = _portable_blender_path(source_filepath) if source_filepath else ""
    portable_base = _portable_blender_path(mesh_path_base) if mesh_path_base else ""
    if not text_name:
        robot_name = str(root_element.get("name", "Robot")).strip() or "Robot"
        robot_object_name = (
            robot_name if robot_name.endswith("_Object") else f"{robot_name}_Object"
        )
        text_name = f"{robot_object_name}_{kind.upper()}"

    return store_robot_description(
        kind,
        element_to_xml_string(stored_root),
        text_name=text_name,
        source_path=portable_source,
        mesh_path_base=portable_base,
    )


def store_robot_xml_from_path(
    kind,
    source_filepath,
    resolve_mesh_paths=True,
    text_name=None,
):
    root, package_dirs = read_robot_xml(
        source_filepath,
        resolve_mesh_paths=resolve_mesh_paths,
    )
    return store_robot_xml(
        kind,
        root,
        source_filepath,
        search_paths=list(package_dirs.values()),
        text_name=text_name,
    )


def store_srdf_from_path(srdf_path, text_name=None):
    return store_robot_xml_from_path(
        RobotDescriptionKind.SRDF,
        srdf_path,
        resolve_mesh_paths=True,
        text_name=text_name,
    )


def store_optional_srdf_from_path(srdf_path, text_name=None):
    srdf_path = srdf_path.strip()
    if not srdf_path:
        return None

    srdf_path = bpy.path.abspath(srdf_path)
    return store_srdf_from_path(srdf_path, text_name=text_name)


def parse_axis_vector(joint):
    axis_elem = joint.find('axis')
    if axis_elem is not None:
        xyz = axis_elem.get('xyz', '0 0 1')
    else:
        xyz = '0 0 1'
    return mathutils.Vector([float(v) for v in xyz.split()])
