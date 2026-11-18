import xml.etree.ElementTree as ET
import tempfile
import os
import re
import contextvars
import functools
from typing import TypedDict, Any, List, Optional
from common.logging_setup import trace
from definitions import GeneratedAppDirectories, MDataLabel, OutputSuffixes, RobotDescriptionKind, RobotJointTypes
import logging
logger = logging.getLogger(__name__)

_resolved_temp_files = contextvars.ContextVar("resolved_temp_files", default=None)


def cleanup_resolved_temp_files(func):
    """Remove resolved XACRO files when the wrapped operation finishes."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        token = _resolved_temp_files.set([])
        try:
            return func(*args, **kwargs)
        finally:
            temp_files = _resolved_temp_files.get()
            for filepath in temp_files or ():
                try:
                    os.remove(filepath)
                    logger.debug(f"Removed temporary resolved robot description: {filepath}")
                except FileNotFoundError:
                    pass
                except OSError as exc:
                    logger.warning(f"Could not remove temporary resolved robot description '{filepath}': {exc}")
            _resolved_temp_files.reset(token)
    return wrapper


def _track_resolved_temp_file(filepath: str):
    temp_files = _resolved_temp_files.get()
    if temp_files is not None:
        temp_files.append(filepath)


class JointData(TypedDict, total=False):
    name: str
    type: str
    parent: str
    child: str
    origin: dict[str, str]
    axis: Any
    limit: dict[str, str]
    mimic: Optional[dict[str, str]]


class LinkData(TypedDict, total=False):
    name: str
    visual: Optional[dict[str, str]]


class URDFData(TypedDict):
    robot_name: str
    joints: List[JointData]
    links: List[LinkData]


def resolve_mesh_path(path: str, urdf_filepath: str, search_paths: list) -> str|None:
    if path.startswith("package://"):
        rel = path[len("package://"):]
        if '/' not in rel:
            return None
        pkg, subpath = rel.split("/", 1)
        for base in search_paths:
            if os.path.basename(base) != pkg:
                continue
            # Try resolving directly or through common mesh-related subdirs
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
                    return os.path.normpath(c)
    else:
        full = os.path.join(os.path.dirname(urdf_filepath), path)
        if os.path.exists(full):
            return os.path.normpath(full)
    return None


def load_urdf_description(filepath: str) -> URDFData:
    tree = ET.parse(filepath)
    root = tree.getroot()

    data: URDFData = {
        MDataLabel.ROBOT_NAME_KEY: root.get('name', 'Robot'),
        'joints': [],
        'links': []
    }

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


def parse_axis_vector(joint):
    axis_elem = joint.find('axis')

    if axis_elem is not None:
        xyz = axis_elem.get('xyz', '0 0 1')
    else:
        xyz = '0 0 1'

    return [float(v) for v in xyz.split()]


def load_srdf_collision_pairs(filepath: str) -> List[tuple[str, str]]:
    tree = ET.parse(filepath)
    root = tree.getroot()
    collision_links = []
    for disable_coll in root.findall('disable_collisions'):
        link1 = disable_coll.get('link1')
        link2 = disable_coll.get('link2')
        if link1 and link2:
            collision_links.append((link1, link2))
    return collision_links


@trace
def load_srdf_joint_groups(filepath: str, urdf_data: URDFData) -> dict:
    """Convert SRDF groups into the MMRT atomic/move-group structure."""
    tree = ET.parse(filepath)
    root = tree.getroot()
    all_joints = {
        joint['name']: joint for joint in urdf_data.get('joints', []) if joint.get('name')
    }
    joints = {
        name: joint for name, joint in all_joints.items()
        if joint.get('type', RobotJointTypes.FIXED) not in RobotJointTypes.NON_MOVING and not joint.get('mimic')
    }
    graph = {}
    for joint in all_joints.values():
        graph.setdefault(joint.get('parent'), []).append(joint)

    def chain_joints(base, tip):
        result = []
        def walk(link):
            if link == tip:
                return True
            for joint in graph.get(link, []):
                if walk(joint.get('child')):
                    if joint.get('type') not in RobotJointTypes.NON_MOVING and not joint.get('mimic'):
                        result.insert(0, joint['name'])
                    return True
            return False
        walk(base)
        return result

    raw = {}
    for group in root.findall('group'):
        name = group.get('name')
        if not name:
            continue
        members = [node.get('name') for node in group.findall('joint') if node.get('name') in joints]
        for chain in group.findall('chain'):
            members.extend(chain_joints(chain.get('base_link'), chain.get('tip_link')))
        raw[name] = {
            "joints": list(dict.fromkeys(members)),
            "groups": [node.get('name') for node in group.findall('group') if node.get('name')],
        }

    atomic_groups = {
        name: value["joints"] for name, value in raw.items() if value["joints"]
    }

    def expand_atomic_groups(name, stack=None):
        stack = set(stack or ())
        if name in stack:
            return []
        stack.add(name)
        result = []
        for member in raw.get(name, {}).get("groups", []):
            if member in atomic_groups and member not in result:
                result.append(member)
            elif member in raw:
                result.extend(expand_atomic_groups(member, stack))
        return result

    move_groups = {name: expand_atomic_groups(name) for name in raw}
    # Direct-joint groups are valid move groups by themselves as well.
    for name in atomic_groups:
        move_groups[name] = [name]
    logger.debug(f"Parsed SRDF groups: {len(atomic_groups)} atomic, {len(move_groups)} move groups")
    logger.debug(f"Atomic groups: {atomic_groups}")
    logger.debug(f"Move groups: {move_groups}")
    return {
        'atomic_groups': atomic_groups,
        'move_groups': move_groups,
        'group_end_effectors': {},
    }


@trace
def resolve_urdf_source(filepath: str) -> str:
    folder_root = os.path.dirname(filepath)
    project_root = os.path.abspath(os.path.join(folder_root, "..", ".."))

    if not os.path.exists(filepath):
        logger.error(f"File not found at {filepath}")
        return ""

    if is_xacro_file(filepath):
        try:
            package_dirs = {
                d: os.path.join(project_root, d)
                for d in os.listdir(project_root)
                if os.path.isdir(os.path.join(project_root, d))
            }
            merged_root = recursively_expand_xacro(filepath, package_dirs)

            # === Resolve mesh paths in the merged XML tree ===
            for mesh in merged_root.findall(".//mesh"):
                original = mesh.get("filename")
                if not original:
                    continue
                resolved = resolve_mesh_path(original, filepath, list(package_dirs.values()))
                if resolved:
                    mesh.set("filename", resolved)
                else:
                    logger.warning(f"Could not resolve mesh path: {original}")

            filepath = save_element_to_temp_file(merged_root, suffix=OutputSuffixes.URDF)
            _track_resolved_temp_file(filepath)
            logger.debug(f"Expanded XACRO to URDF with resolved meshes: {filepath}")
        except Exception as e:
            logger.error(f"XACRO expansion failed: {e}")
            return ""

    return filepath


@trace
def resolve_srdf_source(filepath: str) -> str:
    folder_root = os.path.dirname(filepath)
    project_root = os.path.abspath(os.path.join(folder_root, "..", ".."))

    if not os.path.exists(filepath):
        logger.error(f"File not found at {filepath}")
        return ""

    if is_xacro_file(filepath):
        try:
            package_dirs = {
                d: os.path.join(project_root, d)
                for d in os.listdir(project_root)
                if os.path.isdir(os.path.join(project_root, d))
            }
            merged_root = recursively_expand_xacro(filepath, package_dirs)
            filepath = save_element_to_temp_file(merged_root, suffix=OutputSuffixes.SRDF)
            _track_resolved_temp_file(filepath)
            logger.info(f"Expanded XACRO to SRDF: {filepath}")
        except Exception as e:
            logger.error(f"XACRO expansion failed: {e}")
            return ""

    return filepath


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
            return include_filename

    tree = ET.parse(filepath)
    root = tree.getroot()

    # Extract <xacro:property> and update shared dictionary
    new_props = extract_properties(root)
    properties.update(new_props)

    # Substitute all element attributes that reference properties
    substitute_properties_in_element(root, properties)

    # Clone root to build the merged tree
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


def save_element_to_temp_file(element: ET.Element, suffix=OutputSuffixes.URDF):
    temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False)
    tree = ET.ElementTree(element)
    tree.write(temp_file, encoding="unicode", xml_declaration=True)
    temp_file.close()
    return temp_file.name


def get_joint_limits(urdf_dict: URDFData) -> dict[str, list[float|None]]:
    limits_dict: dict = {}
    for joint in urdf_dict['joints']:
        joint_name = joint.get('name')
        joint_limit = joint.get('limit')
        if joint_name and joint_limit:
            # Convert string to float if not None
            lower = joint_limit.get('lower')
            upper = joint_limit.get('upper')
            try:
                lower = float(lower) if lower is not None else None
            except ValueError:
                lower = None
            try:
                upper = float(upper) if upper is not None else None
            except ValueError:
                upper = None
            limits_dict[joint_name] = [lower, upper]
    return limits_dict
