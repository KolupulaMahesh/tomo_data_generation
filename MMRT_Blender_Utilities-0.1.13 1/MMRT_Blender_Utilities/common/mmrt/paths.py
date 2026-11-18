import os

from ..config import AddonDefaults
from ...processes.mmrt.definitions import GeneratedAppDirectories, OutputSuffixes
from ..system.paths import normalize_output_dir


def get_mmrt_basefilepath(scene):
    """
    Returns:
        /abs/path/output/<basename>
    without extension
    """
    props = scene.mmrt_settings

    output_dir = normalize_output_dir(props.output_dir)
    basename = props.output_basename.strip()

    if not basename:
        raise ValueError("MMRT output basename is empty.")

    return os.path.join(output_dir, basename)


def get_mmrt_xml_filepath(scene):
    return os.path.join(
        get_local_mmrt_app_dir(scene), GeneratedAppDirectories.TREE_DIR,
        scene.mmrt_settings.output_basename.strip() + OutputSuffixes.BEHAVIOR_TREE_XML
    )


def get_mmrt_yaml_filepath(scene):
    return os.path.join(
        get_local_mmrt_app_dir(scene), GeneratedAppDirectories.BLACKBOARD_DIR,
        scene.mmrt_settings.output_basename.strip() + OutputSuffixes.BLACKBOARD_YAML
    )


def get_local_mmrt_app_dir(scene=None):
    """Return the configured local MMRT app directory used for deployment."""
    if scene is not None:
        return normalize_output_dir(scene.mmrt_settings.output_dir, create=False)
    return normalize_output_dir(AddonDefaults.MMRT_OUTPUT_APP_DIR, create=False)
