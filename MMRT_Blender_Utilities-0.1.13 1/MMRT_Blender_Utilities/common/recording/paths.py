"""Output paths shared by recording and environment export code."""

import os

from ...processes.mmrt.definitions import OutputSuffixes
from ..system.paths import normalize_output_dir


def get_recorder_mmrt_filepath(context):
    config = context.scene.mmrt_recorder.config
    output_dir = normalize_output_dir(config.output_dir)
    return os.path.join(
        output_dir,
        config.base_filename + OutputSuffixes.YAML_RECORDING,
    )


def get_recorder_environment_filepath(context):
    config = context.scene.mmrt_recorder.config
    output_dir = normalize_output_dir(config.output_dir)
    return os.path.join(
        output_dir,
        config.base_filename + OutputSuffixes.ENVIRONMENT_JSON,
    )


def get_recorder_urdf_filepath(context):
    config = context.scene.mmrt_recorder.config
    output_dir = normalize_output_dir(config.output_dir)
    return os.path.join(output_dir, config.base_filename + OutputSuffixes.URDF)


def get_recorder_srdf_filepath(context):
    config = context.scene.mmrt_recorder.config
    output_dir = normalize_output_dir(config.output_dir)
    return os.path.join(output_dir, config.base_filename + OutputSuffixes.SRDF)
