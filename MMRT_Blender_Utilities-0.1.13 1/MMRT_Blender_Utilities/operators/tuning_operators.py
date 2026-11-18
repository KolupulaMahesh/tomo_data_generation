import json
import os
import re
import tempfile

import bpy

from ..common.blender.utils import get_bl_idname
from ..processes.mmrt.definitions import (
    GeneratedAppDirectories,
    GeneratedAppFilenames,
    MDataLabel,
    MeasurementUnits as MeasurementConstants,
)
from ..common.recording.frame_domain import get_frame_domain
from ..common.recording.recorder import MMRTRecorder

MDataLabel = MDataLabel


def _app_paths(app_dir):
    app_dir = os.path.abspath(os.path.expanduser(bpy.path.abspath(app_dir)))
    metadata_path = os.path.join(app_dir, GeneratedAppFilenames.ADDON_TUNING_METADATA_FILE)
    blackboard_path = os.path.join(
        app_dir,
        GeneratedAppDirectories.BLACKBOARD_DIR,
        GeneratedAppFilenames.MOTION_BLACKBOARD_FILE,
    )
    return app_dir, metadata_path, blackboard_path


def _load_app_metadata(app_dir):
    app_dir, metadata_path, blackboard_path = _app_paths(app_dir)
    if not os.path.isdir(app_dir):
        raise ValueError("Select an existing generated app folder.")
    if not os.path.isfile(metadata_path):
        raise ValueError(
            "This app folder has no Blender tuning metadata. Generate it again with this addon."
        )
    if not os.path.isfile(blackboard_path):
        raise ValueError(
            "The app folder is missing "
            f"{GeneratedAppDirectories.BLACKBOARD_DIR}/{GeneratedAppFilenames.MOTION_BLACKBOARD_FILE}."
        )
    with open(metadata_path, "r", encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)
    if metadata.get(MDataLabel.SCHEMA_VERSION_KEY) != MDataLabel.SCHEMA_VERSION:
        raise ValueError("Unsupported Blender tuning metadata version.")
    if not isinstance(metadata.get(MDataLabel.WAYPOINTS_KEY), dict):
        raise ValueError("The tuning metadata does not contain a waypoint map.")
    return app_dir, metadata_path, blackboard_path, metadata


def _validate_scene_frame_domain(scene, tracking):
    if tracking.get(MDataLabel.SAMPLE_FRAME_DOMAIN_KEY) != MDataLabel.SAMPLE_FRAME_DOMAIN_SCENE_FRAME_SET:
        raise ValueError("The app does not describe a supported Blender sample-frame domain.")
    expected = {
        MDataLabel.SCENE_FRAME_START_KEY: int(scene.frame_start),
        MDataLabel.SCENE_FRAME_END_KEY: int(scene.frame_end),
        MDataLabel.FRAME_MAP_OLD_KEY: int(scene.render.frame_map_old),
        MDataLabel.FRAME_MAP_NEW_KEY: int(scene.render.frame_map_new),
    }
    for field, current in expected.items():
        tracked = tracking.get(field)
        if tracked is None or int(tracked) != current:
            raise ValueError(
                "The Blender frame range or time mapping changed since generation. "
                "Regenerate the complete app before tuning this motion."
            )
    domain = get_frame_domain(scene)
    if (
        int(tracking.get(MDataLabel.SAMPLE_FRAME_START_KEY, -1))
        != domain["stretched_start"]
        or int(tracking.get(MDataLabel.SAMPLE_FRAME_END_KEY, -1))
        != domain["stretched_end"]
    ):
        raise ValueError(
            "The tracked sample-frame range no longer matches the scene. "
            "Regenerate the complete app before tuning this motion."
        )


def _parse_yaml_key(token):
    token = token.strip()
    if token.startswith('"') and token.endswith('"'):
        return json.loads(token)
    if token.startswith("'") and token.endswith("'"):
        return token[1:-1].replace("''", "'")
    return token


def _mapping_key_and_value(line):
    quote = None
    escaped = False
    index = 0
    while index < len(line):
        char = line[index]
        if quote == '"':
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quote = None
        elif quote == "'":
            if char == "'":
                if index + 1 < len(line) and line[index + 1] == "'":
                    index += 2
                    continue
                quote = None
        elif char in ('"', "'"):
            quote = char
        elif char == ":":
            return _parse_yaml_key(line[:index]), line[index + 1:], line[:index]
        index += 1
    return None, None, None


def _format_yaml_csv_value(key_prefix, values, width=82):
    """Format a plain YAML CSV scalar, folding long values like PyYAML."""
    tokens = [str(float(value)) for value in values]
    if not tokens:
        raise ValueError("Cannot write an empty waypoint joint value.")

    lines = []
    current_line = key_prefix + tokens[0]
    for token in tokens[1:]:
        if len(current_line) + 2 + len(token) > width:
            lines.append(current_line + ",")
            current_line = "      " + token
        else:
            current_line += ", " + token
    lines.append(current_line)
    return lines


def _rewrite_waypoint_blackboard(path, updates):
    """Patch generated waypoint scalar lines and atomically replace the file."""
    with open(path, "r", encoding="utf-8", newline="") as source_file:
        lines = source_file.readlines()

    in_waypoints = False
    current_waypoint = None
    skip_scalar_continuation = False
    found = set()
    rewritten = []
    for line in lines:
        body = line.rstrip("\r\n")
        newline = line[len(body):]
        indent = len(body) - len(body.lstrip(" "))
        key, _value, raw_key = _mapping_key_and_value(body.lstrip(" "))

        if skip_scalar_continuation:
            if indent > 4 and key is None and not body.lstrip().startswith("#"):
                continue
            skip_scalar_continuation = False

        if indent == 0:
            in_waypoints = key == MDataLabel.WAYPOINTS_KEY
            current_waypoint = None
            rewritten.append(line)
            continue

        if in_waypoints and indent == 2 and key is not None:
            current_waypoint = key
            rewritten.append(line)
            continue

        if in_waypoints and indent == 4 and current_waypoint in updates and key in updates[current_waypoint]:
            entry = (current_waypoint, key)
            if entry in found:
                raise ValueError(
                    f"Waypoint data '{current_waypoint}.{key}' appears more than once in the motion blackboard."
                )
            values = updates[current_waypoint][key]
            prefix = body[:indent]
            output_key = key if re.fullmatch(r"[A-Za-z0-9_-]+", key) else raw_key
            scalar_prefix = f"{prefix}{output_key}: "
            formatted_lines = _format_yaml_csv_value(scalar_prefix, values)
            rewritten.extend(
                line + ((newline or "\n") if index < len(formatted_lines) - 1 else newline)
                for index, line in enumerate(formatted_lines)
            )
            skip_scalar_continuation = True
            found.add(entry)
        else:
            rewritten.append(line)

    expected = {
        (waypoint_name, joint_key)
        for waypoint_name, joint_values in updates.items()
        for joint_key in joint_values
    }
    missing = expected - found
    if missing:
        waypoint_name, joint_key = sorted(missing)[0]
        raise ValueError(
            f"Waypoint data '{waypoint_name}.{joint_key}' was not found in the motion blackboard."
        )

    directory = os.path.dirname(path)
    fd, temporary_path = tempfile.mkstemp(prefix=".motion_waypoints_", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as output_file:
            output_file.writelines(rewritten)
        os.replace(temporary_path, path)
    except Exception:
        try:
            os.unlink(temporary_path)
        except OSError:
            pass
        raise


class MMRT_OT_TuningLoadApp(bpy.types.Operator):
    bl_idname = get_bl_idname("tuning_load_app")
    bl_label = "Load App Folder"
    bl_description = "Load the generated waypoint and Blender frame tracking data"

    def execute(self, context):
        tuning = context.scene.mmrt_tuning
        tuning.loaded = False
        tuning.loaded_app_dir = ""
        tuning.scene_frame_range_known = False
        tuning.waypoints.clear()
        try:
            app_dir, _metadata_path, _blackboard_path, metadata = _load_app_metadata(tuning.app_dir)
            tracking = metadata.get(MDataLabel.RECORDING_KEY, {})
            entries = sorted(
                metadata[MDataLabel.WAYPOINTS_KEY].items(),
                key=lambda pair: (
                    min(pair[1].get(MDataLabel.SAMPLE_FRAMES_KEY) or [0]),
                    pair[0],
                ),
            )
            for name, data in entries:
                frames = sorted({
                    int(frame)
                    for frame in data.get(MDataLabel.SAMPLE_FRAMES_KEY, [])
                })
                item = tuning.waypoints.add()
                item.name = name
                item.movegroup = str(data.get(MDataLabel.MOVEGROUP_KEY) or "")
                item.first_frame = frames[0] if frames else 0
                item.occurrence_count = len(frames)
                item.sample_frames = ", ".join(map(str, frames))
                item.tracking_error = str(data.get(MDataLabel.TRACKING_ERROR_KEY) or "")
            scene_frame_start = tracking.get(MDataLabel.SCENE_FRAME_START_KEY)
            scene_frame_end = tracking.get(MDataLabel.SCENE_FRAME_END_KEY)
            tuning.scene_frame_range_known = (
                scene_frame_start is not None and scene_frame_end is not None
            )
            tuning.scene_frame_start = int(scene_frame_start or 0)
            tuning.scene_frame_end = int(scene_frame_end or 0)
            tuning.waypoints_index = min(max(0, tuning.waypoints_index), len(tuning.waypoints) - 1)
            tuning.loaded_app_dir = app_dir
            tuning.loaded = True
            tuning.status = f"Loaded {len(entries)} tracked waypoints."
            self.report({'INFO'}, tuning.status)
            return {'FINISHED'}
        except Exception as exc:
            tuning.status = str(exc)
            self.report({'ERROR'}, tuning.status)
            return {'CANCELLED'}


class MMRT_OT_TuningGoToWaypoint(bpy.types.Operator):
    bl_idname = get_bl_idname("tuning_go_to_waypoint")
    bl_label = "Go to Waypoint Frame"
    bl_description = "Move the Blender timeline to the selected waypoint's tracked frame"

    def execute(self, context):
        tuning = context.scene.mmrt_tuning
        if not tuning.loaded or not (0 <= tuning.waypoints_index < len(tuning.waypoints)):
            self.report({'WARNING'}, "Load an app folder and select a waypoint first.")
            return {'CANCELLED'}
        scene = context.scene
        scene.frame_set(tuning.waypoints[tuning.waypoints_index].first_frame)
        return {'FINISHED'}


class MMRT_OT_TuningApplyMotion(bpy.types.Operator):
    bl_idname = get_bl_idname("tuning_apply_motion")
    bl_label = "Apply Blender Motion to App"
    bl_description = "Resample tracked Blender frames and update waypoint values in the app blackboard"

    def execute(self, context):
        tuning = context.scene.mmrt_tuning
        scene = context.scene
        old_frame = scene.frame_current
        recorder = None
        original_angle_unit = None
        try:
            if not tuning.loaded:
                raise ValueError("Load the selected app folder before applying motion.")
            app_dir, _metadata_path, blackboard_path, metadata = _load_app_metadata(tuning.app_dir)
            if os.path.normcase(os.path.realpath(app_dir)) != os.path.normcase(
                os.path.realpath(tuning.loaded_app_dir)
            ):
                raise ValueError("Load the changed app folder before applying motion.")
            tracking = metadata.get(MDataLabel.RECORDING_KEY, {})
            _validate_scene_frame_domain(scene, tracking)
            if not metadata[MDataLabel.WAYPOINTS_KEY]:
                raise ValueError("This app has no tracked waypoints to update.")
            for waypoint_name, waypoint in metadata[MDataLabel.WAYPOINTS_KEY].items():
                if waypoint.get(MDataLabel.TRACKING_ERROR_KEY):
                    raise ValueError(
                        f"Waypoint '{waypoint_name}' cannot be tuned: "
                        f"{waypoint[MDataLabel.TRACKING_ERROR_KEY]}"
                    )

            recorder = MMRTRecorder(context)
            target_angle_units = str(
                tracking.get(MDataLabel.ANGLE_UNITS_KEY)
                or MeasurementConstants.RADIAN_UNIT
            ).upper()
            if target_angle_units not in {
                MeasurementConstants.RADIAN_UNIT,
                MeasurementConstants.DEGREE_UNIT,
            }:
                raise ValueError(f"Unsupported angle unit '{target_angle_units}'.")
            original_angle_unit = recorder._cfg.angle_unit
            recorder._cfg.angle_unit = target_angle_units
            recorder.prepare()
            sampled_frames = {}
            all_frames = sorted({
                int(frame)
                for waypoint in metadata[MDataLabel.WAYPOINTS_KEY].values()
                for frame in waypoint.get(MDataLabel.SAMPLE_FRAMES_KEY, [])
            })
            if not all_frames:
                raise ValueError("The app metadata contains no tracked waypoint frames.")

            for frame in all_frames:
                scene.frame_set(frame)
                sampled_frames[frame] = recorder.evaluate_current_frame()

            updates = {}
            for waypoint_name, waypoint in metadata[MDataLabel.WAYPOINTS_KEY].items():
                frames = sorted({
                    int(frame)
                    for frame in waypoint.get(MDataLabel.SAMPLE_FRAMES_KEY, [])
                })
                joint_keys = waypoint.get(MDataLabel.JOINT_KEYS_KEY) or {}
                if not frames or not joint_keys:
                    raise ValueError(f"Waypoint '{waypoint_name}' has incomplete tuning metadata.")
                waypoint_updates = {}
                for blackboard_key, joint_names in joint_keys.items():
                    if not joint_names:
                        raise ValueError(
                            f"Waypoint '{waypoint_name}' has no recorded joints for '{blackboard_key}'."
                        )
                    occurrence_values = []
                    for frame in frames:
                        frame_data = sampled_frames[frame]
                        values = []
                        for joint_name in joint_names:
                            if joint_name not in frame_data:
                                raise ValueError(
                                    f"Recorder data is missing joint '{joint_name}' at frame {frame}."
                                )
                            values.append(float(frame_data[joint_name]))
                        occurrence_values.append(values)

                    reference = occurrence_values[0]
                    if any(
                        any(f"{a:.6f}" != f"{b:.6f}" for a, b in zip(reference, candidate))
                        for candidate in occurrence_values[1:]
                    ):
                        raise ValueError(
                            f"Waypoint '{waypoint_name}' occurs at multiple frames with different poses. "
                            "Its generated tree shares one waypoint value, so regenerate the app to split it."
                        )
                    waypoint_updates[blackboard_key] = reference
                updates[waypoint_name] = waypoint_updates

            _rewrite_waypoint_blackboard(blackboard_path, updates)
            tuning.loaded = True
            tuning.status = f"Updated {len(updates)} waypoints in the app motion blackboard."
            self.report({'INFO'}, tuning.status)
            return {'FINISHED'}
        except Exception as exc:
            tuning.status = str(exc)
            self.report({'ERROR'}, tuning.status)
            return {'CANCELLED'}
        finally:
            if recorder is not None and original_angle_unit is not None:
                recorder._cfg.angle_unit = original_angle_unit
            scene.frame_set(old_frame)


classes = (
    MMRT_OT_TuningLoadApp,
    MMRT_OT_TuningGoToWaypoint,
    MMRT_OT_TuningApplyMotion,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
