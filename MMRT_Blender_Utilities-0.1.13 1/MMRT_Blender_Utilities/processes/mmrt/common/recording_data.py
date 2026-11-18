import os
import yaml
import numpy as np
import math
import json
from enum import Enum
import logging
from scipy.spatial.transform import Rotation as R
from definitions import (
    BlackboardDefaults,
    EpisodeConfig,
    EnvironmentSchema,
    FlagTypes,
    GeneratedAppDirectories,
    GeneratedAppFilenames,
    MDataLabel,
    MarkerEventSections,
    MeasurementUnits,
    MMRTPlannerConfig,
    OutputSuffixes,
    RotationFormat,
    RobotDescriptionKind,
    WaypointMergeMethod,
)
logger = logging.getLogger(__name__)


class GeneratedAppConfig:
    """Hardcoded runtime configuration required by the robot MMRT host."""

    PLANNER_CONFIG = {
        "manager_type": "realtime",
        "velocity_scaling_factor": BlackboardDefaults.VELOCITY_SCALE,
        "acceleration_scaling_factor": BlackboardDefaults.ACCELERATION_SCALE,
        "pipeline": MMRTPlannerConfig.DEFAULT_PIPELINE,
        "id": MMRTPlannerConfig.DEFAULT_PLANNER_ID,
    }

    @classmethod
    def create(
        cls,
        tree_id: str,
        packages,
        velocity_scaling_factor: float,
        acceleration_scaling_factor: float,
        plan_pipeline: str = None,
        plan_id: str = None,
        tree_files=None,
        blackboard_files=None,
    ) -> dict:
        planner_config = dict(cls.PLANNER_CONFIG)
        planner_config["velocity_scaling_factor"] = velocity_scaling_factor
        planner_config["acceleration_scaling_factor"] = acceleration_scaling_factor
        if plan_pipeline is not None:
            planner_config["pipeline"] = plan_pipeline
        if plan_id is not None:
            planner_config["id"] = plan_id
        bt_config = {
            "tree_id": tree_id,
            "packages": sorted(packages),
        }
        if tree_files:
            bt_config["tree_files"] = list(tree_files)
        if blackboard_files:
            bt_config["blackboard_files"] = list(blackboard_files)
        return {
            "planner_config": planner_config,
            "bt_config": bt_config,
        }


class RecordingData:
    def __init__(self, metafile_path: str):
        self.metafile_path = metafile_path
        self.base_dir = os.path.dirname(os.path.abspath(metafile_path))
        self._load_metadata()
        self.episodes = self._parse_episodes()
        self.metadata = {
            MDataLabel.FPS_KEY: self.fps,
            MDataLabel.FRAME_START_KEY: self.frame_start,
            MDataLabel.FRAME_END_KEY: self.frame_end,
            MDataLabel.FRAME_COUNT_KEY: self.frame_count,
            MDataLabel.SOURCE_START_FRAME_KEY: self.source_frame_start,
            MDataLabel.SOURCE_END_FRAME_KEY: self.source_frame_end,
            MDataLabel.FRAME_MAP_OLD_KEY: self.frame_map_old,
            MDataLabel.FRAME_MAP_NEW_KEY: self.frame_map_new,
            MDataLabel.FRAME_STRETCH_FACTOR_KEY: self.frame_stretch_factor,
            MDataLabel.LABELS_KEY: self.labels,
            MDataLabel.ANGLE_UNITS_KEY: self.angle_units,
            MDataLabel.LOCATION_UNITS_KEY: self.location_units,
            MDataLabel.ROBOT_NAME_KEY: self.robot_name,
            'urdf_path': self.urdf_path,
            'srdf_path': self.srdf_path,
            'environment_path': self.environment_path,
            MDataLabel.EPISODES_KEY: self.episodes,
        }

        self.flags_general = self._parse_general_flags()
        self.flags_custom = self._parse_custom_flags()
        self.flags_io      = self._parse_io_events()
        self.flags_led     = self._parse_led_events()
        self.flags_wp_names = self._parse_wp_names()

        self.data = self._load_frames()
        self.labeled_data = self._label_data()

    def _load_metadata(self):
        with open(self.metafile_path, 'r') as f:
            self._meta = yaml.safe_load(f)
        f.close()
        
        md = self._meta[MDataLabel.METADATA_KEY]
        raw_labels = self._meta[MDataLabel.LABELS_KEY]
        if isinstance(raw_labels, dict):
            self.label_sections = {
                MDataLabel.ROBOTS_KEY: list(raw_labels.get(MDataLabel.ROBOTS_KEY, [])),
                MDataLabel.ACTIVE_OBJECTS_KEY: list(raw_labels.get(MDataLabel.ACTIVE_OBJECTS_KEY, [])),
            }
            self.labels = self.label_sections[MDataLabel.ROBOTS_KEY] + self.label_sections[MDataLabel.ACTIVE_OBJECTS_KEY]
        else:
            self.label_sections = {
                MDataLabel.ROBOTS_KEY: [],
                MDataLabel.ACTIVE_OBJECTS_KEY: list(raw_labels or []),
            }
            self.labels = list(raw_labels or [])
        # Recorded samples are indexed in the recorder/output domain.  The
        # optional source and map fields describe where that domain came from;
        # they must not be applied again to sample indices here.
        self.frame_start = int(md[MDataLabel.START_FRAME_KEY])
        self.frame_count = int(md[MDataLabel.TOTAL_FRAMES_KEY])
        self.frame_end = int(md.get(
            MDataLabel.END_FRAME_KEY,
            self.frame_start + self.frame_count - 1,
        ))
        expected_frame_count = self.frame_end - self.frame_start + 1
        if expected_frame_count != self.frame_count:
            raise ValueError(
                "Recording metadata frame range does not match total_frames: "
                f"{self.frame_start}-{self.frame_end} vs {self.frame_count}."
            )
        self.fps = float(md[MDataLabel.FRAME_RATE_KEY])
        self.source_frame_start = md.get(MDataLabel.SOURCE_START_FRAME_KEY)
        self.source_frame_end = md.get(MDataLabel.SOURCE_END_FRAME_KEY)
        self.frame_map_old = int(md.get(MDataLabel.FRAME_MAP_OLD_KEY, 1))
        self.frame_map_new = int(md.get(MDataLabel.FRAME_MAP_NEW_KEY, 1))
        if self.frame_map_old < 1 or self.frame_map_new < 1:
            raise ValueError("Recording metadata contains invalid frame-map values.")
        self.frame_stretch_factor = self.frame_map_new / self.frame_map_old
        self.addon_frame_tracking = md.get(MDataLabel.ADDON_FRAME_TRACKING_KEY, {}) or {}
        self.angle_units = md.get(
            MDataLabel.ANGLE_UNITS_KEY,
            MeasurementUnits.RADIAN_UNIT,
        )
        self.location_units = md.get(
            MDataLabel.LOCATION_UNITS_KEY,
            MeasurementUnits.METER_UNIT,
        )
        robot_description = md.get('robot_description', {}) or {}
        self.robot_name = md.get(MDataLabel.ROBOT_NAME_KEY)
        self.urdf_path = self.resolve_bundle_path(
            md.get(MDataLabel.URDF_FILE_KEY) or robot_description.get(MDataLabel.URDF_FILE_KEY) or robot_description.get(RobotDescriptionKind.URDF)
        )
        self.srdf_path = self.resolve_bundle_path(
            md.get(MDataLabel.SRDF_FILE_KEY) or robot_description.get(MDataLabel.SRDF_FILE_KEY) or robot_description.get(RobotDescriptionKind.SRDF)
        )
        self.environment_path = self.resolve_bundle_path(md.get(MDataLabel.ENVIRONMENT_FILE_KEY))
        self.num_items = len(self.labels)
        self.is_binary = self._meta.get('binary', False)
        if self.is_binary:
            self.bin_path = os.path.join(self.base_dir, self._meta['bin_filename'])
        else:
            self.frames = self._meta.get('frames', [])

    def resolve_bundle_path(self, path):
        """Resolve an absolute or recording-relative bundle resource path."""
        if not path:
            return None
        path = os.path.expandvars(os.path.expanduser(str(path).strip()))
        native_path = path.replace('\\', os.sep).replace('/', os.sep)
        if os.path.isabs(native_path):
            return os.path.abspath(os.path.normpath(native_path))
        return os.path.abspath(os.path.normpath(os.path.join(self.base_dir, native_path)))

    def _parse_general_flags(self):
        flags = {}
        raw_flags = self._meta.get(MarkerEventSections.FLAGS, [])
        for entry in raw_flags:
            if not isinstance(entry, list) or len(entry) != 2:
                continue
            frame, code = entry
            flags.setdefault(frame, []).append(code)
        return flags

    def _parse_custom_flags(self):
        flags = {}
        for entry in self._meta.get(MarkerEventSections.CUSTOM_FLAGS, []) or []:
            if isinstance(entry, list) and len(entry) == 2:
                frame, name = entry
                flags.setdefault(int(frame), []).append(str(name))
        return flags

    def _parse_io_events(self):
        io_dict = {}
        raw = self._meta.get(MarkerEventSections.IO_EVENTS, [])
        if not raw:
            return io_dict

        for entry in raw:
            if isinstance(entry, list) and len(entry) == 3:
                frame, byte_val, delay = entry
                # Only one per frame; overwrite if duplicated
                io_dict[frame] = [int(byte_val), int(delay)]

        return io_dict
    
    def _parse_led_events(self):
        led_dict = {}
        raw = self._meta.get(MarkerEventSections.LED_EVENTS, [])
        if not raw:
            return led_dict

        for entry in raw:
            # Expected format:
            # [frame, [[led_id, mode, color, value], ...]]
            if not isinstance(entry, list) or len(entry) != 2:
                raise ValueError(f"Invalid LED event entry: {entry}")

            frame, leds = entry
            frame = int(frame)
            parsed = {}
            for led_entry in leds:
                if not isinstance(led_entry, list) or len(led_entry) != 4:
                    raise ValueError(f"Invalid LED payload: {led_entry}")

                led_id, mode, color, value = led_entry
                parsed[int(led_id)] = {
                    "mode": str(mode),
                    "color": str(color),
                    "value": int(value),
                }
            led_dict[frame] = parsed
        return led_dict
    
    def _parse_wp_names(self):
        wp_names = {}
        raw = self._meta.get(MarkerEventSections.FORCED_WAYPOINT_NAMES, {})
        if not raw:
            return wp_names

        for frame_str, name in raw.items():
            try:
                frame = int(frame_str)
                wp_names[frame] = str(name)
            except ValueError:
                continue
        return wp_names

    def _parse_episodes(self):
        """Load and validate named episode ranges from recording metadata."""
        raw = self._meta.get(MDataLabel.METADATA_KEY, {}).get(MDataLabel.EPISODES_KEY, []) or []
        episodes = []
        seen_names = set()
        seen_ids = set()
        valid_logic = EpisodeConfig.LOGIC_CHOICES
        valid_start_waypoint_modes = EpisodeConfig.START_WAYPOINT_CHOICES
        for entry in raw:
            if not isinstance(entry, dict):
                raise ValueError(f"Invalid episode entry: {entry!r}")
            episode_name = str(entry.get(EpisodeConfig.NAME_KEY, entry.get(EpisodeConfig.ID_KEY, ""))).strip()
            episode_id = str(entry.get(EpisodeConfig.ID_KEY, episode_name)).strip()
            if not episode_name:
                raise ValueError("Episode names must not be empty.")
            if not episode_id:
                raise ValueError("Episode IDs must not be empty.")
            folded_name = episode_name.casefold()
            folded_id = episode_id.casefold()
            if folded_name in seen_names:
                raise ValueError(f"Episode name '{episode_name}' is duplicated.")
            if folded_id in seen_ids:
                raise ValueError(f"Episode ID '{episode_id}' is duplicated.")
            seen_names.add(folded_name)
            seen_ids.add(folded_id)
            try:
                start_frame = int(entry[EpisodeConfig.START_FRAME_KEY])
                end_frame = int(entry[EpisodeConfig.END_FRAME_KEY])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Episode '{episode_id}' has invalid frame bounds.") from exc
            if start_frame > end_frame:
                raise ValueError(f"Episode '{episode_id}' starts after it ends.")
            logic = str(entry.get(EpisodeConfig.LOGIC_KEY, EpisodeConfig.DEFAULT_LOGIC)).strip().upper()
            if logic not in valid_logic:
                raise ValueError(f"Episode '{episode_name}' has unsupported logic '{logic}'.")
            repeat_count = int(entry.get(EpisodeConfig.REPEAT_COUNT_KEY, EpisodeConfig.DEFAULT_REPEAT_COUNT))
            retry_count = int(entry.get(EpisodeConfig.RETRY_COUNT_KEY, EpisodeConfig.DEFAULT_RETRY_COUNT))
            delay_msec = int(entry.get(EpisodeConfig.DELAY_MSEC_KEY, EpisodeConfig.DEFAULT_DELAY_MSEC))
            start_with_waypoint = entry.get(EpisodeConfig.START_WITH_WAYPOINT_KEY, EpisodeConfig.DEFAULT_START_WITH_WAYPOINT)
            if isinstance(start_with_waypoint, str):
                start_with_waypoint = start_with_waypoint.strip().lower() in {
                    "1", "true", "yes", "on"
                }
            else:
                start_with_waypoint = bool(start_with_waypoint)
            start_waypoint_mode = str(
                entry.get(EpisodeConfig.START_WAYPOINT_MODE_KEY, EpisodeConfig.DEFAULT_START_WAYPOINT_MODE)
            ).strip().upper()
            if start_waypoint_mode not in valid_start_waypoint_modes:
                raise ValueError(
                    f"Episode '{episode_name}' has unsupported starting waypoint mode "
                    f"'{start_waypoint_mode}'."
                )
            if repeat_count < 1 or retry_count < 1 or delay_msec < 0:
                raise ValueError(f"Episode '{episode_name}' has invalid logic settings.")
            raw_delay_enabled = entry.get(EpisodeConfig.DELAY_ENABLED_KEY, delay_msec > 0)
            if isinstance(raw_delay_enabled, str):
                delay_enabled = raw_delay_enabled.strip().lower() in {"1", "true", "yes", "on"}
            else:
                delay_enabled = bool(raw_delay_enabled)
            episodes.append({
                EpisodeConfig.NAME_KEY: episode_name,
                EpisodeConfig.ID_KEY: episode_id,
                EpisodeConfig.START_FRAME_KEY: start_frame,
                EpisodeConfig.END_FRAME_KEY: end_frame,
                EpisodeConfig.START_WITH_WAYPOINT_KEY: start_with_waypoint,
                EpisodeConfig.START_WAYPOINT_MODE_KEY: start_waypoint_mode,
                EpisodeConfig.LOGIC_KEY: logic,
                EpisodeConfig.REPEAT_COUNT_KEY: repeat_count,
                EpisodeConfig.RETRY_COUNT_KEY: retry_count,
                EpisodeConfig.DELAY_ENABLED_KEY: delay_enabled,
                EpisodeConfig.DELAY_MSEC_KEY: delay_msec,
            })

        # Keep the source/list order for behavior-tree construction. Use a
        # frame-sorted copy only for validating non-overlap.
        frame_ordered = sorted(episodes, key=lambda episode: episode[EpisodeConfig.START_FRAME_KEY])
        for previous, current in zip(frame_ordered, frame_ordered[1:]):
            if current[EpisodeConfig.START_FRAME_KEY] <= previous[EpisodeConfig.END_FRAME_KEY]:
                raise ValueError(
                    f"Episodes '{previous[EpisodeConfig.NAME_KEY]}' and '{current[EpisodeConfig.NAME_KEY]}' overlap."
                )
        return episodes

    def _load_frames(self):
        if self.is_binary:
            raw = np.fromfile(self.bin_path, dtype=np.float32)
            data = raw.reshape((self.frame_count, self.num_items))
        else:
            data = np.array(self.frames, dtype=np.float32)
            assert data.shape == (self.frame_count, self.num_items)
        return data

    def _label_data(self):
        labeled_data = {}
        for idx, label in enumerate(self.labels):
            labeled_data[label] = self.data[:, idx]
        return labeled_data


class JointGroupDefinition:
    def __init__(self, definition_dict: dict):
        self.atomic_groups: dict = definition_dict.get('atomic_groups', {})
        self.move_groups: dict = definition_dict.get('move_groups', {})
        self.group_end_effectors: dict = definition_dict.get('group_end_effectors', {})
    
    def atomic_group_names(self) -> list:
        return list(self.atomic_groups.keys())
    
    def move_group_names(self) -> list:
        return list(self.move_groups.keys())
    
    def get_atomic_group(self, group_name: str) -> list:
        return self.atomic_groups.get(group_name, [])
    
    def get_move_group(self, group_name: str) -> list:
        return self.move_groups.get(group_name, [])


class JointSceneMapping:
    def __init__(self, joint_name: str, paired_with: str):
        self.joint_name = joint_name
        self.scene_joint_name = paired_with
        self.is_inverted = paired_with.startswith("-")
        if self.is_inverted:
            self.scene_joint_name = paired_with[1:]

    def __str__(self):
        return f"JointPair(joint_name={self.joint_name}, paired_with={self.scene_joint_name})"


class KeypointType(Enum):
    MOTION = "motion"
    IO = "io"
    LED = "led"
    FLAG = "flag"


class Keypoint:
    def __init__(self,
                 *,
                 type: KeypointType,
                 frame: int,
                 movegroup: str|None = None,
                 joint_positions: dict|None = None,
                 is_waypoint: bool = False,
                 flag_code: str|None = None
                 ):
        
        self.type = type
        self.frame = frame
        self.movegroup = movegroup
        self.joint_positions = joint_positions or {}
        self.is_waypoint = is_waypoint
        self.flag_code = flag_code
        

    def get_atomic_joint_positions(self, joint_definition: JointGroupDefinition) -> dict:
        if self.type != KeypointType.MOTION or not self.movegroup:
            raise ValueError("Keypoint is not a motion type with a defined movegroup.")
        atomic_positions = {}
        for atomic_group in joint_definition.get_move_group(self.movegroup):
            joint_names = joint_definition.get_atomic_group(atomic_group)
            positions = [
                self.joint_positions[j]
                for j in joint_names if j in self.joint_positions
            ]
            if positions:
                atomic_positions[atomic_group] = positions
        return atomic_positions


def load_mapping_file(map_path) -> dict:
    ext = map_path.lower().split('.')[-1]
    with open(map_path, 'r') as f:
        return yaml.safe_load(f) if ext in ('yaml','yml') else json.load(f)


def load_recording_data(record_filepath: str) -> RecordingData:
    return RecordingData(record_filepath)


def load_joint_group_definitions(definition_path: str) -> JointGroupDefinition:
    definition_dict = load_mapping_file(definition_path)
    return JointGroupDefinition(definition_dict)

def load_joint_scene_mapping(joint_scene_path: str) -> list[JointSceneMapping]:
    joint_scene_dict = load_mapping_file(joint_scene_path)
    joint_scene_map = []
    for joint_name, scene_obj_name in joint_scene_dict.items():
        joint_scene_map.append(JointSceneMapping(joint_name, scene_obj_name))
    return joint_scene_map

def format_io_byte(value: int) -> str:
    """
    Convert an integer (0–255) into an 8-bit binary string.
    Examples:
        0   -> "00000000"
        192 -> "11000000"
        5   -> "00000101"
    """
    # Safety clamp to byte range
    v = max(0, min(255, int(value)))

    return format(v, "08b")


class EnvironmentObject:
    """Parsed and standardized environment asset."""
    def __init__(self, name: str, position: list[float], rotation_euler: list[float], 
                 quaternion: list[float], scale: list[float], mesh_path: str, serialized_pose: str,
                 follow_robot: str = "", follow_link: str = "",
                 collision_links: list[str] | None = None,
                 collision_movegroup: list[str] | None = None, allow_collision: bool = False):
        self.name = name
        self.position = position            # [x, y, z] in METERS
        self.rotation_euler = rotation_euler  # [roll, pitch, yaw] in RADIANS (for PyBullet)
        self.quaternion = quaternion        # [qx, qy, qz, qw]
        self.scale = scale                  # [sx, sy, sz]
        self.mesh_path = mesh_path          # Mesh relative file path
        self.serialized_pose = serialized_pose  # "x,y,z,qx,qy,qz,qw"
        self.follow_robot = follow_robot
        self.follow_link = follow_link
        self.collision_links = collision_links or []
        self.collision_movegroups = collision_movegroup or []
        self.allow_collision = allow_collision

    def __repr__(self):
        return f"<EnvironmentObject name='{self.name}' serialized_pose='{self.serialized_pose}'>"


class EnvironmentData:
    """
    Handles environment JSON loading, schema parsing, unit conversions,
    and rotation format transformations.
    """
    def __init__(self, env_path: str):
        self.env_path = env_path
        self.base_dir = os.path.dirname(os.path.abspath(env_path)) if env_path else ""
        self.units = {
            EnvironmentSchema.LENGTH_UNIT_KEY: MeasurementUnits.METER_UNIT,
            EnvironmentSchema.ANGLE_UNIT_KEY: MeasurementUnits.RADIAN_UNIT,
            EnvironmentSchema.ROTATION_FORMAT_KEY: RotationFormat.XYZ
        }
        self.schema = []
        self.objects: list[EnvironmentObject] = []
        self.collision_rules: list[dict] = []

        if env_path:
            self._load_and_parse()

    def _load_and_parse(self):
        if not os.path.exists(self.env_path):
            logger.warning(f"Environment file {self.env_path} not found")
            return

        try:
            with open(self.env_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load environment data from {self.env_path}: {e}")
            return

        # 1. Parse Units
        raw_units = data.get(EnvironmentSchema.UNITS_KEY, {})
        if isinstance(raw_units, dict):
            self.units[EnvironmentSchema.LENGTH_UNIT_KEY] = str(raw_units.get(EnvironmentSchema.LENGTH_UNIT_KEY, MeasurementUnits.METER_UNIT)).upper()
            self.units[EnvironmentSchema.ANGLE_UNIT_KEY] = str(raw_units.get(EnvironmentSchema.ANGLE_UNIT_KEY, MeasurementUnits.RADIAN_UNIT)).upper()
            self.units[EnvironmentSchema.ROTATION_FORMAT_KEY] = str(
                raw_units.get(EnvironmentSchema.ROTATION_FORMAT_KEY, raw_units.get(EnvironmentSchema.LEGACY_ROTATION_KEY, RotationFormat.XYZ))
            ).upper()

        # 2. Parse Schema
        self.schema = data.get(EnvironmentSchema.SCHEMA_KEY, EnvironmentSchema.COLUMNS)
        
        # Map schema fields to indices dynamically
        idx_map = {key: idx for idx, key in enumerate(self.schema)}
        name_idx = idx_map.get(EnvironmentSchema.NAME, 0)
        loc_idx = idx_map.get(EnvironmentSchema.LOCATION, 1)
        rot_idx = idx_map.get(EnvironmentSchema.ROTATION, 2)
        scale_idx = idx_map.get(EnvironmentSchema.SCALE, 3)
        mesh_idx = idx_map.get(EnvironmentSchema.MESH, 6)
        follow_robot_idx = idx_map.get(EnvironmentSchema.FOLLOW_ROBOT)
        follow_link_idx = idx_map.get(EnvironmentSchema.FOLLOW_LINK)
        collision_links_idx = idx_map.get(EnvironmentSchema.COLLISION_LINKS)
        collision_movegroup_idx = idx_map.get(EnvironmentSchema.COLLISION_MOVEGROUP)
        allow_collision_idx = idx_map.get(EnvironmentSchema.ALLOW_COLLISION)

        def value_at(entry, index, default=""):
            return entry[index] if index is not None and len(entry) > index else default

        def as_string_list(value):
            if isinstance(value, list):
                return [str(item).strip() for item in value if str(item).strip()]
            return [item.strip() for item in str(value or "").split(",") if item.strip()]

        def as_bool(value):
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes", "on"}
            return bool(value)

        # 3. Parse Objects
        raw_env = data.get(EnvironmentSchema.ENVIRONMENT_KEY, [])
        for entry in raw_env:
            try:
                name = entry[name_idx]
                raw_loc = entry[loc_idx]
                raw_rot = entry[rot_idx]
                scale = entry[scale_idx] if len(entry) > scale_idx else [1.0, 1.0, 1.0]
                mesh_path = entry[mesh_idx] if len(entry) > mesh_idx else name
                follow_robot = value_at(entry, follow_robot_idx)
                follow_link = value_at(entry, follow_link_idx)
                collision_links = as_string_list(value_at(entry, collision_links_idx))
                collision_movegroup = as_string_list(value_at(entry, collision_movegroup_idx))
                allow_collision = as_bool(value_at(entry, allow_collision_idx, False))

                # Convert translation to Meters
                position = self._convert_length(raw_loc)

                # Convert rotation convention to standard Quaternion & Euler (radians)
                quaternion, euler_rad = self._convert_rotation(raw_rot)

                # Runtime pose format: 7 comma-separated values without brackets.
                serialized_pose = f"{position[0]},{position[1]},{position[2]},{quaternion[0]},{quaternion[1]},{quaternion[2]},{quaternion[3]}"

                obj = EnvironmentObject(
                    name=name,
                    position=position,
                    rotation_euler=euler_rad,
                    quaternion=quaternion,
                    scale=scale,
                    mesh_path=mesh_path,
                    serialized_pose=serialized_pose,
                    follow_robot=str(follow_robot or ""),
                    follow_link=str(follow_link or ""),
                    collision_links=collision_links,
                    collision_movegroup=collision_movegroup,
                    allow_collision=allow_collision,
                )
                self.objects.append(obj)
            except Exception as e:
                logger.warning(f"Failed to parse environment object entry '{entry}': {e}")

        raw_rules = data.get(EnvironmentSchema.COLLISION_RULES_KEY, [])
        if isinstance(raw_rules, list):
            self.collision_rules = [rule for rule in raw_rules if isinstance(rule, dict)]

    def _convert_length(self, raw_loc: list[float]) -> list[float]:
        """Convert translation coordinates to standard Meters."""
        length_unit = self.units.get(EnvironmentSchema.LENGTH_UNIT_KEY, MeasurementUnits.METER_UNIT)
        if length_unit == MeasurementUnits.MILLIMETER_UNIT:
            return [x * 0.001 for x in raw_loc]
        return [float(x) for x in raw_loc]

    def _convert_rotation(self, raw_rot: list[float]) -> tuple[list[float], list[float]]:
        """
        Converts rotation based on rotation_format and angle_units.
        Returns: (quaternion [qx, qy, qz, qw], euler_xyz [roll, pitch, yaw] in radians)
        """
        rot_fmt = self.units.get(EnvironmentSchema.ROTATION_FORMAT_KEY, RotationFormat.XYZ)
        angle_unit = self.units.get(EnvironmentSchema.ANGLE_UNIT_KEY, MeasurementUnits.RADIAN_UNIT)

        # Parse Euler angles across conventions
        if rot_fmt in [
            RotationFormat.XYZ, RotationFormat.XZY, RotationFormat.YXZ,
            RotationFormat.YZX, RotationFormat.ZXY, RotationFormat.ZYX
        ]:
            angles = [np.radians(v) if angle_unit == MeasurementUnits.DEGREE_UNIT else v for v in raw_rot]
            r = R.from_euler(rot_fmt.lower(), angles)

        elif rot_fmt == RotationFormat.QUATERNION:
            # Assumed [qx, qy, qz, qw]
            r = R.from_quat(raw_rot)

        elif rot_fmt == RotationFormat.AXIS_ANGLE:
            # Assumed [ax, ay, az, angle]
            axis = np.array(raw_rot[:3], dtype=float)
            angle = raw_rot[3]
            if angle_unit == MeasurementUnits.DEGREE_UNIT:
                angle = np.radians(angle)
            norm = np.linalg.norm(axis)
            if norm > 0:
                axis = axis / norm
            r = R.from_rotvec(axis * angle)

        else:
            angles = [np.radians(v) if angle_unit == MeasurementUnits.DEGREE_UNIT else v for v in raw_rot]
            r = R.from_euler('xyz', angles)

        quat = r.as_quat().tolist()            # [qx, qy, qz, qw]
        euler_xyz = r.as_euler('xyz').tolist() # Standard [roll, pitch, yaw] in radians
        return quat, euler_xyz

    # --- Public Accessor Functions ---

    def get_units(self) -> dict:
        """Returns the parsed unit configuration."""
        return self.units

    def get_schema(self) -> list[str]:
        """Returns the schema header list."""
        return self.schema

    def get_objects(self) -> list[EnvironmentObject]:
        """Returns the list of parsed EnvironmentObject instances."""
        return self.objects

    def get_collision_rules(self) -> list[dict]:
        """Return optional robot-to-robot collision rules from the environment file."""
        return self.collision_rules

    def resolve_mesh_path(self, mesh_rel_path: str) -> str:
        """Resolves relative mesh paths against the environment JSON location."""
        if os.path.isabs(mesh_rel_path):
            return mesh_rel_path
        potential_path = os.path.join(self.base_dir, mesh_rel_path)
        if os.path.exists(potential_path):
            return potential_path
        return mesh_rel_path
