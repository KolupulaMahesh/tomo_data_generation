"""Recording and tuning metadata values shared across the process boundary."""

from enum import Enum


class OutputSuffixes:
    BINARY_RECORDING = ".mmrt.bin"
    YAML_RECORDING = ".mmrt.yaml"
    ENVIRONMENT_JSON = ".env.json"
    URDF = ".urdf"
    SRDF = ".srdf"
    BEHAVIOR_TREE_XML = ".bt.xml"
    BLACKBOARD_YAML = ".btbb.yaml"
    APP_CONFIG_YAML = ".apcfg.yaml"


class MDataLabel:
    """Recording and Blender tuning sidecar schema identifiers."""

    SCHEMA_VERSION_KEY = "schema_version"
    SCHEMA_VERSION = 1
    METADATA_KEY = "metadata"
    ADDON_FRAME_TRACKING_KEY = "addon_frame_tracking"
    RECORDING_KEY = "recording"
    WAYPOINTS_KEY = "waypoints"
    TRAJECTORIES_KEY = "trajectories"
    MOVEGROUP_KEY = "movegroup"
    PLANGROUP_KEY = "plangroup"
    SAMPLE_FRAMES_KEY = "sample_frames"
    JOINT_KEYS_KEY = "joint_keys"
    TRACKING_ERROR_KEY = "tracking_error"
    SCENE_FRAME_START_KEY = "scene_frame_start"
    SCENE_FRAME_END_KEY = "scene_frame_end"
    SAMPLE_FRAME_START_KEY = "sample_frame_start"
    SAMPLE_FRAME_END_KEY = "sample_frame_end"
    SAMPLE_FRAME_DOMAIN_KEY = "sample_frame_domain"
    SAMPLE_FRAME_DOMAIN_SCENE_FRAME_SET = "scene_frame_set"
    FRAME_MAP_OLD_KEY = "frame_map_old"
    FRAME_MAP_NEW_KEY = "frame_map_new"
    SOURCE_START_FRAME_KEY = "source_start_frame"
    SOURCE_END_FRAME_KEY = "source_end_frame"
    START_FRAME_KEY = "start_frame"
    END_FRAME_KEY = "end_frame"
    FRAME_START_KEY = "frame_start"
    FRAME_END_KEY = "frame_end"
    FRAME_COUNT_KEY = "frame_count"
    FRAME_RATE_KEY = "frame_rate"
    TOTAL_FRAMES_KEY = "total_frames"
    FRAME_STRETCH_FACTOR_KEY = "frame_stretch_factor"
    FPS_KEY = "fps"
    ANGLE_UNITS_KEY = "angle_units"
    LOCATION_UNITS_KEY = "location_units"
    LABELS_KEY = "labels"
    ROBOTS_KEY = "robots"
    ACTIVE_OBJECTS_KEY = "active_objects"
    EPISODES_KEY = "episodes"
    ROBOT_NAME_KEY = "robot_name"
    ROTATION_FORMAT_KEY = "rotation_format"
    URDF_FILE_KEY = "urdf_file"
    SRDF_FILE_KEY = "srdf_file"
    ENVIRONMENT_FILE_KEY = "environment_file"


class MeasurementUnits:
    METER_UNIT = "METERS"
    MILLIMETER_UNIT = "MILLIMETERS"
    DEGREE_UNIT = "DEGREES"
    RADIAN_UNIT = "RADIANS"

    METER_DECIMAL_PLACES = 4
    MILLIMETER_DECIMAL_PLACES = 2
    RAD_DECIMAL_PLACES = 6
    DEG_DECIMAL_PLACES = 4


class WaypointMergeMethod:
    MEAN = "MEAN"
    MEDIAN = "MEDIAN"
    FIRST = "FIRST"
    CHOICES = (MEAN, MEDIAN, FIRST)


class EpisodeConfig:
    NAME_KEY = "name"
    ID_KEY = "id"
    START_FRAME_KEY = MDataLabel.START_FRAME_KEY
    END_FRAME_KEY = MDataLabel.END_FRAME_KEY
    START_WITH_WAYPOINT_KEY = "start_with_waypoint"
    START_WAYPOINT_MODE_KEY = "start_waypoint_mode"
    LOGIC_KEY = "logic"
    REPEAT_COUNT_KEY = "repeat_count"
    RETRY_COUNT_KEY = "retry_count"
    DELAY_ENABLED_KEY = "delay_enabled"
    DELAY_MSEC_KEY = "delay_msec"

    LOGIC_NONE = "NONE"
    LOGIC_REPEAT = "REPEAT"
    LOGIC_RETRY_UNTIL_SUCCESSFUL = "RETRY_UNTIL_SUCCESSFUL"
    LOGIC_KEEP_RUNNING_UNTIL_FAILURE = "KEEP_RUNNING_UNTIL_FAILURE"
    LOGIC_CHOICES = (
        LOGIC_NONE,
        LOGIC_REPEAT,
        LOGIC_RETRY_UNTIL_SUCCESSFUL,
        LOGIC_KEEP_RUNNING_UNTIL_FAILURE,
    )
    DEFAULT_LOGIC = LOGIC_NONE

    START_WAYPOINT_FIRST = "FIRST"
    START_WAYPOINT_LAST = "LAST"
    START_WAYPOINT_CHOICES = (START_WAYPOINT_FIRST, START_WAYPOINT_LAST)
    DEFAULT_START_WAYPOINT_MODE = START_WAYPOINT_FIRST

    DEFAULT_START_WITH_WAYPOINT = False
    DEFAULT_REPEAT_COUNT = 1
    DEFAULT_RETRY_COUNT = 1
    DEFAULT_DELAY_ENABLED = False
    DEFAULT_DELAY_MSEC = 0


class RotationFormat:
    QUATERNION = "QUATERNION"
    AXIS_ANGLE = "AXIS_ANGLE"
    XYZ = "XYZ"
    XZY = "XZY"
    YXZ = "YXZ"
    YZX = "YZX"
    ZXY = "ZXY"
    ZYX = "ZYX"

    @classmethod
    def items(cls):
        return [
            (cls.XYZ, "XYZ Euler", "Record rotation as XYZ Euler angles (3 values)"),
            (cls.XZY, "XZY Euler", "Record rotation as XZY Euler angles (3 values)"),
            (cls.YXZ, "YXZ Euler", "Record rotation as YXZ Euler angles (3 values)"),
            (cls.YZX, "YZX Euler", "Record rotation as YZX Euler angles (3 values)"),
            (cls.ZXY, "ZXY Euler", "Record rotation as ZXY Euler angles (3 values)"),
            (cls.ZYX, "ZYX Euler", "Record rotation as ZYX Euler angles (3 values)"),
            (cls.QUATERNION, "Quaternion", "Record rotation as quaternion (4 values)"),
            (cls.AXIS_ANGLE, "Axis Angle", "Record rotation as axis angle (4 values)"),
        ]


class FlagTypes(Enum):
    CAMERA_CAPTURE = "CCP"
    CAMERA_RECORD_START = "CRS"
    CAMERA_RECORD_STOP = "CRP"
    IO_CONTROL = "IOC"
    LED_CONTROL = "LDC"
    RETRY_TRIGGER = "RTG"
    RETRY_EVENT_START = "RSE"
    RETRY_EVENT_END = "REE"
    FORCED_SPLIT = "FSS"
    FORCED_WP_NAME = "FWN"
    CUSTOM = "CUS"

    @classmethod
    def from_name(cls, name):
        try:
            return cls[name].value
        except KeyError:
            return name[:3].upper()

    @classmethod
    def parse_marker_name(cls, marker_name):
        """Return ``(flag_type, custom_flag_name, uid)`` for a marker name."""
        name_stripped = marker_name.strip()
        parts = name_stripped.rsplit("_", 1)
        is_valid_uid = (
            len(parts) > 1
            and len(parts[1]) == 6
            and all(c in "0123456789abcdefABCDEF" for c in parts[1])
        )
        if is_valid_uid:
            base_name = parts[0].upper()
            original_case_name = parts[0]
            uid = parts[1]
        else:
            base_name = name_stripped.upper()
            original_case_name = name_stripped
            uid = ""
        if base_name in cls.__members__:
            return base_name, "", uid
        for member in cls:
            if member.value == base_name:
                return member.name, "", uid
        return cls.CUSTOM.name, original_case_name, uid

    @classmethod
    def canonical_name(cls, marker_name):
        flag_type, custom_name, _ = cls.parse_marker_name(marker_name)
        return custom_name if flag_type == cls.CUSTOM.name else flag_type

    @classmethod
    def is_builtin(cls, name):
        return name in cls.__members__


class MarkerEventSections:
    FLAGS = "flags"
    CUSTOM_FLAGS = "custom_flags"
    IO_EVENTS = "io_events"
    LED_EVENTS = "led_events"
    FORCED_WAYPOINT_NAMES = "forced_wp_names"


class LEDModes(Enum):
    SET = "SET"
    FADE = "FADE"
    SPIN = "SPIN"
    RANDOM = "RANDOM"
    PROGRESS = "PROGRESS"
    HEARTBEAT = "HEARTBEAT"
    BPM = "BPM"

    @classmethod
    def items(cls):
        return [(mode.name, mode.name.title(), "") for mode in cls]


class LEDColors(Enum):
    OFF = "OFF"
    RED = "RED"
    GREEN = "GREEN"
    BLUE = "BLUE"
    YELLOW = "YELLOW"
    CYAN = "CYAN"
    MAGENTA = "MAGENTA"
    WHITE = "WHITE"
    RAINBOW = "RAINBOW"
    CUSTOM = "CUSTOM"

    @classmethod
    def items(cls):
        return [(color.name, color.name.title(), "") for color in cls]
