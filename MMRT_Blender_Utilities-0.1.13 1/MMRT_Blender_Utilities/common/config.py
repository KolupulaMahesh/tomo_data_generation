from .system.paths import current_username
from .robot_server_config import RobotServerConfig
from ..processes.mmrt.definitions import (
    MeasurementUnits,
    MMRTProcessDefaults,
)


class ObjectMode:
    OBJECT = 'OBJECT'
    EDIT = 'EDIT'
    POSE = 'POSE'


class ObjectParentType:
    OBJECT = ObjectMode.OBJECT

class EnvironmentExportMode:
    AUTO = "AUTO"
    MANUAL = "MANUAL"
    OFF = "OFF"


class ObjectOriginMode:
    WORLD: str = "WORLD"
    ORIGINAL: str = "ORIGINAL"
    CUSTOM: str = "CUSTOM"

    @classmethod
    def items(cls):
        return [
            (cls.WORLD, "World", "Bake object with world origin (resets transforms)"),
            (cls.ORIGINAL, "Original", "Bake object with its original origin (applies transforms)"),
            (cls.CUSTOM, "Custom", "Bake relative to another reference object"),
        ]

class RecordTransformSpace:
    WORLD: str = "WORLD"
    LOCAL: str = "LOCAL"
    CUSTOM: str = "CUSTOM"
    
    @classmethod
    def items(cls):
        return [
            (cls.WORLD, "World", "Record location/rotation in world space"),
            (cls.LOCAL, "Local", "Record location/rotation in object local space"),
            (cls.CUSTOM, "Custom", "Record location/rotation relative to reference object"),
        ]


class RecorderOperatorOptions:
    TRANSFORM_ENV = "ENV"
    TRANSFORM_RECORD = "RECORD"
    FRAME_START = "START"
    FRAME_END = "END"
    MOVE_UP = "UP"
    MOVE_DOWN = "DOWN"
    TARGET_ACTIVE = "ACTIVE"
    TARGET_BATCH = "BATCH"
    TARGET_ALL = "ALL"


class SynthesisRenderOutput:
    MP4 = "MP4"
    BLEND = "BLEND"


class SceneRecognitionConfig:
    """Type-based scene discovery with no robot-name assumptions."""

    ROBOT_TYPE = "ARMATURE"
    MESH_TYPE = "MESH"
    CURVE_TYPE = "CURVE"
    ROBOT_TYPES = {ROBOT_TYPE}
    ENV_TYPES = {MESH_TYPE, CURVE_TYPE}


class AddonDefaults:
    # The released add-on uses the bundled PyInstaller backend. An external
    # Python path is kept as an explicit developer-only option.
    DEVELOPMENT_PYTHON_PATH: str = ""
    USE_DEVELOPMENT_PYTHON: bool = False
    GROOT2_PATH: str = ""

    XACRO_PATH: str = ""
    IS_IMPORT_URDF_MESHES: bool = True

    IS_INCLUDE_HIDDEN_OBJECTS: bool = False
    IS_EXPORT_RECORD_AS_BINARY: bool = False
    IS_RECORD_IN_FAST_MODE: bool = True
    IS_EXPORT_ENVIRONMENT: bool = True
    ENV_EXPORT_MODE: str = EnvironmentExportMode.AUTO
    ANGLE_UNIT: str = MeasurementUnits.RADIAN_UNIT
    LENGTH_UNIT: str = MeasurementUnits.METER_UNIT
    RECORD_OUTPUT_DIR: str = "//output/"
    RECORD_OUTPUT_BASENAME: str = "recording_data"

    IS_SAVE_PLOT: bool = False
    IS_USING_RECORDER_OUTPUT: bool = True
    IS_SHOW_BACKEND_SIMULATION: bool = False
    IS_SHOW_PLOT: bool = True
    SPEED_THRESHOLD: float = MMRTProcessDefaults.MOTION_SPEED_THRESHOLD_RAD_PER_SEC
    MINIMUM_SEGMENT_DURATION_S: float = MMRTProcessDefaults.MINIMUM_SEGMENT_DURATION_SEC
    SEGMENT_COMBINE_THRESHOLD_S: float = MMRTProcessDefaults.SEGMENT_MERGE_GAP_SEC
    BT_SLEEP_THRESHOLD_MS: int = MMRTProcessDefaults.BEHAVIOR_TREE_SLEEP_GAP_MS
    WAYPOINT_DECIMATION_FACTOR: int = MMRTProcessDefaults.WAYPOINT_MERGE_WINDOW
    WAYPOINT_DECIMATION_ITERATIONS: int = MMRTProcessDefaults.WAYPOINT_MERGE_ITERATIONS
    WAYPOINT_DECIMATION_METHOD: str = MMRTProcessDefaults.WAYPOINT_MERGE_METHOD
    COLLISION_CHECK_THRESHOLD: float = MMRTProcessDefaults.COLLISION_DISTANCE_THRESHOLD_M

    MMRT_OUTPUT_APP_DIR: str = f"{RECORD_OUTPUT_DIR}untitled_app/"
    MMRT_APP_FILENAME: str = "app"

    # Blender property defaults use the shared process definitions.
    CONFIGURE_TREE_NAME: str = MMRTProcessDefaults.CONFIGURE_TREE_NAME
    SCENE_SETUP_TREE_NAME: str = MMRTProcessDefaults.SCENE_SETUP_TREE_NAME
    MOVE_SEGMENT_TREE_NAME: str = MMRTProcessDefaults.MOVE_SEGMENT_TREE_NAME

    IS_USING_MMRT_OUTPUT: bool = True
    ROBOT_HOST_IP: str = RobotServerConfig.DEFAULT_REMOTE_HOST
    ROBOT_HOST_PORT: int = RobotServerConfig.ROSBRIDGE_DEFAULT_PORT
    ROBOT_HOST_USERNAME: str = current_username()
    ROBOT_HOST_APP_DIR: str = RobotServerConfig.DEFAULT_REMOTE_APP_DIR


class UIConfig:
    ADDON_CATEGORY = "MMRT Utilities"
    ADDON_NAMESPACE = "mmrt"
    ADDON_SPACE = "VIEW_3D"
    ADDON_REGION = "UI"

    PANEL_GLOBAL_LABEL = "Configuration"
    PANEL_IMPORTER_LABEL = "URDF Importer"
    PANEL_MMRT_LABEL = "MMRT"
    PANEL_RECORDER_LABEL = "Scene Recorder"
    PANEL_CONTROL_LABEL = "Remote Controller"
    PANEL_SYNTHESIZER_LABEL = "Synthesizer"

    # Number of remote log entries retained by the Remote Controller.
    REMOTE_LOG_COUNT = 100

    REC_OBJ_LIST_LABEL = "Recorded Objects"
    REC_ENV_LIST_LABEL = "Environment Objects"
    REC_MARKER_LIST_LABEL = "Markers"


class URDFImportConfig:
    BONE_LENGTH: float = 0.05
    CAMERA_LINK_KEYWORD: str = "camera"
