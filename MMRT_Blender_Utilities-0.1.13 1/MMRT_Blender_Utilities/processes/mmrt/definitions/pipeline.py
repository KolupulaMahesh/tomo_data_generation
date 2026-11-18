"""Defaults and supported options for MMRT behavior-tree generation."""

from .recording import WaypointMergeMethod


class MMRTProcessDefaults:
    MOTION_SPEED_THRESHOLD_RAD_PER_SEC = 0.005
    MINIMUM_SEGMENT_DURATION_SEC = 0.05
    SEGMENT_MERGE_GAP_SEC = 0.1
    BEHAVIOR_TREE_SLEEP_GAP_MS = 1000
    WAYPOINT_MERGE_WINDOW = 1
    WAYPOINT_MERGE_ITERATIONS = 1
    WAYPOINT_MERGE_METHOD = WaypointMergeMethod.FIRST
    COLLISION_DISTANCE_THRESHOLD_M = 0.001

    TREE_ID = "MainTree"
    CONFIGURE_TREE_NAME = "Main_LoadConfig"
    SCENE_SETUP_TREE_NAME = "Main_SceneSetup"
    MOVE_SEGMENT_TREE_NAME = "Main_StartSequence"


class MoveTomoTrajectoryConfig:
    """MoveTomoTrajectory port choices and defaults."""

    DEFAULT_VISUALIZE = False
    DEFAULT_RESOLUTION = 0.01  # radians

    STRATEGY_NONE = "none"
    STRATEGY_BASIC = "basic"
    STRATEGY_QUINTIC = "quintic"
    STRATEGIES = (STRATEGY_NONE, STRATEGY_BASIC, STRATEGY_QUINTIC)
    DEFAULT_STRATEGY = STRATEGY_BASIC

    RESAMPLE_NONE = "none"
    RESAMPLE_DISTANCE = "distance"
    RESAMPLE_METHODS = (RESAMPLE_NONE, RESAMPLE_DISTANCE)
    DEFAULT_RESAMPLE = RESAMPLE_DISTANCE

    TIME_PARAMETERIZATION_NONE = "none"
    TIME_PARAMETERIZATION_OPTIMAL = "optimal"
    TIME_PARAMETERIZATION_PARABOLIC = "parabolic"
    TIME_PARAMETERIZATION_SPLINE = "spline"
    TIME_PARAMETERIZATIONS = (
        TIME_PARAMETERIZATION_NONE,
        TIME_PARAMETERIZATION_OPTIMAL,
        TIME_PARAMETERIZATION_PARABOLIC,
        TIME_PARAMETERIZATION_SPLINE,
    )
    DEFAULT_TIME_PARAMETERIZATION = TIME_PARAMETERIZATION_OPTIMAL


class PipelineCLIOptions:
    """Command-line contract between the Blender launcher and MMRT process."""

    RECORD = "--record"
    PROGRESS_FILE = "--progress_file"
    URDF = "--urdf"
    BEHAVIOR_TREE_MODELS_PATH = "--behavior_tree_models_path"
    OUTPUT = "--output"
    APP_DIR = "--app_dir"
    TREE_ID = "--tree_id"
    CONFIGURE_TREE_NAME = "--configure_tree_name"
    SCENE_SETUP_TREE_NAME = "--scene_setup_tree_name"
    MOVE_SEGMENT_TREE_NAME = "--move_segment_tree_name"
    SRDF = "--srdf"
    ENV = "--env"
    ROBOT_NAME = "--robot_name"
    MMRT_SPEED_THR = "--mmrt_speed_thr"
    MMRT_MIN_SEGMENT_TIME = "--mmrt_min_segment_time"
    MMRT_SEGMENT_COMBINE_THR = "--mmrt_segment_combine_thr"
    MMRT_SLEEP_THR = "--mmrt_sleep_thr"
    MMRT_WP_DECIMATION_WINDOW = "--mmrt_wp_decimation_window"
    MMRT_WP_DECIMATION_ITER = "--mmrt_wp_decimation_iter"
    MMRT_WP_DECIMATION_METHOD = "--mmrt_wp_decimation_method"
    MMRT_COLLISION_THR = "--mmrt_collision_thr"
    BT_VEL_SCALE = "--bt_vel_scale"
    BT_ACC_SCALE = "--bt_acc_scale"
    BT_COLLISION_CHECKS = "--bt_collision_checks"
    BT_BLEND_RADIUS = "--bt_blend_radius"
    BT_SAVE_PLAN = "--bt_save_plan"
    BT_PLAN_PIPELINE = "--bt_plan_pipeline"
    BT_PLAN_ID = "--bt_plan_id"
    BT_ENABLE_LOG = "--bt_enable_log"
    BT_ENABLE_LOG_TO_FILE = "--bt_enable_log_to_file"
    BT_DEBUG_TOPIC_NAME = "--bt_debug_topic_name"
    BT_TRAJECTORY_STRATEGY = "--bt_trajectory_strategy"
    BT_TRAJECTORY_RESAMPLE = "--bt_trajectory_resample"
    BT_TRAJECTORY_TIME_PARAMETERIZATION = "--bt_trajectory_time_parameterization"
    BT_TRAJECTORY_VISUALIZE = "--bt_trajectory_visualize"
    BT_TRAJECTORY_RESOLUTION = "--bt_trajectory_resolution"
    SIM = "--sim"
    PLOT = "--plot"
    SAVE_PLOT = "--save_plot"
    NO_EXPORT = "--no_export"
    VERBOSITY = "--verbosity"
    MMRT_SHOW_CONSOLE = "--mmrt_show_console"


class LoggingVerbosity:
    SILENT = "silent"
    NORMAL = "normal"
    DEBUG = "debug"
    VERBOSE = "verbose"
    CHOICES = (SILENT, NORMAL, DEBUG, VERBOSE)
    DEFAULT = NORMAL
