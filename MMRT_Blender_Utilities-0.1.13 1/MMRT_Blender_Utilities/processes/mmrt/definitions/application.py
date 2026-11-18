"""Generated application layout, planner choices, and blackboard defaults."""

from pathlib import Path

from .recording import OutputSuffixes
from .pipeline import MoveTomoTrajectoryConfig


class BehaviorTreeModelConfig:
    DEFAULT_MODELS_PATH = Path(__file__).resolve().with_name("behavior_tree_default_models.xml")


class GeneratedAppDirectories:
    TREE_DIR = "trees"
    BLACKBOARD_DIR = "blackboards"
    CONFIG_DIR = "configs"
    MESH_DIR = "meshes"
    ROBOT_MESH_DIR = "robot"


class GeneratedAppFilenames:
    BEHAVIOR_TREE_FILE = f"behavior_tree{OutputSuffixes.BEHAVIOR_TREE_XML}"
    EPISODE_BEHAVIOR_TREE_FILE = f"episode_subtrees{OutputSuffixes.BEHAVIOR_TREE_XML}"
    SYSTEM_BLACKBOARD_FILE = f"system_settings{OutputSuffixes.BLACKBOARD_YAML}"
    MOTION_BLACKBOARD_FILE = f"motion_waypoints{OutputSuffixes.BLACKBOARD_YAML}"
    ENVIRONMENT_BLACKBOARD_FILE = f"environment_settings{OutputSuffixes.BLACKBOARD_YAML}"
    EPISODE_SETTINGS_BLACKBOARD_FILE = f"episode_settings{OutputSuffixes.BLACKBOARD_YAML}"
    APP_CONFIG_FILE = f"config{OutputSuffixes.APP_CONFIG_YAML}"
    ADDON_TUNING_METADATA_FILE = "blender_tuning.json"


class MMRTPlannerConfig:
    """Planning pipeline and planner IDs supported by the MMRT UI."""

    PILZ = "pilz"
    OMPL = "ompl"
    PIPELINES = (PILZ, OMPL)
    PILZ_PLANNER_IDS = ("PTP", "CIRC", "LIN")
    OMPL_PLANNER_IDS = (
        "SBLkConfigDefault", "ESTkConfigDefault", "LBKPIECEkConfigDefault",
        "BKPIECEkConfigDefault", "KPIECEkConfigDefault", "RRTkConfigDefault",
        "RRTConnectkConfigDefault", "RRTstarkConfigDefault", "TRRTkConfigDefault",
        "PRMkConfigDefault", "PRMstarkConfigDefault", "FMTkConfigDefault",
        "BFMTkConfigDefault", "PDSTkConfigDefault", "STRIDEkConfigDefault",
        "BiTRRTkConfigDefault", "LBTRRTkConfigDefault", "BiESTkConfigDefault",
        "ProjESTkConfigDefault", "LazyPRMkConfigDefault", "LazyPRMstarkConfigDefault",
        "SPARSkConfigDefault", "SPARStwokConfigDefault",
    )
    DEFAULT_PIPELINE = PILZ
    DEFAULT_PLANNER_ID = PILZ_PLANNER_IDS[0]

    @classmethod
    def planner_ids(cls, pipeline):
        if pipeline == cls.PILZ:
            return cls.PILZ_PLANNER_IDS
        if pipeline == cls.OMPL:
            return cls.OMPL_PLANNER_IDS
        return ()


class BlackboardDefaults:
    """Default values and generated keys for deployable MMRT blackboards."""

    PLAN_PIPELINE_KEY = "plan_pipeline"
    PLAN_ID_KEY = "plan_id"
    VELOCITY_SCALE_KEY = "global_vel_scale"
    ACCELERATION_SCALE_KEY = "global_acc_scale"
    BLEND_RADIUS_KEY = "global_blending_radius"
    COLLISION_CHECKS_KEY = "global_collision_checks"
    SAVE_PLAN_KEY = "global_save_plan"
    ENABLE_LOG_KEY = "enable_log"
    ENABLE_LOG_TO_FILE_KEY = "enable_log_to_file"
    DEBUG_TOPIC_NAME_KEY = "debug_topic_name"
    TRAJECTORY_STRATEGY_KEY = "trajectory_strategy"
    TRAJECTORY_RESAMPLE_KEY = "trajectory_resample"
    TRAJECTORY_TIME_PARAMETERIZATION_KEY = "trajectory_time_parameterization"
    TRAJECTORY_VISUALIZE_KEY = "trajectory_visualize"
    TRAJECTORY_RESOLUTION_KEY = "trajectory_resolution"

    VELOCITY_SCALE = 1.0
    ACCELERATION_SCALE = 1.0
    COLLISION_CHECKS = 0
    BLEND_RADIUS = 0.05
    SAVE_PLAN = False
    ENABLE_LOG = True
    ENABLE_LOG_TO_FILE = True
    DEBUG_TOPIC_NAME = "BTLog"

    SYSTEM_SETTINGS = {
        VELOCITY_SCALE_KEY: VELOCITY_SCALE,
        ACCELERATION_SCALE_KEY: ACCELERATION_SCALE,
        COLLISION_CHECKS_KEY: COLLISION_CHECKS,
        BLEND_RADIUS_KEY: BLEND_RADIUS,
        SAVE_PLAN_KEY: str(SAVE_PLAN).lower(),
        ENABLE_LOG_KEY: ENABLE_LOG,
        ENABLE_LOG_TO_FILE_KEY: ENABLE_LOG_TO_FILE,
        DEBUG_TOPIC_NAME_KEY: DEBUG_TOPIC_NAME,
        TRAJECTORY_RESOLUTION_KEY: MoveTomoTrajectoryConfig.DEFAULT_RESOLUTION,
        TRAJECTORY_STRATEGY_KEY: MoveTomoTrajectoryConfig.DEFAULT_STRATEGY,
        TRAJECTORY_RESAMPLE_KEY: MoveTomoTrajectoryConfig.DEFAULT_RESAMPLE,
        TRAJECTORY_TIME_PARAMETERIZATION_KEY: MoveTomoTrajectoryConfig.DEFAULT_TIME_PARAMETERIZATION,
        TRAJECTORY_VISUALIZE_KEY: str(MoveTomoTrajectoryConfig.DEFAULT_VISUALIZE).lower(),
        PLAN_PIPELINE_KEY: MMRTPlannerConfig.DEFAULT_PIPELINE,
        PLAN_ID_KEY: MMRTPlannerConfig.DEFAULT_PLANNER_ID,
    }

    @staticmethod
    def reference(key):
        return f"{{{key}}}"

    @classmethod
    def system_settings(cls, velocity_scale=None, acceleration_scale=None, blend_radius=None,
                        collision_checks=None, save_plan=None, plan_pipeline=None, plan_id=None,
                        enable_log=None, enable_log_to_file=None, debug_topic_name=None,
                        trajectory_strategy=None, trajectory_resample=None,
                        trajectory_time_parameterization=None, trajectory_visualize=None,
                        trajectory_resolution=None):
        """Apply explicit overrides to the shared defaults; None keeps the default."""
        values = dict(cls.SYSTEM_SETTINGS)
        for key, value in {
            cls.VELOCITY_SCALE_KEY: velocity_scale,
            cls.ACCELERATION_SCALE_KEY: acceleration_scale,
            cls.BLEND_RADIUS_KEY: blend_radius,
            cls.COLLISION_CHECKS_KEY: collision_checks,
            cls.SAVE_PLAN_KEY: str(save_plan).lower() if save_plan is not None else None,
            cls.PLAN_PIPELINE_KEY: plan_pipeline,
            cls.PLAN_ID_KEY: plan_id,
            cls.ENABLE_LOG_KEY: enable_log,
            cls.ENABLE_LOG_TO_FILE_KEY: enable_log_to_file,
            cls.DEBUG_TOPIC_NAME_KEY: debug_topic_name,
            cls.TRAJECTORY_RESOLUTION_KEY: trajectory_resolution,
            cls.TRAJECTORY_STRATEGY_KEY: trajectory_strategy,
            cls.TRAJECTORY_RESAMPLE_KEY: trajectory_resample,
            cls.TRAJECTORY_TIME_PARAMETERIZATION_KEY: trajectory_time_parameterization,
            cls.TRAJECTORY_VISUALIZE_KEY: (
                str(trajectory_visualize).lower() if trajectory_visualize is not None else None
            ),
        }.items():
            if value is not None:
                values[key] = value
        return values

    @staticmethod
    def environment_object_values(objects):
        values = {}
        for index, obj in enumerate(objects):
            transform = obj["transform"]
            pose = transform["position"] + transform["rotation"]
            values.update({
                f"env_obj_{index}_name": obj["name"],
                f"env_obj_{index}_mesh": obj["mesh"],
                f"env_obj_{index}_pose": ",".join(str(value) for value in pose),
            })
        return values

    @staticmethod
    def environment_object_key(index, field):
        return f"{{env_obj_{index}_{field}}}"
