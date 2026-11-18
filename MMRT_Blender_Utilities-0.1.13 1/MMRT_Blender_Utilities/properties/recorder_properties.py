import bpy
import xml.etree.ElementTree as ET
from ..common.config import (
    EnvironmentExportMode,
    ObjectOriginMode,
    RecordTransformSpace,
    AddonDefaults,
)
from ..processes.mmrt.definitions import EpisodeConfig, MeasurementUnits as MeasurementConstants, RobotDescriptionKind, RotationFormat
from ..common.marker_utils import FlagTypes, LEDColors, LEDModes


def _selected_robot_description(self, context, kind):
    if context is None or not hasattr(context.scene, "mmrt_recorder"):
        return ""
    selected_robot = self.follow_robot
    if not selected_robot:
        return ""
    recorder = context.scene.mmrt_recorder
    for robot in recorder.robots:
        if robot.obj and robot.obj.name == selected_robot:
            text_block = robot.urdf_text if kind == RobotDescriptionKind.URDF else robot.srdf_text
            return text_block.as_string() if text_block else ""
    return ""


def _robot_items(self, context):
    if context is None or not hasattr(context.scene, "mmrt_recorder"):
        return [("", "None", "No recorder robots available")]
    items = [
        (robot.obj.name, robot.obj.name, "Use this recorder robot")
        for robot in context.scene.mmrt_recorder.robots
        if robot.obj
    ]
    return items or [("", "None", "No recorder robots available")]


def _robot_link_items(self, context):
    try:
        root = ET.fromstring(_selected_robot_description(self, context, RobotDescriptionKind.URDF))
        names = sorted({link.get("name") for link in root.findall("link") if link.get("name")})
    except (ET.ParseError, TypeError, ValueError):
        names = []
    return [(name, name, f"Robot link: {name}") for name in names] or [("", "None", "No robot links available")]


def _robot_movegroup_items(self, context):
    try:
        root = ET.fromstring(_selected_robot_description(self, context, RobotDescriptionKind.SRDF))
        names = sorted({group.get("name") for group in root.findall("group") if group.get("name")})
    except (ET.ParseError, TypeError, ValueError):
        names = []
    return [(name, name, f"SRDF movegroup: {name}") for name in names] or [("", "None", "No SRDF movegroups available")]


def _robot_collision_link_items(self, context):
    return _robot_link_items(self, context)


def _reset_robot_dependent_selections(self, context):
    self.follow_link = ""
    self.collision_link_choice = ""
    self.collision_movegroup_choice = ""
    self.collision_links.clear()
    self.collision_links_index = -1
    self.collision_movegroups.clear()
    self.collision_movegroups_index = -1

# --------------------------------------------------------
#   Per-Bone Settings (PropertyGroup)
# --------------------------------------------------------
class RecordedBoneItem(bpy.types.PropertyGroup):
    name: bpy.props.StringProperty(
        name="Bone Name",
        description="Name of the bone in the Armature",
    )
    record: bpy.props.BoolProperty(
        name="Rec",
        description="Include this bone in the YAML output",
        default=False
    )
    record_location: bpy.props.BoolProperty(name="Location", default=True)
    record_rotation: bpy.props.BoolProperty(name="Rotation", default=True)


class RobotJointItem(bpy.types.PropertyGroup):
    joint_name: bpy.props.StringProperty(name="URDF Joint")
    bone_name: bpy.props.StringProperty(name="Bone")
    record: bpy.props.BoolProperty(name="Record", default=True)
    rot_axis: bpy.props.EnumProperty(
        name="Axis", items=[("X", "X", ""), ("Y", "Y", ""), ("Z", "Z", "")], default="X"
    )
    direction: bpy.props.BoolProperty(name="Invert", default=False)
    offset: bpy.props.FloatProperty(name="Offset", default=0.0, precision=3, subtype='ANGLE', unit='ROTATION')
    initial_quat: bpy.props.FloatVectorProperty(
        name="Initial Quaternion", size=4, subtype=RotationFormat.QUATERNION, default=(1.0, 0.0, 0.0, 0.0)
    )


# --------------------------------------------------------
#   Per-Object / Per-Armature Item (PropertyGroup)
# --------------------------------------------------------
class RecordedObjectItem(bpy.types.PropertyGroup):
    obj: bpy.props.PointerProperty(
        name="Object",
        description="Choose any Blender object to record from",
        type=bpy.types.Object,
    )

    joints: bpy.props.CollectionProperty(
        type=RecordedBoneItem,
        name="Joint Items",
    )
    joints_index: bpy.props.IntProperty(default=0)

    show_joints: bpy.props.BoolProperty(
        name="Show Details",
        description="Toggle display of this object's settings",
        default=True
    )

    # Transform-recording booleans (for ANY object):
    record_location: bpy.props.BoolProperty(
        name="Record Location",
        description="Include this object’s location in the YAML output",
        default=False
    )
    record_rotation: bpy.props.BoolProperty(
        name="Record Rotation",
        description="Include this object’s rotation in the YAML output",
        default=False
    )
    space: bpy.props.EnumProperty(
        name="Space",
        description="Use World or Local space when recording location/rotation",
        items=RecordTransformSpace.items(),
        default=RecordTransformSpace.LOCAL
    )

    custom_ref: bpy.props.PointerProperty(
        name="Reference Object",
        description="Use this object as the reference frame for recording location/rotation",
        type=bpy.types.Object
    )


class RobotItem(bpy.types.PropertyGroup):
    obj: bpy.props.PointerProperty(name="Robot Armature", type=bpy.types.Object)
    urdf_text: bpy.props.PointerProperty(name="URDF", type=bpy.types.Text)
    srdf_text: bpy.props.PointerProperty(name="SRDF", type=bpy.types.Text)
    meshes_collection: bpy.props.PointerProperty(
        name="Original Meshes",
        description="Imported source meshes archived for portable recording export",
        type=bpy.types.Collection,
    )
    joints: bpy.props.CollectionProperty(type=RobotJointItem, name="Robot Joints")
    joints_index: bpy.props.IntProperty(default=0)
    show_joints: bpy.props.BoolProperty(name="Show Details", default=True)


class EnvironmentRuleItem(bpy.types.PropertyGroup):
    name: bpy.props.StringProperty(name="Rule Name")


# --------------------------------------------------------
#   Per-Object Environment Item (PropertyGroup)
class EnvironmentObjectItem(bpy.types.PropertyGroup):
    obj: bpy.props.PointerProperty(
        name="Environment Object",
        description="Choose any Blender object to be included as environment",
        type=bpy.types.Object,
    )

    # UI toggle
    show_settings: bpy.props.BoolProperty(
        name="Show Settings",
        default=True
    )

    origin_mode: bpy.props.EnumProperty(
        name="Export Origin",
        description="Choose how the object origin is treated when exporting STL",
        items=ObjectOriginMode.items(),
        default=ObjectOriginMode.ORIGINAL
    )

    custom_ref: bpy.props.PointerProperty(
        name="Reference Object",
        description="Use this object as the reference frame for recording location/rotation",
        type=bpy.types.Object
    )

    follow_robot: bpy.props.EnumProperty(
        name="Follow Robot",
        description="Robot armature whose link should carry this environment object",
        items=_robot_items,
        update=_reset_robot_dependent_selections,
    )

    follow_link: bpy.props.EnumProperty(
        name="Follow Link",
        description="Robot link to attach this environment object to",
        items=_robot_link_items,
    )

    collision_link_choice: bpy.props.EnumProperty(
        name="Link",
        description="Select a robot link to add to this object's collision rule",
        items=_robot_collision_link_items,
    )

    collision_links: bpy.props.CollectionProperty(type=EnvironmentRuleItem)
    collision_links_index: bpy.props.IntProperty(default=-1)

    collision_movegroup_choice: bpy.props.EnumProperty(
        name="Movegroup",
        description="Select a robot movegroup to add to this object's collision rules",
        items=_robot_movegroup_items,
    )

    collision_movegroups: bpy.props.CollectionProperty(type=EnvironmentRuleItem)
    collision_movegroups_index: bpy.props.IntProperty(default=-1)

    allow_collision: bpy.props.BoolProperty(
        name="Allow Collision",
        description="Allow collision with the declared robot links or movegroup",
        default=False,
    )

    apply_modifiers: bpy.props.BoolProperty(
        name="Apply Modifiers",
        description="Apply modifiers when exporting mesh",
        default=True
    )

    include_for_export: bpy.props.BoolProperty(
        name="Include for Export (STL)",
        default=True
    )

# --------------------------------------------------------
#   Recorder Config
# --------------------------------------------------------
class RecorderConfig(bpy.types.PropertyGroup):
    angle_unit: bpy.props.EnumProperty(
        name="Angle Units",
        description="Choose between radians and degrees for exported angles",
        items=[
            (MeasurementConstants.RADIAN_UNIT, "Radians", ""),
            (MeasurementConstants.DEGREE_UNIT, "Degrees", "")
        ],
        default=AddonDefaults.ANGLE_UNIT
    )

    length_unit: bpy.props.EnumProperty(
        name="Location Units",
        description="Choose between meters and millimeters for exported locations",
        items=[
            (MeasurementConstants.METER_UNIT, "Meters", ""),
            (MeasurementConstants.MILLIMETER_UNIT, "Millimeters", "")
        ],
        default=AddonDefaults.LENGTH_UNIT
    )

    rotation_format: bpy.props.EnumProperty(
        name="Rotation Format",
        description="Choose rotation representation for all recorded objects",
        items=[
            (RotationFormat.QUATERNION, "Quaternion (WXYZ)", ""),
            (RotationFormat.XYZ, "XYZ Euler", ""),
            (RotationFormat.XZY, "XZY Euler", ""),
            (RotationFormat.YXZ, "YXZ Euler", ""),
            (RotationFormat.YZX, "YZX Euler", ""),
            (RotationFormat.ZXY, "ZXY Euler", ""),
            (RotationFormat.ZYX, "ZYX Euler", ""),
            (RotationFormat.AXIS_ANGLE, "Axis Angle", ""),
        ],
        default=RotationFormat.XYZ
    )

    export_binary: bpy.props.BoolProperty(
        name="Export as Binary",
        description="Export the data in a binary format. "
                    "Binary files are smaller but not human-readable. "
                    "YAML files are human-readable but larger in size.",
        default=AddonDefaults.IS_EXPORT_RECORD_AS_BINARY
    )

    output_dir: bpy.props.StringProperty(
        name="Output Directory",
        description="Folder to store YAML, BIN, STL and other outputs",
        default=AddonDefaults.RECORD_OUTPUT_DIR,
        subtype='DIR_PATH',
        options={'ANIMATABLE', 'PATH_SUPPORTS_BLEND_RELATIVE'},
    )

    base_filename: bpy.props.StringProperty(
        name="Base Filename",
        description="Base name for exported files (without extension)",
        default=AddonDefaults.RECORD_OUTPUT_BASENAME
    )

    env_export_mode: bpy.props.EnumProperty(
        name="Environment Mode",
        description="Control how environment data is handled during recording",
        items=[
            (EnvironmentExportMode.AUTO, "Auto Export", "Export environment automatically before recording"),
            (EnvironmentExportMode.MANUAL, "Manual Only", "Use existing environment file (no export)"),
            (EnvironmentExportMode.OFF, "Disabled", "Do not use environment at all"),
        ],
        default=AddonDefaults.ENV_EXPORT_MODE
    )

    include_hidden: bpy.props.BoolProperty(
        name="Include Hidden Objects",
        description="Also consider hidden objects when auto-identifying",
        default=AddonDefaults.IS_INCLUDE_HIDDEN_OBJECTS
    )


# --------------------------------------------------------
#   Recorder Episodes
# --------------------------------------------------------
class EpisodeItem(bpy.types.PropertyGroup):
    """A named, non-overlapping range used as an exported BT subtree."""

    batch_selected: bpy.props.BoolProperty(
        name="Batch Offset",
        description="Include this episode when applying a batch offset",
        default=False,
    )

    episode_name: bpy.props.StringProperty(
        name="Episode Name",
        description="Behavior-tree subtree name for this episode",
        default="",
    )
    episode_id: bpy.props.StringProperty(
        name="Episode ID",
        description="Unique prefix used for episode waypoints and trajectories",
        default="Episode",
    )
    start_with_waypoint: bpy.props.BoolProperty(
        name="Start with MoveWaypoint",
        description="Configure the episode to start with a MoveTomoWaypoint",
        default=EpisodeConfig.DEFAULT_START_WITH_WAYPOINT,
    )
    start_waypoint_mode: bpy.props.EnumProperty(
        name="Starting Waypoint",
        description="Waypoint used when this episode starts with MoveTomoWaypoint",
        items=[
            (
                EpisodeConfig.START_WAYPOINT_FIRST,
                "First WP of First Trajectory",
                "Move to the first waypoint of the episode's first trajectory",
            ),
            (
                EpisodeConfig.START_WAYPOINT_LAST,
                "Last WP of First Trajectory",
                "Move to the last waypoint of the episode's first trajectory",
            ),
        ],
        default=EpisodeConfig.DEFAULT_START_WAYPOINT_MODE,
    )
    start_frame: bpy.props.IntProperty(
        name="Start Frame",
        description="First Blender frame included in this episode",
        default=1,
    )
    end_frame: bpy.props.IntProperty(
        name="End Frame",
        description="Last Blender frame included in this episode",
        default=250,
    )
    logic: bpy.props.EnumProperty(
        name="Episode Logic",
        description="At most one primary control logic can wrap this episode",
        items=[
            (EpisodeConfig.LOGIC_NONE, "None", "Run the episode sequence once"),
            (EpisodeConfig.LOGIC_REPEAT, "Repeat", "Repeat the episode sequence a fixed number of times"),
            (EpisodeConfig.LOGIC_RETRY_UNTIL_SUCCESSFUL, "Retry Until Successful", "Retry until the episode succeeds"),
            (EpisodeConfig.LOGIC_KEEP_RUNNING_UNTIL_FAILURE, "Keep Running Until Failure", "Keep running until the episode fails"),
        ],
        default=EpisodeConfig.DEFAULT_LOGIC,
    )
    repeat_count: bpy.props.IntProperty(
        name="Repeat Count",
        description="Number of executions for Repeat logic",
        default=EpisodeConfig.DEFAULT_REPEAT_COUNT,
        min=1,
    )
    retry_count: bpy.props.IntProperty(
        name="Retry Count",
        description="Maximum attempts for Retry Until Successful logic",
        default=EpisodeConfig.DEFAULT_RETRY_COUNT,
        min=1,
    )
    delay_enabled: bpy.props.BoolProperty(
        name="Enable Delay",
        description="Delay the episode before its primary logic runs",
        default=EpisodeConfig.DEFAULT_DELAY_ENABLED,
    )
    delay_msec: bpy.props.IntProperty(
        name="Delay (ms)",
        description="Delay applied to this episode",
        default=EpisodeConfig.DEFAULT_DELAY_MSEC,
        min=0,
    )


# --------------------------------------------------------
#   Recorder (Main Controller)
# --------------------------------------------------------
class RecorderProperties(bpy.types.PropertyGroup):
    # If multiple objects must be supported:
    objs: bpy.props.CollectionProperty(type=RecordedObjectItem)
    objs_index: bpy.props.IntProperty(default=-1)

    robots: bpy.props.CollectionProperty(type=RobotItem)
    robots_index: bpy.props.IntProperty(default=-1)

    env_objs: bpy.props.CollectionProperty(type=EnvironmentObjectItem)
    env_objs_index: bpy.props.IntProperty(default=-1)

    episodes: bpy.props.CollectionProperty(type=EpisodeItem, name="Episodes")
    episodes_index: bpy.props.IntProperty(default=-1)
    episode_batch_start_with_waypoint: bpy.props.BoolProperty(
        name="Start with MoveWaypoint",
        description="Apply the first-trajectory start mode to the marked episodes",
        default=EpisodeConfig.DEFAULT_START_WITH_WAYPOINT,
    )
    episode_batch_waypoint_mode: bpy.props.EnumProperty(
        name="Starting Waypoint",
        description="Apply this starting waypoint choice to the marked episodes",
        items=[
            (
                EpisodeConfig.START_WAYPOINT_FIRST,
                "First WP of First Trajectory",
                "Use the first waypoint of each episode's first trajectory",
            ),
            (
                EpisodeConfig.START_WAYPOINT_LAST,
                "Last WP of First Trajectory",
                "Use the last waypoint of each episode's first trajectory",
            ),
        ],
        default=EpisodeConfig.DEFAULT_START_WAYPOINT_MODE,
    )
    episode_offset: bpy.props.IntProperty(
        name="Episode Offset",
        description="Number of original Blender frames to add to the selected episode ranges",
        default=0,
    )

    config: bpy.props.PointerProperty(
        type=RecorderConfig,
        name="Recorder Config"
    )

    show_ul_box: bpy.props.BoolProperty(
        name="Show Recorder List",
        default=False
    )

    show_env_box: bpy.props.BoolProperty(
        name="Show Environment List",
        default=False
        )

    show_robot_box: bpy.props.BoolProperty(name="Show Robot List", default=False)

    show_marker_box: bpy.props.BoolProperty(
        name="Show Marker List",
        default=False
    )

    show_episode_box: bpy.props.BoolProperty(
        name="Show Episode List",
        default=False,
    )

    show_episode_batch_box: bpy.props.BoolProperty(
        name="Show Batch Episode Operations",
        description="Show batch offset and start-setting controls for episodes",
        default=False,
    )

    show_transform_box: bpy.props.BoolProperty(
        name="Show Diagnostics and Transform Tools",
        default=False,
    )

    show_scene_content: bpy.props.BoolProperty(
        name="Show Scene Contents and Discovery",
        default=False,
    )

    show_recording_tools: bpy.props.BoolProperty(
        name="Show Recording Timeline and Tools",
        default=False,
    )

    show_advanced: bpy.props.BoolProperty(
        name="Show Advanced Settings",
        default=False
    )

    is_running: bpy.props.BoolProperty(
        name="Is Recording",
        default=False
    )

    is_fast_mode: bpy.props.BoolProperty(
        name="Fast Mode",
        description="Capture the full animation without intermediate screen refreshes",
        default=False
    )

    interrupted: bpy.props.BoolProperty(
        name="Recording Interrupted",
        description="True if a recording was interrupted",
        default=False
    )


# --------------------------------------------------------
#   Timeline Marker Flags (Enum from Python Enum)
# --------------------------------------------------------
def marker_type_items(self, context):
    items = []
    for flag in FlagTypes:
        full  = flag.name                      # e.g. AUTO_START
        label = full.replace("_", " ").title() # "Auto Start"
        short = flag.value                     # e.g. "AST"
        items.append((full, label, f"Short code: {short}"))
    return items


def _marker_name_update(self, context):
    """Rename the timeline marker while keeping its generated UID intact."""
    uid = self.uid.strip()
    label = self.marker_name.strip()
    suffix = label.rsplit("_", 1)
    if len(suffix) == 2 and len(suffix[1]) == 6 and all(
        char in "0123456789abcdefABCDEF" for char in suffix[1]
    ):
        label = suffix[0].strip() or "Marker"
        self.marker_name = label
    if not uid or not label or context is None or not hasattr(context, "scene"):
        return

    target = None
    for marker in context.scene.timeline_markers:
        _, _, marker_uid = FlagTypes.parse_marker_name(marker.name)
        if marker_uid == uid:
            target = marker
            break
    if target is None:
        target = next(
            (marker for marker in context.scene.timeline_markers if marker.frame == self.frame),
            None,
        )
    if target is not None:
        target.name = f"{label}_{uid}"


class MarkerEventProperties(bpy.types.PropertyGroup):
    """One event attached to a marker group."""

    flag_type: bpy.props.EnumProperty(name="Flag Type", items=marker_type_items)
    io0: bpy.props.BoolProperty(name="0")
    io1: bpy.props.BoolProperty(name="1")
    io2: bpy.props.BoolProperty(name="2")
    io3: bpy.props.BoolProperty(name="3")
    io4: bpy.props.BoolProperty(name="4")
    io5: bpy.props.BoolProperty(name="5")
    io6: bpy.props.BoolProperty(name="6")
    io7: bpy.props.BoolProperty(name="7")
    io_delay: bpy.props.IntProperty(
        name="I/O Delay (ms)",
        description="Delay after setting I/O states before continuing",
        default=0,
        min=0,
    )

    led1_is_send: bpy.props.BoolProperty(name="LED 1 Send", default=True)
    led2_is_send: bpy.props.BoolProperty(name="LED 2 Send", default=True)
    led1_mode: bpy.props.EnumProperty(name="LED 1 Mode", items=LEDModes.items(), default=LEDModes.SET.name)
    led2_mode: bpy.props.EnumProperty(name="LED 2 Mode", items=LEDModes.items(), default=LEDModes.SET.name)
    led1_color_preset: bpy.props.EnumProperty(name="LED 1 Color Preset", items=LEDColors.items(), default=LEDColors.WHITE.name)
    led2_color_preset: bpy.props.EnumProperty(name="LED 2 Color Preset", items=LEDColors.items(), default=LEDColors.WHITE.name)
    led1_color: bpy.props.FloatVectorProperty(name="LED 1 Color", subtype="COLOR", size=3, min=0.0, max=1.0, default=(1.0, 1.0, 1.0))
    led2_color: bpy.props.FloatVectorProperty(name="LED 2 Color", subtype="COLOR", size=3, min=0.0, max=1.0, default=(1.0, 1.0, 1.0))
    led1_progress: bpy.props.IntProperty(name="LED 1 Progress", min=0, max=100, default=0)
    led2_progress: bpy.props.IntProperty(name="LED 2 Progress", min=0, max=100, default=0)
    led1_extra: bpy.props.IntProperty(name="LED 1 Extra", description="Extra parameter for certain LED modes", default=0, min=0, max=999)
    led2_extra: bpy.props.IntProperty(name="LED 2 Extra", description="Extra parameter for certain LED modes", default=0, min=0, max=999)

    forced_wp_name: bpy.props.StringProperty(name="Waypoint Name", description="Name of the waypoint to force at this marker", default="")
    custom_flag_name: bpy.props.StringProperty(name="Custom Flag", description="Custom marker name", default="")

    def led_hex(self, idx):
        col = self.led1_color if idx == 1 else self.led2_color
        r, g, b = [int(c * 255) for c in col]
        return f"{r:02X}{g:02X}{b:02X}"

    def io_byte(self):
        return (
            (self.io7 << 0) | (self.io6 << 1) | (self.io5 << 2) | (self.io4 << 3) |
            (self.io3 << 4) | (self.io2 << 5) | (self.io1 << 6) | (self.io0 << 7)
        )

    def short_code(self):
        return FlagTypes.from_name(self.flag_type)


class MarkerProperties(bpy.types.PropertyGroup):
    """Marker metadata; event-specific state lives in ``events``."""

    marker_name: bpy.props.StringProperty(
        name="Marker Name",
        description="Editable marker label; the hexadecimal UID is managed automatically",
        default="",
        update=_marker_name_update,
    )
    uid: bpy.props.StringProperty(name="Unique ID")
    frame: bpy.props.IntProperty(name="Frame", default=0)

    events: bpy.props.CollectionProperty(
        name="Events",
        type=MarkerEventProperties,
    )
    events_index: bpy.props.IntProperty(name="Active Event", default=0)


# --------------------------------------------------------
#   Register / Unregister
# --------------------------------------------------------
classes = (
    RecordedBoneItem,
    RobotJointItem,
    RecordedObjectItem,
    RobotItem,
    EnvironmentRuleItem,
    EnvironmentObjectItem,
    RecorderConfig,
    EpisodeItem,
    RecorderProperties,
    MarkerEventProperties,
    MarkerProperties,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.Scene.mmrt_recorder = bpy.props.PointerProperty(type=RecorderProperties)
    # Popup selection instance
    bpy.types.Scene.mmrt_marker_draft = bpy.props.PointerProperty(type=MarkerEventProperties)

    # UIList backing collection
    bpy.types.Scene.mmrt_markers = bpy.props.CollectionProperty(type=MarkerProperties)
    bpy.types.Scene.mmrt_markers_index = bpy.props.IntProperty(default=0)



def unregister():
    del bpy.types.Scene.mmrt_recorder
    del bpy.types.Scene.mmrt_marker_draft
    del bpy.types.Scene.mmrt_markers
    del bpy.types.Scene.mmrt_markers_index

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
