import bpy
from ..common.config import AddonDefaults, UIConfig


class RemoteLogEntry(bpy.types.PropertyGroup):
    message: bpy.props.StringProperty(name="Message")


class RemoteControllerProperties(bpy.types.PropertyGroup):
    use_mmrt_xml: bpy.props.BoolProperty(
        name=f"Use {UIConfig.PANEL_MMRT_LABEL} Output",
        description="Use the generated MMRT application folder as the upload source",
        default=AddonDefaults.IS_USING_MMRT_OUTPUT
        )

    xml_path: bpy.props.StringProperty(
        name="Remote BT Config YAML",
        description=(
            "Path to a standalone BT config YAML on the robot server; "
            "this is not a local Blender path"
        ),
    )

    local_app_path: bpy.props.StringProperty(
        name="Local Application Folder",
        description="Local application folder to upload when MMRT output is disabled",
        subtype='DIR_PATH',
    )

    show_connection_settings: bpy.props.BoolProperty(
        name="Show Connection Settings",
        default=True,
    )

    show_send_section: bpy.props.BoolProperty(
        name="Show Send Application Folder",
        default=True,
    )

    show_controller_section: bpy.props.BoolProperty(
        name="Show Robot Console Controller",
        default=True,
    )

    use_sender_destination: bpy.props.BoolProperty(
        name="Use Sender Destination Path",
        description="Use the sender's destination path as the remote application path",
        default=True,
    )

    is_application: bpy.props.BoolProperty(
        name="Is Application",
        description="Load an application folder instead of a standalone BT config YAML",
        default=True,
    )

    remote_ip: bpy.props.StringProperty(
        name="IP",
        description="IP address of the remote machine running rosbridge and SSH",
        default=AddonDefaults.ROBOT_HOST_IP
        )

    remote_port: bpy.props.IntProperty(
        name="Port",
        description="WebSocket port for the remote rosbridge server",
        default=AddonDefaults.ROBOT_HOST_PORT, min=1, max=65535
        )

    remote_user: bpy.props.StringProperty(
        name="Username",
        description="Username for SSH connection to the remote machine",
        default=AddonDefaults.ROBOT_HOST_USERNAME
        )

    remote_password: bpy.props.StringProperty(
        name="Password",
        description="Password for SSH connection to the remote machine",
        default="", subtype='PASSWORD'
        )

    ros_domain_id: bpy.props.IntProperty(
        name="ROS_DOMAIN_ID",
        description="ROS 2 domain ID used by the remote ROS Bridge server",
        default=0,
        min=0,
        max=232,
    )

    remote_ros_commands: bpy.props.StringProperty(
        name="Extra ROS Commands",
        description=(
            "Commands run in a remote login shell before checking and launching "
            "rosbridge_server, such as sourcing ROS and robot workspace setup files "
            "or invoking an alias from the remote user's .bashrc"
        ),
        default="",
    )

    remote_dest: bpy.props.StringProperty(
        name="Destination Path",
        description="Destination directory for the uploaded application folder",
        default=AddonDefaults.ROBOT_HOST_APP_DIR,
    )

    remote_app_path: bpy.props.StringProperty(
        name="Remote Application Path",
        description="Application directory on the robot used by the console controller",
        default=AddonDefaults.ROBOT_HOST_APP_DIR,
    )

    last_console_status: bpy.props.StringProperty(
        name="Console Status",
        description="Most recent response from the robot console service",
        default="No console response received",
        options={'TEXTEDIT_UPDATE'},
    )

    last_console_success: bpy.props.BoolProperty(
        name="Last Request Successful",
        default=False,
    )

    bridge_connected: bpy.props.BoolProperty(
        name="ROS Bridge Connected",
        default=False,
    )

    bridge_status: bpy.props.StringProperty(
        name="ROS Bridge Status",
        default="Disconnected",
    )

    remote_bridge_status: bpy.props.StringProperty(
        name="ROS Bridge Server Status",
        default="Not started from MMRT",
    )

    remote_bridge_running: bpy.props.BoolProperty(
        name="Remote ROS Bridge Running",
        description="The remote ROS Bridge server was started from this controller",
        default=False,
    )

    last_remote_log: bpy.props.StringProperty(
        name="Last Remote Log",
        description="Most recent BTLog or /rosout message received through the bridge",
        default="No remote log received",
        options={'TEXTEDIT_UPDATE'},
    )

    remote_logs: bpy.props.CollectionProperty(type=RemoteLogEntry)

    remote_log_index: bpy.props.IntProperty(default=0, min=0)

    # remote_log_visible_rows: bpy.props.IntProperty(
    #     name="Lines",
    #     description="Number of remote log lines shown at once; scroll the list to browse history",
    #     default=10,
    #     min=1,
    #     max=50,
    # )

    
classes = (
    RemoteLogEntry,
    RemoteControllerProperties,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    
    bpy.types.Scene.mmrt_remote_controller = bpy.props.PointerProperty(type=RemoteControllerProperties)


def unregister():
    del bpy.types.Scene.mmrt_remote_controller

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
