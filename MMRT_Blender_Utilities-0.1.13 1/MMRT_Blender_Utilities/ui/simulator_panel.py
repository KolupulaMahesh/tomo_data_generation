import bpy

from ..common.blender import get_bl_idname
from ..common.config import UIConfig
from ..processes.process_manager import ProcessManager


class MMRT_UL_RemoteLog(bpy.types.UIList):
    bl_idname = "MMRT_UL_RemoteLog"

    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        layout.label(text=item.message, icon='TEXT')


class MMRT_PT_SimulatorPanel(bpy.types.Panel):
    bl_idname = "MMRT_PT_SimulatorPanel"
    bl_label = UIConfig.PANEL_CONTROL_LABEL
    bl_category = UIConfig.ADDON_CATEGORY
    bl_space_type = UIConfig.ADDON_SPACE
    bl_region_type = UIConfig.ADDON_REGION
    bl_options = {'DEFAULT_CLOSED'}

    @staticmethod
    def draw_section_header(box, props, property_name, label, icon):
        header = box.row()
        expanded = getattr(props, property_name)
        header.prop(
            props,
            property_name,
            icon="TRIA_DOWN" if expanded else "TRIA_RIGHT",
            icon_only=True,
            emboss=False,
        )
        header.label(text=label, icon=icon)
        return expanded

    def draw(self, context):
        layout = self.layout
        sim_props = context.scene.mmrt_remote_controller

        connection_box = layout.box()
        if self.draw_section_header(
            connection_box,
            sim_props,
            "show_connection_settings",
            "Connection Settings",
            'WORLD',
        ):
            connection_box.prop(sim_props, "remote_ip")
            connection_box.prop(sim_props, "remote_port")
            connection_box.prop(sim_props, "remote_user")
            connection_box.prop(sim_props, "remote_password")

        deploy_box = layout.box()
        if self.draw_section_header(
            deploy_box,
            sim_props,
            "show_send_section",
            "Send Application Folder",
            'EXPORT',
        ):
            deploy_box.operator(get_bl_idname("send_tree_files"), icon='EXPORT')
            deploy_box.prop(sim_props, "use_mmrt_xml", text="Use MMRT Output")
            if not sim_props.use_mmrt_xml:
                deploy_box.prop(sim_props, "local_app_path")
            deploy_box.prop(sim_props, "remote_dest")

        controller_box = layout.box()
        if self.draw_section_header(
            controller_box,
            sim_props,
            "show_controller_section",
            "Robot Console Controller",
            'NODETREE',
        ):
            server_start = controller_box.row()
            server_process = ProcessManager.get_process("ros_bridge_server_session")
            # Keep the toggle clickable while its SSH session is active so the
            # same button can send Ctrl-C to the remote ros2 launch process.
            server_start.enabled = server_process is None or sim_props.remote_bridge_running
            server_start.operator(
                get_bl_idname("start_bridge_server"),
                text=(
                    "Stop ROS Bridge server"
                    if sim_props.remote_bridge_running
                    else "Start ROS Bridge server"
                ),
                icon='CANCEL' if sim_props.remote_bridge_running else 'PLAY',
            )

            domain_row = controller_box.row()
            domain_row.prop(sim_props, "ros_domain_id")

            controller_box.label(text="Extra commands before server launch")
            if hasattr(controller_box, "textbox"):
                controller_box.textbox(
                    sim_props,
                    "remote_ros_commands",
                    initial_visible_lines=3,
                    placeholder=(
                        "source /opt/ros/humble/setup.bash; "
                        "source ~/robot_ws/install/setup.bash"
                    ),
                )
            else:
                controller_box.prop(
                    sim_props,
                    "remote_ros_commands",
                    text="",
                )

            controller_box.label(
                text=f"Server: {sim_props.remote_bridge_status}",
                icon='LINKED' if sim_props.remote_bridge_running else 'UNLINKED',
            )

            bridge_controls = controller_box.row(align=True)
            bridge_proc = ProcessManager.get_process("ros_bridge")
            if bridge_proc and bridge_proc.is_running():
                bridge_controls.operator(get_bl_idname("stop_bridge"), icon='CANCEL')
            else:
                bridge_controls.operator(
                    get_bl_idname("start_bridge"),
                    text="Start ROS Bridge client",
                    icon='PLAY',
                )

            controller_box.label(
                text=f"ROS bridge: {sim_props.bridge_status}",
                icon='LINKED' if sim_props.bridge_connected else 'UNLINKED',
            )
            controller_box.prop(sim_props, "use_sender_destination")
            controller_box.prop(sim_props, "is_application")

            if sim_props.is_application:
                remote_path_row = controller_box.row()
                remote_path_row.enabled = not sim_props.use_sender_destination
                remote_path_row.prop(sim_props, "remote_app_path")
                if sim_props.use_sender_destination:
                    controller_box.label(
                        text=f"Using sender destination: {sim_props.remote_dest}",
                        icon='FILE_FOLDER',
                    )
            else:
                controller_box.prop(sim_props, "xml_path")

            service_controls = controller_box.column()
            service_controls.enabled = sim_props.bridge_connected
            row1 = service_controls.row(align=True)
            if sim_props.is_application:
                row1.operator(get_bl_idname("load_application"), icon='FILE_FOLDER')
            else:
                row1.operator(get_bl_idname("load_config"), icon='FILE')

            row2 = service_controls.row(align=True)
            row2.operator(get_bl_idname("configure_tree"), text="Configure", icon='MODIFIER')
            row2.operator(get_bl_idname("gen_tree"), text="Generate Tree", icon='FILE_REFRESH')

            row3 = service_controls.row(align=True)
            row3.operator(get_bl_idname("run_tree"), icon='PLAY')
            row3.operator(get_bl_idname("pause_tree"), icon='PAUSE')
            row3.operator(get_bl_idname("stop_tree"), icon='CANCEL')

            controller_box.label(text=f"Status: {sim_props.last_console_status}", icon='INFO')
            log_header = controller_box.row(align=True)
            log_header.label(text="Remote log", icon='TEXT')
            # log_header.prop(sim_props, "remote_log_visible_rows", text="Lines")
            controller_box.label(
                text="BTLog messages appear when the tree runs; ROS node logs are also received.",
                icon='INFO',
            )
            controller_box.template_list(
                "MMRT_UL_RemoteLog",
                "mmrt_remote_log",
                sim_props,
                "remote_logs",
                sim_props,
                "remote_log_index",
                rows=2,
                maxrows=UIConfig.REMOTE_LOG_COUNT,
            )


classes = (
    MMRT_UL_RemoteLog,
    MMRT_PT_SimulatorPanel,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
