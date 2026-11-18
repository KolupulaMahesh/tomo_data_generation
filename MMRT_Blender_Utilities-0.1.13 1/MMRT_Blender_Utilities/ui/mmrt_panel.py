import bpy
from ..common.blender import draw_collapsible_box, get_bl_idname
from ..common.config import UIConfig
from ..processes.process_manager import ProcessManager


class MMRT_UL_GenerationLog(bpy.types.UIList):
    bl_idname = "MMRT_UL_GenerationLog"

    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        message = item.message
        icon_name = 'ERROR' if '[ERROR]' in message or 'Traceback' in message else 'TEXT'
        layout.label(text=message, icon=icon_name)


class MMRT_PT_MMRTPanel(bpy.types.Panel):
    bl_idname = "MMRT_PT_MMRTPanel"
    bl_label = UIConfig.PANEL_MMRT_LABEL
    bl_category = UIConfig.ADDON_CATEGORY
    bl_space_type = UIConfig.ADDON_SPACE
    bl_region_type = UIConfig.ADDON_REGION
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        props = scene.mmrt_settings

        row = layout.row()
        process = ProcessManager.get_process("mmrt_processor")
        row.enabled = not (process and process.is_running())
        row.operator(get_bl_idname("run_mmrt_process"), text="Generate Behavior Tree XML", icon='FILE_SCRIPT')
        layout.operator(get_bl_idname("plot_joint_data"), text="Plot Exported Joint Trajectories", icon='FORCE_CURVE')
        self.draw_generation_log(layout, context.window_manager)
        draw_collapsible_box(
            layout,
            scene,
            props,
            "show_advanced",
            self.draw_input_output,
            label="Input and Addon Output Options",
        )
        draw_collapsible_box(
            layout,
            scene,
            props,
            "show_tuning",
            self.draw_tuning_parameters,
            label="MMRT Process Tuning",
        )
        draw_collapsible_box(
            layout,
            scene,
            props,
            "show_behavior_tree_settings",
            self.draw_behavior_tree_settings,
            label="Behavior Tree Settings",
        )
    def draw_generation_log(self, layout, wm):
        box = layout.box()
        row = box.row()
        row.prop(
            wm,
            "mmrt_generation_log_expanded",
            icon='TRIA_DOWN' if wm.mmrt_generation_log_expanded else 'TRIA_RIGHT',
            icon_only=True,
            emboss=False,
        )
        row.label(text="Generation Log", icon='CONSOLE')
        if not wm.mmrt_generation_log_expanded:
            return
        box.label(text=wm.mmrt_generation_log_status)
        box.template_list(
            "MMRT_UL_GenerationLog", "", wm, "mmrt_generation_log",
            wm, "mmrt_generation_log_index", rows=8,
        )
        row = box.row(align=True)
        row.prop(wm, "mmrt_generation_log_follow", text="Follow Latest", toggle=True)
        row.operator(get_bl_idname("copy_generation_log"), text="Copy Full Log", icon='COPYDOWN')

    
    def draw_input_output(self, box, scene):
        props = scene.mmrt_settings
        box.use_property_split = True
        box.use_property_decorate = False 

        input_box = box.box()
        input_box.label(text="Inputs", icon='IMPORT')
        input_box.prop(props, "use_recorder_path")
        if not props.use_recorder_path:
            input_box.prop(props, "record_path", text="Record File")
        input_box.label(text="Robot descriptions and meshes load from the recording folder.", icon='INFO')
        
        output_box = box.box()
        output_box.label(text="Addon Output", icon='OUTPUT')
        output_box.prop(props, "output_dir", text="Output Directory")
        output_box.prop(props, "output_basename", text="Output Basename")

        backend_box = box.box()
        backend_box.label(text="Backend Simulation and Plots", icon='MOD_PHYSICS')
        backend_box.prop(props, "show_backend_simulation", text="Show Backend Simulation")
        backend_box.prop(props, "save_plot")
        backend_box.prop(props, "show_plot")

    def draw_tuning_parameters(self, box, scene):
        props = scene.mmrt_settings
        box.use_property_split = True
        box.use_property_decorate = False

        segment_box = box.box()
        segment_box.label(text="Segmentation", icon='IPO_BEZIER')
        segment_box.prop(props, "speed_thr")
        segment_box.prop(props, "min_segment_time")
        segment_box.prop(props, "segment_combine_thr")
        segment_box.prop(props, "wp_decimation_window")
        segment_box.prop(props, "wp_decimation_iter")
        segment_box.prop(props, "wp_decimation_method")

        collision_box = box.box()
        collision_box.label(text="Collision Checking", icon='MOD_BUILD')
        collision_box.prop(props, "collision_thr")

    def draw_behavior_tree_settings(self, box, scene):
        props = scene.mmrt_settings
        box.use_property_split = True
        box.use_property_decorate = False

        subtree_box = box.box()
        subtree_box.label(text="Behavior Tree Output IDs", icon='NODETREE')
        subtree_box.prop(props, "tree_id", text="Main Tree ID")
        subtree_box.prop(props, "configure_tree_name", text="ConfigureTree")
        subtree_box.prop(props, "scene_setup_tree_name", text="SceneSetup")
        subtree_box.prop(props, "move_segment_tree_name", text="moveSegmentTree")

        pipeline_box = box.box()
        pipeline_box.label(text="Planning Pipeline", icon='SETTINGS')
        pipeline_box.prop(props, "plan_pipeline")

        planner_box = box.box()
        planner_box.label(text="Planner ID", icon='PREFERENCES')
        planner_box.prop(props, "plan_id")

        trajectory_box = box.box()
        trajectory_box.label(text="Trajectory Settings", icon='CURVE_PATH')
        trajectory_box.prop(props, "trajectory_strategy")
        trajectory_box.prop(props, "trajectory_resample")
        trajectory_box.prop(props, "trajectory_resolution")
        trajectory_box.prop(props, "trajectory_time_parameterization")
        trajectory_box.prop(props, "trajectory_visualize")

        global_box = box.box()
        global_box.label(text="Global Planning", icon='SETTINGS')
        global_box.prop(props, "global_trajectory_collision_check")
        global_box.prop(props, "global_velocity_scaling")
        global_box.prop(props, "global_acceleration_scaling")
        global_box.prop(props, "global_trajectory_blending_radius")
        global_box.prop(props, "global_save_plan")

        generation_box = box.box()
        generation_box.label(text="Behavior Tree Timing", icon='NODETREE')
        generation_box.prop(props, "bt_sleep_thr")

        self.draw_system_settings(box, scene)

    def draw_system_settings(self, output_box, scene):
        props = scene.mmrt_settings
        settings_box = output_box.box()
        settings_box.label(text="Behavior Tree Runtime", icon='PREFERENCES')
        settings_box.use_property_split = True
        settings_box.use_property_decorate = False
        settings_box.prop(props, "enable_log")
        settings_box.prop(props, "enable_log_to_file")
        settings_box.prop(props, "debug_topic_name")


# Register/unregister
classes = (
    MMRT_UL_GenerationLog,
    MMRT_PT_MMRTPanel,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
