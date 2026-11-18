import bpy

from ..common.blender.utils import get_bl_idname
from ..common.config import UIConfig


class MMRT_UL_TuningWaypointList(bpy.types.UIList):
    bl_idname = "MMRT_UL_TuningWaypointList"

    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        row.alert = bool(item.tracking_error)
        row.label(text=item.name, icon='ERROR' if item.tracking_error else 'KEYFRAME_HLT')
        row.label(text=f"F{item.first_frame}")
        row.label(text=item.movegroup)


class MMRT_PT_WaypointTuning(bpy.types.Panel):
    bl_idname = "MMRT_PT_WaypointTuning"
    bl_label = "Waypoint Tuning"
    bl_category = UIConfig.ADDON_CATEGORY
    bl_space_type = UIConfig.ADDON_SPACE
    bl_region_type = UIConfig.ADDON_REGION
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        tuning = context.scene.mmrt_tuning
        layout.prop(tuning, "app_dir", text="App Folder")
        layout.operator(get_bl_idname("tuning_load_app"), icon='FILE_FOLDER')
        layout.label(text=tuning.status, icon='INFO')
        layout.label(text="Uses the Scene Recorder robot and joint mapping.")

        if not tuning.loaded:
            layout.label(text="Generate the app with this addon to include frame tracking.", icon='QUESTION')
            return

        if tuning.scene_frame_range_known:
            layout.label(
                text=f"Tracked scene range: {tuning.scene_frame_start}–{tuning.scene_frame_end}",
                icon='TIME',
            )
        else:
            layout.label(text="Scene frame range was not tracked; regenerate this app.", icon='ERROR')
        row = layout.row()
        row.template_list(
            "MMRT_UL_TuningWaypointList",
            "",
            tuning,
            "waypoints",
            tuning,
            "waypoints_index",
            rows=5,
        )
        actions = row.column(align=True)
        actions.operator(get_bl_idname("tuning_go_to_waypoint"), text="", icon='VIEWZOOM')

        if 0 <= tuning.waypoints_index < len(tuning.waypoints):
            waypoint = tuning.waypoints[tuning.waypoints_index]
            layout.label(text=f"Frames: {waypoint.sample_frames or 'None'}")
            if waypoint.tracking_error:
                layout.label(text=waypoint.tracking_error, icon='ERROR')
            if waypoint.occurrence_count > 1:
                layout.label(
                    text=f"Shared waypoint at {waypoint.occurrence_count} frames; edits must agree at each frame.",
                    icon='INFO',
                )

        layout.separator()
        layout.operator(get_bl_idname("tuning_apply_motion"), icon='FILE_REFRESH')
        layout.label(text="Updates waypoint values and keeps the generated tree timing.")
        layout.label(text="Changing the frame range requires full app regeneration.", icon='ERROR')


classes = (
    MMRT_UL_TuningWaypointList,
    MMRT_PT_WaypointTuning,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
