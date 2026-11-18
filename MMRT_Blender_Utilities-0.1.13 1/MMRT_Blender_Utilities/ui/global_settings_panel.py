import bpy
from ..common.blender import draw_collapsible_box, get_bl_idname
from ..common.config import UIConfig
from ..processes.process_manager import ProcessManager


class MMRT_PT_GlobalSettingsPanel(bpy.types.Panel):
    bl_idname = "MMRT_PT_GlobalSettingsPanel"
    bl_label = UIConfig.PANEL_GLOBAL_LABEL
    bl_category = UIConfig.ADDON_CATEGORY
    bl_space_type = UIConfig.ADDON_SPACE
    bl_region_type = UIConfig.ADDON_REGION

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        layout.use_property_split = True
        layout.use_property_decorate = False

        layout.separator()
        layout.operator(get_bl_idname("verify_mmrt_backend"), text="Verify MMRT Backend", icon="CONSOLE")
        if ProcessManager.get_process("groot2") and ProcessManager.get_process("groot2").is_running():
            layout.operator(get_bl_idname("stop_groot2"), icon='CANCEL')
        else:
            layout.operator(get_bl_idname("launch_groot2"), icon='PLAY')

        draw_collapsible_box(
            layout,
            scene,
            scene,
            "mmrt_show_advanced",
            self.draw_advanced,
            label="Advanced Settings",
        )

    @staticmethod
    def draw_advanced(box, scene):
        box.use_property_split = True
        box.use_property_decorate = False

        box.prop(scene, "mmrt_use_development_python", text="Use Development Python")
        python_row = box.row()
        python_row.enabled = scene.mmrt_use_development_python
        python_row.prop(scene, "mmrt_external_python_path", text="Development Python", icon='FILE_SCRIPT')
        box.prop(scene, "mmrt_groot2_path", text="Groot 2 Path", icon='NODETREE')

# Register/unregister
classes = (
    MMRT_PT_GlobalSettingsPanel,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
