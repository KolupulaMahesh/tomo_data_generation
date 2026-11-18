import bpy
from ..common.blender import draw_collapsible_box, get_bl_idname
from ..common.config import UIConfig


class MMRT_PT_RobotImportPanel(bpy.types.Panel):
    bl_idname = "MMRT_PT_RobotImportPanel"
    bl_label = UIConfig.PANEL_IMPORTER_LABEL
    bl_category = UIConfig.ADDON_CATEGORY
    bl_space_type = UIConfig.ADDON_SPACE
    bl_region_type = UIConfig.ADDON_REGION
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        props = scene.robot_importer

        layout.use_property_split = True
        layout.use_property_decorate = False  # hides animation dots

        # Filepath + Import buttons
        layout.prop(props, "filepath", text="URDF/XACRO")
        layout.prop(props, "srdf_filepath", text="SRDF/XACRO")
        row = layout.row(align=True)
        row.operator(get_bl_idname("store_robot_descriptions"), text="Store Only", icon='TEXT')
        row.operator(get_bl_idname("import_xacro_urdf"), text="Build Armature", icon='IMPORT')

        # --- Advanced Section ---
        def draw_advanced(box, scene):
            box.prop(props, "import_meshes")
            if not props.import_meshes:
                box.label(text="Source meshes will be archived but hidden.")

            links_row = box.row()

            end_links_column = links_row.column()
            end_links_column.label(text="End Links for IK:")
            for item in props.end_links:
                end_links_column.prop(item, "name", text="")
            row = end_links_column.row(align=True)
            row.operator(get_bl_idname("add_end_link"), text="Add")
            row.operator(get_bl_idname("remove_end_link"), text="Remove")

            camera_links_column = links_row.column()
            camera_links_column.label(text="Camera Links for IK:")
            for item in props.camera_links:
                camera_links_column.prop(item, "name", text="")
            row = camera_links_column.row(align=True)
            row.operator(get_bl_idname("add_camera_link"), text="Add")
            row.operator(get_bl_idname("remove_camera_link"), text="Remove")

        # Use the unified group + property
        draw_collapsible_box(
            layout,
            scene,
            props,
            "show_advanced",
            draw_advanced,
        )


classes = (
    MMRT_PT_RobotImportPanel,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
