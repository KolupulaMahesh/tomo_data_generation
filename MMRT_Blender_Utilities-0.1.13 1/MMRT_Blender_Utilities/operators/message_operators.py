import bpy
from ..common.blender import get_bl_idname

class MMRT_OT_ShowMessage(bpy.types.Operator):
    """Display a popup message in Blender"""
    bl_idname = get_bl_idname("show_message")
    bl_label = "MMRT Process Message"

    message: bpy.props.StringProperty()
    level: bpy.props.EnumProperty(
        items=[
            ("INFO", "Info", ""),
            ("WARNING", "Warning", ""),
            ("ERROR", "Error", ""),
        ],
        default="INFO",
    )

    def draw(self, context):
        icon = {
            "INFO": "INFO",
            "WARNING": "ERROR",
            "ERROR": "CANCEL",
        }.get(self.level, "INFO")

        lines = self.message.splitlines() or [""]
        for index, line in enumerate(lines):
            self.layout.label(text=line, icon=icon if index == 0 else 'BLANK1')

    def execute(self, context):
        # Optional: keep if you want Info log entry
        self.report({self.level}, self.message)
        return {'FINISHED'}

    def invoke(self, context, event):
        return context.window_manager.invoke_popup(self)

    
classes = (
    MMRT_OT_ShowMessage,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
