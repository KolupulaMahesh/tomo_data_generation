import bpy
from ..common.config import AddonDefaults

def register():
    bpy.types.Scene.mmrt_show_advanced = bpy.props.BoolProperty(
        name="Show Advanced Settings",
        default=False,
    )

    bpy.types.Scene.mmrt_use_development_python = bpy.props.BoolProperty(
        name="Use Development Python",
        description=(
            "Use the explicitly configured development Python instead of the "
            "bundled MMRT backend."
        ),
        default=AddonDefaults.USE_DEVELOPMENT_PYTHON,
    )

    bpy.types.Scene.mmrt_external_python_path = bpy.props.StringProperty(
        name="Development Python Executable",
        description=(
            "Python executable used when Use Development Python is enabled. "
            "The add-on never installs packages into this environment."
        ),
        subtype='FILE_PATH',
        default=AddonDefaults.DEVELOPMENT_PYTHON_PATH,
    )

    bpy.types.Scene.mmrt_groot2_path = bpy.props.StringProperty(
        name="Groot 2 Executable",
        description="Path to the Groot 2 executable. Leave empty to use 'groot2' from system PATH.",
        subtype='FILE_PATH',
        default=AddonDefaults.GROOT2_PATH
    )

def unregister():
    del bpy.types.Scene.mmrt_groot2_path
    del bpy.types.Scene.mmrt_external_python_path
    del bpy.types.Scene.mmrt_use_development_python
    del bpy.types.Scene.mmrt_show_advanced
