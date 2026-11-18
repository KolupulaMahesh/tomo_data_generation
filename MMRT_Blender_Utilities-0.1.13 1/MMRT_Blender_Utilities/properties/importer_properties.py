import bpy
from bpy.props import (
    StringProperty,
    PointerProperty,
    CollectionProperty,
    BoolProperty,
)
from bpy.types import PropertyGroup
from ..common.config import AddonDefaults


class MMRT_UL_IkEndLinkItem(PropertyGroup):
    """Item storing an IK end link name"""
    name: StringProperty(name="Link Name")


class RobotImporterProperties(PropertyGroup):
    """Main importer settings group"""

    # File options
    filepath: StringProperty(
        name="XACRO/URDF File Path",
        description="File path to the XACRO or URDF file",
        default=AddonDefaults.XACRO_PATH,
        subtype='FILE_PATH',
    )

    srdf_filepath: StringProperty(
        name="SRDF/XACRO File Path",
        description="Optional file path to the SRDF or SRDF XACRO file",
        default="",
        subtype='FILE_PATH',
    )

    import_meshes: BoolProperty(
        name="Show Imported Meshes",
        description="Show imported visual meshes; source geometry is always archived for portable recording",
        default=AddonDefaults.IS_IMPORT_URDF_MESHES,
    )

    # Advanced toggle
    show_advanced: BoolProperty(
        name="Show Advanced Options",
        default=False,
    )

    # IK end links
    end_links: CollectionProperty(
        type=MMRT_UL_IkEndLinkItem,
        name="IK End Links",
        description="Names of end links to attach IK targets",
    )

    camera_links: CollectionProperty(
        type=MMRT_UL_IkEndLinkItem,
        name="Camera Links",
        description="Names of camera links to attach IK tracking targets",
    )


classes = (
    MMRT_UL_IkEndLinkItem,
    RobotImporterProperties,
)


# --- Registration ---
def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.Scene.robot_importer = PointerProperty(type=RobotImporterProperties)
    bpy.types.Object.mmrt_robot_mesh_collection = PointerProperty(type=bpy.types.Collection)
    bpy.types.Object.mmrt_robot_urdf_text = PointerProperty(type=bpy.types.Text)
    bpy.types.Object.mmrt_robot_srdf_text = PointerProperty(type=bpy.types.Text)

def unregister():
    del bpy.types.Object.mmrt_robot_srdf_text
    del bpy.types.Object.mmrt_robot_urdf_text
    del bpy.types.Object.mmrt_robot_mesh_collection
    del bpy.types.Scene.robot_importer

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
