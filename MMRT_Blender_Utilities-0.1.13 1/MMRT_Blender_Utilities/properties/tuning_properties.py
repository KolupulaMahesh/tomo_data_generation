import bpy


def _app_dir_changed(self, _context):
    self.loaded = False
    self.loaded_app_dir = ""
    self.waypoints.clear()
    self.waypoints_index = 0
    self.status = "Load the selected app folder."


class TuningWaypointItem(bpy.types.PropertyGroup):
    name: bpy.props.StringProperty(name="Waypoint")
    movegroup: bpy.props.StringProperty(name="Move Group")
    sample_frames: bpy.props.StringProperty(name="Sample Frames")
    first_frame: bpy.props.IntProperty(name="First Frame")
    occurrence_count: bpy.props.IntProperty(name="Occurrences")
    tracking_error: bpy.props.StringProperty(name="Tracking Error")


class TuningProperties(bpy.types.PropertyGroup):
    app_dir: bpy.props.StringProperty(
        name="App Folder",
        description="Generated MMRT app folder with blackboards and Blender tuning metadata",
        subtype='DIR_PATH',
        update=_app_dir_changed,
    )
    loaded: bpy.props.BoolProperty(default=False, options={'HIDDEN'})
    loaded_app_dir: bpy.props.StringProperty(default="", options={'HIDDEN'})
    status: bpy.props.StringProperty(default="Choose a generated app folder.")
    waypoints: bpy.props.CollectionProperty(type=TuningWaypointItem)
    waypoints_index: bpy.props.IntProperty(default=0)
    scene_frame_start: bpy.props.IntProperty(default=0)
    scene_frame_end: bpy.props.IntProperty(default=0)
    scene_frame_range_known: bpy.props.BoolProperty(default=False, options={'HIDDEN'})


classes = (
    TuningWaypointItem,
    TuningProperties,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.mmrt_tuning = bpy.props.PointerProperty(type=TuningProperties)


def unregister():
    del bpy.types.Scene.mmrt_tuning
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
