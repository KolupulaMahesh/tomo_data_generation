import os
import uuid
import xml.etree.ElementTree as ET

import bpy
from bpy.props import IntProperty
import json
from ..common.recording import MMRTRecorder, get_recorder_environment_filepath
from ..common.config import (
    EnvironmentExportMode,
    ObjectOriginMode,
    RecordTransformSpace,
    RecorderOperatorOptions,
    SceneRecognitionConfig,
)
from ..common.exporting import export_environment, export_environment_metadata
from ..common.blender import OPS, ensure_object_mode, get_bl_idname, show_report
from ..common.marker_utils import FlagTypes, HANDLER_REGISTRY, LEDColors, LEDModes, MarkerNaming
from ..processes.mmrt.definitions import RobotJointTypes
from ..common.robot_description.storage import (
    MESH_SOURCE_PATH_PROP,
    ROBOT_MESH_COLLECTION_PROP,
    ROBOT_SRDF_TEXT_PROP,
    ROBOT_URDF_TEXT_PROP,
)
_active_recorder = None


def _relative_transform_entries(context):
    """Return selected-object transforms expressed in active-object space."""
    selected_objects = list(context.selected_objects or [])
    active_object = context.active_object
    if len(selected_objects) <= 1 or active_object is None:
        return None

    active_inverse = active_object.matrix_world.inverted_safe()
    entries = [
        (obj, active_inverse @ obj.matrix_world)
        for obj in selected_objects
        if obj != active_object
    ]
    return active_object, entries


def _format_vector(vector):
    return f"({vector.x:.6f}, {vector.y:.6f}, {vector.z:.6f})"


def _populate_robot_sources(item, armature_obj):
    """Use importer associations when a newly-added recorder robot has them."""
    linked_urdf = getattr(armature_obj, "mmrt_robot_urdf_text", None)
    linked_srdf = getattr(armature_obj, "mmrt_robot_srdf_text", None)
    linked_meshes = getattr(armature_obj, "mmrt_robot_mesh_collection", None)
    if linked_urdf:
        item.urdf_text = linked_urdf
    if linked_srdf:
        item.srdf_text = linked_srdf
    if linked_meshes:
        item.meshes_collection = linked_meshes

    urdf_text_name = armature_obj.get(ROBOT_URDF_TEXT_PROP, "")
    srdf_text_name = armature_obj.get(ROBOT_SRDF_TEXT_PROP, "")
    if not item.urdf_text and urdf_text_name:
        item.urdf_text = bpy.data.texts.get(urdf_text_name)
    if not item.srdf_text and srdf_text_name:
        item.srdf_text = bpy.data.texts.get(srdf_text_name)
    collection_name = armature_obj.get(ROBOT_MESH_COLLECTION_PROP, "")
    if not item.meshes_collection and collection_name:
        item.meshes_collection = bpy.data.collections.get(collection_name)


# --------------------------------------------------------
#   Add / Remove Object Slots
# --------------------------------------------------------
class MMRT_OT_AddObject(bpy.types.Operator):
    bl_idname = get_bl_idname("add_object")
    bl_label = "Add Object"
    bl_description = "Append the currently selected object (or None if nothing selected)"

    def execute(self, context):
        recorder = context.scene.mmrt_recorder
        obj = context.active_object
        if obj:
            # Check for duplicates
            for item in recorder.objs:
                if item.obj == obj:
                    self.report({'WARNING'}, f"Object '{obj.name}' already in list")
                    return {'CANCELLED'}

            new_item = recorder.objs.add()
            new_item.obj = obj
            if obj.type == SceneRecognitionConfig.ROBOT_TYPE:
                for bone in obj.data.bones:
                    entry = new_item.joints.add()
                    entry.name = bone.name
            self.report({'INFO'}, f"Added object '{obj.name}'")

        else:
            # Add None entry
            new_item = recorder.objs.add()
            new_item.obj = None
            self.report({'INFO'}, "Added empty slot (None)")

        # Select the new entry
        recorder.objs_index = len(recorder.objs) - 1
        return {'FINISHED'}


class MMRT_OT_AddRobot(bpy.types.Operator):
    bl_idname = get_bl_idname("add_robot")
    bl_label = "Add Robot"
    bl_description = "Add the selected armature as the recording robot"

    def execute(self, context):
        recorder = context.scene.mmrt_recorder
        obj = context.active_object
        if not obj or obj.type != SceneRecognitionConfig.ROBOT_TYPE:
            self.report({'WARNING'}, "Select an armature to add as a robot.")
            return {'CANCELLED'}
        if any(item.obj == obj for item in recorder.robots):
            self.report({'WARNING'}, f"Robot '{obj.name}' is already in the list.")
            return {'CANCELLED'}
        item = recorder.robots.add()
        item.obj = obj
        _populate_robot_sources(item, obj)
        recorder.robots_index = len(recorder.robots) - 1
        self.report({'INFO'}, f"Added robot '{obj.name}'.")
        return {'FINISHED'}


class MMRT_OT_RemoveRobot(bpy.types.Operator):
    bl_idname = get_bl_idname("remove_robot")
    bl_label = "Remove Robot"
    index: IntProperty()

    def execute(self, context):
        rec = context.scene.mmrt_recorder
        if 0 <= self.index < len(rec.robots):
            rec.robots.remove(self.index)
            rec.robots_index = min(self.index, len(rec.robots) - 1)
        return {'FINISHED'}


class MMRT_OT_ClearRobots(bpy.types.Operator):
    bl_idname = get_bl_idname("clear_robots")
    bl_label = "Clear Robots"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        rec = context.scene.mmrt_recorder
        rec.robots.clear()
        rec.robots_index = -1
        return {'FINISHED'}


class MMRT_OT_RefreshRobotJoints(bpy.types.Operator):
    bl_idname = get_bl_idname("refresh_robot_joints")
    bl_label = "Refresh Joints"
    robot_index: IntProperty()

    def execute(self, context):
        rec = context.scene.mmrt_recorder
        if not 0 <= self.robot_index < len(rec.robots):
            return {'CANCELLED'}
        robot = rec.robots[self.robot_index]
        if not robot.obj or robot.obj.type != SceneRecognitionConfig.ROBOT_TYPE:
            self.report({'WARNING'}, "Select a valid robot armature first.")
            return {'CANCELLED'}
        if not robot.urdf_text or not robot.urdf_text.as_string().strip():
            self.report({'WARNING'}, "Select the robot's URDF text first.")
            return {'CANCELLED'}
        try:
            root = ET.fromstring(robot.urdf_text.as_string())
            joint_names = [
                joint.get('name') for joint in root.findall('joint')
                if joint.get('name') and joint.get('type', RobotJointTypes.FIXED) not in RobotJointTypes.NON_MOVING and joint.find('mimic') is None
            ]
        except ET.ParseError as exc:
            self.report({'ERROR'}, f"Invalid URDF XML: {exc}")
            return {'CANCELLED'}
        # RNA entries are invalidated by clear(); retain plain values, not entries.
        fields = ('bone_name', 'record', 'rot_axis', 'direction', 'offset')
        old = {entry.joint_name: {field: getattr(entry, field) for field in fields} for entry in robot.joints}
        robot.joints.clear()
        for joint_name in joint_names:
            entry = robot.joints.add()
            entry.joint_name = joint_name
            if joint_name in old:
                for field, value in old[joint_name].items():
                    setattr(entry, field, value)
        self.report({'INFO'}, f"Loaded {len(joint_names)} URDF joints for '{robot.obj.name}'.")
        return {'FINISHED'}


class MMRT_OT_AutoDetectRobotAxes(bpy.types.Operator):
    bl_idname = get_bl_idname("auto_detect_robot_axes")
    bl_label = "Auto-Detect Axes"
    robot_index: IntProperty()

    def execute(self, context):
        rec = context.scene.mmrt_recorder
        if not 0 <= self.robot_index < len(rec.robots):
            return {'CANCELLED'}
        robot = rec.robots[self.robot_index]
        if not robot.obj or robot.obj.type != SceneRecognitionConfig.ROBOT_TYPE:
            self.report({'WARNING'}, "Select a valid robot armature first.")
            return {'CANCELLED'}
        detected_count = 0
        for joint in robot.joints:
            # URDF joint names and the names of the corresponding imported
            # bones are normally identical.  Resolve that pairing here so
            # Auto-Detect Axes also works immediately after refreshing the
            # selected URDF joint list, when bone_name is still empty.
            matching_bone = robot.obj.data.bones.get(joint.joint_name)
            if matching_bone is not None:
                joint.bone_name = matching_bone.name

            pose_bone = robot.obj.pose.bones.get(joint.bone_name)
            if pose_bone is None:
                # A refreshed URDF list starts with recording enabled by
                # default.  Do not leave an unavailable joint enabled just
                # because it has no corresponding armature bone.
                joint.record = False
                continue

            for axis, unlocked in zip(('X', 'Y', 'Z'), pose_bone.lock_rotation):
                if not unlocked:
                    joint.rot_axis = axis
                    detected_count += 1
                    break
        self.report({'INFO'}, f"Auto-detected axes for {detected_count} joint(s) on '{robot.obj.name}'.")
        return {'FINISHED'}


class MMRT_OT_RemoveObject(bpy.types.Operator):
    bl_idname = get_bl_idname("remove_object")
    bl_label = "Remove Object"
    bl_description = "Remove this object slot and its settings"
    index: IntProperty()

    def execute(self, context):
        recorder = context.scene.mmrt_recorder
        if 0 <= self.index < len(recorder.objs):
            recorder.objs.remove(self.index)
        return {'FINISHED'}


class MMRT_OT_ClearObjects(bpy.types.Operator):
    bl_idname = get_bl_idname("clear_objects")
    bl_label = "Clear Objects"
    bl_description = "Remove all objects from the list"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        rec = context.scene.mmrt_recorder

        rec.objs.clear()
        rec.objs_index = -1

        return {'FINISHED'}

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width=300)

    def draw(self, context):
        layout = self.layout
        rec = context.scene.mmrt_recorder

        layout.label(text="Clear all recorded objects?", icon='TRASH')

        count = len(rec.objs)
        layout.label(text=f"{count} item(s) will be removed.")

        layout.separator()
        layout.label(text="This action can be undone (Ctrl+Z).", icon='INFO')


# --------------------------------------------------------
#   Refresh Joints for Armature Slots
# --------------------------------------------------------
class MMRT_OT_RefreshJoints(bpy.types.Operator):
    bl_idname = get_bl_idname("refresh_joints")
    bl_label = "Refresh Joint List"
    bl_description = "Populate this armature’s bone list"
    obj_index: IntProperty()

    def execute(self, context):
        recorder = context.scene.mmrt_recorder
        if not (0 <= self.obj_index < len(recorder.objs)):
            return {'CANCELLED'}

        item = recorder.objs[self.obj_index]
        ob = item.obj
        item.joints.clear()
        if ob is None or ob.type != SceneRecognitionConfig.ROBOT_TYPE:
            self.report({'WARNING'}, "Select a valid Armature first.")
            return {'CANCELLED'}

        for bone in ob.data.bones:
            entry = item.joints.add()
            entry.name = bone.name

        self.report({'INFO'}, f"Found {len(ob.data.bones)} bones in '{ob.name}'.")
        return {'FINISHED'}


# --------------------------------------------------------
#   Auto-Detect Axes (for Armatures)
# --------------------------------------------------------
class MMRT_OT_AutoDetectAxes(bpy.types.Operator):
    bl_idname = get_bl_idname("auto_detect_axes")
    bl_label = "Auto-Detect Axes"
    bl_description = "Set each bone’s rot_axis to its first unlocked axis (X→Y→Z)"
    obj_index: IntProperty()

    def execute(self, context):
        recorder = context.scene.mmrt_recorder
        if not (0 <= self.obj_index < len(recorder.objs)):
            return {'CANCELLED'}

        item = recorder.objs[self.obj_index]
        ob = item.obj
        if ob is None or ob.type != SceneRecognitionConfig.ROBOT_TYPE:
            self.report({'WARNING'}, "Select a valid Armature first.")
            return {'CANCELLED'}

        for bone_item in item.joints:
            pb = ob.pose.bones.get(bone_item.name)
            if pb is None:
                continue
            if not pb.lock_rotation[0]:
                bone_item.rot_axis = 'X'
            elif not pb.lock_rotation[1]:
                bone_item.rot_axis = 'Y'
            elif not pb.lock_rotation[2]:
                bone_item.rot_axis = 'Z'

        self.report({'INFO'}, f"Auto-detected axes for '{ob.name}'.")
        return {'FINISHED'}


class MMRT_OT_SetAllTransformSpace(bpy.types.Operator):
    bl_idname = get_bl_idname("set_all_transform_space")
    bl_label = "Set Transform Space for All"
    bl_options = {'REGISTER', 'UNDO'}

    target: bpy.props.EnumProperty(
        items=[
            (RecorderOperatorOptions.TRANSFORM_ENV, "Environment", ""),
            (RecorderOperatorOptions.TRANSFORM_RECORD, "Record", ""),
        ]
    )

    env_mode: bpy.props.EnumProperty(
        name="Transform Space",
        items=ObjectOriginMode.items(),
        default=ObjectOriginMode.ORIGINAL
    )

    record_mode: bpy.props.EnumProperty(
        name="Transform Space",
        items=RecordTransformSpace.items(),
        default=RecordTransformSpace.WORLD
    )

    def execute(self, context):
        recorder = context.scene.mmrt_recorder

        if self.target == RecorderOperatorOptions.TRANSFORM_ENV:
            for item in recorder.env_objs:
                item.origin_mode = self.env_mode
            self.report({'INFO'}, f"Set all {self.target} to {self.env_mode}")

        elif self.target == RecorderOperatorOptions.TRANSFORM_RECORD:
            for item in recorder.objs:
                item.space = self.record_mode
            self.report({'INFO'}, f"Set all {self.target} to {self.record_mode}")
        return {'FINISHED'}

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width=250)
        
    def draw(self, context):
        layout = self.layout
        if self.target == RecorderOperatorOptions.TRANSFORM_ENV:
            layout.prop(self, "env_mode", text="Environment Object Origin", icon='ORIENTATION_GLOBAL')
        if self.target == RecorderOperatorOptions.TRANSFORM_RECORD:
            layout.prop(self, "record_mode", text="Record Space", icon='ORIENTATION_GLOBAL')

class MMRT_OT_PrintCurrentFrame(bpy.types.Operator):
    bl_idname = get_bl_idname("print_current_frame")
    bl_label = "Print Current Frame Data"

    def execute(self, context):
        try:
            logic = MMRTRecorder(context)
            logic.prepare()
            data = logic.evaluate_current_frame()
            print("\n=== CURRENT FRAME DATA ===")
            for k, v in data.items():
                print(f"{k}: {v}")

            # === Clipboard (JSON style, same structure) ===
            text = json.dumps(data, indent=2)
            context.window_manager.clipboard = text
            self.report({'INFO'}, f"Copied {len(data)} values to clipboard")
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}

        return {'FINISHED'}


class MMRT_OT_PrintRelativeTransforms(bpy.types.Operator):
    bl_idname = get_bl_idname("print_relative_transforms")
    bl_label = "Print Relative Transforms"
    bl_description = "Print selected object transforms in the active object's coordinate space"

    def execute(self, context):
        result = _relative_transform_entries(context)
        if result is None:
            show_report(
                "Select at least two objects, including an active object, "
                "to print relative transforms.",
                level="WARNING",
            )
            return {'FINISHED'}
        active_object, entries = result
        if not entries:
            show_report(
                "Select at least two objects, including an active object, "
                "to print relative transforms.",
                level="WARNING",
            )
            return {'FINISHED'}

        output = [f"=== TRANSFORMS IN '{active_object.name}' SPACE ==="]
        for obj, relative_matrix in entries:
            location = relative_matrix.to_translation()
            rotation = relative_matrix.to_euler('XYZ')
            quaternion = relative_matrix.to_quaternion()
            scale = relative_matrix.to_scale()

            output.append(f"{obj.name}:")
            output.append(f"  Location: {_format_vector(location)}")
            output.append(
                "  Rotation Euler XYZ (radians): "
                f"({rotation.x:.6f}, {rotation.y:.6f}, {rotation.z:.6f})"
            )
            output.append(
                "  Rotation Quaternion XYZW: "
                f"({quaternion.x:.6f}, {quaternion.y:.6f}, "
                f"{quaternion.z:.6f}, {quaternion.w:.6f})"
            )
            output.append(f"  Scale: {_format_vector(scale)}")
            output.append("  Matrix:")
            for row in relative_matrix:
                output.append("    " + " ".join(f"{value:.6f}" for value in row))

        text = "\n".join(output)
        context.window_manager.clipboard = text
        show_report(
            f"{text}\n\nCopied {len(entries)} transform(s) to clipboard."
        )

        return {'FINISHED'}


class MMRT_OT_PrintRelativeTransformArray(bpy.types.Operator):
    bl_idname = get_bl_idname("print_relative_transform_array")
    bl_label = "Print Flat Transform Arrays"
    bl_description = "Print translation and quaternion arrays in the active object's coordinate space"

    def execute(self, context):
        result = _relative_transform_entries(context)
        if result is None:
            show_report(
                "Select at least two objects, including an active object, "
                "to print flat transform arrays.",
                level="WARNING",
            )
            return {'FINISHED'}
        active_object, entries = result
        if not entries:
            show_report(
                "Select at least two objects, including an active object, "
                "to print flat transform arrays.",
                level="WARNING",
            )
            return {'FINISHED'}

        output = [
            f"=== FLAT TRANSFORMS IN '{active_object.name}' SPACE ===",
            "# [tx, ty, tz, qx, qy, qz, qw]",
        ]
        for obj, relative_matrix in entries:
            location = relative_matrix.to_translation()
            quaternion = relative_matrix.to_quaternion()
            values = (
                location.x, location.y, location.z,
                quaternion.x, quaternion.y, quaternion.z, quaternion.w,
            )
            flat_array = "[" + ", ".join(f"{value:.6f}" for value in values) + "]"
            output.append(f"{obj.name}: {flat_array}")

        text = "\n".join(output)
        context.window_manager.clipboard = text
        show_report(
            f"{text}\n\nCopied {len(entries)} flat transform array(s) to clipboard."
        )

        return {'FINISHED'}
    

class MMRT_OT_AddEnvObject(bpy.types.Operator):
    bl_idname = get_bl_idname("add_env_object")
    bl_label = "Add Environment Object"

    def execute(self, context):
        recorder = context.scene.mmrt_recorder
        obj = context.active_object

        if obj:
            if obj.type not in SceneRecognitionConfig.ENV_TYPES:
                self.report({'WARNING'}, "Only mesh or curve objects can be added as environment")
                return {'CANCELLED'}
            # prevent duplicates
            for item in recorder.env_objs:
                if item.obj == obj:
                    self.report({'WARNING'}, f"{obj.name} already in env list")
                    return {'CANCELLED'}

            new_item = recorder.env_objs.add()
            new_item.obj = obj
            self.report({'INFO'}, f"Added env object '{obj.name}'")
        else:
            new_item = recorder.env_objs.add()
            new_item.obj = None
            self.report({'INFO'}, "Added empty env slot")

        recorder.env_objs_index = len(recorder.env_objs) - 1
        return {'FINISHED'}


class MMRT_OT_AddEnvCollisionLink(bpy.types.Operator):
    bl_idname = get_bl_idname("add_env_collision_link")
    bl_label = "Add Collision Link"

    env_index: IntProperty()

    def execute(self, context):
        recorder = context.scene.mmrt_recorder
        if not (0 <= self.env_index < len(recorder.env_objs)):
            return {'CANCELLED'}
        item = recorder.env_objs[self.env_index]
        name = item.collision_link_choice
        if not name:
            return {'CANCELLED'}
        if not any(rule.name == name for rule in item.collision_links):
            rule = item.collision_links.add()
            rule.name = name
        item.collision_links_index = len(item.collision_links) - 1
        return {'FINISHED'}


class MMRT_OT_RemoveEnvCollisionLink(bpy.types.Operator):
    bl_idname = get_bl_idname("remove_env_collision_link")
    bl_label = "Remove Collision Link"

    env_index: IntProperty()
    rule_index: IntProperty()

    def execute(self, context):
        recorder = context.scene.mmrt_recorder
        if not (0 <= self.env_index < len(recorder.env_objs)):
            return {'CANCELLED'}
        item = recorder.env_objs[self.env_index]
        if 0 <= self.rule_index < len(item.collision_links):
            item.collision_links.remove(self.rule_index)
            item.collision_links_index = min(self.rule_index, len(item.collision_links) - 1)
        return {'FINISHED'}


class MMRT_OT_AddEnvCollisionMovegroup(bpy.types.Operator):
    bl_idname = get_bl_idname("add_env_collision_movegroup")
    bl_label = "Add Collision Movegroup"

    env_index: IntProperty()

    def execute(self, context):
        recorder = context.scene.mmrt_recorder
        if not (0 <= self.env_index < len(recorder.env_objs)):
            return {'CANCELLED'}
        item = recorder.env_objs[self.env_index]
        name = item.collision_movegroup_choice
        if not name:
            return {'CANCELLED'}
        if not any(rule.name == name for rule in item.collision_movegroups):
            rule = item.collision_movegroups.add()
            rule.name = name
        item.collision_movegroups_index = len(item.collision_movegroups) - 1
        return {'FINISHED'}


class MMRT_OT_RemoveEnvCollisionMovegroup(bpy.types.Operator):
    bl_idname = get_bl_idname("remove_env_collision_movegroup")
    bl_label = "Remove Collision Movegroup"

    env_index: IntProperty()
    rule_index: IntProperty()

    def execute(self, context):
        recorder = context.scene.mmrt_recorder
        if not (0 <= self.env_index < len(recorder.env_objs)):
            return {'CANCELLED'}
        item = recorder.env_objs[self.env_index]
        if 0 <= self.rule_index < len(item.collision_movegroups):
            item.collision_movegroups.remove(self.rule_index)
            item.collision_movegroups_index = min(self.rule_index, len(item.collision_movegroups) - 1)
        return {'FINISHED'}


class MMRT_OT_RemoveEnvObject(bpy.types.Operator):
    bl_idname = get_bl_idname("remove_env_object")
    bl_label = "Remove Environment Object"

    index: IntProperty()

    def execute(self, context):
        recorder = context.scene.mmrt_recorder
        if 0 <= self.index < len(recorder.env_objs):
            recorder.env_objs.remove(self.index)
        return {'FINISHED'}
    

class MMRT_OT_ClearEnvObjects(bpy.types.Operator):
    bl_idname = get_bl_idname("clear_env_objects")
    bl_label = "Clear Environment Objects"
    bl_description = "Remove all environment objects from the list"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        rec = context.scene.mmrt_recorder

        rec.env_objs.clear()
        rec.env_objs_index = -1

        return {'FINISHED'}

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width=300)

    def draw(self, context):
        layout = self.layout
        rec = context.scene.mmrt_recorder

        layout.label(text="Clear all environment objects?", icon='TRASH')

        count = len(rec.env_objs)
        layout.label(text=f"{count} item(s) will be removed.")

        layout.separator()
        layout.label(text="This action can be undone (Ctrl+Z).", icon='INFO')


class MMRT_OT_ExportEnvironment(bpy.types.Operator):
    bl_idname = get_bl_idname("export_environment")
    bl_label = "Export Environment Only"

    def execute(self, context):
        try:
            ensure_object_mode()
            export_environment(context)
            export_environment_metadata(context)
            self.report({'INFO'}, "Environment exported successfully")
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}

        return {'FINISHED'}
    

# --------------------------------------------------------
#   Start Recording (wraps MMRTRecorder)
# --------------------------------------------------------
class MMRT_OT_StartRecording(bpy.types.Operator):
    bl_idname = get_bl_idname("start_recording")
    bl_label = "Start Recording"
    bl_description = "Start or stop recording animation data"

    def execute(self, context):
        global _active_recorder
        recorder = context.scene.mmrt_recorder
        
        try:
            recorder.interrupted = False
            _active_recorder = MMRTRecorder(context)
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        
        mode = recorder.config.env_export_mode

        if mode != EnvironmentExportMode.OFF:
            try:
                ensure_object_mode()

                if mode == EnvironmentExportMode.AUTO:
                    # Full export (safe + reproducible)
                    env_path = get_recorder_environment_filepath(context)
                    if os.path.exists(env_path):
                        self.report({'INFO'}, "Overwriting existing environment snapshot...")
                    
                    export_environment(context)
                    export_environment_metadata(context)

                elif mode == EnvironmentExportMode.MANUAL:
                    # Validate only (fast + flexible)
                    env_path = get_recorder_environment_filepath(context)

                    if not os.path.exists(env_path):
                        self.report(
                            {'ERROR'},
                            "Environment file not found.\n"
                            "Please click 'Export Environment' first or switch to AUTO mode."
                        )
                        return {'CANCELLED'}

            except Exception as e:
                self.report({'ERROR'}, f"Environment step failed: {e}")
                return {'CANCELLED'}

        if recorder.is_fast_mode:
            def run_fast(recorder: MMRTRecorder=_active_recorder):
                # Process the complete frame range in one callback. Returning
                # between chunks lets Blender refresh the screen every 250
                # frames, which adds overhead in Fast Mode.
                while recorder.update_frame() is not None:
                    pass
                return None

            _active_recorder.set_fast_mode(True)
            _active_recorder.run_pipeline()
            bpy.app.timers.register(run_fast)
            self.report({'INFO'}, "Recording started (fast mode). " + _active_recorder.robot_recording_summary())
        else:
            _active_recorder.run_pipeline()
            bpy.app.timers.register(_active_recorder.update_frame)
            self.report({'INFO'}, "Recording started (real time). " + _active_recorder.robot_recording_summary())
        return {'FINISHED'}


class MMRT_OT_StopRecording(bpy.types.Operator):
    bl_idname = get_bl_idname("stop_recording")
    bl_label = "Stop Recording"
    bl_description = "Stop recording early and decide whether to save data"

    def execute(self, context):
        global _active_recorder
        recorder = context.scene.mmrt_recorder

        if not recorder.is_running:
            self.report({'WARNING'}, "No active recording to stop.")
            return {'CANCELLED'}
        recorder.interrupted = True

        # Popup with Yes/No
        OPS.invoke("stop_recording_dialog")
        return {'FINISHED'}


class MMRT_OT_StopRecordingDialog(bpy.types.Operator):
    bl_idname = get_bl_idname("stop_recording_dialog")
    bl_label = "Save Recording?"
    save: bpy.props.BoolProperty()

    # Callback for popup
    def save_or_discard(self, context_popup):
        global _active_recorder
        if self.save:
            exported = _active_recorder._export_data()
            _active_recorder.set_fast_mode(False)
            if exported:
                self.report({'INFO'}, "Recording stopped and data saved.")
            else:
                self.report({'ERROR'}, "Recording stopped, but bundle export failed.")
        else:
            _active_recorder.set_fast_mode(False)
            self.report({'INFO'}, "Recording stopped, data discarded.")

        _active_recorder = None
        return {'FINISHED'}
    
    def execute(self, context):
        return self.save_or_discard(context)

    def invoke(self, context, event):
        wm = context.window_manager
        return wm.invoke_props_dialog(self)

    def draw(self, context):
        layout = self.layout
        layout.label(text="Do you want to save recorded data?")
        row = layout.row()
        row.prop(self, "save", text="Save data")


class MMRT_OT_AutoIdentifyScene(bpy.types.Operator):
    bl_idname = get_bl_idname("auto_identify_scene")
    bl_label = "Auto-Identify Scene"
    bl_description = "Add visible armatures as robots and supported geometry as environment objects"

    def execute(self, context):
        recorder = context.scene.mmrt_recorder
        view_layer = context.view_layer

        include_hidden = recorder.config.include_hidden

        visible_objects = [
            ob for ob in view_layer.objects
            if (include_hidden or not ob.hide_viewport)
            and not ob.get(MESH_SOURCE_PATH_PROP)
        ]

        existing_objs = {item.obj for item in recorder.objs if item.obj}
        existing_robots = {item.obj for item in recorder.robots if item.obj}
        existing_envs = {item.obj for item in recorder.env_objs if item.obj}
        discovered_robots = existing_robots | {
            ob for ob in visible_objects
            if ob.type in SceneRecognitionConfig.ROBOT_TYPES
        }
        added_robots = 0
        added_env = 0
        cfg = SceneRecognitionConfig
        for ob in visible_objects:
            # Skip duplicates
            if ob in existing_objs or ob in existing_robots or ob in existing_envs:
                continue

            if ob.type in cfg.ROBOT_TYPES:
                robot = recorder.robots.add()
                robot.obj = ob
                _populate_robot_sources(robot, ob)
                existing_robots.add(ob)
                added_robots += 1
                continue

            if ob.type in cfg.ENV_TYPES:
                parent = ob.parent
                while parent is not None and parent not in discovered_robots:
                    parent = parent.parent
                if parent in discovered_robots:
                    continue

                env = recorder.env_objs.add()
                env.obj = ob
                added_env += 1

        self.report(
            {'INFO'},
            f"Added {added_robots} robots and {added_env} environment objects"
        )
        return {'FINISHED'}

# --------------------------------------------------------
#   Flag Popup + Add Marker
# --------------------------------------------------------
def _copy_event_values(source, target):
    """Copy the event settings shared by the popup and nested event rows."""
    target.flag_type = source.flag_type
    target.custom_flag_name = source.custom_flag_name
    target.io_delay = source.io_delay
    for bit in range(8):
        setattr(target, f"io{bit}", getattr(source, f"io{bit}"))

    for led in (1, 2):
        for suffix in (
            "is_send", "mode", "color_preset", "color", "progress", "extra"
        ):
            name = f"led{led}_{suffix}"
            setattr(target, name, getattr(source, name))
    target.forced_wp_name = source.forced_wp_name


def _backup_event(event):
    code = FlagTypes.from_name(event.flag_type)
    handler = HANDLER_REGISTRY.get(code)
    return {
        "flag_type": event.flag_type,
        "custom_flag_name": event.custom_flag_name,
        "data": handler.backup_data(event) if handler else {},
    }


def _backup_marker(item):
    nested = getattr(item, "events", None)
    if nested and len(nested):
        return [_backup_event(event) for event in nested]
    return []


def _marker_label(marker_name, uid):
    """Return the editable label portion of ``label_uid``."""
    parts = marker_name.rsplit("_", 1)
    if len(parts) == 2 and parts[1].lower() == uid.lower():
        return parts[0] or "Marker"
    return marker_name or "Marker"


def _restore_event(event, saved):
    event.flag_type = saved.get("flag_type", FlagTypes.CUSTOM.name)
    event.custom_flag_name = saved.get("custom_flag_name", "")
    code = FlagTypes.from_name(event.flag_type)
    handler = HANDLER_REGISTRY.get(code)
    if handler:
        handler.restore_data(event, saved.get("data", {}))


class MMRT_OT_AddFlagPopup(bpy.types.Operator):
    bl_idname = get_bl_idname("add_flag_popup")
    bl_label = "Add Marker"

    marker_uid: bpy.props.StringProperty(
        name="Marker UID",
        description="Append this event to an existing marker group",
        default="",
        options={'HIDDEN'},
    )
    use_selected_marker: bpy.props.BoolProperty(
        name="Use Selected Marker",
        description="Use the marker selected in the list instead of the current timeline frame",
        default=False,
        options={'HIDDEN'},
    )
    add_initial_event: bpy.props.BoolProperty(
        name="Add Initial Event",
        description="Attach the selected event to the marker",
        default=True,
    )

    def execute(self, context):
        scene = context.scene
        marker_pg = scene.mmrt_marker_draft
        if self.add_initial_event and marker_pg.flag_type == FlagTypes.CUSTOM.name:
            flag = marker_pg.custom_flag_name.strip().upper()
            if not flag:
                self.report({'ERROR'}, "Custom flag name required")
                return {'CANCELLED'}
        else:
            flag = marker_pg.flag_type

        selected_marker = next(
            (item for item in scene.mmrt_markers if item.uid == self.marker_uid),
            None,
        )
        if not self.use_selected_marker:
            selected_marker = None
        frame = selected_marker.frame if selected_marker else scene.frame_current
        marker_existed = selected_marker is not None or any(
            marker.frame == frame for marker in scene.timeline_markers
        )

        # Rebuild first so markers added manually in the timeline are available
        # as groups. Adding another event at the same frame reuses that group.
        OPS.exec("refresh_marker_list")
        marker = next((item for item in scene.mmrt_markers if item.frame == frame), None)
        if marker is None:
            uid = uuid.uuid4().hex[:6]
            scene.timeline_markers.new(name=f"Marker_{uid}", frame=frame)
            OPS.exec("refresh_marker_list")
            marker = next((item for item in scene.mmrt_markers if item.frame == frame), None)

        if marker is None:
            self.report({'ERROR'}, "Could not create a timeline marker")
            return {'CANCELLED'}

        if not self.add_initial_event:
            if marker_existed:
                self.report({'WARNING'}, "A marker already exists at this frame; choose an event to add")
                return {'CANCELLED'}
            marker.events_index = -1
            self.report({'INFO'}, f"Added empty marker at frame {frame}")
            return {'FINISHED'}

        event = marker.events.add()
        _copy_event_values(marker_pg, event)
        marker.events_index = len(marker.events) - 1

        self.report({'INFO'}, f"Added flag {flag} at frame {frame}")
        return {'FINISHED'}


    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        marker_pg = scene.mmrt_marker_draft

        layout.prop(self, "add_initial_event")
        if not self.add_initial_event:
            layout.label(text="Creates a placeholder marker with no events.", icon='INFO')
            return

        layout.prop(marker_pg, "flag_type")

        if marker_pg.flag_type == FlagTypes.IO_CONTROL.name:
            row = layout.row(align=True)
            for bit in range(4):
                row.prop(marker_pg, f"io{bit}", text=str(bit), toggle=True)

            row = layout.row(align=True)
            for bit in range(4, 8):
                row.prop(marker_pg, f"io{bit}", text=str(bit), toggle=True)

            layout.prop(marker_pg, "io_delay", text="Delay (ms)")
            layout.label(text=f"Raw Byte: {marker_pg.io_byte():08b}")
        
        if marker_pg.flag_type == FlagTypes.LED_CONTROL.name:
            COLOR_MODES = {
                LEDModes.SET.name,
                LEDModes.FADE.name,
                LEDModes.SPIN.name,
                LEDModes.HEARTBEAT.name,
                LEDModes.BPM.name,
                LEDModes.PROGRESS.name,
            }

            EXTRA_MODES = {
                LEDModes.FADE.name,
                LEDModes.SPIN.name,
                LEDModes.HEARTBEAT.name,
                LEDModes.BPM.name,
                LEDModes.RANDOM.name,
            }

            col = layout.column(align=True)

            # ===================== Row 1: checkboxes =====================
            r = col.row(align=True)
            r.prop(marker_pg, "led1_is_send", text="LED 1")
            r.prop(marker_pg, "led2_is_send", text="LED 2")

            # ===================== Row 2: LED 1 =====================
            if marker_pg.led1_is_send:
                r = col.row(align=True)
                r.label(text="L1")
                r.prop(marker_pg, "led1_mode", text="")

                if marker_pg.led1_mode in COLOR_MODES:
                    r.prop(marker_pg, "led1_color_preset", text="")
                    if marker_pg.led1_color_preset == LEDColors.CUSTOM.name:
                        r.prop(marker_pg, "led1_color", text="")

                if marker_pg.led1_mode == LEDModes.PROGRESS.name:
                    r.prop(marker_pg, "led1_progress", text="%", slider=True)

                if marker_pg.led1_mode in EXTRA_MODES:
                    r.prop(marker_pg, "led1_extra", text="")

            # ===================== Row 3: LED 2 =====================
            if marker_pg.led2_is_send:
                r = col.row(align=True)
                r.label(text="L2")
                r.prop(marker_pg, "led2_mode", text="")

                if marker_pg.led2_mode in COLOR_MODES:
                    r.prop(marker_pg, "led2_color_preset", text="")
                    if marker_pg.led2_color_preset == LEDColors.CUSTOM.name:
                        r.prop(marker_pg, "led2_color", text="")

                if marker_pg.led2_mode == LEDModes.PROGRESS.name:
                    r.prop(marker_pg, "led2_progress", text="%", slider=True)

                if marker_pg.led2_mode in EXTRA_MODES:
                    r.prop(marker_pg, "led2_extra", text="")

        
        if marker_pg.flag_type == FlagTypes.FORCED_WP_NAME.name:
            layout.prop(marker_pg, "forced_wp_name", text="Waypoint Name")

        if marker_pg.flag_type == FlagTypes.CUSTOM.name:
            layout.prop(marker_pg, "custom_flag_name")


# ============================================================
#   MARKER MONITOR OPERATORS
# ============================================================
class MMRT_OT_AddEpisode(bpy.types.Operator):
    bl_idname = get_bl_idname("add_episode")
    bl_label = "Add Episode"
    bl_description = "Add a non-overlapping episode after the selected episode"

    def execute(self, context):
        scene = context.scene
        recorder = scene.mmrt_recorder
        used_names = {
            (item.episode_name.strip() or item.episode_id.strip()).casefold()
            for item in recorder.episodes
        }
        used_ids = {item.episode_id.strip().casefold() for item in recorder.episodes}
        index = 0
        episode_name = f"Episode_{index}"
        while episode_name.casefold() in used_names:
            index += 1
            episode_name = f"Episode_{index}"
        episode_id = f"episode_{index}"
        while episode_id.casefold() in used_ids:
            index += 1
            episode_id = f"episode_{index}"

        ordered = sorted(recorder.episodes, key=lambda item: item.start_frame)
        cursor = scene.frame_start
        free_range = None
        for existing in ordered:
            if cursor <= existing.start_frame - 1:
                free_range = (cursor, existing.start_frame - 1)
                break
            cursor = max(cursor, existing.end_frame + 1)
        if free_range is None and cursor <= scene.frame_end:
            free_range = (cursor, scene.frame_end)
        if free_range is None:
            self.report({'WARNING'}, "There is no free frame range for another episode.")
            return {'CANCELLED'}

        selected_index = recorder.episodes_index
        if 0 <= selected_index < len(recorder.episodes):
            insert_index = selected_index + 1
        else:
            insert_index = len(recorder.episodes)

        item = recorder.episodes.add()
        item.episode_name = episode_name
        item.episode_id = episode_id
        item.start_frame, item.end_frame = free_range

        new_index = len(recorder.episodes) - 1
        if new_index != insert_index:
            recorder.episodes.move(new_index, insert_index)
        recorder.episodes_index = insert_index
        self.report({'INFO'}, f"Added episode '{episode_name}'. Set its name and ID.")
        return {'FINISHED'}


class MMRT_OT_SetEpisodeFrameToCurrent(bpy.types.Operator):
    bl_idname = get_bl_idname("set_episode_frame_to_current")
    bl_label = "Set Episode Frame to Current"
    bl_description = "Set this episode boundary to the current timeline frame"

    boundary: bpy.props.EnumProperty(
        name="Boundary",
        items=[
            (RecorderOperatorOptions.FRAME_START, "Start", "Set the original start frame"),
            (RecorderOperatorOptions.FRAME_END, "End", "Set the original end frame"),
        ],
        options={'HIDDEN'},
    )

    def execute(self, context):
        recorder = context.scene.mmrt_recorder
        index = recorder.episodes_index
        if not 0 <= index < len(recorder.episodes):
            self.report({'WARNING'}, "Select an episode first.")
            return {'CANCELLED'}

        episode = recorder.episodes[index]
        if self.boundary == RecorderOperatorOptions.FRAME_START:
            episode.start_frame = context.scene.frame_current
        else:
            episode.end_frame = context.scene.frame_current
        return {'FINISHED'}


class MMRT_OT_RemoveEpisode(bpy.types.Operator):
    bl_idname = get_bl_idname("remove_episode")
    bl_label = "Remove Episode"
    index: IntProperty(default=-1)

    def execute(self, context):
        recorder = context.scene.mmrt_recorder
        if 0 <= self.index < len(recorder.episodes):
            recorder.episodes.remove(self.index)
            recorder.episodes_index = min(self.index, len(recorder.episodes) - 1)
        return {'FINISHED'}


class MMRT_OT_MoveEpisode(bpy.types.Operator):
    bl_idname = get_bl_idname("move_episode")
    bl_label = "Move Episode"
    bl_description = "Move the active episode in the behavior-tree order"

    direction: bpy.props.EnumProperty(
        name="Direction",
        items=[
            (RecorderOperatorOptions.MOVE_UP, "Move Up", "Move the active episode earlier in the list"),
            (RecorderOperatorOptions.MOVE_DOWN, "Move Down", "Move the active episode later in the list"),
        ],
        default=RecorderOperatorOptions.MOVE_UP,
    )

    def execute(self, context):
        recorder = context.scene.mmrt_recorder
        index = recorder.episodes_index
        if not 0 <= index < len(recorder.episodes):
            self.report({'WARNING'}, "Select an episode first.")
            return {'CANCELLED'}

        new_index = index - 1 if self.direction == RecorderOperatorOptions.MOVE_UP else index + 1
        if not 0 <= new_index < len(recorder.episodes):
            self.report({'WARNING'}, "The episode is already at that end of the list.")
            return {'CANCELLED'}

        recorder.episodes.move(index, new_index)
        recorder.episodes_index = new_index
        return {'FINISHED'}


class MMRT_OT_ConfigureEpisodeBatch(bpy.types.Operator):
    bl_idname = get_bl_idname("configure_episode_batch")
    bl_label = "Configure Episode Batch"
    bl_description = "Apply the batch episode start settings to marked or all episodes"

    target: bpy.props.EnumProperty(
        name="Target",
        items=[
            (RecorderOperatorOptions.TARGET_BATCH, "Marked Episodes", "Apply settings to episodes marked for batch editing"),
            (RecorderOperatorOptions.TARGET_ALL, "All Episodes", "Apply settings to every episode"),
        ],
        default=RecorderOperatorOptions.TARGET_BATCH,
    )

    def execute(self, context):
        recorder = context.scene.mmrt_recorder
        if self.target == RecorderOperatorOptions.TARGET_BATCH:
            targets = [episode for episode in recorder.episodes if episode.batch_selected]
            target_label = "marked episodes"
        else:
            targets = list(recorder.episodes)
            target_label = "all episodes"

        if not targets:
            self.report({'WARNING'}, f"There are no {target_label} to configure.")
            return {'CANCELLED'}

        for episode in targets:
            episode.start_with_waypoint = recorder.episode_batch_start_with_waypoint
            episode.start_waypoint_mode = recorder.episode_batch_waypoint_mode

        self.report(
            {'INFO'},
            f"Configured {len(targets)} {target_label}.",
        )
        return {'FINISHED'}


class MMRT_OT_OffsetEpisodes(bpy.types.Operator):
    bl_idname = get_bl_idname("offset_episodes")
    bl_label = "Offset Episodes"
    bl_description = "Shift one or more episode ranges by the configured number of frames"

    target: bpy.props.EnumProperty(
        name="Target",
        items=[
            (RecorderOperatorOptions.TARGET_ACTIVE, "Active Episode", "Offset the currently active episode"),
            (RecorderOperatorOptions.TARGET_BATCH, "Marked Episodes", "Offset episodes marked for batch editing"),
            (RecorderOperatorOptions.TARGET_ALL, "All Episodes", "Offset every episode"),
        ],
        default=RecorderOperatorOptions.TARGET_ACTIVE,
    )

    def execute(self, context):
        recorder = context.scene.mmrt_recorder
        offset = int(recorder.episode_offset)

        if offset == 0:
            self.report({'WARNING'}, "Episode offset is zero; no ranges were changed.")
            return {'CANCELLED'}

        if self.target == RecorderOperatorOptions.TARGET_ACTIVE:
            if not 0 <= recorder.episodes_index < len(recorder.episodes):
                self.report({'WARNING'}, "Select an active episode first.")
                return {'CANCELLED'}
            targets = [recorder.episodes[recorder.episodes_index]]
            target_label = "active episode"
        elif self.target == RecorderOperatorOptions.TARGET_BATCH:
            targets = [episode for episode in recorder.episodes if episode.batch_selected]
            if not targets:
                self.report({'WARNING'}, "Mark at least one episode for the batch offset.")
                return {'CANCELLED'}
            target_label = "marked episodes"
        else:
            targets = list(recorder.episodes)
            if not targets:
                self.report({'WARNING'}, "There are no episodes to offset.")
                return {'CANCELLED'}
            target_label = "all episodes"

        # Blender integer properties use signed 32-bit values. Validate every
        # result before changing anything so a failed offset is atomic.
        int_min = -2147483648
        int_max = 2147483647
        updated_ranges = []
        for episode in targets:
            start_frame = episode.start_frame + offset
            end_frame = episode.end_frame + offset
            if not int_min <= start_frame <= int_max or not int_min <= end_frame <= int_max:
                self.report({'ERROR'}, "The offset would exceed Blender's frame-number limits.")
                return {'CANCELLED'}
            updated_ranges.append((episode, start_frame, end_frame))

        for episode, start_frame, end_frame in updated_ranges:
            episode.start_frame = start_frame
            episode.end_frame = end_frame

        self.report(
            {'INFO'},
            f"Offset {len(updated_ranges)} {target_label} by {offset:+d} frame(s).",
        )
        return {'FINISHED'}


class MMRT_OT_RefreshMarkerList(bpy.types.Operator):
    bl_idname = get_bl_idname("refresh_marker_list")
    bl_label = "Refresh Marker List"

    def execute(self, context):
        scene = context.scene

        # Back up nested events, and also understand the old one-event shape.
        old_data = {}
        old_by_frame = {}
        for item in scene.mmrt_markers:
            if item.uid:
                old_data[item.uid] = {
                    "uid": item.uid,
                    "events": _backup_marker(item),
                    "empty_placeholder": bool(item.marker_name),
                }
                old_by_frame.setdefault(item.frame, []).append(old_data[item.uid])

        scene.mmrt_markers.clear()

        groups = {}
        for m in sorted(scene.timeline_markers, key=lambda x: x.frame):
            flag_type, custom_name, uid = FlagTypes.parse_marker_name(m.name)

            if not uid:
                uid = uuid.uuid4().hex[:6]
                m.name = f"{m.name}_{uid}"

            group = groups.setdefault(m.frame, {
                "marker": m,
                "uid": uid,
                "events": [],
                "marker_name": _marker_label(m.name, uid),
            })
            saved = old_data.get(uid)
            if saved is None:
                candidates = old_by_frame.get(m.frame, [])
                if candidates and not group["events"]:
                    saved = candidates[0]
                    group["uid"] = saved["uid"]
                    m.name = f"{group['marker_name']}_{group['uid']}"

            if saved and saved["events"]:
                group["events"].extend(saved["events"])
            elif (
                not (saved and saved["empty_placeholder"])
                and not group["events"]
                and m.name.rsplit("_", 1)[0].upper() != MarkerNaming.DEFAULT_BASENAME
            ):
                group["events"].append({
                    "flag_type": flag_type,
                    "custom_flag_name": custom_name,
                    "data": {},
                })

            # A marker group is the unit shown in the list, so collapse any
            # legacy duplicate Blender markers that share a frame.
            if group["marker"] != m:
                scene.timeline_markers.remove(m)

        for group in groups.values():
            new_item = scene.mmrt_markers.add()
            marker = group["marker"]
            new_item.marker_name = group["marker_name"]
            new_item.frame = marker.frame
            new_item.uid = group["uid"]
            new_item.name = marker.name
            for saved_event in group["events"]:
                event = new_item.events.add()
                _restore_event(event, saved_event)

        return {'FINISHED'}
    

class MMRT_OT_MarkerJump(bpy.types.Operator):
    bl_idname = get_bl_idname("marker_jump")
    bl_label = "Go to Marker"

    frame: bpy.props.IntProperty()

    def execute(self, context):
        context.scene.frame_set(self.frame)
        return {'FINISHED'}


class MMRT_OT_MarkerDelete(bpy.types.Operator):
    bl_idname = get_bl_idname("marker_delete")
    bl_label = "Delete Marker"

    frame: bpy.props.IntProperty()
    marker_name: bpy.props.StringProperty() 
    marker_uid: bpy.props.StringProperty()

    def execute(self, context):
        markers = context.scene.timeline_markers
        target = None
        for m in markers:
            _, _, uid = FlagTypes.parse_marker_name(m.name)
            if m.frame == self.frame and (
                m.name == self.marker_name or uid == self.marker_uid
            ):
                target = m
                break

        if target:
            markers.remove(target)

        OPS.exec("refresh_marker_list")
        return {'FINISHED'}


class MMRT_OT_MarkerEventDelete(bpy.types.Operator):
    bl_idname = get_bl_idname("marker_event_delete")
    bl_label = "Remove Marker Event"

    marker_uid: bpy.props.StringProperty()
    event_index: bpy.props.IntProperty()

    def execute(self, context):
        markers = context.scene.mmrt_markers
        marker = next((item for item in markers if item.uid == self.marker_uid), None)
        if marker is None or not 0 <= self.event_index < len(marker.events):
            return {'CANCELLED'}

        marker.events.remove(self.event_index)
        marker.events_index = min(marker.events_index, len(marker.events) - 1)
        return {'FINISHED'}


# --------------------------------------------------------
#   Register / Unregister for Operator classes
# --------------------------------------------------------
classes = (
    MMRT_OT_AddObject,
    MMRT_OT_AddEpisode,
    MMRT_OT_SetEpisodeFrameToCurrent,
    MMRT_OT_RemoveEpisode,
    MMRT_OT_MoveEpisode,
    MMRT_OT_ConfigureEpisodeBatch,
    MMRT_OT_OffsetEpisodes,
    MMRT_OT_AddRobot,
    MMRT_OT_RemoveRobot,
    MMRT_OT_ClearRobots,
    MMRT_OT_RefreshRobotJoints,
    MMRT_OT_AutoDetectRobotAxes,
    MMRT_OT_RemoveObject,
    MMRT_OT_ClearObjects,
    MMRT_OT_RefreshJoints,
    MMRT_OT_AutoDetectAxes,
    MMRT_OT_PrintCurrentFrame,
    MMRT_OT_PrintRelativeTransforms,
    MMRT_OT_PrintRelativeTransformArray,
    MMRT_OT_AddEnvObject,
    MMRT_OT_AddEnvCollisionLink,
    MMRT_OT_RemoveEnvCollisionLink,
    MMRT_OT_AddEnvCollisionMovegroup,
    MMRT_OT_RemoveEnvCollisionMovegroup,
    MMRT_OT_RemoveEnvObject,
    MMRT_OT_ClearEnvObjects,
    MMRT_OT_SetAllTransformSpace,
    MMRT_OT_ExportEnvironment,
    MMRT_OT_StartRecording,
    MMRT_OT_StopRecording,
    MMRT_OT_StopRecordingDialog,
    MMRT_OT_AutoIdentifyScene,
    MMRT_OT_AddFlagPopup,
    MMRT_OT_RefreshMarkerList,
    MMRT_OT_MarkerJump,
    MMRT_OT_MarkerDelete,
    MMRT_OT_MarkerEventDelete,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
