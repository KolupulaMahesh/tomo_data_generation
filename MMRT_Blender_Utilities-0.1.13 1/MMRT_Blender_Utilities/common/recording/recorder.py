import bpy
import math
import mathutils
import os
import datetime
import struct
import logging
import json
import xml.etree.ElementTree as ET
from ..blender.utils import show_report
from ..exporting.robot import export_robot_bundle
from ..config import (
    EnvironmentExportMode,
    RecordTransformSpace,
    SceneRecognitionConfig,
)
from ...processes.mmrt.definitions import (
    MDataLabel,
    EpisodeConfig,
    MeasurementUnits as MeasurementConstants,
    OutputSuffixes,
    RotationFormat,
)
from ..marker_utils import MarkerEvents, collect_marker_events, write_marker_events
from ..robot_description.storage import ROBOT_MESH_COLLECTION_PROP
from ..system.paths import normalize_output_dir
from .frame_domain import (
    EpisodeRangeStatus,
    episode_range_status,
    get_frame_domain,
    remap_frame,
)

logger = logging.getLogger(__name__)


class RecordDescriptorTypes:
    ACTIVE_BONE = "active_bone"
    BONE = "bone"


class MMRTRecorder:
    def __init__(self, context: bpy.types.Context):
        self._context = context
        self.scene = context.scene

        # === Access main recorder PG ===
        self.recorder = self.scene.mmrt_recorder
        self._cfg = self.recorder.config

        # --- Time Stretching Setup ---
        domain = get_frame_domain(self.scene)
        self.map_old = domain["map_old"]
        self.map_new = domain["map_new"]
        self._stretch_factor = domain["stretch_factor"]

        # Animation params
        self.frame_rate = self.scene.render.fps / self.scene.render.fps_base
        self.delay = 1.0 / self.frame_rate

        # Keep source bounds for animation evaluation and use the stretched
        # bounds for the recorder/output domain.
        self.source_start_frame = domain["source_start"]
        self.source_end_frame = domain["source_end"]
        self.start_frame = domain["stretched_start"]
        self.end_frame = domain["stretched_end"]
        self.frame_tracking = {
            MDataLabel.SCHEMA_VERSION_KEY: MDataLabel.SCHEMA_VERSION,
            MDataLabel.SCENE_FRAME_START_KEY: self.source_start_frame,
            MDataLabel.SCENE_FRAME_END_KEY: self.source_end_frame,
            MDataLabel.SAMPLE_FRAME_START_KEY: self.start_frame,
            MDataLabel.SAMPLE_FRAME_END_KEY: self.end_frame,
            MDataLabel.FRAME_MAP_OLD_KEY: self.map_old,
            MDataLabel.FRAME_MAP_NEW_KEY: self.map_new,
            MDataLabel.SAMPLE_FRAME_DOMAIN_KEY: MDataLabel.SAMPLE_FRAME_DOMAIN_SCENE_FRAME_SET,
        }

        # The exported recording is dense in its stretched frame domain.
        # Blender's scene frame step is a source-timeline playback setting;
        # applying it here would create gaps in the implicit MMRT frame list.
        self.frame_step = 1
        
        self._frame_id = self.start_frame
        self._fast_mode = False

        self._wm = context.window_manager
        self._total_frames = (self.end_frame - self.start_frame) // self.frame_step + 1

        # === Build descriptors ===
        self.descriptors = []
        self.robot_label_bases = set()
        self.active_label_bases = set()

        self.output_dir = normalize_output_dir(self._cfg.output_dir)
        self.base_filename = self._cfg.base_filename
        self._basefilepath = os.path.join(self.output_dir, self.base_filename)
        self._mmrt_yaml_path = self._basefilepath + OutputSuffixes.YAML_RECORDING
        self._mmrt_bin_path = self._basefilepath + OutputSuffixes.BINARY_RECORDING
        self._mmrt_environment_path = self._basefilepath + OutputSuffixes.ENVIRONMENT_JSON
    
        self.is_binary = self._cfg.export_binary

        # Buffers
        self.records = []
        self.num_recorded_items = 0
        self.export_succeeded = None
        self.marker_events = MarkerEvents()
        self._sync_marker_event_attributes()

        self._need_warmup = False
        self._validate_episodes()

    # ======================================================================
    # Time Stretching Map Helpers
    # ======================================================================
    def _remap_frame_forward(self, frame: float) -> int:
        """Map an original frame to the stretched recorder domain."""
        return remap_frame(
            frame,
            stretch_factor=self._stretch_factor,
        )

    # ======================================================================
    # Warmup logic
    # ======================================================================
    def _need_warmup_check(self):
        armatures = [item.obj for item in self.recorder.objs]
        armatures.extend(item.obj for item in self.recorder.robots)
        for ob in armatures:
            if ob and ob.type == SceneRecognitionConfig.ROBOT_TYPE:
                if ob.pose.ik_solver == 'ITASC':
                    self._need_warmup = True

    def _warmup_depsgraph(self):
        # Warm up in the same output domain used by the recording loop.  A
        # source-domain warmup would be remapped by Blender and would stop
        # before the actual stretched start frame.
        if self.start_frame <= 0:
            return

        self._wm.progress_begin(0, self.start_frame)
        depsgraph = bpy.context.evaluated_depsgraph_get()

        for f in range(0, self.start_frame):
            self.scene.frame_set(f)
            bpy.context.view_layer.update()
            depsgraph.update()
            self._wm.progress_update(f)

        self._wm.progress_end()
    
    def prepare(self):
        """Build descriptors + warmup (shared by recording & single-frame capture)."""
        self.descriptors.clear()
        self.robot_label_bases.clear()
        self.active_label_bases.clear()
        self._validate_robot_sources()

        for item in self.recorder.objs:
            ob = item.obj
            if not ob:
                continue

            if ob.type == SceneRecognitionConfig.ROBOT_TYPE:
                for bone_item in item.joints:
                    if not bone_item.record:
                        continue

                    pose_bone = ob.pose.bones.get(bone_item.name)
                    if not pose_bone:
                        continue

                    self.descriptors.append({
                        "type": RecordDescriptorTypes.ACTIVE_BONE,
                        "obj_name": ob.name,
                        "bone_name": bone_item.name,
                        "record_location": bone_item.record_location,
                        "record_rotation": bone_item.record_rotation,
                        "space": item.space,
                        "custom_ref": item.custom_ref.name if item.custom_ref else None,
                        "label": f"{ob.name}.{bone_item.name}",
                        "section": "active_objects",
                    })
                    self.active_label_bases.add(f"{ob.name}.{bone_item.name}")

            else:
                if item.record_location or item.record_rotation:
                    self.descriptors.append({
                        "type": "obj",
                        "obj_name": ob.name,
                        "record_location": item.record_location,
                        "record_rotation": item.record_rotation,
                        "space": item.space,
                        "custom_ref": item.custom_ref.name if item.custom_ref else None,
                        "label": ob.name,
                        "section": "active_objects",
                    })
                    self.active_label_bases.add(ob.name)

        for robot in self.recorder.robots:
            ob = robot.obj
            if not ob or ob.type != SceneRecognitionConfig.ROBOT_TYPE:
                continue
            # The robot transform is always part of the recording.
            self.descriptors.append({
                "type": "obj", "obj_name": ob.name,
                "record_location": True, "record_rotation": True,
                "space": RecordTransformSpace.WORLD, "custom_ref": None,
                "label": ob.name,
                "section": "robots",
            })
            self.robot_label_bases.add(ob.name)
            for joint in robot.joints:
                if not joint.record:
                    continue
                if not joint.bone_name or ob.pose.bones.get(joint.bone_name) is None:
                    logger.warning(
                        "Robot '%s': URDF joint '%s' has no paired bone and will not be recorded.",
                        ob.name, joint.joint_name,
                    )
                    continue
                pose_bone = ob.pose.bones[joint.bone_name]
                bone = pose_bone.bone
                rel_matrix = (
                    bone.parent.matrix_local.inverted() @ bone.matrix_local
                    if bone.parent else bone.matrix_local
                )
                quat = rel_matrix.to_quaternion()
                joint.initial_quat = (quat.w, quat.x, quat.y, quat.z)
                self.descriptors.append({
                    "type": RecordDescriptorTypes.BONE,
                    "obj_name": ob.name,
                    "bone_name": joint.bone_name,
                    "axis_idx": {"X": 0, "Y": 1, "Z": 2}[joint.rot_axis],
                    "sign": -1.0 if joint.direction else 1.0,
                    "initial_quat": joint.initial_quat,
                    "initial_quat_inverse": quat.inverted(),
                    "offset_rad": joint.offset,
                    "label": joint.joint_name,
                    "section": "robots",
                })
                self.robot_label_bases.add(joint.joint_name)

        if not self.descriptors:
            raise RuntimeError("No bones or transforms selected.")

        self.num_recorded_items = len(self.descriptors)

        self._need_warmup_check()
        if self._need_warmup:
            self._warmup_depsgraph()

    def _validate_robot_sources(self):
        robots = [item for item in self.recorder.robots if item.obj]
        if not robots:
            return
        if any(item.obj.type != SceneRecognitionConfig.ROBOT_TYPE for item in robots):
            raise RuntimeError("Every Recorder robot entry must reference an armature.")
        if any(not item.urdf_text or not item.srdf_text for item in robots):
            raise RuntimeError("Every Recorder robot needs a URDF and SRDF Text datablock.")
        for item in robots:
            expected_collection = str(
                item.urdf_text.get(ROBOT_MESH_COLLECTION_PROP, "")
            ).strip()
            if expected_collection and (
                not item.meshes_collection
                or item.meshes_collection.name != expected_collection
            ):
                raise RuntimeError(
                    f"Robot '{item.obj.name}' must use its associated Original Meshes collection."
                )

        first_urdf = robots[0].urdf_text.as_string()
        first_srdf = robots[0].srdf_text.as_string()
        if any(
            item.urdf_text.as_string() != first_urdf
            or item.srdf_text.as_string() != first_srdf
            for item in robots[1:]
        ):
            raise RuntimeError("All Recorder robots must share the same URDF/SRDF pair.")
        try:
            ET.fromstring(first_urdf)
            ET.fromstring(first_srdf)
        except ET.ParseError as exc:
            raise RuntimeError(f"Stored robot description is invalid XML: {exc}") from exc

    def evaluate_current_frame(self) -> dict:
        """Core evaluation used by both recording loop and print."""
        depsgraph = self._context.evaluated_depsgraph_get()
        frame_data = {}

        # Descriptors often contain several joints from the same armature and
        # may share a custom transform reference. Resolve/evaluate each once
        # for this frame; evaluated datablocks and matrices are frame-specific.
        objects = {}
        evaluated_armatures = {}
        pose_bones = {}
        reference_inverses = {}

        def get_object(name):
            if name not in objects:
                objects[name] = bpy.data.objects.get(name)
            return objects[name]

        def get_pose_bone(obj_name, bone_name):
            cache_key = (obj_name, bone_name)
            if cache_key not in pose_bones:
                armature = evaluated_armatures.get(obj_name)
                if armature is None:
                    obj = get_object(obj_name)
                    armature = obj.evaluated_get(depsgraph) if obj else None
                    evaluated_armatures[obj_name] = armature
                pose_bones[cache_key] = (
                    armature.pose.bones.get(bone_name) if armature else None
                )
            return pose_bones[cache_key]

        def get_reference_inverse(ref):
            if ref.name not in reference_inverses:
                reference_inverses[ref.name] = ref.matrix_world.inverted()
            return reference_inverses[ref.name]

        length_mul = 1000.0 if self._cfg.length_unit == MeasurementConstants.MILLIMETER_UNIT else 1.0

        for desc in self.descriptors:
            if desc["type"] == RecordDescriptorTypes.ACTIVE_BONE:
                ob = get_object(desc["obj_name"])
                if not ob:
                    continue
                pose_b = get_pose_bone(desc["obj_name"], desc["bone_name"])
                if not pose_b:
                    continue
                if desc["space"] == RecordTransformSpace.LOCAL:
                    mat = pose_b.matrix
                elif desc["space"] == RecordTransformSpace.CUSTOM:
                    ref = get_object(desc.get("custom_ref")) if desc.get("custom_ref") else None
                    mat = get_reference_inverse(ref) @ (ob.matrix_world @ pose_b.matrix) if ref else ob.matrix_world @ pose_b.matrix
                else:
                    mat = ob.matrix_world @ pose_b.matrix
                loc = mat.to_translation()
                if desc["record_location"]:
                    frame_data[f"{desc['label']}.location.x"] = round(loc.x * length_mul, MeasurementConstants.MILLIMETER_DECIMAL_PLACES if length_mul != 1.0 else MeasurementConstants.METER_DECIMAL_PLACES)
                    frame_data[f"{desc['label']}.location.y"] = round(loc.y * length_mul, MeasurementConstants.MILLIMETER_DECIMAL_PLACES if length_mul != 1.0 else MeasurementConstants.METER_DECIMAL_PLACES)
                    frame_data[f"{desc['label']}.location.z"] = round(loc.z * length_mul, MeasurementConstants.MILLIMETER_DECIMAL_PLACES if length_mul != 1.0 else MeasurementConstants.METER_DECIMAL_PLACES)
                if desc["record_rotation"]:
                    rot_format = self._cfg.rotation_format
                    if rot_format == RotationFormat.QUATERNION:
                        q = mat.to_quaternion()
                        for axis, value in zip(('w', 'x', 'y', 'z'), (q.w, q.x, q.y, q.z)):
                            frame_data[f"{desc['label']}.rotation.{axis}"] = round(value, 6)
                    elif rot_format == RotationFormat.AXIS_ANGLE:
                        axis, angle = mat.to_quaternion().to_axis_angle()
                        if self._cfg.angle_unit == MeasurementConstants.DEGREE_UNIT:
                            angle = math.degrees(angle)
                        for component in ('x', 'y', 'z'):
                            frame_data[f"{desc['label']}.rotation.axis.{component}"] = round(getattr(axis, component), 6)
                        frame_data[f"{desc['label']}.rotation.angle"] = round(angle, 6)
                    else:
                        euler = mat.to_euler(rot_format)
                        values = (euler.x, euler.y, euler.z)
                        if self._cfg.angle_unit == MeasurementConstants.DEGREE_UNIT:
                            values = tuple(math.degrees(value) for value in values)
                        decimals = (MeasurementConstants.RAD_DECIMAL_PLACES if self._cfg.angle_unit == MeasurementConstants.RADIAN_UNIT
                                    else MeasurementConstants.DEG_DECIMAL_PLACES)
                        for component, value in zip(('x', 'y', 'z'), values):
                            frame_data[f"{desc['label']}.rotation.{component}"] = round(value, decimals)
            elif desc["type"] == RecordDescriptorTypes.BONE:
                ob: bpy.types.Object = get_object(desc["obj_name"])
                if not ob:
                    continue

                pose_b: bpy.types.PoseBone = get_pose_bone(
                    desc["obj_name"], desc["bone_name"]
                )
                if not pose_b:
                    continue

                rel_mat = (
                    pose_b.parent.matrix.inverted() @ pose_b.matrix
                    if pose_b.parent else pose_b.matrix
                )

                current_q: mathutils.Quaternion = rel_mat.to_quaternion()
                delta_q = desc["initial_quat_inverse"] @ current_q

                raw_rad = (
                    delta_q.to_euler("XZY").x if desc["axis_idx"] == 0 else
                    delta_q.to_euler("YZX").y if desc["axis_idx"] == 1 else
                    delta_q.to_euler("ZXY").z
                )

                adjusted: float = (raw_rad - desc["offset_rad"]) * desc["sign"]

                frame_data[desc["label"]] = (
                    round(adjusted, MeasurementConstants.RAD_DECIMAL_PLACES)
                    if self._cfg.angle_unit == MeasurementConstants.RADIAN_UNIT
                    else round(math.degrees(adjusted), MeasurementConstants.DEG_DECIMAL_PLACES)
                )

            else:
                ob: bpy.types.Object = get_object(desc["obj_name"])
                if not ob:
                    continue

                space = desc["space"]
                if space == RecordTransformSpace.LOCAL:
                    mat = ob.matrix_local

                elif space == RecordTransformSpace.CUSTOM:
                    ref_name = desc.get("custom_ref")
                    ref: bpy.types.Object|None = get_object(ref_name) if ref_name else None

                    if ref is None or ref == ob:
                        mat = ob.matrix_world
                    else:
                        try:
                            mat = get_reference_inverse(ref) @ ob.matrix_world
                        except:
                            mat = ob.matrix_world
                else:  # WORLD
                    mat = ob.matrix_world

                loc: mathutils.Vector = mat.to_translation()
                if desc["record_location"]:
                    if self._cfg.length_unit == MeasurementConstants.MILLIMETER_UNIT:
                        frame_data[f"{desc['label']}.location.x"] = round(loc.x * length_mul, MeasurementConstants.MILLIMETER_DECIMAL_PLACES)
                        frame_data[f"{desc['label']}.location.y"] = round(loc.y * length_mul, MeasurementConstants.MILLIMETER_DECIMAL_PLACES)
                        frame_data[f"{desc['label']}.location.z"] = round(loc.z * length_mul, MeasurementConstants.MILLIMETER_DECIMAL_PLACES)
                    else:
                        frame_data[f"{desc['label']}.location.x"] = round(loc.x * length_mul, MeasurementConstants.METER_DECIMAL_PLACES)
                        frame_data[f"{desc['label']}.location.y"] = round(loc.y * length_mul, MeasurementConstants.METER_DECIMAL_PLACES)
                        frame_data[f"{desc['label']}.location.z"] = round(loc.z * length_mul, MeasurementConstants.METER_DECIMAL_PLACES)
                
                rot_format = self._cfg.rotation_format
                if desc["record_rotation"]:
                    if rot_format == RotationFormat.QUATERNION:
                        q = mat.to_quaternion()
                        frame_data[f"{desc['label']}.rotation.w"] = round(q.w, 6)
                        frame_data[f"{desc['label']}.rotation.x"] = round(q.x, 6)
                        frame_data[f"{desc['label']}.rotation.y"] = round(q.y, 6)
                        frame_data[f"{desc['label']}.rotation.z"] = round(q.z, 6)

                    elif rot_format == RotationFormat.AXIS_ANGLE:
                        q = mat.to_quaternion()
                        axis, angle = q.to_axis_angle()

                        if self._cfg.angle_unit == MeasurementConstants.DEGREE_UNIT:
                            angle = math.degrees(angle)

                        frame_data[f"{desc['label']}.rotation.axis.x"] = round(axis.x, 6)
                        frame_data[f"{desc['label']}.rotation.axis.y"] = round(axis.y, 6)
                        frame_data[f"{desc['label']}.rotation.axis.z"] = round(axis.z, 6)
                        frame_data[f"{desc['label']}.rotation.angle"] = round(angle, 6)

                    else:
                        euler = mat.to_euler(rot_format)

                        if self._cfg.angle_unit == MeasurementConstants.RADIAN_UNIT:
                            frame_data[f"{desc['label']}.rotation.x"] = round(euler.x, MeasurementConstants.RAD_DECIMAL_PLACES)
                            frame_data[f"{desc['label']}.rotation.y"] = round(euler.y, MeasurementConstants.RAD_DECIMAL_PLACES)
                            frame_data[f"{desc['label']}.rotation.z"] = round(euler.z, MeasurementConstants.RAD_DECIMAL_PLACES)
                        else:
                            frame_data[f"{desc['label']}.rotation.x"] = round(math.degrees(euler.x), MeasurementConstants.DEG_DECIMAL_PLACES)
                            frame_data[f"{desc['label']}.rotation.y"] = round(math.degrees(euler.y), MeasurementConstants.DEG_DECIMAL_PLACES)
                            frame_data[f"{desc['label']}.rotation.z"] = round(math.degrees(euler.z), MeasurementConstants.DEG_DECIMAL_PLACES)
        return frame_data

    def run_pipeline(self):
        self._validate_episodes()
        self.prepare()
        self._gather_flags()
        self._wm.progress_begin(0, self._total_frames)

    def robot_recording_summary(self):
        """Describe actual prepared robot channels, independent of selection."""
        counts = {}
        for desc in self.descriptors:
            if desc['section'] != 'robots':
                continue
            counts.setdefault(desc['obj_name'], 0)
            if desc['type'] == RecordDescriptorTypes.BONE:
                counts[desc['obj_name']] += 1
        return ', '.join(f'{name}: {count} joints' for name, count in counts.items())

    # ======================================================================
    # Flags / Events (With Scaled Marker Frames)
    # ======================================================================
    def _sync_marker_event_attributes(self):
        """Keep the recorder's legacy event attributes as compatible aliases."""
        self.flags = self.marker_events.flags
        self.custom_flags = self.marker_events.custom_flags
        self.io_events = self.marker_events.io_events
        self.led_events = self.marker_events.led_events
        self.forced_wp_names = self.marker_events.forced_wp_names

    def _gather_flags(self):
        self.marker_events = collect_marker_events(
            self.scene.mmrt_markers,
            remap_frame=self._remap_frame_forward,
            start_frame=self.start_frame,
            end_frame=self.end_frame,
        )
        self._sync_marker_event_attributes()

    def _validate_episodes(self):
        """Validate the Blender-side episode declaration before recording."""
        episodes = sorted(self.recorder.episodes, key=lambda item: item.start_frame)
        active_episodes = []

        for item in episodes:
            if item.start_frame > item.end_frame:
                episode_id = item.episode_id.strip() or "<unnamed>"
                raise RuntimeError(f"Episode '{episode_id}' starts after it ends.")

            range_status = episode_range_status(
                item.start_frame,
                item.end_frame,
                record_start=self.source_start_frame,
                record_end=self.source_end_frame,
            )
            if range_status == EpisodeRangeStatus.OUTSIDE:
                # Episodes wholly outside the recording range do not become
                # part of this recording and should not block it.
                continue

            if range_status == EpisodeRangeStatus.PARTIAL:
                episode_id = item.episode_id.strip() or "<unnamed>"
                raise RuntimeError(
                    f"Episode '{episode_id}' partially overlaps the recording "
                    f"range ({self.source_start_frame}-{self.source_end_frame})."
                )
            active_episodes.append(item)

        seen_names = set()
        seen_ids = set()
        previous_end = None
        for item in active_episodes:
            episode_name = item.episode_name.strip() or item.episode_id.strip()
            episode_id = item.episode_id.strip()
            if not episode_name:
                raise RuntimeError("Every episode must have a non-empty name.")
            if not episode_id:
                raise RuntimeError("Every episode must have a non-empty ID.")
            folded_name = episode_name.casefold()
            folded_id = episode_id.casefold()
            if folded_name in seen_names:
                raise RuntimeError(f"Episode name '{episode_name}' is duplicated.")
            if folded_id in seen_ids:
                raise RuntimeError(f"Episode ID '{episode_id}' is duplicated.")
            seen_names.add(folded_name)
            seen_ids.add(folded_id)
            if previous_end is not None and item.start_frame <= previous_end:
                raise RuntimeError(f"Episode '{episode_id}' overlaps another episode.")
            previous_end = item.end_frame

    # ======================================================================
    # Recording loop
    # ======================================================================
    def set_fast_mode(self, fast: bool):
        self._fast_mode = fast
        self.delay = 0.0 if fast else 1.0 / self.frame_rate
        self.scene.render.use_lock_interface = fast

    def update_frame(self):
        if self._frame_id > self.end_frame:
            self._wm.progress_end()
            self.recorder.is_running = False
            self.set_fast_mode(False)
            if self._export_data():
                show_report("Recording completed successfully.", level="INFO")
            return None

        if self.recorder.interrupted:
            self._wm.progress_end()
            self.recorder.is_running = False
            self.set_fast_mode(False)
            show_report("Recording interrupted by user.", level="WARNING")
            return None

        self.recorder.is_running = True
        
        # The recorder frame is already in Blender's stretched/output domain.
        # Blender applies the Old/New time remap while evaluating the
        # dependency graph.  Do not inverse-remap here: doing so would apply
        # the time stretch twice and extend motion by factor**2.
        self.scene.frame_set(self._frame_id)
        
        frame_data = self.evaluate_current_frame()
        self.records.append(frame_data)
        self._wm.progress_update(self._frame_id - self.start_frame)
        self._frame_id += self.frame_step
        return self.delay

    # ======================================================================
    # Export
    # ======================================================================
    def _export_data(self):
        if not self.records:
            self.export_succeeded = False
            show_report("No recorded frames are available to export.", level="ERROR")
            return False

        parent = os.path.dirname(self._basefilepath)
        if parent and not os.path.isdir(parent):
            os.makedirs(parent, exist_ok=True)

        env_path = self._mmrt_environment_path
        if self._cfg.env_export_mode != EnvironmentExportMode.OFF and not os.path.isfile(env_path):
            self.export_succeeded = False
            show_report("The declared environment snapshot is missing.", level="ERROR")
            return False

        now = datetime.datetime.now()
        all_labels = list(self.records[0].keys())
        robot_labels = [
            label for label in all_labels
            if any(label == base or label.startswith(base + ".") for base in self.robot_label_bases)
        ]
        active_labels = [label for label in all_labels if label not in robot_labels]
        labels = robot_labels + active_labels
        n_digits = MeasurementConstants.RAD_DECIMAL_PLACES if self._cfg.angle_unit == MeasurementConstants.RADIAN_UNIT else MeasurementConstants.DEG_DECIMAL_PLACES

        yaml_temporary_path = self._mmrt_yaml_path + ".tmp"
        binary_temporary_path = self._mmrt_bin_path + ".tmp"
        try:
            robot_bundle = export_robot_bundle(
                self._context,
                self.output_dir,
                self.base_filename,
            )
            with open(yaml_temporary_path, "w", encoding="utf-8") as f:
                f.write(f"{MDataLabel.METADATA_KEY}:\n")
                f.write(f"  date: {now:%Y-%m-%d}\n")
                f.write(f"  time: {now:%H:%M:%S}\n")
                f.write(f"  {MDataLabel.FRAME_RATE_KEY}: {self.frame_rate}\n")
                f.write(f"  {MDataLabel.START_FRAME_KEY}: {self.start_frame}\n")
                f.write(f"  {MDataLabel.END_FRAME_KEY}: {self.end_frame}\n")
                f.write(f"  {MDataLabel.SOURCE_START_FRAME_KEY}: {self.source_start_frame}\n")
                f.write(f"  {MDataLabel.SOURCE_END_FRAME_KEY}: {self.source_end_frame}\n")
                f.write(f"  {MDataLabel.FRAME_MAP_OLD_KEY}: {self.map_old}\n")
                f.write(f"  {MDataLabel.FRAME_MAP_NEW_KEY}: {self.map_new}\n")
                f.write(f"  {MDataLabel.ADDON_FRAME_TRACKING_KEY}:\n")
                for key, value in self.frame_tracking.items():
                    f.write(f"    {key}: {json.dumps(value)}\n")
                f.write(f"  {MDataLabel.TOTAL_FRAMES_KEY}: {len(self.records)}\n")
                f.write(f"  recorded_items: {self.num_recorded_items}\n")
                f.write(f"  {MDataLabel.ROTATION_FORMAT_KEY}: {self._cfg.rotation_format}\n")
                f.write(f"  {MDataLabel.ANGLE_UNITS_KEY}: {self._cfg.angle_unit.upper()}\n")
                f.write(f"  {MDataLabel.LOCATION_UNITS_KEY}: {self._cfg.length_unit.upper()}\n")
                if self._cfg.env_export_mode != EnvironmentExportMode.OFF:
                    f.write(f"  {MDataLabel.ENVIRONMENT_FILE_KEY}: \"{os.path.basename(self._mmrt_environment_path)}\"\n")
                if robot_bundle:
                    f.write(f"  {MDataLabel.ROBOT_NAME_KEY}: {json.dumps(robot_bundle[MDataLabel.ROBOT_NAME_KEY])}\n")
                    f.write(f"  {MDataLabel.URDF_FILE_KEY}: {json.dumps(robot_bundle[MDataLabel.URDF_FILE_KEY])}\n")
                    f.write(f"  {MDataLabel.SRDF_FILE_KEY}: {json.dumps(robot_bundle[MDataLabel.SRDF_FILE_KEY])}\n")
                    f.write(f"  robot_mesh_dir: {json.dumps(robot_bundle['mesh_dir'])}\n")

                export_episodes = [
                    episode for episode in self.recorder.episodes
                    if episode_range_status(
                        episode.start_frame,
                        episode.end_frame,
                        record_start=self.source_start_frame,
                        record_end=self.source_end_frame,
                    ) == EpisodeRangeStatus.INSIDE
                ]
                if export_episodes:
                    f.write(f"  {MDataLabel.EPISODES_KEY}:\n")
                    # Preserve the collection order: it is also the desired
                    # episode order in the generated behavior-tree sequence.
                    for episode in export_episodes:
                        # Episode bounds are exported in the same scaled frame
                        # domain as marker events and recorded metadata.
                        start_frame = self._remap_frame_forward(episode.start_frame)
                        end_frame = self._remap_frame_forward(episode.end_frame)
                        episode_name = episode.episode_name.strip() or episode.episode_id.strip()
                        f.write(f"    - {EpisodeConfig.NAME_KEY}: {json.dumps(episode_name)}\n")
                        f.write(f"      {EpisodeConfig.ID_KEY}: {json.dumps(episode.episode_id.strip())}\n")
                        f.write(f"      {EpisodeConfig.START_FRAME_KEY}: {start_frame}\n")
                        f.write(f"      {EpisodeConfig.END_FRAME_KEY}: {end_frame}\n")
                        f.write(f"      {EpisodeConfig.START_WITH_WAYPOINT_KEY}: {'true' if episode.start_with_waypoint else 'false'}\n")
                        f.write(f"      {EpisodeConfig.START_WAYPOINT_MODE_KEY}: {episode.start_waypoint_mode}\n")
                        f.write(f"      {EpisodeConfig.LOGIC_KEY}: {episode.logic}\n")
                        f.write(f"      {EpisodeConfig.REPEAT_COUNT_KEY}: {episode.repeat_count}\n")
                        f.write(f"      {EpisodeConfig.RETRY_COUNT_KEY}: {episode.retry_count}\n")
                        f.write(f"      {EpisodeConfig.DELAY_ENABLED_KEY}: {'true' if episode.delay_enabled else 'false'}\n")
                        f.write(f"      {EpisodeConfig.DELAY_MSEC_KEY}: {episode.delay_msec}\n")

                write_marker_events(f, self.marker_events)

                f.write(f"{MDataLabel.LABELS_KEY}:\n")
                f.write(f"  {MDataLabel.ROBOTS_KEY}:\n")
                for label in robot_labels:
                    f.write(f"    - {label}\n")

                if active_labels:
                    f.write(f"  {MDataLabel.ACTIVE_OBJECTS_KEY}:\n")
                    for label in active_labels:
                        f.write(f"    - {label}\n")

                if self.is_binary:
                    f.write("binary: true\n")
                    f.write(f"bin_filename: {os.path.basename(self._mmrt_bin_path)}\n")

                    with open(binary_temporary_path, "wb") as fb:
                        for frame in self.records:
                            for k in labels:
                                fb.write(struct.pack("f", round(frame[k], n_digits)))
                else:
                    f.write("binary: false\n")
                    f.write("frames:\n")
                    for frame in self.records:
                        vals = [f"{frame[k]:.{n_digits}f}" for k in labels]
                        f.write("  - [" + ", ".join(vals) + "]\n")
            
            if self.is_binary:
                os.replace(binary_temporary_path, self._mmrt_bin_path)
            os.replace(yaml_temporary_path, self._mmrt_yaml_path)
            show_report(f"Recorded path:\n{self._mmrt_yaml_path}", level="INFO")
            self.export_succeeded = True
            return True

        except Exception as e:
            for temporary_path in (yaml_temporary_path, binary_temporary_path):
                try:
                    os.remove(temporary_path)
                except FileNotFoundError:
                    pass
                except OSError:
                    pass
            self.export_succeeded = False
            show_report(f"Failed to export recording data: {str(e)}", level="ERROR")
            return False
