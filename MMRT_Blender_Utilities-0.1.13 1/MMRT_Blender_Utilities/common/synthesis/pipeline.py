import bpy
import random
import pprint
import time
from pathlib import Path
from mathutils import Vector, Euler, Matrix
from math import radians, degrees, floor

from ..config import EnvironmentExportMode, SynthesisRenderOutput
from ...processes.mmrt.definitions import OutputSuffixes, RotationFormat
from ..exporting.environment import export_environment, export_environment_metadata
from ..recording.recorder import MMRTRecorder


def print_synthesis_progress(message):
    """Write progress immediately, even while Blender's UI thread is busy."""
    print(f"[mmrt][SYNTH] {message}", flush=True)


def format_elapsed(start_time):
    elapsed = max(0.0, time.monotonic() - start_time)
    if elapsed < 60.0:
        return f"{elapsed:.1f}s"
    minutes, seconds = divmod(int(elapsed), 60)
    return f"{minutes}m {seconds:02d}s"


def print_pose(title, matrix):
    loc = matrix.translation
    euler = matrix.to_quaternion().to_euler("XYZ")
    print(title)
    print(f"  Location : ({loc.x:.6f}, {loc.y:.6f}, {loc.z:.6f})")
    print(f"  Rotation : ({degrees(euler.x):.3f}, {degrees(euler.y):.3f}, {degrees(euler.z):.3f})")

def axis_range_to_tuple(range_config):
    if not range_config: return None
    return (tuple(range_config["x"]), tuple(range_config["y"]), tuple(range_config["z"]))

def sample_offset(rng, range_tuple):
    if not range_tuple: return (0.0, 0.0, 0.0)
    return tuple(rng.uniform(float(low), float(high)) for low, high in range_tuple)

def offset_matrix(source_matrix, translation_offset, offset_quat):
    loc, source_rot, source_scale = source_matrix.decompose()
    loc = loc + Vector(translation_offset)
    rot = offset_quat @ source_rot
    return Matrix.LocRotScale(loc, rot, source_scale)

def insert_transform_key(obj, frame):
    obj.keyframe_insert(data_path="location", frame=frame)
    if obj.rotation_mode == RotationFormat.QUATERNION:
        obj.keyframe_insert(data_path="rotation_quaternion", frame=frame)
    else:
        obj.keyframe_insert(data_path="rotation_euler", frame=frame)
        
def set_source_animation_frame(scene, source_frame):
    """ setting annimation frame: 
        scene_time = frame_map_new / frame_map_old
    """
    old = max(1, int(scene.render.frame_map_old))
    new = max(1, int(scene.render.frame_map_new))
    scene_time = float(source_frame) * new / old
    whole_frame = floor(scene_time)
    scene.frame_set(whole_frame, subframe=scene_time - whole_frame)

class ActionManager:
    # Stored on each object so Reset Actions also works after saving/reopening.
    SOURCE_ACTION = "_mmrt_synth_source_action"
    SOURCE_SLOT = "_mmrt_synth_source_slot"
    PREVIEW_ACTION = "_mmrt_synth_preview_action"

    def __init__(self, object_names, is_dry_run=False):
        self.object_names = list(object_names)
        self.is_dry_run = is_dry_run
        self.initial_actions = {}
        self.initial_slots = {}

    def protect_all_actions(self):
        # Preserve the existing synthesizer's action-retention behavior.
        for action in bpy.data.actions:
            action.use_fake_user = True

    @staticmethod
    def _assign_action(obj, action, slot_identifier):
        # Validate before replacing the object's current assignment.
        slot = None
        if slot_identifier:
            slot = next(
                (item for item in action.slots
                 if item.identifier == slot_identifier),
                None,
            )
            if slot is None:
                raise RuntimeError(
                    f"Action '{action.name}' has no slot '{slot_identifier}' "
                    f"for '{obj.name}'."
                )
        obj.animation_data_create()
        obj.animation_data.action = action
        obj.animation_data.action_slot = slot

    @classmethod
    def restore_preview(cls, obj):
        if cls.SOURCE_ACTION not in obj:
            return False

        source = obj.get(cls.SOURCE_ACTION)
        if not isinstance(source, bpy.types.Action):
            raise RuntimeError(f"Original action is missing for '{obj.name}'.")

        preview = obj.get(cls.PREVIEW_ACTION)
        cls._assign_action(obj, source, obj.get(cls.SOURCE_SLOT, ""))

        for key in (cls.SOURCE_ACTION, cls.SOURCE_SLOT, cls.PREVIEW_ACTION):
            if key in obj:
                del obj[key]

        # Remove only the tracked preview, and only if nothing else uses it.
        if isinstance(preview, bpy.types.Action) and preview != source:
            preview.use_fake_user = False
            if preview.users == 0:
                bpy.data.actions.remove(preview)
        return True

    def capture_baselines(self):
        self.initial_actions.clear()
        self.initial_slots.clear()

        # Recover previous previews before discovering any source actions.
        for name in self.object_names:
            obj = bpy.data.objects.get(name)
            if obj is not None:
                self.restore_preview(obj)

        self.protect_all_actions()
        for name in self.object_names:
            obj = bpy.data.objects.get(name)
            if obj is None or obj.animation_data is None:
                continue
            animation = obj.animation_data
            action = animation.action
            if action is None:
                continue
            slot = animation.action_slot
            if len(action.slots) and slot is None:
                raise RuntimeError(
                    f"Choose an action slot for '{obj.name}' before synthesis."
                )

            # Original actions remain unchanged and serve as the baselines.
            self.initial_actions[name] = action
            self.initial_slots[name] = slot.identifier if slot else ""

    def assign_episode_actions(self, episode_name):
        assigned_actions = {}
        for name, source in self.initial_actions.items():
            obj = bpy.data.objects.get(name)
            if obj is None:
                continue

            # A new copy for EVERY object, even if source actions are shared.
            action = source.copy()
            suffix = "preview" if self.is_dry_run else episode_name
            action.name = f"{name}_{suffix}"
            action.use_fake_user = True
            slot_identifier = self.initial_slots[name]

            if self.is_dry_run:
                obj[self.SOURCE_ACTION] = source
                obj[self.SOURCE_SLOT] = slot_identifier
                obj[self.PREVIEW_ACTION] = action

            self._assign_action(obj, action, slot_identifier)
            assigned_actions[name] = action
        return assigned_actions

    def restore_initial_actions(self):
        for name, source in self.initial_actions.items():
            obj = bpy.data.objects.get(name)
            if obj is None:
                continue
            if not self.restore_preview(obj):
                self._assign_action(obj, source, self.initial_slots[name])


class MaterialManager:
    PREVIEW_STATE = "_mmrt_synth_material_preview"

    def __init__(self, randomization_configs, is_dry_run=False):
        self.configs = randomization_configs
        self.is_dry_run = is_dry_run
        self.caches = []
        self.original_states = {}

    @staticmethod
    def _capture_slot(slot):
        # Preserve the inactive object override too, even for DATA-linked slots.
        link = slot.link
        material = slot.material
        try:
            slot.link = "OBJECT"
            override = slot.material
        finally:
            slot.link = link
        state = {"link": link}
        if material is not None:
            state["material"] = material
        if override is not None:
            state["override"] = override
        return state

    @staticmethod
    def _restore_state(obj, state):
        if obj.data != state["data"] or len(obj.material_slots) != state["slot_count"]:
            raise RuntimeError(
                f"Material slots changed on '{obj.name}'; restore its original "
                "mesh and slot layout before resetting this preview."
            )
        for index, original in state["slots"].items():
            slot = obj.material_slots[int(index)]
            slot.link = "OBJECT"
            slot.material = original.get("override")
            slot.link = original["link"]

    @classmethod
    def restore_preview(cls, obj):
        state = obj.get(cls.PREVIEW_STATE)
        if state is None:
            return False
        cls._restore_state(obj, state)
        del obj[cls.PREVIEW_STATE]
        return True

    def cache_initial_states(self):
        self.caches.clear()
        self.original_states.clear()

        # Recover prior dry runs before looking for the source material.
        for conf in self.configs:
            obj = bpy.data.objects.get(conf["object_name"])
            if obj is None:
                raise ValueError(f"Material target '{conf['object_name']}' is missing.")
            self.restore_preview(obj)

        caches = []
        states = {}
        occupied = set()
        for conf in self.configs:
            obj = bpy.data.objects[conf["object_name"]]
            source = bpy.data.materials.get(conf["target_material_name"])
            indices = [
                idx for idx, slot in enumerate(obj.material_slots)
                if source is not None and slot.material == source
            ]
            if not indices:
                raise ValueError(
                    f"'{obj.name}' has no slot using '{conf['target_material_name']}'. "
                    "Select its original material in the synthesizer."
                )

            pool = []
            for name in dict.fromkeys(conf["material_pool"]):
                mat = bpy.data.materials.get(name)
                if mat is None:
                    raise ValueError(f"Pool material '{name}' is missing for '{obj.name}'.")
                pool.append(mat)
            if not pool:
                raise ValueError(f"Material pool is empty for '{obj.name}'.")

            state = states.setdefault(obj.name, {
                "data": obj.data,
                "slot_count": len(obj.material_slots),
                "slots": {},
            })
            for idx in indices:
                key = (obj.name, idx)
                if key in occupied:
                    raise ValueError(
                        f"Multiple material targets overlap at '{obj.name}', slot {idx}."
                    )
                occupied.add(key)
                state["slots"][str(idx)] = self._capture_slot(obj.material_slots[idx])

            caches.append({
                "object_name": obj.name,
                "slot_indices": indices,
                "original_material": source,
                "pool": pool,
            })

        self.caches = caches
        self.original_states = states
        for cache in caches:
            for mat in cache["pool"]:
                mat.use_fake_user = True

    def apply_random_materials(self, rng):
        if self.is_dry_run:
            for name, state in self.original_states.items():
                obj = bpy.data.objects[name]
                if self.PREVIEW_STATE not in obj:
                    # ID pointers survive material renames and saving/reopening.
                    obj[self.PREVIEW_STATE] = state

        meta = []
        for cache in self.caches:
            obj = bpy.data.objects[cache["object_name"]]
            selected = rng.choice(cache["pool"])
            # One choice per target; replace ALL slots using its source material.
            for idx in cache["slot_indices"]:
                slot = obj.material_slots[idx]
                # Object overrides prevent linked mesh users overwriting each other.
                slot.link = "OBJECT"
                slot.material = selected
                meta.append({
                    "object_name": obj.name,
                    "slot_index": idx,
                    "original_material_name": cache["original_material"].name,
                    "selected_material_name": selected.name,
                })
        return meta

    def restore_initial_materials(self):
        for name, state in self.original_states.items():
            obj = bpy.data.objects.get(name)
            if obj is not None:
                self._restore_state(obj, state)
                if self.PREVIEW_STATE in obj:
                    del obj[self.PREVIEW_STATE]

class TransformManager:
    def __init__(self, scene, objects_config):
        self.scene = scene
        self.config = objects_config
        self.original_matrices = {} 
        self.relative_transforms = {}
        self.all_frames = set()

        for name, spec in self.config.items():
            obj = bpy.data.objects.get(name)
            fmin = spec.get("frame_min", scene.frame_start)
            fmax = spec.get("frame_max", scene.frame_end)
            
            obj_frames = set()
            if obj and obj.animation_data and obj.animation_data.action:
                action = obj.animation_data.action
                
                for layer in action.layers:
                    for strip in layer.strips:
                        for channelbag in strip.channelbags:
                            if channelbag.slot_handle != obj.animation_data.action_slot_handle:
                                continue
                            for fcurve in channelbag.fcurves:
                                for kp in fcurve.keyframe_points:
                                    f = int(round(kp.co[0]))
                                    if fmin <= f <= fmax:
                                        obj_frames.add(f)
            
            spec["computed_frames"] = sorted(list(obj_frames))
            self.all_frames.update(obj_frames)

    def cache_baseline_transforms(self):
        sorted_frames = sorted(list(self.all_frames))
        for name in self.config.keys():
            self.original_matrices[name] = {}
            self.relative_transforms[name] = {}

        for frame in sorted_frames:
            set_source_animation_frame(self.scene, frame)
            bpy.context.view_layer.update()
            for name in self.config.keys():
                obj = bpy.data.objects.get(name)
                if obj: self.original_matrices[name][frame] = obj.matrix_world.copy()

        for frame in sorted_frames:
            for name, spec in self.config.items():
                p_name = spec.get("parent_offset_object")
                if p_name and frame in self.original_matrices[p_name] and frame in self.original_matrices[name]:
                    p_mat = self.original_matrices[p_name][frame]
                    c_mat = self.original_matrices[name][frame]
                    self.relative_transforms[name][frame] = p_mat.inverted() @ c_mat

    def evaluate_and_apply_offsets(self, rng, episode_offsets=None):
        if episode_offsets is None:
            episode_offsets = {}
            for name, spec in self.config.items():
                t_tuple = axis_range_to_tuple(spec.get("translation_range"))
                r_tuple = axis_range_to_tuple(spec.get("rotation_range_deg"))
                t_off = sample_offset(rng, t_tuple) if t_tuple else (0.0, 0.0, 0.0)
                r_off = sample_offset(rng, r_tuple) if r_tuple else (0.0, 0.0, 0.0)
                episode_offsets[name] = (t_off, r_off)

        for frame in sorted(list(self.all_frames)):
            calculated_matrices = {}

            def resolve_matrix(obj_name):
                if obj_name in calculated_matrices: return calculated_matrices[obj_name]
                spec = self.config[obj_name]
                p_name = spec.get("parent_offset_object")

                if p_name and frame in self.original_matrices[p_name]:
                    p_mat = resolve_matrix(p_name)
                    base_mat = p_mat @ self.relative_transforms[obj_name][frame]
                else:
                    base_mat = self.original_matrices[obj_name][frame]

                t_off, r_deg = episode_offsets[obj_name]
                r_rad = Euler(tuple(radians(v) for v in r_deg), "XYZ").to_quaternion()
                final_mat = offset_matrix(base_mat, t_off, r_rad)
                calculated_matrices[obj_name] = final_mat
                return final_mat

            for name in self.config.keys():
                if frame in self.config[name].get("computed_frames", []):
                    new_mat = resolve_matrix(name)
                    set_source_animation_frame(self.scene, frame)
                    obj = bpy.data.objects.get(name)
                    if obj:
                        obj.matrix_world = new_mat
                        bpy.context.view_layer.update()
                        insert_transform_key(obj, frame)

        return episode_offsets

class BlenderSynthesizer:
    def __init__(self, config):
        self.config = config.copy()
        self.scene = bpy.context.scene
        self.is_dry_run = self.config.get("dry_run", False)

        # FIX: Changed 'tree' to 'self.tree' so other class methods can access it
        if hasattr(self.scene, "compositing_node_group"):
            self.tree = self.scene.compositing_node_group
            # If no compositor tree exists yet, create a new data block
            if self.tree is None:
                self.tree = bpy.data.node_groups.new("CompositorTree", "CompositorNodeTree")
                self.scene.compositing_node_group = self.tree
        else:
            # Fallback for Blender 4.x backward compatibility
            self.scene.use_nodes = True
            self.tree = self.scene.node_tree

        if self.is_dry_run:
            self.config["episode_count"] = 1
            self.config["render_animation"] = False
            self.config["record_mmrt"] = False
            self.config["record_mmrt_before_render"] = False

        self.rng = random.Random(self.config["random_seed"]) if not self.is_dry_run else random.Random()
        self.action_mgr = ActionManager(list(self.config["objects"].keys()), self.is_dry_run)
        self.material_mgr = MaterialManager(
            self.config.get("material_randomization", []), self.is_dry_run
        )
        self.transform_mgr = None

    def get_output_root(self):
        if self.config["output_root"]: return Path(bpy.path.abspath(self.config["output_root"]))
        return Path(bpy.path.abspath(self.scene.render.filepath)).parent

    def configure_render_paths(self, episode_dir, episode_name):
        render = self.scene.render
        if self.config["render_output"].upper() == SynthesisRenderOutput.MP4:
            # Select video output
            if hasattr(render.image_settings, "media_type"):
                render.image_settings.media_type = "VIDEO"
            else:
                render.image_settings.file_format = "FFMPEG"

            # Configure FFmpeg
            render.ffmpeg.format = "MPEG4"
            render.ffmpeg.codec = "H264"
            render.ffmpeg.constant_rate_factor = "MEDIUM"
            render.ffmpeg.ffmpeg_preset = "GOOD"

            # Output path
            render.filepath = str(episode_dir / f"{episode_name}.mp4")
        else:
            render.filepath = str(episode_dir / episode_name)

    def record_mmrt(self, context, episode_dir, episode_name):
        scene = context.scene
        rec_config = scene.mmrt_recorder.config
        orig_dir, orig_file = rec_config.output_dir, rec_config.base_filename
        orig_fast = scene.mmrt_recorder.is_fast_mode
        orig_interrupted = scene.mmrt_recorder.interrupted
        record_started = time.monotonic()

        rec_config.output_dir = str(episode_dir)
        rec_config.base_filename = episode_name
        scene.mmrt_recorder.is_fast_mode = bool(self.config["record_mmrt_fast_mode"])
        scene.mmrt_recorder.interrupted = False

        recorder = None
        try:
            if rec_config.env_export_mode == EnvironmentExportMode.AUTO:
                print_synthesis_progress(f"{episode_name}: exporting MMRT environment")
                export_environment(context)
                export_environment_metadata(context)

            print_synthesis_progress(f"{episode_name}: preparing MMRT recorder")
            recorder = MMRTRecorder(context)
            recorder.set_fast_mode(bool(self.config["record_mmrt_fast_mode"]))
            recorder.run_pipeline()

            total_frames = max(1, recorder._total_frames)
            # Cap routine output at about 20 frame updates per episode while
            # still reporting every frame for very short recordings.
            report_interval = max(1, (total_frames + 19) // 20)
            next_report = report_interval
            print_synthesis_progress(
                f"{episode_name}: recording frames 0/{total_frames} (0%)"
            )

            while True:
                if recorder._frame_id > recorder.end_frame:
                    print_synthesis_progress(
                        f"{episode_name}: frames captured; writing MMRT bundle"
                    )

                result = recorder.update_frame()
                if result is None:
                    break

                completed = min(
                    total_frames,
                    max(0, recorder._frame_id - recorder.start_frame),
                )
                if completed >= next_report or completed == total_frames:
                    percent = round(completed * 100 / total_frames)
                    print_synthesis_progress(
                        f"{episode_name}: recording frames "
                        f"{completed}/{total_frames} ({percent}%)"
                    )
                    next_report = completed + report_interval

            if not recorder.export_succeeded:
                raise RuntimeError("MMRT recording bundle export failed")
            print_synthesis_progress(
                f"{episode_name}: MMRT recording complete "
                f"({format_elapsed(record_started)})"
            )
            return str(Path(episode_dir) / f"{episode_name}{OutputSuffixes.YAML_RECORDING}")
        finally:
            if recorder: recorder.set_fast_mode(False)
            rec_config.output_dir, rec_config.base_filename = orig_dir, orig_file
            scene.mmrt_recorder.is_fast_mode = orig_fast
            scene.mmrt_recorder.interrupted = orig_interrupted

    def export_metadata(self, dest_dir, data):
        meta_path = dest_dir / "metadata.py"
        with meta_path.open("w", encoding="utf-8") as f:
            f.write("metadata = ")
            f.write(pprint.pformat(data, sort_dicts=False))
            f.write("\n")
        return meta_path

    def configure_depth_output(self, ep_dir):
        """Configures the File Output node for Blender 5.x scene compositor."""
        # Ensure Z pass is enabled for all view layers
        for vl in self.scene.view_layers:
            vl.use_pass_z = True

        # Reuse whichever Render Layers node already exists.  IDG creates a
        # managed node when its render handlers initialize; older versions of
        # this method then created a second node named ``Render Layers``.  Two
        # source nodes in the shared Blender 5.2 compositor can destabilize
        # depsgraph copies during long animation renders.  Sharing one node
        # also keeps this addon compatible with IDG-enabled blend files.
        rl_node = self.tree.nodes.get("Render Layers")
        if not rl_node or getattr(rl_node, "bl_idname", "") != "CompositorNodeRLayers":
            rl_node = next(
                (
                    node
                    for node in self.tree.nodes
                    if getattr(node, "bl_idname", "") == "CompositorNodeRLayers"
                ),
                None,
            )
        if rl_node is None:
            rl_node = self.tree.nodes.new("CompositorNodeRLayers")
            rl_node.name = "Render Layers"
        rl_node.scene = self.scene

        # Find or create File Output node
        depth_node = self.tree.nodes.get("SynthDepthOutput")
        if not depth_node:
            depth_node = self.tree.nodes.new("CompositorNodeOutputFile")
            depth_node.name = "SynthDepthOutput"
            
            # FIX: Blender 5.x uses 'FLOAT' instead of 'VALUE'
            depth_node.file_output_items.clear()
            depth_node.file_output_items.new('FLOAT', 'Depth')

        # Apply folder routes dynamically per episode
        depth_path = ep_dir / "depth"
        depth_path.mkdir(parents=True, exist_ok=True)
        
        depth_node.directory = str(depth_path)
        depth_node.file_name = "depth_"
        
        # Determine valid depth/Z token string mapping from render layer
        z_socket_name = "Depth" if "Depth" in rl_node.outputs else "Z"
        z_output = rl_node.outputs[z_socket_name]
        
        # The first input now dynamically aligns with your VALUE item layer
        file_input = depth_node.inputs[0]
        
        already_connected = any(
            link.from_socket == z_output and link.to_socket == file_input
            for link in self.tree.links
        )
        
        if not already_connected:
            self.tree.links.new(z_output, file_input)
            
        return depth_node

    def execute_generation(self):
        if "_mmrt_joint_retime" in self.scene:
            raise RuntimeError(
                "Regional timing is active. Restore Original Timing before batch synthesis, "
                "or use Export Current Animation as Dataset Episode to preserve current timing."
            )
        batch_started = time.monotonic()
        render = self.scene.render
        orig_path = render.filepath
        orig_media_type = getattr(render.image_settings, "media_type", None)
        orig_format = render.image_settings.file_format
        orig_ffmpeg = {
            name: getattr(render.ffmpeg, name)
            for name in ("format", "codec", "constant_rate_factor", "ffmpeg_preset")
        }

        output_root = self.get_output_root()
        output_root.mkdir(parents=True, exist_ok=True)

        episode_count = self.config["episode_count"]
        print_synthesis_progress(
            f"Starting {episode_count} episode(s); output: {output_root}"
        )
        print_synthesis_progress("Preparing source actions and materials")
        self.action_mgr.capture_baselines()
        self.transform_mgr = TransformManager(self.scene, self.config["objects"])
        self.material_mgr.cache_initial_states()
        print_synthesis_progress("Caching source transforms")
        self.transform_mgr.cache_baseline_transforms()
        print_synthesis_progress("Source preparation complete")

        batch_metadata = {
            "blend_file": bpy.data.filepath,
            "output_root": str(output_root),
            "frame_start": self.scene.frame_start,
            "frame_end": self.scene.frame_end,
            "episodes": {},
        }

        generation_succeeded = False
        try:
            start_idx = self.config["episode_start_index"]
            for episode_number, idx in enumerate(
                range(start_idx, start_idx + episode_count),
                start=1,
            ):
                episode_started = time.monotonic()
                ep_name = f"{self.config['episode_prefix']}_{idx:03d}"
                ep_dir = output_root / ep_name
                ep_dir.mkdir(parents=True, exist_ok=True)
                episode_label = f"[{episode_number}/{episode_count}] {ep_name}"
                print_synthesis_progress(f"{episode_label}: started")

                # Dynamically set up depth routing for the specific episode path
                self.configure_depth_output(ep_dir)

                print_synthesis_progress(f"{episode_label}: applying randomization")
                mat_meta = self.material_mgr.apply_random_materials(self.rng)
                ep_actions = self.action_mgr.assign_episode_actions(ep_name)
                applied_offsets = self.transform_mgr.evaluate_and_apply_offsets(self.rng)

                self.configure_render_paths(ep_dir, ep_name)
                mmrt_path = None
                if self.config["record_mmrt"] and self.config["record_mmrt_before_render"]:
                    mmrt_path = self.record_mmrt(bpy.context, ep_dir, ep_name)

                episode_metadata = {
                    "episode_name": ep_name,
                    "episode_dir": str(ep_dir),
                    "offsets_applied": applied_offsets,
                    "materials_swapped": mat_meta,
                    "mmrt_record_filepath": mmrt_path
                }

                if self.config["save_episode_blend"]:
                    print_synthesis_progress(f"{episode_label}: saving .blend file")
                    bpy.ops.wm.save_as_mainfile(filepath=str(ep_dir / f"{ep_name}.blend"))
                if self.config["render_animation"]:
                    print_synthesis_progress(
                        f"{episode_label}: rendering frames "
                        f"{self.scene.frame_start}-{self.scene.frame_end}"
                    )
                    bpy.ops.render.render(animation=True, write_still=False)
                    print_synthesis_progress(f"{episode_label}: rendering complete")

                if self.config["record_mmrt"] and not self.config["record_mmrt_before_render"]:
                    mmrt_path = self.record_mmrt(bpy.context, ep_dir, ep_name)
                    episode_metadata["mmrt_record_filepath"] = mmrt_path

                print_synthesis_progress(f"{episode_label}: writing metadata")
                self.export_metadata(ep_dir, episode_metadata)
                batch_metadata["episodes"][ep_name] = episode_metadata
                self.action_mgr.protect_all_actions()
                batch_percent = round(episode_number * 100 / episode_count)
                print_synthesis_progress(
                    f"{episode_label}: complete ({batch_percent}% batch, "
                    f"{format_elapsed(episode_started)})"
                )
            generation_succeeded = True
        finally:
            print_synthesis_progress("Restoring source scene state")
            try:
                # Keep successful previews visible; undo failed attempts.
                if not self.is_dry_run or not generation_succeeded:
                    try:
                        self.material_mgr.restore_initial_materials()
                    finally:
                        self.action_mgr.restore_initial_actions()
            finally:
                render.filepath = orig_path
                # Restore the media category BEFORE its dependent file-format enum.
                if orig_media_type is not None:
                    render.image_settings.media_type = orig_media_type
                render.image_settings.file_format = orig_format
                for name, value in orig_ffmpeg.items():
                    setattr(render.ffmpeg, name, value)
                bpy.context.view_layer.update()

        print_synthesis_progress("Writing batch metadata")
        self.export_metadata(output_root, batch_metadata)
        print_synthesis_progress(
            f"Synthesis complete: {episode_count} episode(s) "
            f"in {format_elapsed(batch_started)}"
        )
