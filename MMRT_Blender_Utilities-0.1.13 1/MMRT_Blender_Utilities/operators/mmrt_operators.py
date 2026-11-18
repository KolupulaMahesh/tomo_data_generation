import atexit
import bpy
import json
import os
import tempfile
from ..processes.process_manager import ProcessManager
from ..common.blender import fmt, get_bl_idname, show_report
from ..common.mmrt import get_local_mmrt_app_dir, get_mmrt_xml_filepath
from ..common.recording import get_recorder_mmrt_filepath
from ..common.system.backend import get_backend_command
from ..processes.mmrt.definitions import LoggingVerbosity, PipelineCLIOptions

_latest_log_path = None
_MAX_VISIBLE_LOG_LINES = 500


def _remove_latest_log():
    global _latest_log_path
    if _latest_log_path:
        try:
            os.remove(_latest_log_path)
        except OSError:
            pass
        _latest_log_path = None


atexit.register(_remove_latest_log)


class MMRT_OT_CopyGenerationLog(bpy.types.Operator):
    bl_idname = get_bl_idname("copy_generation_log")
    bl_label = "Copy Generation Output"
    bl_description = "Copy the full output from the latest Behavior Tree generation"

    def execute(self, context):
        if _latest_log_path and os.path.isfile(_latest_log_path):
            with open(_latest_log_path, "r", encoding="utf-8", errors="replace") as log_file:
                content = log_file.read()
        else:
            content = "\n".join(line.message for line in context.window_manager.mmrt_generation_log)
        if not content:
            content = "\n".join(line.message for line in context.window_manager.mmrt_generation_log)
        if not content:
            self.report({'WARNING'}, "No generation output to copy yet.")
            return {'CANCELLED'}
        context.window_manager.clipboard = content
        self.report({'INFO'}, "Generation output copied to clipboard.")
        return {'FINISHED'}

class MMRT_OT_MMRT(bpy.types.Operator):
    bl_idname = get_bl_idname("run_mmrt_process")
    bl_label = "Generate Behavior Tree XML"
    bl_description = "Generate Behavior Tree XML"
    bl_options = {'REGISTER'}

    _timer = None
    _progress = 0.0
    _finished = False
    _succeeded = False
    _progress_file = None
    _progress_message = "Starting MMRT process"
    _log_file = None
    _log_offset = 0
    _log_pending = b""

    @classmethod
    def poll(cls, context):
        process = ProcessManager.get_process("mmrt_processor")
        return not (process and process.is_running())

    @staticmethod
    def _append_log(wm, message):
        lines = wm.mmrt_generation_log
        if len(lines) >= _MAX_VISIBLE_LOG_LINES:
            lines.remove(0)
            wm.mmrt_generation_log_index = max(0, wm.mmrt_generation_log_index - 1)
        entry = lines.add()
        entry.message = message or " "
        if wm.mmrt_generation_log_follow:
            wm.mmrt_generation_log_index = len(lines) - 1

    def _fail(self, context, message):
        context.window_manager.mmrt_generation_log_status = "Generation could not start"
        self._append_log(context.window_manager, f"[ERROR] {message}")
        self.report({'ERROR'}, message)
        return {'CANCELLED'}

    def _read_log(self, wm, final=False):
        if not self._log_file:
            return
        try:
            with open(self._log_file, "rb") as log_file:
                log_file.seek(self._log_offset)
                chunk = log_file.read()
                self._log_offset = log_file.tell()
        except OSError:
            return
        data = self._log_pending + chunk
        parts = data.split(b"\n")
        self._log_pending = parts.pop()
        for part in parts:
            self._append_log(wm, part.decode("utf-8", errors="replace").rstrip("\r"))
        if final and self._log_pending:
            self._append_log(wm, self._log_pending.decode("utf-8", errors="replace"))
            self._log_pending = b""

    def execute(self, context):
        _remove_latest_log()
        wm = context.window_manager
        wm.mmrt_generation_log.clear()
        wm.mmrt_generation_log_index = 0
        wm.mmrt_generation_log_status = "Preparing generation..."
        self._finished = False
        self._succeeded = False
        self._progress = 0.0
        self._progress_message = "Starting MMRT process"
        self._log_file = None
        self._log_offset = 0
        self._log_pending = b""
        addon_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        script_path = os.path.join(addon_dir, "processes", "mmrt", "main.py")
        behavior_tree_models_path = os.path.join(
            addon_dir, "processes", "mmrt", "definitions", "behavior_tree_models.xml"
        )

        if not os.path.isfile(script_path):
            return self._fail(context, f"Processor script not found: {script_path}")

        scene = context.scene
        props = scene.mmrt_settings
        
        if props.use_recorder_path:
            record_path = get_recorder_mmrt_filepath(context)
        else:
            record_path = bpy.path.abspath(props.record_path)
        
        if not os.path.isfile(record_path):
            return self._fail(context, f"Record file not found: {record_path}. Did you run recording first?")

        try:
            output_path = get_mmrt_xml_filepath(scene)
        except ValueError as e:
            return self._fail(context, str(e))
        
        global_velocity_scaling = props.global_velocity_scaling
        global_acceleration_scaling = props.global_acceleration_scaling
        external_python = (
            bpy.path.abspath(scene.mmrt_external_python_path)
            if scene.mmrt_external_python_path
            else None
        )
        use_development_python = scene.mmrt_use_development_python
        try:
            backend_command = get_backend_command(
                "process",
                external_python,
                use_development_python,
            )
        except (RuntimeError, ValueError) as exc:
            return self._fail(context, str(exc))

        args = [
            PipelineCLIOptions.RECORD, record_path,
            PipelineCLIOptions.BEHAVIOR_TREE_MODELS_PATH, behavior_tree_models_path,
            PipelineCLIOptions.OUTPUT, output_path,
            PipelineCLIOptions.APP_DIR, get_local_mmrt_app_dir(scene),
            PipelineCLIOptions.TREE_ID, props.tree_id,
            PipelineCLIOptions.CONFIGURE_TREE_NAME, props.configure_tree_name,
            PipelineCLIOptions.SCENE_SETUP_TREE_NAME, props.scene_setup_tree_name,
            PipelineCLIOptions.MOVE_SEGMENT_TREE_NAME, props.move_segment_tree_name,
            PipelineCLIOptions.BT_TRAJECTORY_STRATEGY, props.trajectory_strategy,
            PipelineCLIOptions.BT_TRAJECTORY_RESAMPLE, props.trajectory_resample,
            PipelineCLIOptions.BT_TRAJECTORY_RESOLUTION, fmt(props.trajectory_resolution),
            PipelineCLIOptions.BT_TRAJECTORY_TIME_PARAMETERIZATION, props.trajectory_time_parameterization,
            PipelineCLIOptions.BT_TRAJECTORY_VISUALIZE, str(props.trajectory_visualize).lower(),

            PipelineCLIOptions.MMRT_SPEED_THR, fmt(props.speed_thr),
            PipelineCLIOptions.MMRT_MIN_SEGMENT_TIME, fmt(props.min_segment_time),
            PipelineCLIOptions.MMRT_SEGMENT_COMBINE_THR, fmt(props.segment_combine_thr),
            PipelineCLIOptions.MMRT_SLEEP_THR, str(props.bt_sleep_thr),
            PipelineCLIOptions.MMRT_WP_DECIMATION_WINDOW, str(props.wp_decimation_window),
            PipelineCLIOptions.MMRT_WP_DECIMATION_ITER, str(props.wp_decimation_iter),
            PipelineCLIOptions.MMRT_WP_DECIMATION_METHOD, props.wp_decimation_method.upper(),
            PipelineCLIOptions.MMRT_COLLISION_THR, fmt(props.collision_thr),

            PipelineCLIOptions.BT_VEL_SCALE, fmt(global_velocity_scaling),
            PipelineCLIOptions.BT_ACC_SCALE, fmt(global_acceleration_scaling),
            PipelineCLIOptions.BT_COLLISION_CHECKS, str(props.global_trajectory_collision_check),
            PipelineCLIOptions.BT_BLEND_RADIUS, fmt(props.global_trajectory_blending_radius),
            PipelineCLIOptions.BT_SAVE_PLAN, str(props.global_save_plan).lower(),
            PipelineCLIOptions.BT_PLAN_PIPELINE, props.plan_pipeline,
            PipelineCLIOptions.BT_PLAN_ID, props.plan_id,
            PipelineCLIOptions.BT_ENABLE_LOG, str(props.enable_log).lower(),
            PipelineCLIOptions.BT_ENABLE_LOG_TO_FILE, str(props.enable_log_to_file).lower(),
            PipelineCLIOptions.BT_DEBUG_TOPIC_NAME, props.debug_topic_name,
            PipelineCLIOptions.VERBOSITY, LoggingVerbosity.DEFAULT
        ]
        progress_fd, self._progress_file = tempfile.mkstemp(
            prefix="mmrt_progress_", suffix=".json"
        )
        os.close(progress_fd)
        args.extend([PipelineCLIOptions.PROGRESS_FILE, self._progress_file])
            
        if props.save_plot:
            args.append(PipelineCLIOptions.SAVE_PLOT)

        if props.show_plot:
            args.append(PipelineCLIOptions.PLOT)
        
        if props.show_backend_simulation:
            args.append(PipelineCLIOptions.SIM)

        global _latest_log_path
        try:
            log_fd, self._log_file = tempfile.mkstemp(
                prefix="mmrt_generation_", suffix=".log"
            )
            os.close(log_fd)
        except OSError as exc:
            self._remove_progress_file()
            return self._fail(context, f"Could not create generation log: {exc}")
        _latest_log_path = self._log_file

        try:
            environment = os.environ.copy()
            environment["PYTHONUNBUFFERED"] = "1"
            ProcessManager.launch_process(
                name="mmrt_processor",
                default_cmd=backend_command[0],
                extra_args=backend_command[1:] + args,
                cwd=addon_dir,
                log_path=self._log_file,
                env=environment,
                on_exit=lambda p: self._on_exit(p)
            )
        
        except Exception as e:
            self._remove_progress_file()
            return self._fail(context, f"Launch failed: {e}")

        wm.mmrt_generation_log_status = "Running: Starting MMRT process (0%)"
        self._append_log(wm, "[MMRT] Behavior Tree generation started.")
        wm.progress_begin(0, 100)
        self._progress = 0.0
        self._set_progress_status(context)
        self._timer = wm.event_timer_add(0.2, window=context.window)
        wm.modal_handler_add(self)

        self.report({'INFO'}, "Behavior Tree generation started.")
        return {'RUNNING_MODAL'}

    def _on_exit(self, proc):
        """Callback when process finishes."""
        self._succeeded = proc.returncode == 0
        self._finished = True

    def _read_progress(self):
        if not self._progress_file:
            return
        try:
            with open(self._progress_file, "r", encoding="utf-8") as progress_file:
                payload = json.load(progress_file)
            progress = float(payload.get("progress", self._progress))
            self._progress = max(self._progress, min(100.0, max(0.0, progress)))
            message = payload.get("message")
            if message:
                self._progress_message = str(message)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            # The process replaces the progress file atomically. Ignore a
            # missing or transiently unreadable update and keep the last value.
            pass

    def _remove_progress_file(self):
        if self._progress_file:
            try:
                os.remove(self._progress_file)
            except OSError:
                pass
            self._progress_file = None

    def _set_progress_status(self, context):
        workspace = getattr(context, "workspace", None)
        if workspace is None:
            return
        try:
            workspace.status_text_set(
                text=f"MMRT: {self._progress_message} ({self._progress:.0f}%)"
            )
        except (AttributeError, RuntimeError):
            pass

    @staticmethod
    def _clear_progress_status(context):
        workspace = getattr(context, "workspace", None)
        if workspace is None:
            return
        try:
            workspace.status_text_set(text=None)
        except (AttributeError, RuntimeError):
            pass

    def modal(self, context, event):
        wm = context.window_manager

        # Allow Esc to cancel
        if event.type == 'ESC':
            proc = ProcessManager.get_process("mmrt_processor")
            if proc and proc.is_running():
                proc.stop()
                self.report({'WARNING'}, "Process cancelled by user.")
            self._read_log(wm, final=True)
            wm.mmrt_generation_log_status = "Generation cancelled"
            self._append_log(wm, "[MMRT] Generation cancelled.")
            wm.progress_end()
            self._clear_progress_status(context)
            wm.event_timer_remove(self._timer)
            self._remove_progress_file()
            return {'CANCELLED'}

        # Periodically check process
        if event.type == 'TIMER':
            self._read_progress()
            self._read_log(wm, final=self._finished)
            if context.area:
                context.area.tag_redraw()
            if self._finished:
                if self._succeeded:
                    self._progress = 100.0
                wm.progress_update(self._progress)
                wm.progress_end()
                self._clear_progress_status(context)
                wm.event_timer_remove(self._timer)
                self._remove_progress_file()
                if self._succeeded:
                    wm.mmrt_generation_log_status = "Generation completed"
                    self._append_log(wm, "[MMRT] Generation completed.")
                    self.report({'INFO'}, "Behavior Tree generation completed.")
                    return {'FINISHED'}
                wm.mmrt_generation_log_status = "Generation failed"
                self._append_log(wm, "[MMRT] Generation failed. See the output above.")
                self.report({'ERROR'}, "Behavior Tree generation failed. Check the Generation Log for details.")
                return {'CANCELLED'}

            wm.progress_update(self._progress)
            self._set_progress_status(context)
            status = f"Running: {self._progress_message} ({self._progress:.0f}%)"
            if wm.mmrt_generation_log_status != status:
                wm.mmrt_generation_log_status = status
        return {'RUNNING_MODAL'}

    def cancel(self, context):
        wm = context.window_manager
        self._read_log(wm, final=True)
        wm.mmrt_generation_log_status = "Generation cancelled"
        self._append_log(wm, "[MMRT] Generation cancelled.")
        wm.progress_end()
        self._clear_progress_status(context)
        self._remove_progress_file()
        ret = ProcessManager.stop_process("mmrt_processor")
        if ret:
            self.report({'WARNING'}, "Process cancelled by user.")


class MMRT_OT_Plot_Joint_Data(bpy.types.Operator):
    bl_idname = get_bl_idname("plot_joint_data")
    bl_label = "Evaluate Joint Trajectories"
    bl_description = "Plot Position, Velocity, and Acceleration limits for joints"
    bl_options = {'REGISTER'}
    
    def execute(self, context):
        scene = context.scene
        props = scene.mmrt_settings
        
        # 1. Resolve Record File Path
        if props.use_recorder_path:
            record_path = get_recorder_mmrt_filepath(context)
        else:
            record_path = bpy.path.abspath(props.record_path)
            
        if not os.path.exists(record_path):
            show_report(f"Record file not found: {record_path}\nDid you run recording first?", level="ERROR")
            return {'CANCELLED'}
            
        # 2. Resolve Python Executable & Script Path
        addon_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        script_path = os.path.join(addon_dir, "processes", "mmrt", "joint_limit_plotter.py")
        
        if not os.path.exists(script_path):
            self.report({'ERROR'}, f"Plotting script not found: {script_path}")
            return {'CANCELLED'}
        
        external_python = getattr(scene, "mmrt_external_python_path", "")
        external_python = bpy.path.abspath(external_python) if external_python else None
        use_development_python = scene.mmrt_use_development_python
        try:
            backend_command = get_backend_command(
                "plot",
                external_python,
                use_development_python,
            )
        except (RuntimeError, ValueError) as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}

        # 3. Launch Process
        try:
            ProcessManager.launch_process(
                name="mmrt_joint_plotter",
                default_cmd=backend_command[0],
                extra_args=backend_command[1:] + [record_path],
                cwd=addon_dir,
                console_output=True
            )
            self.report({'INFO'}, "Launched joint evaluation tool.")
        except Exception as e:
            self.report({'ERROR'}, f"Failed to launch plotter: {e}")
            return {'CANCELLED'}
            
        return {'FINISHED'}

classes = (
   MMRT_OT_CopyGenerationLog,
   MMRT_OT_MMRT,
   MMRT_OT_Plot_Joint_Data,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    _remove_latest_log()
