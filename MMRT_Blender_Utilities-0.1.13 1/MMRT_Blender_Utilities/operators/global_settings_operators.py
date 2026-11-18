import bpy
import os
from ..processes.process_manager import ProcessManager
from ..common.blender import get_bl_idname, show_report
from ..common.system.backend import backend_location_text, get_backend_command


class MMRT_OT_VerifyMMRTBackend(bpy.types.Operator):
    bl_idname = get_bl_idname("verify_mmrt_backend")
    bl_label = "Verify MMRT Backend"

    def execute(self, context):
        scene = context.scene
        addon_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        external_python = (
            bpy.path.abspath(scene.mmrt_external_python_path)
            if scene.mmrt_external_python_path
            else None
        )
        use_development_python = scene.mmrt_use_development_python

        try:
            command = get_backend_command(
                "health-check",
                external_python,
                use_development_python,
            )
        except (RuntimeError, ValueError) as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}

        def on_finished(proc):
            if proc is None:
                show_report("MMRT backend verification did not return a process.", level='ERROR')
                return

            if proc.returncode == 0:
                show_report(
                    "MMRT backend is ready: "
                    f"{backend_location_text(external_python, use_development_python)}",
                    level='INFO',
                )
                return

            details = ""
            if proc.stderr:
                try:
                    details = proc.stderr.read().strip()
                except Exception as exc:
                    details = f"Error reading backend output: {exc}"
            show_report(
                details or "MMRT backend verification failed.",
                level='ERROR',
            )

        try:
            ProcessManager.launch_process(
                name="mmrt_backend_check",
                default_cmd=command[0],
                extra_args=command[1:],
                cwd=addon_dir,
                console_output=False,
                on_exit=on_finished
            )

            self.report({'INFO'}, "Verifying MMRT backend...")

            return {'FINISHED'}

        except Exception as e:
            self.report({'ERROR'}, f"Backend verification failed: {e}")
            return {'CANCELLED'}



class MMRT_OT_LaunchGroot2(bpy.types.Operator):
    bl_idname = get_bl_idname("launch_groot2")
    bl_label = "Launch Groot2 Behavior Tree Editor"

    def execute(self, context):
        scene = context.scene
        groot_path = bpy.path.abspath(scene.mmrt_groot2_path) if scene.mmrt_groot2_path else "groot2"

        try:
            ProcessManager.launch_process(
                name="groot2",
                default_cmd=groot_path,
                use_alias=False,
                console_output=True
            )
            self.report({'INFO'}, f"Groot2 launched using: {groot_path}")
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        return {'FINISHED'}


class MMRT_OT_StopGroot2(bpy.types.Operator):
    bl_idname = get_bl_idname("stop_groot2")
    bl_label = "Stop Groot2"

    def execute(self, context):
        ret = ProcessManager.stop_process("groot2")
        if ret:
            self.report({'INFO'}, "Groot2 stopped.")
            return {'FINISHED'}
        else:
            self.report({'WARNING'}, "Groot2 was not running.")
            

classes = (
    MMRT_OT_VerifyMMRTBackend,
    MMRT_OT_LaunchGroot2,
    MMRT_OT_StopGroot2,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
