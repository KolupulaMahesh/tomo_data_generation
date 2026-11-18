import bpy
import json
import os
import tempfile
import time
import uuid

from ..common.blender import get_bl_idname, show_report
from ..common.config import UIConfig
from ..processes.mmrt.definitions import GeneratedAppDirectories, GeneratedAppFilenames
from ..common.system.backend import get_backend_command
from ..common.mmrt import get_local_mmrt_app_dir, get_mmrt_xml_filepath
from ..common.robot_server_config import (
    RobotServerConfig,
    control_request,
    load_application_request,
    load_config_request,
)
from ..processes.helpers.bridge_client import LocalBridgeClient
from ..processes.process_manager import ProcessManager


_bridge_client = LocalBridgeClient()
_bridge_timer_registered = False
_bridge_connection_timer = None
_bridge_scene = None
_service_response_timers = {}
_bridge_poll_tick_logged = False
_panel_btlog_event_count = 0


def _remote_path(value):
    """Return a robot-side path without Blender/local path interpretation."""
    return str(value or "").strip().replace("\\", "/")


def _controller_properties(scene=None):
    """Resolve controller properties even when a timer has no context."""
    global _bridge_scene
    if scene is not None:
        _bridge_scene = scene
    else:
        scene = getattr(bpy.context, "scene", None) or _bridge_scene
        if scene is not None:
            _bridge_scene = scene
    return getattr(scene, "mmrt_remote_controller", None)


def _tag_ui_redraw():
    window_manager = getattr(bpy.context, "window_manager", None)
    for window in getattr(window_manager, "windows", ()):
        screen = getattr(window, "screen", None)
        for area in getattr(screen, "areas", ()):
            try:
                area.tag_redraw()
            except (AttributeError, RuntimeError):
                pass


def _service_response_details(event):
    values = event.get("values")
    if not isinstance(values, dict):
        values = {}

    result = event.get("result")
    success = values.get("success")
    if success is None:
        success = values.get("result")
    if success is None:
        success = result
    if success is None:
        # A few bridge/server combinations omit both the top-level result
        # and a response success field. An error response is still explicit;
        # otherwise the service response itself is the successful ack.
        success = not bool(event.get("error"))

    message = values.get("message")
    if not message:
        message = event.get("error") or (
            "Service request succeeded" if success else "Service request failed"
        )
    return bool(success), str(message)


def _set_console_status(message, success=None):
    props = _controller_properties()
    if props is None:
        return
    props.last_console_status = message
    if success is not None:
        props.last_console_success = success
    _tag_ui_redraw()


def _set_bridge_status(status, connected=False):
    props = _controller_properties()
    if props is None:
        return
    props.bridge_status = status
    props.bridge_connected = connected
    _tag_ui_redraw()


def _bridge_is_running():
    process = ProcessManager.get_process("ros_bridge")
    return process is not None and process.is_running()


def _bridge_is_connected():
    props = _controller_properties()
    return bool(props is not None and props.bridge_connected and _bridge_client.running)


def _clear_bridge_connection_check():
    global _bridge_connection_timer
    timer = _bridge_connection_timer
    _bridge_connection_timer = None
    if timer is not None:
        try:
            bpy.app.timers.unregister(timer)
        except (ValueError, RuntimeError):
            pass


def _clear_service_response_timeout(request_id):
    timer = _service_response_timers.pop(request_id, None)
    if timer is not None:
        try:
            bpy.app.timers.unregister(timer)
        except (ValueError, RuntimeError):
            pass


def _clear_service_response_timeouts(message=None):
    request_ids = list(_service_response_timers)
    for request_id in request_ids:
        _clear_service_response_timeout(request_id)
        _bridge_client.cancel_request(request_id)
    if request_ids and message:
        _set_console_status(message, False)


def _begin_bridge_connection_check(status_file):
    global _bridge_connection_timer
    _clear_bridge_connection_check()
    # Give the helper a short margin to print its final retry error and exit.
    deadline = time.monotonic() + RobotServerConfig.BRIDGE_CONNECTION_TIMEOUT_SECONDS + 2.0

    def check_bridge_process():
        global _bridge_connection_timer
        if not _bridge_is_running():
            _bridge_connection_timer = None
            _set_bridge_status("Disconnected")
            return None
        try:
            with open(status_file, "r", encoding="utf-8") as marker:
                status = json.load(marker)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            status = None
        if isinstance(status, dict) and status.get("state") == "connected":
            _bridge_connection_timer = None
            try:
                os.remove(status_file)
            except OSError:
                pass
            _set_bridge_status("Connected", connected=True)
            print("[mmrt] ROS Bridge WebSocket handshake received by Blender.")
            return None
        if time.monotonic() >= deadline:
            _bridge_connection_timer = None
            ProcessManager.stop_process("ros_bridge")
            try:
                os.remove(status_file)
            except OSError:
                pass
            message = "Connection timed out before the WebSocket handshake completed"
            _set_bridge_status(message)
            _set_console_status(message, False)
            return None
        return 0.1

    _bridge_connection_timer = check_bridge_process
    bpy.app.timers.register(
        check_bridge_process,
        first_interval=0.1,
    )


def send_bridge_request(req, action, on_success=None):
    """Send a configured rosbridge request and handle its async response."""
    if not _bridge_is_connected():
        raise RuntimeError("ROS bridge is not connected. Start the bridge and wait for it to connect.")
    _ensure_bridge_event_timer()

    def on_response(event):
        _clear_service_response_timeout(request_id)
        if event.get("op") == "status":
            success = False
            message = event.get("msg") or event.get("message") or "rosbridge rejected the request"
        else:
            success, message = _service_response_details(event)
        _set_console_status(f"{action}: {message}", success)
        print(f"[mmrt] Resolved {action} request {request_id}: {message}")
        if success and on_success:
            on_success(message)

    request_id = _bridge_client.send(req, callback=on_response)
    _set_console_status(f"{action}: waiting for response")

    def on_response_timeout():
        _service_response_timers.pop(request_id, None)
        _bridge_client.cancel_request(request_id)
        _set_console_status(
            f"{action}: no response from the robot after "
            f"{RobotServerConfig.BRIDGE_SERVICE_RESPONSE_TIMEOUT_SECONDS:.0f} seconds",
            False,
        )
        return None

    _service_response_timers[request_id] = on_response_timeout
    bpy.app.timers.register(
        on_response_timeout,
        first_interval=RobotServerConfig.BRIDGE_SERVICE_RESPONSE_TIMEOUT_SECONDS,
    )
    return request_id


def _remote_log_text(event):
    msg = event.get("msg", {})
    if not isinstance(msg, dict):
        return str(msg)

    # BTLog is std_msgs/msg/String.
    if msg.get("data"):
        return str(msg["data"])

    # /rosout is rcl_interfaces/msg/Log.
    text = msg.get("msg") or msg.get("message")
    if not text:
        return ""
    node_name = msg.get("name")
    return f"[{node_name}] {text}" if node_name else str(text)


def _handle_bridge_events():
    global _panel_btlog_event_count
    for event in _bridge_client.poll():
        try:
            operation = event.get("op")
            if operation == "status":
                level = str(event.get("level", "")).lower()
                message = event.get("msg") or event.get("message")
                if message and level in {"error", "warning"}:
                    detail = f"rosbridge {level}: {message}"
                    _set_bridge_status(
                        f"Connected; {detail}",
                        connected=_bridge_is_running(),
                    )
                    show_report(
                        detail,
                        level="ERROR" if level == "error" else "WARNING",
                    )
                continue
            if operation == RobotServerConfig.ROSBRIDGE_SERVICE_RESPONSE_OP:
                success, message = _service_response_details(event)
                _set_console_status(f"Service response: {message}", success)
                print(f"[mmrt] Updated panel status from service response: {message}")
                continue
            if operation != RobotServerConfig.ROSBRIDGE_PUBLISH_OP:
                continue
            text = _remote_log_text(event)
            if not text:
                continue
            props = _controller_properties()
            if props is not None:
                text = text[-4096:]
                props.last_remote_log = text
                log_item = props.remote_logs.add()
                log_item.message = text
                while len(props.remote_logs) > UIConfig.REMOTE_LOG_COUNT:
                    props.remote_logs.remove(0)
                _tag_ui_redraw()
                if event.get("topic") == RobotServerConfig.BT_LOG_TOPIC:
                    _panel_btlog_event_count += 1
                    if _panel_btlog_event_count == 1 or _panel_btlog_event_count % 100 == 0:
                        print(
                            "[mmrt] Added "
                            f"{_panel_btlog_event_count} BTLog messages to the panel."
                        )
        except Exception as exc:
            # Drop only the malformed event; keep processing later responses
            # and log messages in the same timer tick.
            print(f"[MMRT] Ignoring malformed bridge event: {exc}")


def _poll_bridge_events():
    global _bridge_poll_tick_logged
    if not _bridge_timer_registered:
        return None
    if not _bridge_poll_tick_logged:
        _bridge_poll_tick_logged = True
        print("[mmrt] Bridge event poll timer is active.")
    try:
        _handle_bridge_events()
        if not _bridge_is_running():
            props = _controller_properties()
            if props is not None and props.bridge_connected:
                _set_bridge_status("Disconnected")
    except Exception as exc:
        # Keep the event timer alive if a UI or message update fails.
        print(f"[MMRT] Bridge event handling error: {exc}")
    return 0.1


def _ensure_bridge_event_timer():
    """Restore the poll timer if Blender removed it during a file load."""
    global _bridge_timer_registered, _bridge_poll_tick_logged
    try:
        is_registered = bpy.app.timers.is_registered(_poll_bridge_events)
    except AttributeError:
        is_registered = _bridge_timer_registered
    if not is_registered:
        _bridge_poll_tick_logged = False
        bpy.app.timers.register(
            _poll_bridge_events,
            first_interval=0.1,
            persistent=True,
        )
        print("[mmrt] Bridge event poll timer registered.")
    _bridge_timer_registered = True


class MMRT_OT_SendTreeFiles(bpy.types.Operator):
    bl_idname = get_bl_idname("send_tree_files")
    bl_label = "Send Application Folder"

    def execute(self, context):
        scene = context.scene
        sim_props = scene.mmrt_remote_controller

        # The robot consumes an application folder. The source can either be
        # the generated MMRT output or a user-selected local folder.
        if sim_props.use_mmrt_xml:
            app_dir = get_local_mmrt_app_dir(scene)
            required_dirs = (
                app_dir,
                os.path.join(app_dir, GeneratedAppDirectories.TREE_DIR),
                os.path.join(app_dir, GeneratedAppDirectories.BLACKBOARD_DIR),
                os.path.join(app_dir, GeneratedAppDirectories.MESH_DIR),
            )
            required_files = (
                get_mmrt_xml_filepath(scene),
                os.path.join(app_dir, GeneratedAppDirectories.BLACKBOARD_DIR, GeneratedAppFilenames.SYSTEM_BLACKBOARD_FILE),
                os.path.join(app_dir, GeneratedAppDirectories.BLACKBOARD_DIR, GeneratedAppFilenames.MOTION_BLACKBOARD_FILE),
                os.path.join(app_dir, GeneratedAppDirectories.BLACKBOARD_DIR, GeneratedAppFilenames.ENVIRONMENT_BLACKBOARD_FILE),
                os.path.join(app_dir, GeneratedAppFilenames.APP_CONFIG_FILE),
            )
            if not all(os.path.isdir(path) for path in required_dirs) or not all(
                os.path.isfile(path) for path in required_files
            ):
                self.report({'ERROR'}, f"Generated MMRT app is incomplete: {app_dir}")
                return {'CANCELLED'}
        else:
            local_app_path = sim_props.local_app_path.strip()
            if not local_app_path:
                self.report({'ERROR'}, "Choose a local application folder first.")
                return {'CANCELLED'}
            app_dir = bpy.path.abspath(local_app_path)
            if not os.path.isdir(app_dir):
                self.report({'ERROR'}, f"Local application folder does not exist: {app_dir}")
                return {'CANCELLED'}

        dest_dir = _remote_path(sim_props.remote_dest)
        if not dest_dir:
            self.report({'ERROR'}, "Remote destination path is empty.")
            return {'CANCELLED'}

        external_python = (
            bpy.path.abspath(scene.mmrt_external_python_path)
            if scene.mmrt_external_python_path
            else None
        )
        use_development_python = scene.mmrt_use_development_python
        try:
            backend_command = get_backend_command(
                "send",
                external_python,
                use_development_python,
            )
        except (RuntimeError, ValueError) as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}

        addon_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        try:
            args = [
                f"--ip={sim_props.remote_ip}",
                f"--user={sim_props.remote_user}",
                f"--password={sim_props.remote_password}",
                f"--local={app_dir}",
                f"--dest={dest_dir}",
            ]

            def on_finished(proc):
                if proc.returncode != 0:
                    err = ""
                    if proc.stderr:
                        try:
                            err = proc.stderr.read().strip()
                        except Exception as exc:
                            err = f"Error reading stderr: {exc}"
                    show_report(err or "Upload failed (no error message).", level='ERROR')
                else:
                    show_report("Upload completed.", level='INFO')

            ProcessManager.launch_process(
                name=f"send {os.path.basename(app_dir)}",
                default_cmd=backend_command[0],
                extra_args=backend_command[1:] + args,
                cwd=addon_dir,
                console_output=False,
                on_exit=on_finished,
            )

            self.report({'INFO'}, "Uploading… (check popup for results)")
            return {'FINISHED'}
        except Exception as exc:
            self.report({'ERROR'}, f"Upload failed: {exc}")
            return {'CANCELLED'}


class MMRT_OT_StartBridge(bpy.types.Operator):
    bl_idname = get_bl_idname("start_bridge")
    bl_label = "Start ROS Bridge client"

    def execute(self, context):
        scene = context.scene
        sim_props = scene.mmrt_remote_controller
        _ensure_bridge_event_timer()

        if not _bridge_client.running:
            message = (
                "Blender's local ROS Bridge event listener is unavailable. "
                "Reload the add-on and check whether another Blender instance is using it."
            )
            _set_bridge_status("Disconnected")
            self.report({'ERROR'}, message)
            return {'CANCELLED'}

        addon_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        external_python = (
            bpy.path.abspath(scene.mmrt_external_python_path)
            if scene.mmrt_external_python_path
            else None
        )
        use_development_python = scene.mmrt_use_development_python
        try:
            backend_command = get_backend_command(
                "bridge",
                external_python,
                use_development_python,
            )
        except (RuntimeError, ValueError) as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}

        args = [
            RobotServerConfig.BRIDGE_REMOTE_IP_ARGUMENT,
            sim_props.remote_ip,
            RobotServerConfig.BRIDGE_REMOTE_PORT_ARGUMENT,
            str(sim_props.remote_port),
            RobotServerConfig.BRIDGE_STATUS_FILE_ARGUMENT,
            os.path.join(
                tempfile.gettempdir(),
                f"mmrt_ros_bridge_{uuid.uuid4().hex}.json",
            ),
        ]
        status_file = args[-1]
        _set_bridge_status("Connecting…")

        def on_finished(proc):
            _clear_bridge_connection_check()
            try:
                os.remove(status_file)
            except OSError:
                pass
            _clear_service_response_timeouts(
                "ROS bridge stopped before the service response arrived"
            )
            if proc is None:
                _set_bridge_status("Disconnected")
                return
            if proc.returncode != 0:
                err = ""
                if proc.stderr:
                    try:
                        err = proc.stderr.read().strip()
                    except Exception as exc:
                        err = f"Error reading stderr: {exc}"
                _set_bridge_status(
                    f"Connection failed: {err or 'Bridge failed (no error message).'}"
                )
                show_report(err or "Bridge failed (no error message).", level='ERROR')
            else:
                _set_bridge_status("Disconnected")
                show_report("Bridge closed", level='INFO')

        try:
            ProcessManager.launch_process(
                name="ros_bridge",
                default_cmd=backend_command[0],
                extra_args=backend_command[1:] + args,
                cwd=addon_dir,
                console_output=True,
                show_launch_report=False,
                on_exit=on_finished,
            )
            _begin_bridge_connection_check(status_file)
            return {'FINISHED'}
        except Exception as exc:
            _clear_bridge_connection_check()
            try:
                os.remove(status_file)
            except OSError:
                pass
            _set_bridge_status(f"Connection failed: {exc}")
            self.report({'ERROR'}, f"Failed: {exc}")
            return {'CANCELLED'}


class MMRT_OT_StartBridgeServer(bpy.types.Operator):
    bl_idname = get_bl_idname("start_bridge_server")
    bl_label = "Start ROS Bridge server"
    bl_description = (
        "Start the remote ROS Bridge server, or stop it if this controller started it"
    )

    def execute(self, context):
        scene = context.scene
        sim_props = scene.mmrt_remote_controller

        if sim_props.remote_bridge_running:
            sim_props.remote_bridge_status = "Closing SSH session and ROS Bridge…"
            _tag_ui_redraw()
            if ProcessManager.stop_process("ros_bridge_server_session"):
                sim_props.remote_bridge_running = False
                sim_props.remote_bridge_status = "ROS Bridge SSH session closed"
                _tag_ui_redraw()
                self.report({'INFO'}, "Remote ROS Bridge session closed.")
                return {'FINISHED'}

            sim_props.remote_bridge_running = False
            sim_props.remote_bridge_status = "No active ROS Bridge SSH session"
            _tag_ui_redraw()
            self.report({'WARNING'}, "No active remote ROS Bridge session was found.")
            return {'CANCELLED'}

        if ProcessManager.get_process("ros_bridge_server_session") is not None:
            self.report({'WARNING'}, "The remote ROS Bridge SSH session is already starting.")
            return {'CANCELLED'}
        if not sim_props.remote_ip.strip():
            self.report({'ERROR'}, "Set the remote IP address first.")
            return {'CANCELLED'}
        if not sim_props.remote_user.strip():
            self.report({'ERROR'}, "Set the remote SSH username first.")
            return {'CANCELLED'}

        addon_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        external_python = (
            bpy.path.abspath(scene.mmrt_external_python_path)
            if scene.mmrt_external_python_path
            else None
        )
        use_development_python = scene.mmrt_use_development_python
        try:
            backend_command = get_backend_command(
                "remote-bridge",
                external_python,
                use_development_python,
            )
        except (RuntimeError, ValueError) as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}

        args = [
            "--ip",
            sim_props.remote_ip.strip(),
            "--user",
            sim_props.remote_user.strip(),
            "--password",
            sim_props.remote_password,
            "--port",
            str(sim_props.remote_port),
            "--domain-id",
            str(sim_props.ros_domain_id),
            "--extra-commands",
            sim_props.remote_ros_commands,
        ]
        sim_props.remote_bridge_status = "Opening remote SSH session…"
        _tag_ui_redraw()

        def on_finished(proc):
            sim_props.remote_bridge_running = False
            if proc is None:
                sim_props.remote_bridge_status = "ROS Bridge SSH session ended"
            elif proc.returncode == 0:
                sim_props.remote_bridge_status = "ROS Bridge SSH session ended"
            else:
                sim_props.remote_bridge_status = (
                    f"SSH session ended (code {proc.returncode}); see Blender console"
                )
                show_report(sim_props.remote_bridge_status, level='WARNING')
            _tag_ui_redraw()

        try:
            ProcessManager.launch_process(
                name="ros_bridge_server_session",
                default_cmd=backend_command[0],
                extra_args=backend_command[1:] + args,
                cwd=addon_dir,
                console_output=True,
                show_launch_report=False,
                stop_timeout=9.0,
                on_exit=on_finished,
            )
            sim_props.remote_bridge_running = True
            sim_props.remote_bridge_status = "SSH session active; remote output is in the console"
            _tag_ui_redraw()
            self.report({'INFO'}, "Remote ROS Bridge SSH session started; see Blender console for output.")
            return {'FINISHED'}
        except Exception as exc:
            sim_props.remote_bridge_status = f"Failed: {exc}"
            sim_props.remote_bridge_running = False
            _tag_ui_redraw()
            self.report({'ERROR'}, f"Failed to start ROS Bridge SSH session: {exc}")
            return {'CANCELLED'}


class MMRT_OT_StopBridge(bpy.types.Operator):
    bl_idname = get_bl_idname("stop_bridge")
    bl_label = "Stop ROS Bridge"

    def execute(self, context):
        if ProcessManager.stop_process("ros_bridge"):
            _clear_bridge_connection_check()
            _clear_service_response_timeouts(
                "ROS bridge stopped before the service response arrived"
            )
            _set_bridge_status("Disconnected")
            self.report({'INFO'}, "ROS Bridge stopped.")
            return {'FINISHED'}
        self.report({'WARNING'}, "ROS Bridge was not running.")
        return {'CANCELLED'}


class MMRT_OT_LoadApplication(bpy.types.Operator):
    bl_idname = get_bl_idname("load_application")
    bl_label = "Load Application"

    def execute(self, context):
        sim_props = context.scene.mmrt_remote_controller
        if not sim_props.is_application:
            self.report({'ERROR'}, "Enable Is Application to load an application folder.")
            return {'CANCELLED'}

        remote_dir = (
            _remote_path(sim_props.remote_dest)
            if sim_props.use_sender_destination
            else _remote_path(sim_props.remote_app_path)
        )
        if not remote_dir:
            self.report({'ERROR'}, "Remote application directory is empty.")
            return {'CANCELLED'}

        try:
            send_bridge_request(load_application_request(remote_dir), "Load application")
        except (RuntimeError, OSError, ValueError) as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        self.report({'INFO'}, f"Sent application load request for {remote_dir}")
        return {'FINISHED'}


class MMRT_OT_LoadConfig(bpy.types.Operator):
    bl_idname = get_bl_idname("load_config")
    bl_label = "Load Tree Config"

    def execute(self, context):
        sim_props = context.scene.mmrt_remote_controller
        if sim_props.is_application:
            self.report({'ERROR'}, "Disable Is Application to load a standalone BT config YAML.")
            return {'CANCELLED'}
        remote_config_path = _remote_path(sim_props.xml_path)
        if not remote_config_path:
            self.report({'ERROR'}, "Set the remote BT config YAML path first.")
            return {'CANCELLED'}

        try:
            send_bridge_request(load_config_request(remote_config_path), "Load config")
        except (RuntimeError, OSError, ValueError) as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        self.report({'INFO'}, f"Sent config load request for {remote_config_path}")
        return {'FINISHED'}


class MMRT_OT_ConfigureTree(bpy.types.Operator):
    bl_idname = get_bl_idname("configure_tree")
    bl_label = "Configure Tree"

    def execute(self, context):
        try:
            send_bridge_request(control_request("configure"), "Configure")
        except (RuntimeError, OSError, ValueError) as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        self.report({'INFO'}, "Sent configure request")
        return {'FINISHED'}


class MMRT_OT_GenTree(bpy.types.Operator):
    bl_idname = get_bl_idname("gen_tree")
    bl_label = "Generate Tree"

    def execute(self, context):
        try:
            send_bridge_request(control_request("generate_tree"), "Generate tree")
        except (RuntimeError, OSError, ValueError) as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        self.report({'INFO'}, "Sent generate-tree request")
        return {'FINISHED'}


class MMRT_OT_RunTree(bpy.types.Operator):
    bl_idname = get_bl_idname("run_tree")
    bl_label = "Run Tree"

    def execute(self, context):
        try:
            send_bridge_request(control_request("run"), "Run tree")
        except (RuntimeError, OSError, ValueError) as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        self.report({'INFO'}, "Sent run request")
        return {'FINISHED'}


class MMRT_OT_PauseTree(bpy.types.Operator):
    bl_idname = get_bl_idname("pause_tree")
    bl_label = "Pause Tree"

    def execute(self, context):
        try:
            send_bridge_request(control_request("pause"), "Pause tree")
        except (RuntimeError, OSError, ValueError) as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        self.report({'INFO'}, "Sent pause request")
        return {'FINISHED'}


class MMRT_OT_StopTree(bpy.types.Operator):
    bl_idname = get_bl_idname("stop_tree")
    bl_label = "Stop Tree"

    def execute(self, context):
        try:
            send_bridge_request(control_request("stop"), "Stop tree")
        except (RuntimeError, OSError, ValueError) as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        self.report({'INFO'}, "Sent stop request")
        return {'FINISHED'}


classes = (
    MMRT_OT_SendTreeFiles,
    MMRT_OT_StartBridgeServer,
    MMRT_OT_StartBridge,
    MMRT_OT_StopBridge,
    MMRT_OT_LoadApplication,
    MMRT_OT_LoadConfig,
    MMRT_OT_ConfigureTree,
    MMRT_OT_GenTree,
    MMRT_OT_RunTree,
    MMRT_OT_PauseTree,
    MMRT_OT_StopTree,
)


def register():
    global _bridge_timer_registered
    for cls in classes:
        bpy.utils.register_class(cls)
    try:
        _bridge_client.start()
    except OSError as exc:
        print(f"[MMRT] Bridge event listener unavailable: {exc}")
    _ensure_bridge_event_timer()


def unregister():
    global _bridge_timer_registered, _bridge_scene
    _bridge_timer_registered = False
    _clear_bridge_connection_check()
    _clear_service_response_timeouts()
    _bridge_client.stop()
    _bridge_scene = None
    try:
        bpy.app.timers.unregister(_poll_bridge_events)
    except (ValueError, RuntimeError):
        pass
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
