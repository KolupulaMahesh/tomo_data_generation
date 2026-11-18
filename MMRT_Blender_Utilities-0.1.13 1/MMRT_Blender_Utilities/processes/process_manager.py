import os
import signal
import subprocess
import bpy
import shutil
import platform
import sys
import shlex
from ..common.blender import show_report
from bpy.app.handlers import persistent


def safe_which(cmd):
    # if sys.version_info >= (3, 12):
    #     return os.which(cmd)
    # else:
    return shutil.which(cmd)
    

class SubprocessHandler:
    def __init__(self, name, default_cmd, use_alias=False):
        self.name = name
        self.cmd = default_cmd
        self.use_alias = use_alias
        self.process = None
        self._timer = None
        self._registry_key = None
        self._on_registry_cleanup = None
        self._stop_timeout = 0.0

    def is_running(self):
        return self.process and self.process.poll() is None

    @staticmethod
    def _command_tokens(command):
        """Convert an executable command into argv without splitting paths."""
        if isinstance(command, (list, tuple)):
            return [os.fspath(token) for token in command]
        if isinstance(command, os.PathLike):
            return [os.fspath(command)]
        if isinstance(command, str):
            # A string represents one executable. Callers that need command
            # arguments should pass a list; this preserves paths such as
            # ``C:\Program Files\...`` on Windows.
            return [command]
        raise TypeError(f"Unsupported command type: {type(command).__name__}")

    @staticmethod
    def _argument_tokens(extra_args):
        if extra_args is None:
            return []
        if isinstance(extra_args, (list, tuple)):
            return [os.fspath(token) for token in extra_args]
        if isinstance(extra_args, str):
            return shlex.split(extra_args, posix=platform.system() != "Windows")
        raise TypeError(f"Unsupported argument type: {type(extra_args).__name__}")

    def launch(self, override_cmd=None, extra_args=None, on_exit=None, console_output=True, **kwargs):
        if self.is_running():
            show_report(f"{self.name} process is already running.", level='WARNING')
            raise RuntimeError(f"{self.name} process is already running.")
        
        command = override_cmd if override_cmd is not None else self.cmd
        cmd_tokens = self._command_tokens(command)
        argument_tokens = self._argument_tokens(extra_args)
        if not cmd_tokens or not cmd_tokens[0]:
            raise RuntimeError(f"{self.name} has no executable command.")

        alias_name = cmd_tokens[0]

        if self.use_alias and platform.system() != "Windows":
            shell_command = f"source ~/.bashrc && alias {alias_name}"
            result = subprocess.run(
                ["bash", "-i", "-c", shell_command],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            if result.returncode != 0:
                raise RuntimeError(f"Alias '{alias_name}' not found.")

            full_command = shlex.join(cmd_tokens + argument_tokens)
            shell_exec = f"source ~/.bashrc && {full_command}"
            full_cmd = ["bash", "-i", "-c", shell_exec]
        else:
            if platform.system() != "Windows":
                if os.path.dirname(alias_name):
                    if not os.path.isfile(alias_name):
                        raise RuntimeError(f"Executable not found: {alias_name}")
                elif not safe_which(alias_name):
                    raise RuntimeError(f"Executable not found: {alias_name}")
            full_cmd = cmd_tokens + argument_tokens

        if platform.system() == 'Windows':
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
            preexec_fn = None
        else:
            creationflags = 0
            preexec_fn = os.setsid

        cwd_dir = kwargs.get("cwd", os.path.expanduser("~"))
        self._stop_timeout = max(0.0, float(kwargs.get("stop_timeout", 0.0)))

        log_path = kwargs.get("log_path")
        if log_path:
            # A file avoids pipe deadlocks while Blender polls the output from
            # its main thread. Close only the parent's handle after spawning.
            with open(log_path, "wb", buffering=0) as log_file:
                self.process = subprocess.Popen(
                    full_cmd,
                    cwd=cwd_dir,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    env=kwargs.get("env"),
                    preexec_fn=preexec_fn,
                    creationflags=creationflags,
                )
        elif console_output:
            self.process = subprocess.Popen(
                full_cmd,
                cwd=cwd_dir,
                stdout=None,
                stderr=None,
                preexec_fn=preexec_fn,
                creationflags=creationflags
            )
        else:
            self.process = subprocess.Popen(
                full_cmd,
                cwd=cwd_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                preexec_fn=preexec_fn,
                creationflags=creationflags,
                bufsize=1
            )
        if kwargs.get("show_launch_report", True):
            show_report(f"Launched {self.name} process.", level='INFO')


        def check_proc():
            if not self.is_running():
                show_report(f"{self.name} process exited.", level='INFO')
                process = self.process
                if on_exit and process is not None:
                    try:
                        on_exit(process)
                    except Exception as e:
                        show_report(f"[MMRT] on_exit() error: {e}", level='ERROR')

                if self._on_registry_cleanup:
                    try:
                        self._on_registry_cleanup(self)
                    except Exception as e:
                        show_report(f"[MMRT] Registry cleanup error: {e}", level='ERROR')

                self.process = None
                self._timer = None
                return None

            return 1.0

        self._timer = bpy.app.timers.register(check_proc)

    def stop(self):
        timer = self._timer
        self._timer = None
        if timer:
            try:
                bpy.app.timers.unregister(timer)
            except (ValueError, RuntimeError):
                pass

        if self.process is not None:
            process = self.process
            if self.is_running():
                try:
                    if platform.system() == "Windows":
                        break_event = getattr(signal, "CTRL_BREAK_EVENT", None)
                        if self._stop_timeout and break_event is not None:
                            try:
                                process.send_signal(break_event)
                            except (OSError, ValueError):
                                process.terminate()
                        else:
                            process.terminate()
                    else:
                        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                except Exception as e:
                    show_report(f"Failed to stop {self.name}: {e}", level='ERROR')
                if self._stop_timeout:
                    try:
                        process.wait(timeout=self._stop_timeout)
                    except subprocess.TimeoutExpired:
                        try:
                            if platform.system() == "Windows":
                                process.kill()
                            else:
                                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                            process.wait(timeout=2.0)
                        except Exception as e:
                            show_report(f"Failed to finish stopping {self.name}: {e}", level='ERROR')
            self.process = None
        
        if self._on_registry_cleanup:
            try:
                self._on_registry_cleanup(self)
            except Exception as e:
                show_report(f"[MMRT] Registry cleanup error: {e}", level='ERROR')
    
    def set_registry_binding(self, registry_key, cleanup_callback):
        self._registry_key = registry_key
        self._on_registry_cleanup = cleanup_callback


class ProcessManager:
    _processes: dict[str, SubprocessHandler] = {}
    _handler_registered = False

    @staticmethod
    @persistent
    def _cleanup_on_load_pre(*args):
        ProcessManager.stop_all_processes()

    @staticmethod
    @persistent
    def _cleanup_on_quit_pre(*args):
        ProcessManager.stop_all_processes()

    @classmethod
    def launch_process(cls, name, default_cmd, **launch_kwargs):
        proc = SubprocessHandler(name=name, default_cmd=default_cmd)
        cls.register_process(name, proc)
        try:
            proc.launch(**launch_kwargs)
        except Exception:
            cls.unregister_process(name)
            raise
        return proc
    
    @classmethod
    def stop_process(cls, name):
        proc = cls.get_process(name)
        if proc:
            proc.stop()
            cls.unregister_process(name)
            return True
        return False

    @classmethod
    def _ensure_handlers_registered(cls):
        if cls._handler_registered:
            return
        
        if cls._cleanup_on_load_pre not in bpy.app.handlers.load_pre:
            bpy.app.handlers.load_pre.append(cls._cleanup_on_load_pre)
        quit_handlers = getattr(bpy.app.handlers, "quit_pre", None)
        if quit_handlers is not None and cls._cleanup_on_quit_pre not in quit_handlers:
            quit_handlers.append(cls._cleanup_on_quit_pre)
        
        cls._handler_registered = True

    @classmethod
    def _remove_if_same(cls, name, process_manager):
        current = cls._processes.get(name)
        if current is process_manager:
            del cls._processes[name]

    @classmethod
    def get_process(cls, name):
        return cls._processes.get(name)

    @classmethod
    def register_process(cls, name, process_manager: SubprocessHandler):
        cls._ensure_handlers_registered()

        existing = cls._processes.get(name)
        if existing and existing is not process_manager:
            existing.stop()
        process_manager.set_registry_binding(
            registry_key=name, 
            cleanup_callback=lambda proc: cls._remove_if_same(name, proc))
        cls._processes[name] = process_manager

    @classmethod
    def unregister_process(cls, name):
        process = cls._processes.pop(name, None)
        if process:
            process.set_registry_binding(registry_key=None, cleanup_callback=None)
    
    @classmethod
    def stop_all_processes(cls):
        for process in list(cls._processes.values()):
            process.stop()
        cls._processes.clear()
    
    @classmethod
    def register_handlers(cls):
        cls._ensure_handlers_registered()

    @classmethod
    def unregister_handlers(cls):
        if cls._cleanup_on_load_pre in bpy.app.handlers.load_pre:
            bpy.app.handlers.load_pre.remove(cls._cleanup_on_load_pre)
        quit_handlers = getattr(bpy.app.handlers, "quit_pre", None)
        if quit_handlers is not None and cls._cleanup_on_quit_pre in quit_handlers:
            quit_handlers.remove(cls._cleanup_on_quit_pre)

        cls._handler_registered = False
