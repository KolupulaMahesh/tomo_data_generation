import os
import getpass
import platform
import subprocess
import tempfile

def current_username():
    # 1) try getpass (cross-platform, env-based)
    try:
        user = getpass.getuser()
        if user:
            return user
    except Exception:
        pass

    # 2) try os.getlogin() (may fail in some non-interactive contexts)
    try:
        user = os.getlogin()
        if user:
            return user
    except Exception:
        pass

    # 3) on Unix, use pwd to get effective/real uid name (reliable for system processes)
    if platform.system() != "Windows":
        try:
            import pwd
            user = pwd.getpwuid(os.geteuid()).pw_name
            if user:
                return user
        except Exception:
            pass

    # 4) environment variables fallback (Windows: USERNAME, Unix: USER/LOGNAME)
    for envvar in ("USERNAME", "USER", "LOGNAME"):
        user = os.environ.get(envvar)
        if user:
            return user

    # 5) last resort: hostname to indicate unknown user
    return f"<unknown:{platform.node()}>"


def get_system_python_path():
    system = platform.system()
    try:
        if system == "Windows":
            # 'where' command lists all python executables
            result = subprocess.run(
                ["where", "python"],
                capture_output=True, text=True, check=True
            )
            paths = result.stdout.strip().splitlines()
            # pick the one not inside Blender
            for p in paths:
                if "blender" not in p.lower():
                    return p
        else:
            # On Linux/macOS, use 'which python3'
            result = subprocess.run(
                ["which", "python3"],
                capture_output=True, text=True, check=True
            )
            path = result.stdout.strip()
            if path and "blender" not in path.lower():
                return path
    except Exception:
        pass

    # fallback: guess system python locations
    if system == "Windows":
        return r"C:\Python311\python.exe"
    else:
        return "/usr/bin/python3"


def normalize_output_dir(path, create=True):
    """
    Normalize output directory:
    - if Blender file unsaved and path starts with // -> use temp dir
    - resolve Blender relative paths
    - optionally create directory if missing
    - return absolute normalized path
    """
    path = path.strip()

    if path.startswith("//"):
        try:
            import bpy
            if not bpy.data.filepath:   # unsaved blend
                rel_path = path[2:]
                path = os.path.join(tempfile.gettempdir(), rel_path)
            else:
                path = bpy.path.abspath(path)
        except Exception:
            path = os.path.join(tempfile.gettempdir(), path[2:])

    path = os.path.abspath(os.path.normpath(path))
    if create:
        os.makedirs(path, exist_ok=True)

    return path
