"""Resolve the bundled MMRT backend or an explicit development Python."""

from __future__ import annotations

import platform
from pathlib import Path
from typing import Sequence
from .paths import get_system_python_path


_MODES = {
    "process",
    "plot",
    "bridge",
    "send",
    "remote-bridge",
    "health-check",
    "version",
}

_HEALTH_CHECK = r'''
import importlib
from pathlib import Path
dependencies = {
    "numpy": "numpy",
    "scipy": "scipy",
    "pybullet": "pybullet",
    "pybullet_data": "pybullet_data",
    "matplotlib": "matplotlib",
    "matplotlib_backend_svg": "matplotlib.backends.backend_svg",
    "websockets": "websockets",
    "paramiko": "paramiko",
    "yaml": "yaml",
    "rich": "rich",
}
missing = []
for display_name, module_name in dependencies.items():
    try:
        module = importlib.import_module(module_name)
        if module_name == "pybullet_data" and not Path(module.getDataPath()).is_dir():
            raise RuntimeError("pybullet_data data path does not exist")
    except Exception as exc:
        missing.append(f"{display_name}: {exc}")
if missing:
    raise SystemExit("MMRT backend dependency check failed: " + "; ".join(missing))
print("MMRT backend dependencies are available.")
'''.strip()


def addon_root() -> Path:
    return Path(__file__).resolve().parents[2]


def platform_tag() -> str:
    system = platform.system()
    machine = platform.machine().lower()
    architecture = {
        "amd64": "x64",
        "x86_64": "x64",
        "arm64": "arm64",
        "aarch64": "arm64",
    }.get(machine, machine)

    if system == "Windows":
        return f"win-{architecture}"
    if system == "Linux":
        return f"linux-{architecture}"
    if system == "Darwin":
        return f"macos-{architecture}"
    return f"{system.lower()}-{architecture}"


def bundled_backend_path() -> Path | None:
    """Return the generated PyInstaller executable, if this release has one."""

    platform_root = addon_root() / "backend" / platform_tag()
    executable_name = "mmrt_backend.exe" if platform.system() == "Windows" else "mmrt_backend"

    # The builders use --onedir. The first candidate also permits a future
    # --onefile build without changing the resolver.
    candidates = (
        platform_root / executable_name,
        platform_root / "mmrt_backend" / executable_name,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _source_script(mode: str) -> Path:
    addon = addon_root()
    scripts = {
        "process": addon / "processes" / "mmrt" / "main.py",
        "plot": addon / "processes" / "mmrt" / "joint_limit_plotter.py",
        "bridge": addon / "processes" / "helpers" / "ros2_bridge.py",
        "send": addon / "processes" / "helpers" / "send_file.py",
        "remote-bridge": addon / "processes" / "helpers" / "start_remote_rosbridge.py",
    }
    return scripts[mode]


def get_backend_command(
    mode: str,
    external_python: str | None = None,
    use_development_python: bool = False,
) -> list[str]:
    """Return an argv prefix for a backend mode.

    The bundled executable is used by default. The explicit Python path is
    selected only when ``use_development_python`` is enabled and is never
    modified or installed into.
    """

    if mode not in _MODES:
        raise ValueError(f"Unknown MMRT backend mode: {mode}")

    if use_development_python:
        if not external_python:
            raise RuntimeError(
                "Use Development Python is enabled, but no development Python "
                "executable has been configured in Advanced Settings."
            )

        if mode == "health-check":
            return [external_python, "-c", _HEALTH_CHECK]
        if mode == "version":
            return [external_python, "--version"]
        return [external_python, str(_source_script(mode))]

    bundled = bundled_backend_path()
    if bundled is not None:
        return [str(bundled), mode]

    # Source distributions use the system Python when no packaged backend
    # exists, since Blender's bundled Python may not include MMRT dependencies.
    system_python = get_system_python_path()
    if mode == "health-check":
        return [system_python, "-c", _HEALTH_CHECK]
    if mode == "version":
        return [system_python, "--version"]
    return [system_python, str(_source_script(mode))]


def backend_location_text(
    external_python: str | None = None,
    use_development_python: bool = False,
) -> str:
    if use_development_python:
        return f"external development Python: {external_python}"

    bundled = bundled_backend_path()
    if bundled is not None:
        return str(bundled)
    return f"system Python fallback: {get_system_python_path()}"


__all__: Sequence[str] = [
    "addon_root",
    "backend_location_text",
    "bundled_backend_path",
    "get_backend_command",
    "platform_tag",
]
