"""Operating-system and filesystem helpers."""

from .backend import (
    addon_root,
    backend_location_text,
    bundled_backend_path,
    get_backend_command,
    platform_tag,
)
from .paths import current_username, get_system_python_path, normalize_output_dir

__all__ = [
    "addon_root",
    "backend_location_text",
    "bundled_backend_path",
    "current_username",
    "get_backend_command",
    "get_system_python_path",
    "normalize_output_dir",
    "platform_tag",
]
