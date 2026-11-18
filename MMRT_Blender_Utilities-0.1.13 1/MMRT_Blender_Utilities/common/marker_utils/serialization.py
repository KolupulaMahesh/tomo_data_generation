from typing import TextIO
from ...processes.mmrt.definitions import MarkerEventSections

from .events import MarkerEvents


def serialize_marker_events(events: MarkerEvents) -> str:
    """Serialize marker sections using the recorder's existing YAML schema."""
    lines = []

    if events.flags:
        lines.append(
            f"{MarkerEventSections.FLAGS}: ["
            + ", ".join(f"[{frame},{code}]" for frame, code in events.flags)
            + "]\n"
        )

    if events.custom_flags:
        lines.append(
            f"{MarkerEventSections.CUSTOM_FLAGS}: ["
            + ", ".join(
                f"[{frame},{name}]" for frame, name in events.custom_flags
            )
            + "]\n"
        )

    if events.io_events:
        lines.append(f"{MarkerEventSections.IO_EVENTS}:\n")
        for frame, byte, delay in events.io_events:
            lines.append(f"  - [{frame},{byte},{delay}]\n")

    if events.led_events:
        lines.append(f"{MarkerEventSections.LED_EVENTS}:\n")
        for frame, leds in events.led_events:
            led_string = ", ".join(
                f'[{led_id},"{mode}","{color}",{value}]'
                for led_id, mode, color, value in leds
            )
            lines.append(f"  - [{frame}, [{led_string}]]\n")

    if events.forced_wp_names:
        lines.append(f"{MarkerEventSections.FORCED_WAYPOINT_NAMES}:\n")
        for frame, name in events.forced_wp_names.items():
            lines.append(f'  {frame}: "{name}"\n')

    return "".join(lines)


def write_marker_events(stream: TextIO, events: MarkerEvents) -> None:
    """Write serialized marker sections to an open text stream."""
    stream.write(serialize_marker_events(events))
