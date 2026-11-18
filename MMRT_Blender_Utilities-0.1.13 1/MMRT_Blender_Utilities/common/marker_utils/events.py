from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from .types import FlagTypes, LEDColors, LEDModes


@dataclass
class MarkerEvents:
    """Marker-derived recording events, independent of Blender state."""

    flags: list[tuple[int, str]] = field(default_factory=list)
    custom_flags: list[tuple[int, str]] = field(default_factory=list)
    io_events: list[tuple[int, int, float]] = field(default_factory=list)
    led_events: list[tuple[int, list[tuple[int, str, str, float]]]] = field(
        default_factory=list
    )
    forced_wp_names: dict[int, str] = field(default_factory=dict)


def collect_marker_events(
    markers: Iterable[Any],
    *,
    remap_frame: Callable[[float], int],
    start_frame: int,
    end_frame: int,
) -> MarkerEvents:
    """Collect marker events without reading or mutating Blender global state."""
    marker_items = list(markers)
    events = MarkerEvents()

    def event_items(marker):
        """Return nested events, with a legacy single-event fallback."""
        nested = getattr(marker, "events", None)
        if nested is not None:
            return nested
        return [marker]

    for marker in marker_items:
        stretched_frame = remap_frame(marker.frame)
        if not start_frame <= stretched_frame <= end_frame:
            continue

        for item in event_items(marker):
            code = FlagTypes.from_name(item.flag_type)

            if code == FlagTypes.IO_CONTROL.value:
                events.io_events.append(
                    (stretched_frame, item.io_byte(), item.io_delay)
                )
                continue

            if code == FlagTypes.LED_CONTROL.value:
                leds = []

                if item.led1_is_send:
                    color = (
                        "#" + item.led_hex(1)
                        if item.led1_color_preset == LEDColors.CUSTOM.value
                        else item.led1_color_preset.lower()
                    )
                    value = (
                        item.led1_progress
                        if item.led1_mode == LEDModes.PROGRESS.value
                        else item.led1_extra
                    )
                    leds.append((1, item.led1_mode.lower(), color, value))

                if item.led2_is_send:
                    color = (
                        "#" + item.led_hex(2)
                        if item.led2_color_preset == LEDColors.CUSTOM.value
                        else item.led2_color_preset.lower()
                    )
                    value = (
                        item.led2_progress
                        if item.led2_mode == LEDModes.PROGRESS.value
                        else item.led2_extra
                    )
                    leds.append((2, item.led2_mode.lower(), color, value))

                if leds:
                    events.led_events.append((stretched_frame, leds))
                continue

            if code == FlagTypes.FORCED_WP_NAME.value:
                events.forced_wp_names[stretched_frame] = item.forced_wp_name
                continue

            if item.flag_type == FlagTypes.CUSTOM.name:
                events.custom_flags.append(
                    (stretched_frame, item.custom_flag_name)
                )
            else:
                events.flags.append((stretched_frame, code))

    return events
