from .base import MarkerHandler
from ..types import LEDColors, LEDModes

class LEDMarkerHandler(MarkerHandler):
    @staticmethod
    def draw_ui(layout, row, item):
        COLOR_MODES = {
            LEDModes.SET.value, LEDModes.FADE.value, LEDModes.SPIN.value,
            LEDModes.HEARTBEAT.value, LEDModes.BPM.value, LEDModes.PROGRESS.value,
        }

        EXTRA_MODES = {
            LEDModes.FADE.value, LEDModes.SPIN.value, LEDModes.HEARTBEAT.value,
            LEDModes.BPM.value, LEDModes.RANDOM.value,
        }

        # Uses layout.column instead of the main row to stack vertically
        col = layout.column(align=True)

        # Row 1: checkboxes
        r = col.row(align=True)
        r.separator(factor=1.2)
        r.prop(item, "led1_is_send", text="LED 1")
        r.prop(item, "led2_is_send", text="LED 2")

        # Row 2: LED 1
        if item.led1_is_send:
            r = col.row(align=True)
            r.separator(factor=2.5)
            r.label(text="L1")
            r.prop(item, "led1_mode", text="")

            if item.led1_mode in COLOR_MODES:
                r.prop(item, "led1_color_preset", text="")
                if item.led1_color_preset == LEDColors.CUSTOM.value:
                    r.prop(item, "led1_color", text="")

            if item.led1_mode == LEDModes.PROGRESS.value:
                r.prop(item, "led1_progress", text="%", slider=True)

            if item.led1_mode in EXTRA_MODES:
                r.prop(item, "led1_extra", text="")

        # Row 3: LED 2
        if item.led2_is_send:
            r = col.row(align=True)
            r.separator(factor=2.5)
            r.label(text="L2")
            r.prop(item, "led2_mode", text="")

            if item.led2_mode in COLOR_MODES:
                r.prop(item, "led2_color_preset", text="")
                if item.led2_color_preset == LEDColors.CUSTOM.value:
                    r.prop(item, "led2_color", text="")

            if item.led2_mode == LEDModes.PROGRESS.value:
                r.prop(item, "led2_progress", text="%", slider=True)

            if item.led2_mode in EXTRA_MODES:
                r.prop(item, "led2_extra", text="")

    @staticmethod
    def backup_data(item):
        return {
            "led1_is_send": item.led1_is_send,
            "led1_mode": item.led1_mode,
            "led1_color_preset": item.led1_color_preset,
            "led1_color": list(item.led1_color),
            "led1_progress": item.led1_progress,
            "led1_extra": item.led1_extra,
            "led2_is_send": item.led2_is_send,
            "led2_mode": item.led2_mode,
            "led2_color_preset": item.led2_color_preset,
            "led2_color": list(item.led2_color),
            "led2_progress": item.led2_progress,
            "led2_extra": item.led2_extra,
        }

    @staticmethod
    def restore_data(item, data):
        for key, value in data.items():
            if hasattr(item, key):
                setattr(item, key, value)
