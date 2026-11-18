from .types import FlagTypes


FLAG_ICONS = {
    FlagTypes.CAMERA_CAPTURE.name: "RENDER_STILL",
    FlagTypes.CAMERA_RECORD_START.name: "REC",
    FlagTypes.CAMERA_RECORD_STOP.name: "FILE_TICK",
    FlagTypes.IO_CONTROL.name: "TOOL_SETTINGS",
    FlagTypes.LED_CONTROL.name: "LIGHT",
    FlagTypes.RETRY_TRIGGER.name: "ERROR",
    FlagTypes.RETRY_EVENT_START.name: "LOOP_FORWARDS",
    FlagTypes.RETRY_EVENT_END.name: "LOOP_BACK",
    FlagTypes.FORCED_SPLIT.name: "SPLIT_HORIZONTAL",
    FlagTypes.FORCED_WP_NAME.name: "FONT_DATA",
    FlagTypes.CUSTOM.name: "BOOKMARKS",
}
