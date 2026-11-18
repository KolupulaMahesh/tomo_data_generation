from .handlers import (
    CustomMarkerHandler,
    IOMarkerHandler,
    LEDMarkerHandler,
    WaypointMarkerHandler,
)
from .types import FlagTypes


HANDLER_REGISTRY = {
    FlagTypes.LED_CONTROL.value: LEDMarkerHandler,
    FlagTypes.IO_CONTROL.value: IOMarkerHandler,
    FlagTypes.CUSTOM.value: CustomMarkerHandler,
    FlagTypes.FORCED_WP_NAME.value: WaypointMarkerHandler,
}
