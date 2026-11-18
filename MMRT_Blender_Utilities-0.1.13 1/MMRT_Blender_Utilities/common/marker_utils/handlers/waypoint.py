from .base import MarkerHandler

class WaypointMarkerHandler(MarkerHandler):
    @staticmethod
    def draw_ui(layout, row, item):
        # Draws inline with the main row
        row.prop(item, "forced_wp_name", text="Name")

    @staticmethod
    def backup_data(item):
        return {"forced_wp_name": item.forced_wp_name}

    @staticmethod
    def restore_data(item, data):
        item.forced_wp_name = data.get("forced_wp_name", "")
