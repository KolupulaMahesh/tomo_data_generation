from .base import MarkerHandler

class CustomMarkerHandler(MarkerHandler):
    @staticmethod
    def draw_ui(layout, row, item):
        row.prop(item, "custom_flag_name", text="", icon='OUTLINER_DATA_FONT')

    @staticmethod
    def backup_data(item):
        return {"custom_flag_name": item.custom_flag_name}

    @staticmethod
    def restore_data(item, data):
        item.custom_flag_name = data.get("custom_flag_name", "")
