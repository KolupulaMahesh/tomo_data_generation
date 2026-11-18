from .base import MarkerHandler

class IOMarkerHandler(MarkerHandler):
    @staticmethod
    def draw_ui(layout, row, item):
        # Draws inline with the main row
        io_row = row.row(align=True)
        io_row.ui_units_x = 4

        for bit in range(8):
            io_row.prop(item, f"io{bit}", text="", toggle=True)

        row.prop(item, "io_delay", text="Delay", emboss=True)

    @staticmethod
    def backup_data(item):
        data = {f"io{i}": getattr(item, f"io{i}") for i in range(8)}
        data["io_delay"] = item.io_delay
        return data

    @staticmethod
    def restore_data(item, data):
        for i in range(8):
            setattr(item, f"io{i}", data.get(f"io{i}", False))
        item.io_delay = int(data.get("io_delay", 0))
