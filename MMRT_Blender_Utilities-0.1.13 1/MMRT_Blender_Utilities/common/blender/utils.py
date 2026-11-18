import bpy

from ..config import ObjectMode, UIConfig


def get_bl_idname(bl_idname:str) -> str:
    """Helper to ensure consistent bl_idname formatting"""
    return f"{UIConfig.ADDON_NAMESPACE}.{bl_idname}"


class OPS:
    """Utility wrapper for calling Blender operators with dynamic addon namespace."""

    @staticmethod
    def _get_op(op_name: str):
        namespace = UIConfig.ADDON_NAMESPACE

        try:
            op_module = getattr(bpy.ops, namespace)
        except AttributeError:
            raise RuntimeError(f"Namespace '{namespace}' not found in bpy.ops")

        try:
            return getattr(op_module, op_name)
        except AttributeError:
            raise RuntimeError(f"Operator '{namespace}.{op_name}' not found")

    # ---- Core Calls ----
    @staticmethod
    def call(op_name: str, invoke_type: str = 'EXEC_DEFAULT', **kwargs):
        """Generic operator call"""
        op = OPS._get_op(op_name)
        return op(invoke_type, **kwargs)

    @staticmethod
    def invoke(op_name: str, **kwargs):
        """Call operator with INVOKE_DEFAULT (UI popup, dialogs, etc.)"""
        return OPS.call(op_name, 'INVOKE_DEFAULT', **kwargs)

    @staticmethod
    def exec(op_name: str, **kwargs):
        """Call operator with EXEC_DEFAULT (direct execution)"""
        return OPS.call(op_name, 'EXEC_DEFAULT', **kwargs)

    # ---- Optional Convenience ----
    @staticmethod
    def poll(op_name: str) -> bool:
        """Check if operator can run in current context"""
        try:
            op = OPS._get_op(op_name)
            return op.poll()
        except Exception:
            return False

    @staticmethod
    def safe_call(op_name: str, invoke_type: str = 'EXEC_DEFAULT', **kwargs):
        """Call operator safely without crashing"""
        try:
            return OPS.call(op_name, invoke_type, **kwargs)
        except Exception as e:
            print(f"[OPS ERROR] {e}")
            return {'CANCELLED'}

    @staticmethod
    def idname(op_name: str) -> str:
        """Return full bl_idname string"""
        return f"{UIConfig.ADDON_NAMESPACE}.{op_name}"
    

def ensure_object_mode():
    if bpy.context.mode != ObjectMode.OBJECT:
        try:
            bpy.ops.object.mode_set(mode=ObjectMode.OBJECT)
        except Exception:
            pass


def switch_mode(obj: bpy.types.Object | None, mode: ObjectMode) -> str:
    """
    Safely switch mode.
    
    - If obj is provided → switch that object to mode
    - If obj is None → just ensure OBJECT mode globally
    - Works even if no active object
    """

    ctx = bpy.context
    view_layer = ctx.view_layer

    # -----------------------------
    # Validate mode
    # -----------------------------
    valid_modes = [m.identifier for m in bpy.types.Context.bl_rna_get_subclass_py("Context").mode.enum_items]
    if mode not in valid_modes:
        raise ValueError(f"Mode '{mode}' isn't valid. Choose from: {valid_modes}")

    # -----------------------------
    # If NO object → fallback to OBJECT mode
    # -----------------------------
    if obj is None:
        try:
            if ctx.mode != ObjectMode.OBJECT:
                bpy.ops.object.mode_set(mode=ObjectMode.OBJECT)
        except Exception:
            pass  # safe fallback if operator fails

        return ctx.mode

    # -----------------------------
    # Ensure object is in view layer
    # -----------------------------
    if obj.name not in view_layer.objects:
        raise RuntimeError(f"Object '{obj.name}' not in current view layer")

    # -----------------------------
    # Set active + selected safely
    # -----------------------------
    view_layer.objects.active = obj

    for o in view_layer.objects:
        o.select_set(False)
    obj.select_set(True)

    # -----------------------------
    # Find VIEW_3D context (optional)
    # -----------------------------
    area = next((a for a in ctx.screen.areas if a.type == 'VIEW_3D'), None)
    region = None
    if area:
        region = next((r for r in area.regions if r.type == 'WINDOW'), None)

    # -----------------------------
    # Try override (preferred)
    # -----------------------------
    try:
        override = {
            'window': ctx.window,
            'screen': ctx.screen,
            'scene': ctx.scene,
            'view_layer': view_layer,
            'active_object': obj,
            'object': obj,
            'selected_objects': [obj],
            'selected_editable_objects': [obj],
        }

        if area and region:
            override.update({'area': area, 'region': region})

        with ctx.temp_override(**override):
            bpy.ops.object.mode_set(mode=mode)

    except Exception:
        # -----------------------------
        # Fallback (no override)
        # -----------------------------
        try:
            bpy.ops.object.mode_set(mode=mode)
        except Exception:
            pass

    return obj.mode if obj else ctx.mode


def draw_collapsible_box(layout, scene, datablock, prop_name, draw_fn, label: str = "Advanced Settings"):
    """
    Draws a collapsible "Advanced Settings" section.

    Args:
        layout (bpy.types.UILayout): The layout to draw into.
        scene (bpy.types.Scene): The Blender scene, passed through to draw_fn.
        datablock (bpy.types.PropertyGroup): The datablock that owns the toggle property.
        prop_name (str): Name of the BoolProperty controlling folding.
        draw_fn (function): A function that takes (box, scene) and draws the advanced UI.
    """
    expanded = getattr(datablock if datablock else scene, prop_name, False)

    box = layout.box()
    row = box.row()
    row.prop(
        datablock if datablock else scene,
        prop_name,
        icon="TRIA_DOWN" if expanded else "TRIA_RIGHT",
        icon_only=True,
        emboss=False,
    )
    row.label(text=label)

    if expanded:
        draw_fn(box, scene)


def show_report(message, level="INFO", delay=0.1):
    """Safely schedule a popup report from timers / background logic."""

    def _report():
        wm = bpy.context.window_manager
        if not wm.windows:
            print(f"[{UIConfig.ADDON_NAMESPACE}][{level}] {message}")
            return None  # stop timer

        window = wm.windows[0]
        screen = window.screen

        override = {
            "window": window,
            "screen": screen,
        }

        try:
            with bpy.context.temp_override(**override):
                OPS.invoke("show_message", message=message, level=level)
                print(f"[{UIConfig.ADDON_NAMESPACE}][{level}] {message}")
        except Exception as e:
            print(f"[{UIConfig.ADDON_NAMESPACE}] Failed to show report: {e}")

        return None  # one-shot

    bpy.app.timers.register(_report, first_interval=delay)
    
    
def fmt(v):
    return format(v, '.6g')
