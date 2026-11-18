"""Reversible animator interface; backend data and implementations stay registered."""
import bpy
from .common.blender import get_bl_idname
from .common.config import UIConfig


def animator_mode(context):
    return bool(getattr(getattr(context, 'scene', None), 'mmrt_animator_mode', True))


def restrict_to_full_mode(cls):
    """Disable advanced UI entry points while keeping their implementation intact."""
    if '_mmrt_original_poll' in cls.__dict__:
        return
    original = getattr(cls, 'poll', None)
    cls._mmrt_original_poll = original
    @classmethod
    def poll(subclass, context):
        if animator_mode(context):
            return False
        return bool(original(context)) if original else True
    cls.poll = poll


class MMRT_PT_AnimatorWorkflow(bpy.types.Panel):
    bl_idname = 'MMRT_PT_AnimatorWorkflow'
    bl_label = 'Animator Workflow'
    bl_category = UIConfig.ADDON_CATEGORY
    bl_space_type = UIConfig.ADDON_SPACE
    bl_region_type = UIConfig.ADDON_REGION
    bl_order = -10

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        layout.prop(scene, 'mmrt_animator_mode')
        if not animator_mode(context):
            layout.label(text='Full toolset enabled.')
            return
        layout.label(text='1. Edit animation.')
        layout.label(text='2. Stretch violating regions (includes checks).')
        layout.label(text='3. Export one dataset episode below.')
        layout.label(text='Disable Animator Mode to configure advanced tools.', icon='INFO')
        row = layout.row(align=True)
        row.prop(scene, 'frame_start', text='Start')
        row.prop(scene, 'frame_end', text='End')
        layout.prop(scene, 'camera')
        if not scene.mmrt_recorder.robots:
            layout.label(text='Open a prepared robot scene first.', icon='ERROR')
        for robot in scene.mmrt_recorder.robots:
            if robot.obj:
                count = sum(j.record and bool(robot.obj.pose.bones.get(j.bone_name)) for j in robot.joints) if robot.obj.type == 'ARMATURE' else 0
                layout.label(text=f'{robot.obj.name}: {count} mapped joints')
        if scene.mmrt_recorder.objs:
            layout.label(text='Recorded objects: ' + ', '.join(item.obj.name for item in scene.mmrt_recorder.objs if item.obj))


class MMRT_PT_AnimatorExport(bpy.types.Panel):
    bl_idname = 'MMRT_PT_AnimatorExport'
    bl_label = 'Export One Dataset Episode'
    bl_category = UIConfig.ADDON_CATEGORY
    bl_space_type = UIConfig.ADDON_SPACE
    bl_region_type = UIConfig.ADDON_REGION
    bl_order = 100

    @classmethod
    def poll(cls, context):
        return animator_mode(context)

    def draw(self, context):
        layout = self.layout
        config = context.scene.mmrt_synthesizer
        layout.prop(config, 'output_root')
        layout.prop(config, 'episode_prefix')
        layout.prop(config, 'episode_start_index', text='Episode Index')
        layout.prop(config, 'save_blend', text='Save Scene Copy')
        layout.label(text='MP4 + depth + MMRT + robot bundle + metadata')
        layout.label(text='Uses current timing and materials.')
        layout.operator(get_bl_idname('export_current_dataset_episode'), text='Export Episode', icon='RENDER_ANIMATION')


classes = (MMRT_PT_AnimatorWorkflow, MMRT_PT_AnimatorExport)


def register():
    bpy.types.Scene.mmrt_animator_mode = bpy.props.BoolProperty(
        name='Animator Mode', default=True,
        description='Show only joint checks, retiming and current-episode export. Disable to restore all tools; scene configuration is preserved')
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.mmrt_animator_mode
