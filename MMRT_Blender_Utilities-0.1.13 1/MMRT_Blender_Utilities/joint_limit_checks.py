"""Interactive, non-exporting checks of the current recorder robot animation."""
import csv
from datetime import datetime, timezone
import io
import math
from pathlib import Path

import bpy
from bpy.app.handlers import persistent

from .common.blender import get_bl_idname
from .common.config import UIConfig
from .common.joint_limits import analyze_samples, limits_from_urdf, parse_overrides
from .common.recording.frame_domain import get_frame_domain
from .joint_limit_retime import MMRT_OT_RetimeJointLimits, MMRT_OT_RestoreJointTiming, STATE, original_source_frame

_BUSY = False


def _changed(self, context):
    self.stale = True


class MMRTJointLimitIssue(bpy.types.PropertyGroup):
    joint: bpy.props.StringProperty()
    frame: bpy.props.FloatProperty()
    source_frame: bpy.props.FloatProperty()
    metric: bpy.props.StringProperty()
    value: bpy.props.FloatProperty()
    bounds: bpy.props.StringProperty()


class MMRTJointLimitSettings(bpy.types.PropertyGroup):
    limits_path: bpy.props.StringProperty(
        name='Joint Limits YAML/JSON', subtype='FILE_PATH', update=_changed,
        description='MoveIt joint_limits file supplying velocity/acceleration overrides in radians and seconds')
    samples_per_frame: bpy.props.IntProperty(
        name='Samples per Frame', default=1, min=1, max=32, update=_changed,
        description='Sample evaluated animation between keys; finite sampling does not prove continuous-time compliance')
    tolerance: bpy.props.FloatProperty(name='Tolerance', default=0.0001, min=0, precision=6, update=_changed)
    stale: bpy.props.BoolProperty(default=True, options={'SKIP_SAVE'})
    status: bpy.props.StringProperty(default='Not checked', options={'SKIP_SAVE'})
    issues: bpy.props.CollectionProperty(type=MMRTJointLimitIssue, options={'SKIP_SAVE'})
    index: bpy.props.IntProperty(default=0, options={'SKIP_SAVE'})
    report: bpy.props.PointerProperty(type=bpy.types.Text)
    retime_margin: bpy.props.FloatProperty(
        name='Limit Utilization', default=0.9, min=0.1, max=0.99, subtype='FACTOR',
        description='Target fraction of velocity and acceleration limits; leaves room for sampled estimates')
    retime_transition: bpy.props.FloatProperty(
        name='Timing Transition (s)', default=0.25, min=0.05, max=10,
        description='Width of regional slowdown transitions in original motion time')
    retime_status: bpy.props.StringProperty(default='')
    retime_method: bpy.props.EnumProperty(
        name='Retiming Method', default='KEYFRAMES',
        items=[('KEYFRAMES', 'Move Existing Keys', 'Move existing keys and handles together; overlapping motion segments share a stretch'),
               ('NLA', 'NLA Timing Curve', 'Preserve source actions using a shared regional NLA clock')])
    retime_speedup: bpy.props.BoolProperty(
        name='Speed Up Under-Limit Blocks', default=False,
        description='Shorten existing-key motion blocks with limit headroom; requires compliant validation in accelerated blocks and their boundaries')
    retime_max_speedup: bpy.props.FloatProperty(
        name='Maximum Speedup', default=2., min=1., max=4.,
        description='Maximum playback speed multiplier for an under-limit block')
    retime_log: bpy.props.PointerProperty(type=bpy.types.Text)
    retime_attempts: bpy.props.IntProperty(name='Maximum Attempts', default=6, min=1, max=20)
    retime_keep_improvement: bpy.props.BoolProperty(
        name='Keep Best Improvement', default=True,
        description='Keep verified timing if peak velocity/acceleration limit ratios improve without either worsening; remaining violations stay visible')
    retime_skip_steps: bpy.props.BoolProperty(
        name='Skip Unrepairable Transitions', default=True,
        description='Leave stepped segments and sharp-corner neighborhoods at original speed; violations remain visible')


def _descriptors(scene, overrides):
    descriptors, notes, known = [], [], set()
    # A full URDF may be split over body/hand armatures. Disabled rows on one
    # part are covered by valid enabled rows on another using the same URDF Text.
    covered = set()
    for robot in scene.mmrt_recorder.robots:
        obj = robot.obj
        if obj is None or obj.type != 'ARMATURE':
            continue
        text = robot.urdf_text or getattr(obj, 'mmrt_robot_urdf_text', None)
        if text:
            covered.update((text.as_pointer(), item.joint_name) for item in robot.joints
                           if item.record and obj.pose.bones.get(item.bone_name))
    for robot in scene.mmrt_recorder.robots:
        obj = robot.obj
        if obj is None or obj.type != 'ARMATURE':
            notes.append('Recorder robot has no armature; not checked.')
            continue
        text = robot.urdf_text or getattr(obj, 'mmrt_robot_urdf_text', None)
        if text is None:
            raise ValueError(f'{obj.name}: select its URDF in the recorder.')
        limits = limits_from_urdf(text.as_string(), overrides)
        known.update(limits)
        paired = {}
        for item in robot.joints:
            if item.joint_name in paired:
                raise ValueError(f'{obj.name}: duplicate pairing for {item.joint_name}.')
            paired[item.joint_name] = item
            if item.joint_name not in limits:
                raise ValueError(f'{obj.name}: {item.joint_name} is not in the URDF.')
        for name, limit in limits.items():
            if limit['type'] == 'fixed' or limit['mimic']:
                continue
            label = f'{obj.name}/{name}'
            item = paired.get(name)
            if (item is None or not item.record) and (text.as_pointer(), name) in covered:
                continue
            if item is None:
                notes.append(f'{label}: no recorder pairing; use Refresh Joints, select its Bone and enable Record; not checked.')
                continue
            if not item.record:
                notes.append(f'{label}: Record is disabled in the recorder; enable it to check this joint; not checked.')
                continue
            if not obj.pose.bones.get(item.bone_name):
                notes.append(f'{label}: recorder Bone "{item.bone_name}" is empty or missing; select the matching pose bone; not checked.')
                continue
            if limit['type'] not in ('revolute', 'continuous'):
                notes.append(f'{label}: recorder rotation mapping cannot check {limit["type"]} joints.')
                continue
            bone = obj.pose.bones[item.bone_name].bone
            rest = bone.parent.matrix_local.inverted() @ bone.matrix_local if bone.parent else bone.matrix_local
            descriptors.append(dict(obj=obj, bone=item.bone_name, label=label, limits=limit,
                                    inverse=rest.to_quaternion().inverted(), axis='XYZ'.index(item.rot_axis),
                                    offset=item.offset, sign=-1 if item.direction else 1))
    unknown = set(overrides) - known
    if unknown:
        notes.append('Limits file joints absent from configured robots: ' + ', '.join(sorted(unknown)))
    if not descriptors:
        raise ValueError('No supported paired robot joints to check. Configure robots and joints in the recorder.')
    return descriptors, notes


def sample_motion(scene, settings, frames=None, on_sample=None):
    """Read evaluated poses without exporting, keying, or modifying animation."""
    overrides = {}
    if settings.limits_path:
        overrides = parse_overrides(Path(bpy.path.abspath(settings.limits_path)).read_text(encoding='utf-8'))
    descriptors, notes = _descriptors(scene, overrides)
    domain = get_frame_domain(scene)
    start, end = domain['stretched_start'], domain['stretched_end']
    count = (end - start) * settings.samples_per_frame + 1
    if count < 1 or count > 100000:
        raise ValueError('Check range must contain 1–100,000 samples; shorten the scene range or reduce sampling.')
    if frames is None:
        frames = [start + index / settings.samples_per_frame for index in range(count)]
    if len(frames) > 100000:
        raise ValueError('Too many motion samples.')
    fps = scene.render.fps / scene.render.fps_base
    samples = [[] for _ in descriptors]
    original = (scene.frame_current, scene.frame_subframe)
    try:
        for sample_index, frame in enumerate(frames):
            whole = math.floor(frame)
            scene.frame_set(whole, subframe=frame - whole)
            graph = bpy.context.evaluated_depsgraph_get()
            if on_sample is not None:
                on_sample(sample_index, graph)
            evaluated = {desc['obj'].name: desc['obj'].evaluated_get(graph) for desc in descriptors}
            for desc, values in zip(descriptors, samples):
                bone = evaluated[desc['obj'].name].pose.bones[desc['bone']]
                relative = bone.parent.matrix.inverted() @ bone.matrix if bone.parent else bone.matrix
                delta = desc['inverse'] @ relative.to_quaternion()
                axis = desc['axis']
                raw = delta.to_euler(('XZY', 'YZX', 'ZXY')[axis])[axis]
                values.append((raw - desc['offset']) * desc['sign'])
    finally:
        scene.frame_set(original[0], subframe=original[1])
    return descriptors, notes, frames, samples, fps, domain


def sample_scene(scene, settings):
    descriptors, notes, frames, samples, fps, domain = sample_motion(scene, settings)
    issues = []
    for desc, values in zip(descriptors, samples):
        found, unchecked = analyze_samples(frames, values, fps, desc['limits'],
                                          angular=True, tolerance=settings.tolerance)
        for issue in found:
            issue.update(joint=desc['label'], source_frame=original_source_frame(scene, issue['frame'] / domain['stretch_factor']))
        issues.extend(found)
        notes.extend(f'{desc["label"]}: {message}' for message in unchecked)
    issues.sort(key=lambda issue: (issue['frame'], issue['joint'], issue['metric']))
    return issues, notes, len(descriptors), len(frames), fps


def _report(settings, content):
    if settings.report is None:
        settings.report = bpy.data.texts.new('MMRT Joint Limit Report.csv')
    settings.report.clear()
    settings.report.write(content)


def run_check(scene):
    global _BUSY
    if _BUSY:
        return False
    settings = scene.mmrt_joint_limits
    _BUSY = True
    try:
        settings.issues.clear()
        issues, notes, joints, samples, fps = sample_scene(scene, settings)
        for issue in issues[:1000]:
            row = settings.issues.add()
            for key in ('joint', 'frame', 'source_frame', 'metric', 'value'):
                setattr(row, key, issue[key])
            row.bounds = f'{issue["lower"] if issue["lower"] is not None else "—"} .. {issue["upper"] if issue["upper"] is not None else "—"}'
        settings.index = 0
        state = 'VIOLATIONS' if issues else ('INCOMPLETE' if notes else 'No sampled violations')
        settings.status = f'{state}: {len(issues)} violations; {joints} joints, {samples} samples/joint'
        if notes and issues:
            settings.status += '; incomplete coverage'
        output = io.StringIO()
        output.write(f'# {datetime.now(timezone.utc).isoformat()}\n# {settings.status}\n')
        output.write(f'# FPS: {fps}; tolerance: {settings.tolerance}; radians, rad/s, rad/s^2\n')
        output.write('# Scope: non-fixed, non-mimic URDF joints; mimic joints are excluded.\n')
        output.write('# Frames use output time; source_frame identifies the original timing before retiming.\n')
        output.write('# Sampled numerical derivatives; increase sampling to inspect narrow peaks.\n')
        for note in notes:
            output.write('# UNCHECKED: ' + note + '\n')
        writer = csv.DictWriter(output, fieldnames=['joint', 'frame', 'source_frame', 'metric', 'value', 'lower', 'upper'])
        writer.writeheader()
        writer.writerows(issues)
        _report(settings, output.getvalue())
        settings.stale = False
        return True
    except Exception as exc:
        settings.status = f'Check failed: {exc}'
        settings.stale = True
        _report(settings, '# ' + settings.status + '\n')
        return False
    finally:
        _BUSY = False
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                area.tag_redraw()


class MMRT_OT_CheckJointLimits(bpy.types.Operator):
    bl_idname = get_bl_idname('check_joint_limits')
    bl_label = 'Check Current Animation'
    bl_description = 'Check evaluated robot motion against URDF and YAML limits without exporting'

    def execute(self, context):
        if context.screen and context.screen.is_animation_playing:
            self.report({'WARNING'}, 'Stop playback before checking the full animation.')
            return {'CANCELLED'}
        ok = run_check(context.scene)
        self.report({'INFO'} if ok else {'ERROR'}, context.scene.mmrt_joint_limits.status)
        return {'FINISHED'} if ok else {'CANCELLED'}


class MMRT_OT_JumpJointLimit(bpy.types.Operator):
    bl_idname = get_bl_idname('jump_joint_limit')
    bl_label = 'Go to Violating Frame'
    frame: bpy.props.FloatProperty()

    def execute(self, context):
        context.scene.frame_set(math.floor(self.frame), subframe=self.frame - math.floor(self.frame))
        return {'FINISHED'}


class MMRT_OT_ShowJointLimitReport(bpy.types.Operator):
    bl_idname = get_bl_idname('show_joint_limit_report')
    bl_label = 'Copy Full Report'

    def execute(self, context):
        report = context.scene.mmrt_joint_limits.report
        if report is None:
            return {'CANCELLED'}
        context.window_manager.clipboard = report.as_string()
        self.report({'INFO'}, 'Full report copied; also available in the Text Editor.')
        return {'FINISHED'}


class MMRT_UL_JointLimitIssues(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        op = row.operator(get_bl_idname('jump_joint_limit'), text=f'{item.frame:g}', icon='PLAY')
        op.frame = item.frame
        row.label(text=f'{item.joint}: {item.metric} = {item.value:.5g}')


class MMRT_PT_JointLimitChecks(bpy.types.Panel):
    bl_idname = 'MMRT_PT_JointLimitChecks'
    bl_label = 'Joint Limit Checks (Current Animation)'
    bl_category = UIConfig.ADDON_CATEGORY
    bl_space_type = UIConfig.ADDON_SPACE
    bl_region_type = UIConfig.ADDON_REGION
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        settings = context.scene.mmrt_joint_limits
        layout.label(text='Uses recorder robots, pairings, and scene frame range.')
        layout.prop(settings, 'limits_path')
        layout.prop(settings, 'samples_per_frame')
        layout.prop(settings, 'tolerance')
        if settings.stale:
            layout.label(text='Not checked or changed since last check', icon='INFO')
        layout.label(text=settings.status)
        layout.label(text='Sampled checks; missing limits are reported as unchecked.')
        layout.template_list('MMRT_UL_JointLimitIssues', '', settings, 'issues', settings, 'index', rows=5)
        if settings.issues and 0 <= settings.index < len(settings.issues):
            selected = settings.issues[settings.index]
            layout.label(text=f'Original timing frame: {selected.source_frame:g}')
            layout.label(text=f'Bounds: {selected.bounds}')
        layout.label(text='List shows first 1,000; full report includes every violation.')
        if settings.report:
            layout.operator(get_bl_idname('show_joint_limit_report'), icon='TEXT')
        box = layout.box()
        box.label(text='Regional Timing — Preserve Motion Path')
        if STATE in context.scene:
            box.operator(get_bl_idname('restore_joint_timing'), icon='LOOP_BACK')
            box.label(text='Restore original timing before editing/reapplying.')
        else:
            box.prop(settings, 'retime_method')
            if settings.retime_method == 'KEYFRAMES':
                box.prop(settings, 'retime_speedup')
                if settings.retime_speedup:
                    box.prop(settings, 'retime_max_speedup')
            box.prop(settings, 'retime_margin')
            box.prop(settings, 'retime_transition')
            box.prop(settings, 'retime_attempts')
            box.prop(settings, 'retime_keep_improvement')
            box.prop(settings, 'retime_skip_steps')
            box.label(text='Synchronizes all keyed scene objects and timeline events.')
            box.operator(get_bl_idname('retime_joint_limits'), icon='TIME')
        if settings.retime_status:
            box.label(text=settings.retime_status)
        if settings.retime_log:
            box.label(text='Text Editor log: ' + settings.retime_log.name, icon='TEXT')


@persistent
def _invalidate(scene, depsgraph):
    if not _BUSY and hasattr(scene, 'mmrt_joint_limits'):
        if not scene.mmrt_joint_limits.stale:
            scene.mmrt_joint_limits.stale = True


@persistent
def _on_load(_):
    for scene in bpy.data.scenes:
        settings = scene.mmrt_joint_limits
        settings.stale = True
        settings.status = 'Not checked since loading'
        settings.issues.clear()


_CLASSES = (MMRTJointLimitIssue, MMRTJointLimitSettings, MMRT_OT_CheckJointLimits,
            MMRT_OT_JumpJointLimit, MMRT_OT_ShowJointLimitReport, MMRT_UL_JointLimitIssues,
            MMRT_PT_JointLimitChecks, MMRT_OT_RetimeJointLimits, MMRT_OT_RestoreJointTiming)


def register():
    for cls in _CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.mmrt_joint_limits = bpy.props.PointerProperty(type=MMRTJointLimitSettings)
    bpy.app.handlers.depsgraph_update_post.append(_invalidate)
    bpy.app.handlers.load_post.append(_on_load)


def unregister():
    bpy.app.handlers.load_post.remove(_on_load)
    bpy.app.handlers.depsgraph_update_post.remove(_invalidate)
    del bpy.types.Scene.mmrt_joint_limits
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)
