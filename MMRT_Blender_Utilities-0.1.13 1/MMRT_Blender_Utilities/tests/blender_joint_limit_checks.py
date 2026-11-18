"""Run: blender --background --factory-startup --python <this file>.

Exercises actual evaluated poses, UI operators, timer lifecycle and frame recovery.
"""
from pathlib import Path
import os
import sys
import tempfile
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import bpy
from MMRT_Blender_Utilities import joint_limit_checks as checks
from MMRT_Blender_Utilities.properties import recorder_properties
from MMRT_Blender_Utilities.common.blender import get_bl_idname


def operator(name):
    namespace, action = get_bl_idname(name).split('.')
    return getattr(getattr(bpy.ops, namespace), action)


def run():
    recorder_properties.register()
    checks.register()
    scene = bpy.context.scene
    from types import SimpleNamespace
    from MMRT_Blender_Utilities.common.marker_utils.events import collect_marker_events
    from MMRT_Blender_Utilities.common.marker_utils.types import FlagTypes
    marker = scene.mmrt_markers.add()
    marker.frame = 3
    def collect(markers):
        return collect_marker_events(markers, remap_frame=lambda frame: frame * 2,
                                     start_frame=0, end_frame=20)
    assert not collect([marker]).custom_flags  # Actual empty Blender collection.
    event = marker.events.add()
    event.flag_type, event.custom_flag_name = FlagTypes.CUSTOM.name, 'test_event'
    assert collect([marker]).custom_flags == [(6, 'test_event')]
    legacy = SimpleNamespace(frame=4, flag_type=FlagTypes.CUSTOM.name, custom_flag_name='legacy')
    assert collect([legacy]).custom_flags == [(8, 'legacy')]
    marker.events.clear()
    assert collect([marker, legacy]).custom_flags == [(8, 'legacy')]
    scene.mmrt_markers.clear()
    print('PASS empty/nested/legacy marker event collection', flush=True)
    scene.frame_start, scene.frame_end = 1, 11
    scene.render.fps = 10
    scene.render.fps_base = 1
    bpy.ops.object.armature_add()
    obj = bpy.context.object
    bone = obj.pose.bones[0]
    bone.rotation_mode = 'XYZ'
    # Evaluated driver rather than raw action curves: q(t)=t^2 radians.
    driver = bone.driver_add('rotation_euler', 0).driver
    driver.expression = '(frame-1)**2/100'
    robot = scene.mmrt_recorder.robots.add()
    robot.obj = obj
    robot.urdf_text = bpy.data.texts.new('LimitTest.urdf')
    robot.urdf_text.write('''<robot name="test"><joint name="j" type="revolute">
        <limit lower="-2" upper="2" velocity="1.5"/></joint></robot>''')
    item = robot.joints.add()
    item.joint_name = 'j'
    item.bone_name = bone.name
    item.rot_axis = 'X'
    settings = scene.mmrt_joint_limits
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'joint_limits.yaml'
        path.write_text('joint_limits:\n  j:\n    has_acceleration_limits: true\n    max_acceleration: 1\n')
        settings.limits_path = str(path)
        scene.frame_set(5, subframe=0.25)
        assert operator('check_joint_limits')() == {'FINISHED'}
        assert scene.frame_current == 5 and abs(scene.frame_subframe - .25) < 1e-6
        metrics = {row.metric for row in settings.issues}
        assert metrics == {'velocity', 'acceleration'}, metrics
        acceleration = [row.value for row in settings.issues if row.metric == 'acceleration']
        assert len(acceleration) == 41
        assert max(abs(value - 2) for value in acceleration) < .02
        report = settings.report.as_string()
        assert 'source_frame' in report and 'acceleration' in report
        print('PASS current evaluated animation, velocity, acceleration, subframes, report, frame restoration', flush=True)

        scene.render.frame_map_new = 200
        assert checks.run_check(scene)
        assert len(settings.issues) == 0, settings.report.as_string()
        assert settings.status.startswith('No sampled violations'), settings.status
        assert scene.frame_current == 5 and abs(scene.frame_subframe - .25) < 1e-6
        print('PASS time remapping halves speed and quarters acceleration', flush=True)

        settings.limits_path = ''
        assert checks.run_check(scene)
        assert settings.status.startswith('INCOMPLETE')
        assert 'acceleration: limit missing' in settings.report.as_string()
        print('PASS missing acceleration limits cannot report a complete pass', flush=True)

        settings.limits_path = str(path)
        scene.render.frame_map_new = 100
        driver.expression = '-(frame-1)**2/100'
        assert checks.run_check(scene)
        assert any(row.metric == 'velocity' and row.value < -1.5 for row in settings.issues)
        assert any(row.metric == 'acceleration' and row.value < -1 for row in settings.issues)
        print('PASS negative speed and acceleration violations', flush=True)

        assert operator('jump_joint_limit')(frame=8.5) == {'FINISHED'}
        assert scene.frame_current == 8 and abs(scene.frame_subframe - .5) < 1e-6
        settings.limits_path = str(Path(directory) / 'missing.yaml')
        assert not checks.run_check(scene)
        assert not settings.issues and settings.stale and settings.status.startswith('Check failed:')
        assert scene.frame_current == 8 and abs(scene.frame_subframe - .5) < 1e-6
        print('PASS navigation and error clears stale results', flush=True)

        settings.limits_path = str(path)
        driver.expression = '0'
        assert not hasattr(settings, 'auto_check')
        assert not hasattr(checks, '_auto_tick')
        assert checks.run_check(scene)
        assert settings.status.startswith('No sampled violations'), settings.status
        checks._invalidate(scene, None)
        assert settings.stale  # Edits flag stale data without running a scan.
        checks._on_load(None)
        assert settings.stale and not settings.issues
        print('PASS manual-only checks, stale invalidation and load reset', flush=True)

        # End keys are in range, but Bezier handles produce an interior overshoot.
        bone.driver_remove('rotation_euler', 0)
        for frame in (1, 11):
            bone.rotation_euler.x = 0
            bone.keyframe_insert('rotation_euler', index=0, frame=frame)
        action = obj.animation_data.action
        bag = action.layers[0].strips[0].channelbag(obj.animation_data.action_slot)
        curve = bag.fcurves[0]
        first, last = curve.keyframe_points
        first.handle_right_type = last.handle_left_type = 'FREE'
        first.handle_right = (4, 2)
        last.handle_left = (8, 2)
        curve.update()
        robot.urdf_text.clear()
        robot.urdf_text.write('<robot name="test"><joint name="j" type="revolute">'
                             '<limit lower="-1" upper="1" velocity="100"/>'
                             '</joint></robot>')
        path.write_text('joint_limits:\n  j:\n    max_acceleration: 1000\n')
        assert checks.run_check(scene)
        position = [row for row in settings.issues if row.metric == 'position']
        assert position and all(1 < row.frame < 11 for row in position)
        assert any(row.frame != int(row.frame) for row in position)
        print('PASS detects Bezier overshoot between valid keyframes', flush=True)

    checks.unregister()
    assert not hasattr(checks, '_auto_tick')
    assert checks._invalidate not in bpy.app.handlers.depsgraph_update_post
    checks.register()
    checks.unregister()
    recorder_properties.unregister()
    import MMRT_Blender_Utilities as addon
    addon.register()
    scene.mmrt_animator_mode = False  # Exercise advanced mapping setup explicitly.
    scene.mmrt_recorder.robots.clear()
    robot = scene.mmrt_recorder.robots.add()
    robot.obj = obj
    robot.urdf_text = bpy.data.texts.new('MimicCoverage.urdf')
    robot.urdf_text.write('<robot name="test">'
        '<joint name="j" type="revolute"><limit lower="-2" upper="2" velocity="1" acceleration="2"/></joint>'
        '<joint name="mimic_j" type="revolute"><mimic joint="j"/>'
        '<limit lower="-2" upper="2" velocity="1" acceleration="2"/></joint></robot>')
    item = robot.joints.add()
    item.joint_name, item.bone_name = 'j', bone.name
    item.rot_axis, item.direction, item.offset = 'Z', True, .25
    assert operator('refresh_robot_joints')(robot_index=0) == {'FINISHED'}
    assert len(robot.joints) == 1
    primary = robot.joints[0]
    assert (primary.bone_name, primary.rot_axis, primary.direction, primary.offset) == (bone.name, 'Z', True, .25)
    descriptors, notes = checks._descriptors(scene, {})
    assert len(descriptors) == 1 and not notes
    primary.bone_name = ''
    try:
        checks._descriptors(scene, {})
    except ValueError as exc:
        assert 'No supported paired' in str(exc)
    primary.bone_name = bone.name
    assert operator('refresh_robot_joints')(robot_index=0) == {'FINISHED'}
    assert len(robot.joints) == 1 and robot.joints[0].bone_name == bone.name
    print('PASS refresh preserves mappings; mimic joints excluded from required coverage', flush=True)
    text = robot.urdf_text
    source = text.as_string().replace('</robot>', '<joint name="hand" type="revolute">'
        '<limit lower="-2" upper="2" velocity="1" acceleration="2"/></joint></robot>')
    text.clear()
    text.write(source)
    hand_obj = obj.copy()
    hand_obj.data = obj.data.copy()
    scene.collection.objects.link(hand_obj)
    hand = scene.mmrt_recorder.robots.add()
    hand.obj, hand.urdf_text = hand_obj, text
    pair = hand.joints.add()
    pair.joint_name, pair.bone_name = 'hand', bone.name
    descriptors, notes = checks._descriptors(scene, {})
    assert len(descriptors) == 2 and not notes, notes
    pair.record = False
    descriptors, notes = checks._descriptors(scene, {})
    assert any('/hand:' in note for note in notes)
    pair.record = True
    hand.urdf_text = text.copy()
    descriptors, notes = checks._descriptors(scene, {})
    assert notes, 'Different URDF Texts must not share coverage by joint name alone'
    print('PASS split armatures share URDF coverage; genuinely missing joints remain flagged', flush=True)
    addon.unregister()
    print('PASS full addon registration lifecycle', flush=True)
    print('ALL BLENDER JOINT LIMIT CHECKS PASSED', flush=True)


try:
    run()
except BaseException:
    traceback.print_exc()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(1)
else:
    # Avoid the test host's audio shutdown hang in background Blender.
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)
