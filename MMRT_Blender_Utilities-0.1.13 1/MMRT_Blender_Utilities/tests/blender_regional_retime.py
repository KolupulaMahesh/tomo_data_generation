"""Background Blender integration tests for reversible regional timing."""
from pathlib import Path
import os
import sys
import tempfile
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import bpy
from MMRT_Blender_Utilities import joint_limit_checks as checks
from MMRT_Blender_Utilities import joint_limit_retime as retime
from MMRT_Blender_Utilities.properties import recorder_properties
from MMRT_Blender_Utilities.common.blender import get_bl_idname


def operator(name):
    namespace, action = get_bl_idname(name).split('.')
    return getattr(getattr(bpy.ops, namespace), action)


def curve_snapshot(action):
    return [(curve.data_path, curve.array_index,
             [(tuple(key.co), tuple(key.handle_left), tuple(key.handle_right), key.interpolation)
              for key in curve.keyframe_points])
            for layer in action.layers for strip in layer.strips
            for bag in strip.channelbags for curve in bag.fcurves]


def check_modifier_preflight(scene):
    cube = scene.objects['Cube']
    mod = cube.modifiers.new('StaticGeometry', 'NODES')
    tree = bpy.data.node_groups.new('StaticGeometryTree', 'GeometryNodeTree')
    mod.node_group = tree
    tree.interface.new_socket(name='Geometry', in_out='INPUT', socket_type='NodeSocketGeometry')
    tree.interface.new_socket(name='Geometry', in_out='OUTPUT', socket_type='NodeSocketGeometry')
    source = tree.nodes.new('NodeGroupInput')
    output = tree.nodes.new('NodeGroupOutput')
    tree.links.new(source.outputs['Geometry'], output.inputs['Geometry'])
    nested = bpy.data.node_groups.new('NestedGeometry', 'GeometryNodeTree')
    group = tree.nodes.new('GeometryNodeGroup')
    group.node_tree = nested
    retime._preflight(scene)
    clock = nested.nodes.new('GeometryNodeInputSceneTime')
    try:
        retime._preflight(scene)
    except ValueError as exc:
        assert all(text in str(exc) for text in ('Cube', 'StaticGeometry', 'NestedGeometry', clock.name, 'scene clock'))
    else:
        raise AssertionError('Nested Scene Time accepted')
    nested.nodes.remove(clock)
    nested['test_value'] = 0.0
    nested.driver_add('["test_value"]').driver.expression = 'frame'
    try:
        retime._preflight(scene)
    except ValueError as exc:
        assert 'NestedGeometry' in str(exc) and 'drivers' in str(exc)
    else:
        raise AssertionError('Nested node-tree driver accepted')
    nested.driver_remove('["test_value"]')
    cloth = cube.modifiers.new('ClothSimulation', 'CLOTH')
    try:
        retime._preflight(scene)
    except ValueError as exc:
        assert 'Cube' in str(exc) and 'ClothSimulation' in str(exc) and 'cache' in str(exc)
    else:
        raise AssertionError('Cloth simulation accepted')
    cube.modifiers.remove(cloth)
    retime._preflight(scene)
    # Leave static Geometry Nodes in place during full retime/save/restore tests.
    print('PASS static geometry accepted; nested clock/driver and named simulation rejected', flush=True)


def run():
    recorder_properties.register()
    checks.register()
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = 1, 51
    scene.render.fps = 24
    bpy.ops.object.armature_add()
    robot = bpy.context.object
    robot.name = 'TestRobot'
    bone = robot.pose.bones[0]
    bone.rotation_mode = 'XYZ'
    for frame, angle in [(1, 0), (11, 1), (51, 1)]:
        bone.rotation_euler.x = angle
        bone.keyframe_insert('rotation_euler', index=0, frame=frame)
    action = robot.animation_data.action
    original = curve_snapshot(action)
    source_slot = robot.animation_data.action_slot.identifier
    robot2 = robot.copy()
    robot2.data = robot.data.copy()
    robot2.name = 'SharedActionRobot'
    scene.collection.objects.link(robot2)
    assert robot2.animation_data.action == action
    robot2.animation_data.use_nla = False
    urdf = bpy.data.texts.new('RetimeTest.urdf')
    urdf.write('<robot name="test"><joint name="j" type="revolute">'
               '<limit lower="-2" upper="2" velocity="0.8" acceleration="3"/>'
               '</joint></robot>')
    srdf = bpy.data.texts.new('RetimeTest.srdf')
    srdf.write('<robot name="test"/>')
    for obj in (robot, robot2):
        item = scene.mmrt_recorder.robots.add()
        item.obj, item.urdf_text, item.srdf_text = obj, urdf, srdf
        pair = item.joints.add()
        pair.joint_name, pair.bone_name, pair.rot_axis = 'j', bone.name, 'X'
    controller = bpy.data.objects.new('CoordinatedObject', None)
    scene.collection.objects.link(controller)
    for frame, value in [(1, 0), (51, 2)]:
        controller.location.x = value
        controller.keyframe_insert('location', index=0, frame=frame)
    controller_action = controller.animation_data.action
    controller_curve = next(retime._curves(controller_action, controller.animation_data.action_slot))
    controller_curve.keyframe_points[0].interpolation = 'CONSTANT'
    try:
        retime._check_corners(controller_action, controller.animation_data.action_slot, 1, 51)
    except ValueError as exc:
        assert all(text in str(exc) for text in ('location[0]', 'frame 51', 'frame 1', 'Constant', 'Bezier'))
    else:
        raise AssertionError('Stepped transform accepted')
    try:
        retime._preflight(scene)
    except ValueError as exc:
        assert 'Object "CoordinatedObject"' in str(exc) and 'location[0]' in str(exc)
    else:
        raise AssertionError('Stepped object accepted by preflight')
    for key in controller_curve.keyframe_points:
        key.interpolation = 'LINEAR'
    scene.timeline_markers.new('grip', frame=11)
    scene.timeline_markers.new('done', frame=51)
    episode = scene.mmrt_recorder.episodes.add()
    episode.name = 'Episode'
    episode.start_frame, episode.end_frame = 1, 51
    settings = scene.mmrt_joint_limits
    settings.retime_method = 'NLA'
    scene.use_preview_range = True
    scene.frame_preview_start, scene.frame_preview_end = 1, 51
    scene.frame_set(5, subframe=.25)
    assert checks.run_check(scene) and settings.issues
    check_modifier_preflight(scene)

    # Failure after mutation must restore every affected object and timing field.
    sampler = checks.sample_motion
    def fail_after_apply(*args, **kwargs):
        if retime.STATE in scene:
            raise RuntimeError('forced verification failure')
        return sampler(*args, **kwargs)
    checks.sample_motion = fail_after_apply
    try:
        try:
            retime.retime_scene(scene)
        except RuntimeError as exc:
            assert str(exc) == 'forced verification failure'
        else:
            raise AssertionError('Expected failure')
    finally:
        checks.sample_motion = sampler
    assert retime.STATE not in scene and scene.frame_end == 51
    assert robot.animation_data.action == action and not robot.animation_data.nla_tracks
    assert curve_snapshot(action) == original
    assert scene.timeline_markers['grip'].frame == 11
    print('PASS verification failure rolls back source actions and range', flush=True)

    assert operator('retime_joint_limits')() == {'FINISHED'}
    assert retime.STATE in scene and scene.frame_end > 51
    assert not settings.issues and settings.status.startswith('No sampled violations'), settings.report.as_string()
    assert curve_snapshot(action) == original
    assert robot.animation_data.action is None
    assert robot.animation_data.nla_tracks[0].strips[0].action == action
    assert robot2.animation_data.nla_tracks[0].strips[0].action == action
    strip = robot.animation_data.nla_tracks[0].strips[0]
    clock = next(curve for curve in strip.fcurves if curve.data_path == 'strip_time')
    points = list(clock.keyframe_points)
    # A safe late portion stays at approximately 1x speed, unlike global scaling.
    late = [(key.co.x, key.co.y) for key in points if 30 <= key.co.y <= 45]
    assert abs((late[-1][0]-late[0][0]) - (late[-1][1]-late[0][1])) < .1, late
    for key in points[::max(1, len(points)//15)]:
        frame, source = key.co
        scene.frame_set(int(frame), subframe=frame-int(frame))
        assert abs(controller.location.x - controller_curve.evaluate(source)) < 1e-4
    angular_curve = next(retime._curves(action, strip.action_slot))
    # Off-clock-key samples check the continuous path, not just sampled knots.
    for fraction in (.123, .287, .419, .631):
        frame = 1 + (scene.frame_end-1)*fraction
        source = clock.evaluate(frame)
        scene.frame_set(int(frame), subframe=frame-int(frame))
        assert abs(robot.pose.bones[0].rotation_euler.x-angular_curve.evaluate(source)) < 1e-4
        assert abs(controller.location.x-controller_curve.evaluate(source)) < 1e-4
    from MMRT_Blender_Utilities.common.recording.recorder import MMRTRecorder
    recorder = MMRTRecorder(bpy.context)
    recorder.prepare()
    assert recorder.end_frame == scene.frame_end
    for frame in (10, 20, 30):
        scene.frame_set(frame)
        recorded = recorder.evaluate_current_frame()['j']
        assert abs(recorded-angular_curve.evaluate(clock.evaluate(frame))) < 1e-4
    print('PASS actual recorder samples follow the regional clock', flush=True)
    assert scene.timeline_markers['grip'].frame > 11
    assert scene.timeline_markers['done'].frame == scene.frame_end
    assert episode.end_frame == scene.frame_end
    assert scene.frame_preview_end == scene.frame_end
    from MMRT_Blender_Utilities.common.synthesis.pipeline import BlenderSynthesizer
    engine = object.__new__(BlenderSynthesizer)
    engine.scene = scene
    try:
        engine.execute_generation()
    except RuntimeError as exc:
        assert 'Regional timing is active' in str(exc)
    else:
        raise AssertionError('Synthesis accepted unsupported retimed strips')
    print('PASS regional limits, safe-region duration, coordinated poses, marker/episode timing, unchanged shared action', flush=True)

    with tempfile.TemporaryDirectory() as directory:
        robot.name = 'RenamedRobot'
        action.name = 'RenamedSource'
        bpy.ops.wm.save_as_mainfile(filepath=str(Path(directory) / 'retimed.blend'))
        bpy.ops.wm.open_mainfile(filepath=str(Path(directory) / 'retimed.blend'))
        scene = bpy.context.scene
        settings = scene.mmrt_joint_limits
        robot = bpy.data.objects['RenamedRobot']
        action = bpy.data.actions['RenamedSource']
        assert retime.STATE in scene
        assert operator('restore_joint_timing')() == {'FINISHED'}
        assert scene.frame_end == 51 and scene.frame_current == 5 and abs(scene.frame_subframe-.25) < 1e-5
        assert robot.animation_data.action == action
        assert robot.animation_data.action_slot.identifier == source_slot
        assert not robot.animation_data.nla_tracks
        assert scene.timeline_markers['grip'].frame == 11 and scene.mmrt_recorder.episodes[0].end_frame == 51
        assert curve_snapshot(action) == original
        assert scene.frame_preview_end == 51 and scene.use_preview_range
        assert not bpy.data.objects['SharedActionRobot'].animation_data.use_nla
        print('PASS save/reopen, renames, original action/slot/events/frame restoration', flush=True)

    scene.render.frame_map_new = 200
    assert operator('retime_joint_limits')() == {'FINISHED'}
    assert not scene.mmrt_joint_limits.issues, scene.mmrt_joint_limits.report.as_string()
    assert scene.render.frame_map_new == 200
    assert operator('restore_joint_timing')() == {'FINISHED'}
    assert scene.frame_end == 51 and scene.render.frame_map_new == 200
    print('PASS regional timing with existing global time stretching', flush=True)

    settings = scene.mmrt_joint_limits
    settings.retime_method = 'KEYFRAMES'
    before_keys = curve_snapshot(robot.animation_data.action)
    source_action = robot.animation_data.action
    checks.sample_motion = fail_after_apply
    try:
        try:
            retime.retime_scene(scene)
        except RuntimeError as exc:
            assert 'forced verification failure' in str(exc)
        else:
            raise AssertionError('Direct keyframe verification failure ignored')
    finally:
        checks.sample_motion = sampler
    assert robot.animation_data.action == source_action and retime.STATE not in scene
    assert curve_snapshot(source_action) == before_keys
    message = retime.retime_scene(scene)
    assert retime.STATE in scene and scene[retime.STATE]['method'] == 'KEYFRAMES'
    assert robot.animation_data.action is not None and not robot.animation_data.nla_tracks
    after_keys = curve_snapshot(robot.animation_data.action)
    assert len(after_keys) == len(before_keys)
    assert [len(c[2]) for c in after_keys] == [len(c[2]) for c in before_keys]
    assert [[k[0][1] for k in c[2]] for c in after_keys] == [[k[0][1] for k in c[2]] for c in before_keys]
    assert after_keys[0][2][1][0][0] > before_keys[0][2][1][0][0]
    assert robot.animation_data.action == bpy.data.objects['SharedActionRobot'].animation_data.action
    assert curve_snapshot(source_action) == before_keys
    assert not retime._verify(scene, settings)[0]
    with tempfile.TemporaryDirectory() as directory:
        bpy.ops.wm.save_as_mainfile(filepath=str(Path(directory)/'retimed_keys.blend'))
        bpy.ops.wm.open_mainfile(filepath=str(Path(directory)/'retimed_keys.blend'))
        scene = bpy.context.scene
        settings = scene.mmrt_joint_limits
        robot = bpy.data.objects['RenamedRobot']
        retime.restore_timing(scene)
        assert curve_snapshot(robot.animation_data.action) == before_keys
        assert not robot.animation_data.nla_tracks
    # Already compliant animations can now contract without adding keys.
    urdf = bpy.data.texts['RetimeTest.urdf']
    saved_urdf = urdf.as_string()
    urdf.clear()
    urdf.write(saved_urdf.replace('velocity="0.8" acceleration="3"', 'velocity="100" acceleration="1000"'))
    settings.retime_speedup = True
    previous_end = scene.frame_end
    previous_key_count = sum(len(c.keyframe_points) for c in retime._curves(robot.animation_data.action, None))
    retime.retime_scene(scene)
    assert scene.frame_end < previous_end
    assert not retime._verify(scene, settings)[0]
    assert sum(len(c.keyframe_points) for c in retime._curves(robot.animation_data.action, None)) == previous_key_count
    retime.restore_timing(scene)
    assert scene.frame_end == previous_end
    # Acceptance may retain unrelated residuals even without Keep Best Improvement.
    real_verify = retime._verify
    saved_keep = settings.retime_keep_improvement
    scene.frame_end = 100
    settings.retime_keep_improvement = False
    def residual_elsewhere(*args, **kwargs):
        found, notes = real_verify(*args, **kwargs)
        return found + [dict(metric='velocity', upper=1., lower=-1., value=2., source_frame=90.)], notes
    retime._verify = residual_elsewhere
    try:
        message = retime.retime_scene(scene)
        assert 'violations remain elsewhere' in message, message
        assert scene.frame_end < 100
        assert '0 violations in accelerated blocks/boundary guards' in settings.retime_log.as_string()
        retime.restore_timing(scene)
    finally:
        retime._verify = real_verify
        settings.retime_keep_improvement = saved_keep
        scene.frame_end = previous_end
    print('PASS speedup retains unrelated residual violations with partial improvement disabled', flush=True)

    # Reject compressed candidates to exercise retry backoff and fallback.
    real_verify = retime._verify
    previous_attempts = settings.retime_attempts
    settings.retime_attempts = 2
    verification_calls = []
    def reject_speedups(*args, **kwargs):
        verification_calls.append(True)
        if len(verification_calls) in (2, 3):
            return ([dict(metric='velocity', upper=1., lower=-1., value=2., source_frame=10.)], [])
        return real_verify(*args, **kwargs)
    retime._verify = reject_speedups
    try:
        message = retime.retime_scene(scene)
        assert 'slowdown-only fallback' in message
        log = settings.retime_log.as_string()
        assert 'Speedup strength: 100.0%' in log and 'Speedup strength: 50.0%' in log
        assert 'Attempt 3/3' in log
        assert scene.frame_end >= previous_end
        assert not real_verify(scene, settings)[0]
        retime.restore_timing(scene)
    finally:
        retime._verify = real_verify
        settings.retime_attempts = previous_attempts
    print('PASS rejected speedup backs off then keeps validated slowdown-only fallback', flush=True)
    urdf.clear()
    urdf.write(saved_urdf)
    settings.retime_speedup = False
    print('PASS compliant existing-key animation speeds up, validates and restores', flush=True)
    settings.retime_method = 'NLA'
    print('PASS direct keys: active actions, unchanged key counts/values, shared action, sampled pose/limit verification, save/reopen restore', flush=True)

    # A keyed Child Of attachment must keep its world pose at corresponding times.
    prop = bpy.data.objects.new('TestPhone', None)
    scene.collection.objects.link(prop)
    constraint = prop.constraints.new('CHILD_OF')
    constraint.target = bpy.data.objects['CoordinatedObject']
    for frame, influence in [(1, 0), (11, 1), (51, 0)]:
        constraint.influence = influence
        constraint.keyframe_insert('influence', frame=frame)
    prop_action = prop.animation_data.action
    verifier = retime._verify
    settings = scene.mmrt_joint_limits
    settings.retime_attempts = 3
    def residual(value):
        return ([dict(metric='velocity', upper=1., lower=-1., value=value, source_frame=10.)], [])
    values = iter([4., 2., 3., 2.5, 2.])
    retime._verify = lambda *args: residual(next(values))
    try:
        message = retime.retime_scene(scene)
        assert 'kept improved attempt 1' in message, message
        assert prop.animation_data.action is None and prop.animation_data.nla_tracks
        assert 'Attempt 3/3' in settings.retime_log.as_string()
        assert 'TestPhone' in settings.retime_log.as_string()
        retime.restore_timing(scene)
        assert prop.animation_data.action == prop_action
        for keep, candidates in [(True, [4., 5., 5., 5.]), (False, [4., 2., 2., 2.])]:
            settings.retime_keep_improvement = keep
            values = iter(candidates)
            try:
                retime.retime_scene(scene)
            except ValueError as exc:
                assert 'original timing restored' in str(exc)
            else:
                raise AssertionError('Unacceptable partial timing retained')
            assert retime.STATE not in scene and prop.animation_data.action == prop_action
    finally:
        retime._verify = verifier
        settings.retime_attempts = 6
        settings.retime_keep_improvement = True
    bpy.data.objects.remove(prop, do_unlink=True)
    print('PASS best partial candidate retained; worse/disabled results restored; attachment poses preserved', flush=True)

    stepped = bpy.data.objects.new('SteppedProp', None)
    scene.collection.objects.link(stepped)
    for frame, value in [(1, 0), (40, 0), (45, .5), (51, .5)]:
        stepped.location.x = value
        stepped.keyframe_insert('location', index=0, frame=frame)
    for channel in retime._curves(stepped.animation_data.action, stepped.animation_data.action_slot):
        for key in channel.keyframe_points:
            key.interpolation = 'CONSTANT'
    scene.mmrt_joint_limits.retime_skip_steps = True
    message = retime.retime_scene(scene)
    assert 'Partial retiming' in message and '40–45' in message, message
    strip = stepped.animation_data.nla_tracks[0].strips[0]
    clock = next(c for c in strip.fcurves if c.data_path == 'strip_time')
    # The inverse clock retains unit speed throughout the skipped segment.
    a, b = 1.0, float(scene.frame_end)
    for _ in range(40):
        mid = (a+b)/2
        if clock.evaluate(mid) < 41:
            a = mid
        else:
            b = mid
    assert abs(clock.evaluate((a+b)/2+2) - clock.evaluate((a+b)/2) - 2) < 1e-4
    retime.restore_timing(scene)
    bpy.data.objects.remove(stepped, do_unlink=True)
    print('PASS partial retiming preserves stepped segment duration', flush=True)

    # Unsafe configurations must be rejected without mutation.
    source = robot.animation_data.action
    robot.driver_add('location', 0).driver.expression = 'frame'
    try:
        retime.retime_scene(scene)
    except ValueError as exc:
        assert 'drivers' in str(exc)
    else:
        raise AssertionError('Frame driver accepted')
    assert robot.animation_data.action == source and retime.STATE not in scene
    robot.driver_remove('location', 0)
    urdf = scene.mmrt_recorder.robots[0].urdf_text
    urdf.clear()
    urdf.write('<robot name="test"><joint name="j" type="revolute">'
               '<limit lower="-2" upper="0.5" velocity="0.8" acceleration="3"/>'
               '</joint></robot>')
    try:
        retime.retime_scene(scene)
    except ValueError as exc:
        assert 'Position limits' in str(exc)
    else:
        raise AssertionError('Position violation accepted')
    assert retime.STATE not in scene
    print('PASS time-dependent drivers and position violations rejected without mutation', flush=True)
    urdf.clear()
    urdf.write('<robot name="test"><joint name="j" type="revolute">'
               '<limit lower="-2" upper="2" velocity="0.8" acceleration="3"/>'
               '</joint></robot>')
    channel = next(retime._curves(robot.animation_data.action, robot.animation_data.action_slot))
    channel.keyframe_points[-1].co.y = 1.2
    key = channel.keyframe_points.insert(40, 1)
    key.interpolation = 'CONSTANT'
    key = channel.keyframe_points.insert(45, 1.2)
    key.interpolation = 'CONSTANT'
    channel.update()
    message = retime.retime_scene(scene)
    assert 'Partial retiming' in message, message
    checks.run_check(scene)
    assert scene.mmrt_joint_limits.issues, 'Skipped joint jump must remain visible'
    retime.restore_timing(scene)
    print('PASS skipped robot jump remains reported by joint checks', flush=True)
    checks.unregister()
    recorder_properties.unregister()
    print('ALL REGIONAL RETIMING CHECKS PASSED', flush=True)


try:
    run()
except BaseException:
    traceback.print_exc()
    sys.stdout.flush(); sys.stderr.flush(); os._exit(1)
else:
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)
