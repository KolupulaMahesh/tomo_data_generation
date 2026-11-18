"""Reversible shared timing for regional joint-limit retiming."""
from .common.pose_validation import pose_errors, poses_match
import math
import time
from types import SimpleNamespace
import numpy as np
import bpy

from .common.blender import get_bl_idname
from .common.joint_limits import analyze_samples
from .common.regional_retime import plan_retime
from .common.keyframe_retime import shared_key_clock, accelerate_blocks, speedup_violations
from .keyframe_retime import motion_intervals, make_working_actions, apply_key_clock
from .common.recording.frame_domain import get_frame_domain

STATE = '_mmrt_joint_retime'


def _curves(action, slot):
    for layer in action.layers:
        for strip in layer.strips:
            for bag in strip.channelbags:
                if slot is None or bag.slot_handle == slot.handle:
                    yield from bag.fcurves


def _check_corners(action, slot, start, end, skipped=None):
    for curve in _curves(action, slot):
        if not any(part in curve.data_path for part in ('location', 'rotation_', 'scale')):
            continue
        keys = list(curve.keyframe_points)
        for i, key in enumerate(keys[:-1]):
            other = keys[i+1]
            if other.co.x < start or key.co.x > end:
                continue
            if key.interpolation == 'CONSTANT' and abs(other.co.y - key.co.y) > 1e-6:
                if skipped is not None:
                    skipped.append((float(key.co.x), float(other.co.x)))
                    continue
                raise ValueError(
                    f'{action.name}: {curve.data_path}[{curve.array_index}] jumps at '
                    f'frame {other.co.x:g} from {key.co.y:g} to {other.co.y:g}. '
                    f'The key at frame {key.co.x:g} uses Constant interpolation. '
                    'Change that segment to a smooth Bezier transition and recheck; '
                    'stretching alone cannot remove a position jump.')
        for i, key in enumerate(keys[1:-1], 1):
            if not start < key.co.x < end:
                continue
            previous, following = keys[i-1], keys[i+1]
            if previous.interpolation not in ('BEZIER', 'LINEAR', 'CONSTANT') or key.interpolation not in ('BEZIER', 'LINEAR', 'CONSTANT'):
                raise ValueError(f'{action.name}: use Bezier/linear transform curves before retiming.')
            def slope(kind, vector):
                if kind == 'CONSTANT':
                    return 0.0
                if abs(vector.x) < 1e-8:
                    if abs(vector.y) > 1e-8:
                        raise ValueError(f'{action.name}: vertical transform handle at frame {key.co.x:g}.')
                    return 0.0
                return vector.y / vector.x
            left = slope(previous.interpolation, key.co - (key.handle_left if previous.interpolation == 'BEZIER' else previous.co))
            right = slope(key.interpolation, (key.handle_right if key.interpolation == 'BEZIER' else following.co) - key.co)
            if abs(left - right) > 1e-4 * max(1, abs(left), abs(right)):
                if skipped is not None:
                    skipped.append((float(previous.co.x), float(following.co.x)))
                    continue
                raise ValueError(f'{action.name}: sharp velocity corner at frame {key.co.x:g}; smooth its handles before retiming.')


def original_source_frame(scene, frame):
    state = scene.get(STATE)
    if state is None:
        return frame
    if state.get('method') == 'KEYFRAMES':
        return float(np.interp(frame, state['target_times'], state['source_times']))
    saved = next(iter(state['objects'].values()))
    obj = saved.get('object')
    track = obj.animation_data.nla_tracks.get(saved['track']) if obj and obj.animation_data else None
    if track is None or not track.strips:
        raise ValueError('Managed retiming track is missing; restore its original layout.')
    curve = next(curve for curve in track.strips[0].fcurves if curve.data_path == 'strip_time')
    return curve.evaluate(frame)


def _check_geometry_nodes(tree, label, ids, visited=None):
    """Allow deterministic geometry; NLA cannot retime a node's scene clock/cache."""
    if tree is None:
        return
    visited = set() if visited is None else visited
    if tree.as_pointer() in visited:
        return
    visited.add(tree.as_pointer())
    ids.append(tree)  # Node-tree actions/drivers must pass the same animation checks.
    for node in tree.nodes:
        path = f'{label} → {tree.name} / {node.name}'
        if node.bl_idname in {
            'GeometryNodeInputSceneTime', 'GeometryNodeSimulationInput',
            'GeometryNodeSimulationOutput', 'GeometryNodeBake',
        }:
            raise ValueError(
                f'{path}: time-dependent node ({node.bl_idname}). Regional timing '
                'cannot retime its scene clock/cache. Remove this dependency or '
                'convert its motion to plain object actions first; a simulation '
                'cache bake alone is not sufficient.')
        image = getattr(node, 'image', None)
        if image and image.source in {'MOVIE', 'SEQUENCE'}:
            raise ValueError(f'{path}: animated image input uses scene time and cannot be retimed.')
        nested = getattr(node, 'node_tree', None)
        if nested is not None:
            _check_geometry_nodes(nested, path, ids, visited)


def _check_modifiers(obj, ids):
    if obj.rigid_body:
        raise ValueError(f'{obj.name}: rigid body physics is unsupported. Convert its motion '
                         'to plain object actions and remove the physics before retiming; '
                         'a physics cache bake alone is not sufficient.')
    for mod in obj.modifiers:
        label = f'{obj.name} / modifier "{mod.name}" ({mod.type})'
        # Do not ignore hidden modifiers: visibility may itself be animated.
        if mod.type == 'NODES':
            _check_geometry_nodes(mod.node_group, label, ids)
        elif mod.type in {'CLOTH', 'SOFT_BODY', 'FLUID', 'DYNAMIC_PAINT',
                          'PARTICLE_SYSTEM', 'MESH_SEQUENCE_CACHE', 'MESH_CACHE'}:
            raise ValueError(
                f'{label}: simulation/cache timing is unsupported. Remove this effect '
                'from the scene being retimed, or convert rigid motion to plain object '
                'actions first. Deforming simulation caches need separate cache retiming; '
                'baking the cache alone does not make this retimer preserve it.')


def _preflight(scene, skipped=None):
    if STATE in scene:
        raise ValueError('Restore Original Timing before editing/reapplying regional retiming.')
    objects = []
    ids = [scene, scene.world]
    for obj in scene.objects:
        if '_mmrt_synth_source_action' in obj:
            raise ValueError('Reset synthesis action previews before regional retiming.')
        ids.extend([obj, obj.data, getattr(obj.data, 'shape_keys', None)])
        _check_modifiers(obj, ids)
        if obj.animation_data and obj.animation_data.action:
            if obj.library and not obj.override_library:
                raise ValueError(f'{obj.name}: make a local override before retiming.')
            animation = obj.animation_data
            if animation.use_tweak_mode or animation.action_blend_type != 'REPLACE' or abs(animation.action_influence - 1) > 1e-6:
                raise ValueError(f'{obj.name}: retiming requires a full-influence Replace action outside NLA tweak mode.')
            first, last = animation.action.frame_range
            if first < scene.frame_start or last > scene.frame_end:
                raise ValueError(f'{obj.name}: extend the scene range to include all action keys before retiming.')
            try:
                _check_corners(animation.action, animation.action_slot, scene.frame_start, scene.frame_end, skipped)
            except ValueError as exc:
                raise ValueError(f'Object "{obj.name}", action {exc}') from exc
            objects.append(obj)
        for material in getattr(obj.data, 'materials', []):
            if material:
                ids.extend([material, material.node_tree])
    if scene.world:
        ids.append(scene.world.node_tree)
    for data in ids:
        animation = getattr(data, 'animation_data', None)
        if not animation:
            continue
        if animation.drivers or animation.nla_tracks:
            raise ValueError(f'{data.name}: existing drivers/NLA need baking to plain object actions before automatic retiming.')
        if animation.action and not isinstance(data, bpy.types.Object):
            raise ValueError(f'{data.name}: animated non-object data is unsupported; bake the coordinated motion first.')
    if not objects:
        raise ValueError('No keyed object actions to retime. Bake driver/constraint motion to actions first.')
    return objects


def _snapshot(scene, objects):
    return {
        'frame_start': scene.frame_start, 'frame_end': scene.frame_end,
        'current': scene.frame_current, 'subframe': scene.frame_subframe,
        'map_old': scene.render.frame_map_old, 'map_new': scene.render.frame_map_new,
        'preview_start': scene.frame_preview_start, 'preview_end': scene.frame_preview_end,
        'use_preview': scene.use_preview_range,
        'objects': {str(i): {'object': obj, 'action': obj.animation_data.action,
                    'slot': obj.animation_data.action_slot.identifier if obj.animation_data.action_slot else '',
                    'use_nla': obj.animation_data.use_nla, 'track': ''} for i, obj in enumerate(objects)},
        'markers': {str(i): {'name': marker.name, 'frame': marker.frame} for i, marker in enumerate(scene.timeline_markers)},
        'episodes': {str(i): {'name': item.episode_name, 'id': item.episode_id, 'start': item.start_frame, 'end': item.end_frame}
                     for i, item in enumerate(scene.mmrt_recorder.episodes)},
    }


def restore_timing(scene):
    state = scene.get(STATE)
    if state is None:
        return False
    # Validate all recovery references before changing anything.
    if len(scene.timeline_markers) != len(state['markers']) or len(scene.mmrt_recorder.episodes) != len(state['episodes']):
        raise ValueError('Marker/episode lists changed. Restore their original list layout before restoring timing.')
    for index, saved in state['markers'].items():
        if scene.timeline_markers[int(index)].name != saved['name']:
            raise ValueError('Marker names/order changed since retiming; restore that layout first.')
    for index, saved in state['episodes'].items():
        item = scene.mmrt_recorder.episodes[int(index)]
        if item.episode_name != saved['name'] or item.episode_id != saved['id']:
            raise ValueError('Episode names/order changed since retiming; restore that layout first.')
    assignments = []
    for saved in state['objects'].values():
        obj, action = saved.get('object'), saved.get('action')
        if obj is None or action is None or obj.animation_data is None:
            raise ValueError('A source object/action is missing; original timing cannot be restored automatically.')
        animation = obj.animation_data
        track = animation.nla_tracks.get(saved['track']) if saved['track'] else None
        if saved['track'] and track is None:
            raise ValueError(f'{obj.name}: the managed NLA track was renamed/removed; restore its name first.')
        if any(other != track for other in animation.nla_tracks) or animation.action not in (None, action, saved.get('working_action')):
            raise ValueError(f'{obj.name}: remove newly added animation layers before restoring timing.')
        slot = next((slot for slot in action.slots if slot.identifier == saved['slot']), None)
        if saved['slot'] and slot is None:
            raise ValueError(f'{obj.name}: original action slot is missing.')
        assignments.append((animation, track, action, slot, saved['use_nla']))
    for animation, track, action, slot, use_nla in assignments:
        if track:
            animation.nla_tracks.remove(track)
        animation.action = action
        animation.action_slot = slot
        animation.use_nla = use_nla
    for index, saved in state['markers'].items():
        scene.timeline_markers[int(index)].frame = saved['frame']
    for index, saved in state['episodes'].items():
        episode = scene.mmrt_recorder.episodes[int(index)]
        episode.start_frame, episode.end_frame = saved['start'], saved['end']
    scene.frame_start, scene.frame_end = state['frame_start'], state['frame_end']
    scene.render.frame_map_old, scene.render.frame_map_new = state['map_old'], state['map_new']
    scene.frame_preview_end = state['preview_end']
    scene.frame_preview_start = state['preview_start']
    scene.use_preview_range = state['use_preview']
    scene.frame_set(state['current'], subframe=state['subframe'])
    working_actions = {saved.get('working_action') for saved in state['objects'].values()} - {None}
    del scene[STATE]
    for action in working_actions:
        if action.users == 0:
            bpy.data.actions.remove(action)
    return True


def _make_strips(scene):
    strips = []
    for saved in scene[STATE]['objects'].values():
        obj, action = saved['object'], saved['action']
        animation = obj.animation_data
        track = animation.nla_tracks.new()
        track.name = 'MMRT Regional Timing'
        saved['track'] = track.name
        strip = track.strips.new('MMRT Regional Timing', scene.frame_start, action)
        if saved['slot']:
            strip.action_slot = next(slot for slot in action.slots if slot.identifier == saved['slot'])
        strip.use_animated_time = True
        strip.use_animated_time_cyclic = False
        strip.blend_type = 'REPLACE'
        strip.extrapolation = 'HOLD'
        animation.action = None
        animation.use_nla = True
        strips.append(strip)
    return strips


def _set_clock(strip, old, new, slopes):
    # Extend the shared clock by identity/translation outside the checked range.
    start = min(float(strip.action.frame_range[0]), old[0])
    end = max(float(strip.action.frame_range[1]), old[-1])
    source, target, derivatives = list(old), list(new), list(slopes)
    if start < old[0]:
        source.insert(0, start); target.insert(0, start); derivatives.insert(0, 1.0)
    if end > old[-1]:
        source.append(end); target.append(end + new[-1] - old[-1]); derivatives.append(1.0)
    strip.action_frame_start = start
    strip.action_frame_end = end
    strip.frame_start = start
    strip.frame_end = target[-1]
    curve = next((curve for curve in strip.fcurves if curve.data_path == 'strip_time'), None)
    if curve is None:
        strip.keyframe_insert('strip_time', frame=target[0])
        curve = next(curve for curve in strip.fcurves if curve.data_path == 'strip_time')
    curve.keyframe_points.clear()
    curve.keyframe_points.add(len(source))
    for i, key in enumerate(curve.keyframe_points):
        x, y, slope = target[i], source[i], derivatives[i]
        left = (x - target[i-1]) / 3 if i else 1/3
        right = (target[i+1] - x) / 3 if i+1 < len(source) else 1/3
        key.co = (x, y)
        key.interpolation = 'BEZIER'
        key.handle_left_type = key.handle_right_type = 'FREE'
        key.handle_left = (x - left, y - slope*left)
        key.handle_right = (x + right, y + slope*right)
    curve.extrapolation = 'LINEAR'
    curve.update()
    return curve


def _forward(frame, old, new, curve):
    if frame <= old[0]:
        return float(frame)
    if frame >= old[-1]:
        return float(frame + new[-1] - old[-1])
    index = int(np.searchsorted(old, frame))
    low, high = new[index-1], new[index]
    for _ in range(30):
        mid = (low + high) / 2
        if curve.evaluate(mid) < frame:
            low = mid
        else:
            high = mid
    return float((low + high) / 2)


def _verify(scene, settings, skipped=(), log=None):
    from . import joint_limit_checks as checks
    if log:
        log(f'Validating at 1 sample/frame, end frame {scene.frame_end}.')
    config = SimpleNamespace(limits_path=settings.limits_path,
                             samples_per_frame=1, tolerance=settings.tolerance)
    issues, notes, _, _, _ = checks.sample_scene(scene, config)
    return _unskipped_issues(issues, skipped), notes


def _severity(issues):
    """Peak limit ratios, independent of the stretched timeline's sample count."""
    peaks = {'velocity': 1.0, 'acceleration': 1.0}
    for issue in issues:
        metric = issue['metric']
        if metric == 'position':
            return (math.inf, math.inf)
        bound = issue['upper']
        peaks[metric] = max(peaks[metric], abs(issue['value']) / bound if bound else math.inf)
    return peaks['velocity'], peaks['acceleration']


def _improves(candidate, reference):
    return (all(a <= b + 1e-6 for a, b in zip(candidate, reference))
            and sum(candidate) < sum(reference) - 1e-6)


def _unskipped_issues(issues, skipped):
    return [issue for issue in issues if issue["metric"] == "position" or not any(
        lo <= issue["source_frame"] <= hi for lo, hi in skipped)]


def retime_scene(scene):
    from . import joint_limit_checks as checks
    settings = scene.mmrt_joint_limits
    settings.samples_per_frame = 1
    started = time.monotonic()
    report = bpy.data.texts.new('MMRT Retiming Log')
    settings.retime_log = report
    def log(message):
        line = f'[MMRT retime +{time.monotonic()-started:.1f}s] {message}'
        print(line, flush=True)
        report.write(line + '\n')
    skipped = []
    objects = _preflight(scene, skipped if settings.retime_skip_steps else None)
    direct = settings.retime_method == 'KEYFRAMES'
    speedup = direct and settings.retime_speedup
    intervals = motion_intervals(objects, _curves, scene.frame_start, scene.frame_end) if direct else []
    log('Method: move existing keys and handles' if direct else 'Method: shared NLA clock')
    log('Synchronized keyed objects: ' + ', '.join(obj.name for obj in objects))
    domain = get_frame_domain(scene)
    ratio = domain['stretch_factor']
    count = int(math.ceil((scene.frame_end-scene.frame_start) * ratio * settings.samples_per_frame)) + 1
    if count < 4 or count > 25000:
        raise ValueError('Regional retiming needs 4–25,000 samples; adjust the range/sampling.')
    frames = np.linspace(scene.frame_start*ratio, scene.frame_end*ratio, count)
    # Also compare world-space objects/bones, including constrained roots and IK
    # controllers. Joint-axis values alone cannot detect a changed world path.
    pose_objects = list(scene.objects)
    interaction_objects = [obj for obj in pose_objects if any(
        constraint.type == 'CHILD_OF' and constraint.target for constraint in obj.constraints)]
    interaction_poses = {}
    log('Attachment objects checked at every source sample: ' +
        (', '.join(obj.name for obj in interaction_objects) or '(none)'))
    pose_count = len(pose_objects) + sum(len(obj.pose.bones) for obj in pose_objects if obj.type == 'ARMATURE')
    indices = set(np.linspace(0, count-1, min(count, 512, max(4, 1000000 // max(1, pose_count*16))), dtype=int))
    reference_poses = {}
    pose_labels = []
    for obj in pose_objects:
        pose_labels.append(obj.name)
        if obj.type == 'ARMATURE':
            pose_labels.extend(f'{obj.name}/{bone.name}' for bone in obj.pose.bones)
    def pose_mismatch(expected, actual, labels, index):
        difference = np.abs(expected - actual)
        translations, angles, stretches = pose_errors(expected, actual)
        scores = np.maximum.reduce((translations / 1e-4, angles / .05, stretches / 1e-4))
        row = int(np.argmax(scores))
        translation = float(translations[row])
        return (f'{labels[row]}: world-space pose changed at original frame '
                f'{frames[index]/ratio:.6f}, retimed frame {scene.frame_current_final:.6f}; '
                f'max matrix difference {difference[row].max():.6g}, '
                f'translation difference {translation:.6g} scene units, '
                f'rotation difference {angles[row]:.6g} degrees, scale/shear difference {stretches[row]:.6g}. '
                'Original timing restored. Inspect this object’s animation and constraints; '
                'this comparison alone does not establish that baking is required.')
    def pose_array(graph):
        matrices = []
        for obj in pose_objects:
            evaluated = obj.evaluated_get(graph)
            matrices.append(np.asarray(evaluated.matrix_world))
            if obj.type == 'ARMATURE':
                matrices.extend(np.asarray(evaluated.matrix_world @ bone.matrix) for bone in evaluated.pose.bones)
        result = np.asarray(matrices, dtype=np.float32)
        if not np.isfinite(result).all():
            raise ValueError('Nonfinite evaluated pose; cannot preserve this motion.')
        return result
    def capture_pose(index, graph):
        if interaction_objects:
            interaction_poses[index] = np.asarray([
                obj.evaluated_get(graph).matrix_world for obj in interaction_objects], dtype=np.float32)
        if index % 1000 == 0:
            log(f'Original poses: sample {index+1}/{count}.')
        if index in indices:
            reference_poses[index] = pose_array(graph)
    def verify_pose(index, graph):
        if interaction_objects:
            actual = np.asarray([obj.evaluated_get(graph).matrix_world for obj in interaction_objects], dtype=np.float32)
            if not poses_match(interaction_poses[index], actual):
                raise ValueError(pose_mismatch(interaction_poses[index], actual,
                                               [obj.name for obj in interaction_objects], index))
        if index % 1000 == 0:
            log(f'Preserved-pose check: sample {index+1}/{count}.')
        if index in indices:
            actual = pose_array(graph)
            if not poses_match(reference_poses[index], actual):
                raise ValueError(pose_mismatch(reference_poses[index], actual, pose_labels, index))
    log(f'Sampling original motion: {count} samples; world-pose checks include all scene objects.')
    descriptors, notes, _, samples, fps, _ = checks.sample_motion(scene, settings, frames=frames, on_sample=capture_pose)
    issues = []
    for desc, values in zip(descriptors, samples):
        found, missing = analyze_samples(frames, values, fps, desc['limits'], angular=True, tolerance=settings.tolerance)
        notes.extend(f'{desc["label"]}: {item}' for item in missing)
        issues.extend(found)
    if notes:
        raise ValueError('Retiming requires complete check coverage: ' + notes[0])
    if any(issue['metric'] == 'position' for issue in issues):
        raise ValueError('Position limits are exceeded. Timing changes cannot repair the motion path.')
    if not issues and not speedup:
        return 'No sampled velocity/acceleration violations; timing unchanged.'
    old = frames / ratio
    # Exclude the finite-difference stencil around each protected segment too.
    padding = 2 * (old[1] - old[0])
    skipped = [(lo-padding, hi+padding) for lo, hi in skipped]
    excluded = np.array([any(lo <= frame <= hi for lo, hi in skipped) for frame in old])
    if excluded.all():
        return "All regions contain unrepairable transitions; timing unchanged. Limit violations remain."
    baseline_issues, missing = _verify(scene, settings, skipped, log)
    if missing:
        raise ValueError('Original validation has incomplete coverage.')
    baseline = _severity(baseline_issues)
    best = None
    log(f'Original peak ratios: velocity {baseline[0]:.4f}x, acceleration {baseline[1]:.4f}x limit.')
    old_end = scene.frame_end
    scene[STATE] = _snapshot(scene, objects)
    scene[STATE]['method'] = settings.retime_method
    try:
        if direct:
            make_working_actions(scene[STATE])
            strips = []
        else:
            strips = _make_strips(scene)
        total_attempts = settings.retime_attempts + int(speedup)
        for attempt in range(total_attempts):
            candidate_speedup = speedup and attempt < settings.retime_attempts
            log(f'Attempt {attempt+1}/{total_attempts}: planning regional stretch.')
            if speedup and not candidate_speedup:
                log('Speedup candidates did not pass; trying slowdown-only fallback.')
            try:
                proposed, slopes = plan_retime(frames, samples, [desc['limits'] for desc in descriptors], fps,
                    margin=settings.retime_margin * 0.8**attempt, transition_seconds=settings.retime_transition,
                    excluded=excluded)
            except ValueError as exc:
                if best is None or not settings.retime_keep_improvement:
                    raise
                log(f'Attempt {attempt+1}: planner stopped ({exc}); retaining earlier verified improvement.')
                break
            clock_source, clock_target = old, proposed / ratio
            if direct:
                clock_source, clock_target = shared_key_clock(old, proposed / ratio, intervals)
                if candidate_speedup:
                    slowdown_target = clock_target.copy()
                    clock_target = accelerate_blocks(clock_source, clock_target, old, samples,
                        [desc['limits'] for desc in descriptors], fps/ratio,
                        settings.retime_margin * 0.8**attempt, settings.retime_max_speedup,
                        intervals, skipped)
                    strength = 0.5**attempt
                    clock_target = slowdown_target + strength * (clock_target-slowdown_target)
                    log(f'Speedup strength: {strength:.1%} of the proposed compression.')
                    for lo, hi, duration in zip(clock_source, clock_source[1:], np.diff(clock_target)):
                        log(f'Block {lo:g}–{hi:g}: playback speed {(hi-lo)/duration:.3f}x.')
                proposed = np.interp(old, clock_source, clock_target) * ratio
                log(f'Attempt {attempt+1}: {len(clock_source)-1} shared timing blocks; key counts unchanged.')
            # Align the new endpoint with an integer source frame without speeding up.
            new = proposed / ratio
            endpoint = math.ceil(new[-1])
            delay = new[-1] - old[-1]
            if speedup:
                # Uniform upward rounding preserves positive block durations even
                # when some blocks contract and the net duration decreases.
                factor = (endpoint-new[0])/(new[-1]-new[0])
                new = new[0]+(new-new[0])*factor
                clock_target = clock_target[0]+(clock_target-clock_target[0])*factor
            else:
                if delay <= 0:
                    restore_timing(scene)
                    return 'No stretch needed outside skipped regions; timing unchanged. Skipped violations remain.'
                factor = (endpoint - old[-1]) / delay
                new = old + (new - old)*factor
                clock_target = clock_source + (clock_target-clock_source)*factor
                slopes = 1 / (1 + (1/slopes - 1)*factor)
            if endpoint > 1048574 or endpoint <= old[0]:
                raise ValueError('Retimed range exceeds Blender frame limits.')
            if direct:
                apply_key_clock(scene[STATE], clock_source, clock_target, _curves)
                curve = SimpleNamespace(evaluate=lambda frame: float(np.interp(frame, clock_target, clock_source)))
            else:
                for strip in strips:
                    curve = _set_clock(strip, old, new, slopes)
            scene.frame_end = endpoint
            log(f'Attempt {attempt+1}: proposed end {endpoint}; checking synchronized poses.')
            # Confirm the SAME original poses at corresponding clock nodes.
            _, _, _, retimed_samples, _, _ = checks.sample_motion(scene, settings, frames=new*ratio, on_sample=verify_pose)
            for before, after in zip(samples, retimed_samples):
                delta = (np.asarray(after) - before + np.pi) % (2*np.pi) - np.pi
                if np.max(np.abs(delta)) > 1e-4:
                    raise ValueError('Evaluated poses changed under retiming. Bake time-dependent constraints first.')
            verified, missing = _verify(scene, settings, () if candidate_speedup else skipped, log)
            if missing:
                raise ValueError('Retimed validation has incomplete coverage.')
            speedup_accepted = False
            if candidate_speedup:
                accelerated, blocking = speedup_violations(verified, clock_source, clock_target, ratio)
                speedup_accepted = bool(accelerated) and not blocking
                log(f'Speedup validation: {len(accelerated)} accelerated blocks; '
                    f'{len(blocking)} violations in accelerated blocks/boundary guards; '
                    f'{len(verified)-len(blocking)} violations elsewhere.')
                for issue in blocking[:5]:
                    log(f'Blocked by {issue.get("joint", "joint")}: {issue["metric"]} '
                        f'at original frame {issue["source_frame"]:g}.')
            score = _severity(verified)
            log(f'Attempt {attempt+1}: {len(verified)} violations at 1 sample/frame outside skipped regions; '
                f'peak velocity {score[0]:.4f}x, acceleration {score[1]:.4f}x limit.')
            if speedup_accepted or not verified or (not candidate_speedup and _improves(score, baseline) and (best is None or _improves(score, best[0]))):
                best = (score, new.copy(), slopes.copy(), endpoint, attempt+1, clock_source.copy(), clock_target.copy(), candidate_speedup)
                log(f'Attempt {attempt+1}: saved best verified timing.')
            if speedup_accepted or not verified:
                break
        else:
            if best is None or not settings.retime_keep_improvement:
                raise ValueError('No acceptable limit-compliant timing found; original timing restored. '
                                 'No speedup candidate or slowdown-only fallback passed the acceptance checks. Increase Maximum Attempts or inspect the logged violations.')
        score, new, slopes, endpoint, chosen_attempt, clock_source, clock_target, chosen_speedup = best
        if direct:
            apply_key_clock(scene[STATE], clock_source, clock_target, _curves)
            curve = SimpleNamespace(evaluate=lambda frame: float(np.interp(frame, clock_target, clock_source)))
        else:
            for strip in strips:
                curve = _set_clock(strip, old, new, slopes)
        scene.frame_end = endpoint
        log(f'Keeping attempt {chosen_attempt}; end frame {endpoint}.')
        state = scene[STATE]
        for index, saved in state['markers'].items():
            scene.timeline_markers[int(index)].frame = round(_forward(saved['frame'], old, new, curve))
        for index, saved in state['episodes'].items():
            episode = scene.mmrt_recorder.episodes[int(index)]
            episode.end_frame = round(_forward(saved['end'], old, new, curve))
            episode.start_frame = round(_forward(saved['start'], old, new, curve))
        scene.frame_preview_end = round(_forward(state['preview_end'], old, new, curve))
        scene.frame_preview_start = round(_forward(state['preview_start'], old, new, curve))
        current = _forward((state['current'] + state['subframe']) / ratio, old, new, curve)*ratio
        scene.frame_set(math.floor(current), subframe=current-math.floor(current))
        final_issues, final_missing = _verify(scene, settings, () if chosen_speedup else skipped, log)
        if chosen_speedup:
            _, final_blocking = speedup_violations(final_issues, clock_source, clock_target, ratio)
            final_rejected = bool(final_blocking)
        else:
            final_rejected = bool(final_issues) and (not settings.retime_keep_improvement or not _improves(_severity(final_issues), baseline))
        if final_missing or final_rejected:
            raise ValueError('Final validation failed after updating event timing; original timing restored.')
        settings.retime_status = f'Validated regional timing: end {old_end} → {endpoint}; {len(objects)} synchronized objects.'
        if final_issues:
            settings.retime_status = (f'Partial retiming: kept improved attempt {chosen_attempt}; end {old_end} → {endpoint}. '
                f'Peak velocity {baseline[0]:.3f}x → {score[0]:.3f}x, acceleration {baseline[1]:.3f}x → {score[1]:.3f}x. '
                'Limit violations remain; see joint checks.')
        if chosen_speedup and final_issues:
            settings.retime_status = (f'Partial retiming: accelerated blocks and boundary guards passed; '
                f'end {old_end} → {endpoint}. {len(final_issues)} violations remain elsewhere; see joint checks.')
        if speedup and not chosen_speedup:
            settings.retime_status += ' Speedup was rejected; kept slowdown-only fallback.'
        if skipped:
            ranges = ', '.join(f'{lo+padding:g}–{hi-padding:g}' for lo, hi in sorted(set(skipped)))
            settings.retime_status = ('Partial retiming. ' + settings.retime_status +
                f' Skipped source segments: {ranges}. Limit violations may remain; see joint checks.')
        log(settings.retime_status)
        return settings.retime_status
    except Exception as exc:
        log(f'Rolling back: {exc}')
        restore_timing(scene)
        raise


class MMRT_OT_RetimeJointLimits(bpy.types.Operator):
    bl_idname = get_bl_idname('retime_joint_limits')
    bl_label = 'Stretch Violating Regions'
    bl_description = 'Apply a shared regional clock to all keyed scene objects; preserve source actions and verify sampled limits'
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        from . import joint_limit_checks as checks
        if (context.screen and context.screen.is_animation_playing) or bpy.app.is_job_running('RENDER'):
            self.report({'ERROR'}, 'Stop playback/rendering before retiming.')
            return {'CANCELLED'}
        checks._BUSY = True
        failure = None
        try:
            message = retime_scene(context.scene)
        except Exception as exc:
            context.scene.mmrt_joint_limits.retime_status = str(exc)
            self.report({'ERROR'}, str(exc))
            failure = exc
        finally:
            checks._BUSY = False
            context.scene.mmrt_joint_limits.stale = True
        checks.run_check(context.scene)
        if failure is not None:
            return {'CANCELLED'}
        self.report({'WARNING'} if ('Partial retiming' in message or 'remain' in message) else {'INFO'}, message)
        return {'FINISHED'}


class MMRT_OT_RestoreJointTiming(bpy.types.Operator):
    bl_idname = get_bl_idname('restore_joint_timing')
    bl_label = 'Restore Original Timing'
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        from . import joint_limit_checks as checks
        if (context.screen and context.screen.is_animation_playing) or bpy.app.is_job_running('RENDER'):
            self.report({'ERROR'}, 'Stop playback/rendering before restoring timing.')
            return {'CANCELLED'}
        checks._BUSY = True
        try:
            restore_timing(context.scene)
            context.scene.mmrt_joint_limits.retime_status = 'Original actions and timing restored.'
        except Exception as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        finally:
            checks._BUSY = False
            context.scene.mmrt_joint_limits.stale = True
        checks.run_check(context.scene)
        return {'FINISHED'}
