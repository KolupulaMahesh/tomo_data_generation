"""Blender action copies with retimed existing keys and unchanged key values."""
from .common.keyframe_retime import map_frame


def motion_intervals(objects, curves, start, end):
    intervals = []
    seen = set()
    for obj in objects:
        action = obj.animation_data.action
        # Retiming copies the entire action (including all shared slots).
        if action.as_pointer() in seen:
            continue
        seen.add(action.as_pointer())
        for curve in curves(action, None):
            if curve.modifiers or len(curve.sampled_points):
                raise ValueError(f'{action.name}: direct keyframe retiming requires key curves without F-curve modifiers or sampled points.')
            keys = list(curve.keyframe_points)
            for left, right in zip(keys, keys[1:]):
                constant = abs(left.co.y-right.co.y) < 1e-10
                if left.interpolation == 'BEZIER':
                    constant = constant and abs(left.handle_right.y-left.co.y) < 1e-10 and abs(right.handle_left.y-left.co.y) < 1e-10
                if not constant:
                    intervals.append((left.co.x, right.co.x))
            if keys and curve.extrapolation == 'LINEAR':
                # Conservatively include extrapolated regions in the common clock.
                intervals.extend([(start, keys[0].co.x), (keys[-1].co.x, end)])
    return intervals


def make_working_actions(state):
    copies = {}
    for saved in state['objects'].values():
        source = saved['action']
        if source.as_pointer() not in copies:
            working = source.copy()
            working.name = source.name + ' [Retimed Keys]'
            working.use_fake_user = False
            copies[source.as_pointer()] = working
        working = copies[source.as_pointer()]
        saved['working_action'] = working
        animation = saved['object'].animation_data
        animation.action = working
        animation.action_slot = next((slot for slot in working.slots if slot.identifier == saved['slot']), None)


def apply_key_clock(state, source_times, target_times, curves):
    state['source_times'] = list(source_times)
    state['target_times'] = list(target_times)
    visited = set()
    for saved in state['objects'].values():
        action = saved['working_action']
        if action.as_pointer() in visited:
            continue
        visited.add(action.as_pointer())
        for original, changed in zip(curves(saved['action'], None), curves(action, None)):
            keys = list(original.keyframe_points)
            for index, (src, dst) in enumerate(zip(keys, changed.keyframe_points)):
                x = src.co.x
                mapped = map_frame(x, source_times, target_times)
                def scale(first, last):
                    return (map_frame(last, source_times, target_times)-map_frame(first, source_times, target_times))/(last-first) if last != first else 1.
                left_scale = scale(keys[index-1].co.x, x) if index else (scale(x, keys[index+1].co.x) if len(keys)>1 else 1.)
                right_scale = scale(x, keys[index+1].co.x) if index+1<len(keys) else left_scale
                # Freeze the existing Bezier geometry; automatic handle recomputation
                # after nonuniform time edits can otherwise alter the motion path.
                dst.handle_left_type = dst.handle_right_type = 'FREE'
                dst.co = (mapped, src.co.y)
                dst.handle_left = (mapped+(src.handle_left.x-x)*left_scale, src.handle_left.y)
                dst.handle_right = (mapped+(src.handle_right.x-x)*right_scale, src.handle_right.y)
            changed.update()
