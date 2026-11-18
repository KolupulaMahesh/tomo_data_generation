# motion_segmentation.py
import numpy as np
from scipy.signal import find_peaks
from typing import List, Tuple, Dict
from common.recording_data import RecordingData
from definitions import FlagTypes, MeasurementUnits, WaypointMergeMethod


def calculate_joint_velocity(data: np.ndarray, fps: float) -> np.ndarray:
    """
    Computes the velocity and speed of multi-dimensional joint data.
    Args:
        data: shape (frames, joints)
        fps: frames per second
    Returns:
        velocity: shape (frames, joints)
    """
    dt = 1.0 / fps
    vel = np.gradient(data, dt, axis=0)
    return vel


def find_motion_segments(speed_data: np.ndarray, threshold=0.1, min_duration_frames=8) -> List[Tuple[int, int]]:
    """
    Segments multi-dimensional joint data based on speed threshold.
    Args:
        speed_data: shape (frames,)
        threshold: speed above which we consider “in motion”
        min_duration_frames: minimum length for a valid segment
    Returns:
        List of (start_frame, end_frame) tuples representing motion segments.
    """
    segments = []
    in_seg = False
    start = 0
    for i, v in enumerate(speed_data):
        moving = (v > threshold)
        if moving and not in_seg:
            in_seg = True
            start = i
        elif not moving and in_seg:
            if i - start >= min_duration_frames:
                segments.append((start, i-1))
            in_seg = False
    # Handle case where motion continues to end
    if in_seg and (len(speed_data) - start) >= min_duration_frames:
        segments.append((start, len(speed_data) - 1))
    return segments


def bin_segments_with_groups(group_segments: Dict[str, List[Tuple[int, int]]]) -> List[Tuple[int, int, List[str]]]:
    # Flatten all segments with group label
    all_labeled_segments = []
    for group, segments in group_segments.items():
        for s, e in segments:
            all_labeled_segments.append((s, e, group))

    # Sort by start time
    all_labeled_segments.sort(key=lambda x: x[0])

    merged = []

    for s, e, group in all_labeled_segments:
        if not merged:
            merged.append([s, e, {group}])
        else:
            last_s, last_e, last_groups = merged[-1]
            if s <= last_e + 1:
                # Merge overlapping or adjacent
                merged[-1][1] = max(last_e, e)
                merged[-1][2].add(group)
            else:
                merged.append([s, e, {group}])

    # Convert group sets to lists for consistency
    return [(s, e, sorted(list(groups))) for s, e, groups in merged]


def merge_group_segments(group_segments: Dict[str, List[Tuple[int, int]]]) -> List[Tuple[int, int, List[str]]]:
    """
    Merge and intersect multiple groups of time segments, producing a timeline that shows
    which groups are active (overlapping) at each period.
    Example:
        Input:
            group_segments = {
                "A": [(1, 3), (6, 8)],
                "B": [(2, 4)],
                "C": [(7, 9)],
            }

        Output:
            [
                (1, 1, ['A']),
                (2, 3, ['A', 'B']),
                (4, 4, ['B']),
                (6, 6, ['A']),
                (7, 8, ['A', 'C']),
                (9, 9, ['C'])
            ]
    """
    # Collect events: (time, is_start, group)
    events = []
    for group, segments in group_segments.items():
        for s, e in segments:
            events.append((s, 1, group))       # start
            events.append((e + 1, -1, group))  # end (exclusive)

    # Sort: end (-1) before start (+1) at same timestamp
    events.sort()

    active = set()
    merged = []
    prev_time = None

    for time, typ, group in events:
        # Emit interval BEFORE changing active-set
        if prev_time is not None and prev_time < time and active:
            merged.append((prev_time, time - 1, sorted(active)))

        # Now apply the event
        if typ == 1:
            active.add(group)
        else:
            active.discard(group)

        prev_time = time

    return merged


def select_best_move_group(active_atoms: List[str], move_groups: Dict[str, List[str]]) -> str:
    active_set = set(active_atoms)
    candidate_groups = []

    for mg_name, mg_atoms in move_groups.items():
        mg_set = set(mg_atoms)
        if active_set.issubset(mg_set):
            candidate_groups.append((mg_name, len(mg_atoms)))

    if not candidate_groups:
        raise ValueError(f"No move group can contain: {active_atoms}")

    # Choose the most specific one (fewest atomic groups)
    candidate_groups.sort(key=lambda x: x[1])
    return candidate_groups[0][0]


def map_atomic_segments_to_move_groups(
        segments: List[Tuple[int, int, List[str], List[int]]],
        move_groups: Dict[str, List[str]]) -> List[Tuple[int, int, str, List[int]]]:
    """
    Assigns a movegroup to each segment, and if consecutive segments are within threshold_frames,
    unifies their movegroup to the largest one that can handle all their active atoms combined.
    """
    binned_segments = [
        (start, end, select_best_move_group(active_atoms, move_groups), waypoints)
        for start, end, active_atoms, waypoints in segments
    ]
    return binned_segments


def merge_nearby_segments(
        segments: List[Tuple[int, int, List[str]]],
        threshold_frames: int = 0
    ) -> List[Tuple[int, int, List[str], List[int]]]:
    """
    Combine segments that are close in time, and collect the junction frames
    as initial waypoints.
    
    Input segment format:
        (start, end, atomic_groups)
    
    Output format:
        (start, end, atomic_groups, waypoints)
    """

    if not segments:
        return []

    segments = sorted(segments, key=lambda x: x[0])
    
    combined = []
    cur_start, cur_end, cur_groups = segments[0]
    cur_waypoints = []

    for start, end, groups in segments[1:]:
        if start - cur_end <= threshold_frames:
            # Merge
            if start != cur_end:
                # if there's a gap, store the join point
                cur_waypoints.append(start)
            cur_end = end
            cur_groups = list(set(cur_groups) | set(groups))
        else:
            # Push previous
            combined.append((cur_start, cur_end, cur_groups, sorted(cur_waypoints)))
            # Reset
            cur_start, cur_end, cur_groups = start, end, groups
            cur_waypoints = []

    # Last segment
    combined.append((cur_start, cur_end, cur_groups, sorted(cur_waypoints)))
    return combined


def find_local_speed_minima(speed_data: np.ndarray, start: int | None = None, end: int | None = None) -> List[int]:
    """
    Identify robust local minima in speed data.
    Returns a list of frame indices where local minima occur.
    Filters out edge artifacts, flat zero regions, and small oscillations.
    """
    if speed_data is None or len(speed_data) < 3:
        return []

    # Safe bounds
    start = max(1, start or 1)
    end = min(len(speed_data), end or len(speed_data))

    segment = speed_data[start:end]

    # Handle all-zero regions: no meaningful minima
    if np.allclose(segment, 0):
        return []

    # Find peaks in the inverted speed signal = local minima
    inverted = -segment

    # Parameters:
    #   - distance: min separation (in frames) between minima
    #   - prominence: ignore tiny dips
    peaks, _ = find_peaks(inverted, distance=8, prominence=0.01)

    # Convert local indices to global frame indices
    minima_indices = (peaks + start).tolist()

    # Optionally filter out minima too close to edges (2 frames from start/end)
    minima_indices = [
        i for i in minima_indices
        if (i - start) > 2 and (end - i) > 2
    ]

    return minima_indices


def merge_nearby_indices(indices, window=10, method=WaypointMergeMethod.MEAN):
    method = str(method).upper()
    """
    Merge indices that are within a certain window of each other.
    Args:        
        indices: list or np.array of integer indices
        window: maximum gap to consider for merging
        method: how to merge ('MEAN', 'MEDIAN', 'FIRST')
    Returns:        list of merged indices
     Example:        indices = [10, 12, 13, 50, 51, 90]
        window = 5
        Output: [11 (mean of 10,12,13), 50.5 (mean of 50,51), 90]
    """
    if len(indices) == 0:
        return []

    indices = np.sort(indices)
    diffs = np.diff(indices)

    # Group start points whenever gap > window
    group_starts = np.r_[0, np.where(diffs > window)[0] + 1]
    group_ends = np.r_[group_starts[1:], len(indices)]

    merged = []
    for s, e in zip(group_starts, group_ends):
        group = indices[s:e]
        if method == WaypointMergeMethod.MEAN:
            merged.append(int(np.mean(group)))
        elif method == WaypointMergeMethod.MEDIAN:
            merged.append(int(np.median(group)))
        elif method == WaypointMergeMethod.FIRST:
            merged.append(int(group[0]))
        else:
            raise ValueError(f"Unknown method: {method}")

    return merged


def smooth_joint_angles_with_limits(values, unit=MeasurementUnits.DEGREE_UNIT, min_val=None, max_val=None):
    """
    Smooths a sequence of angular values by removing wraparound jumps, with optional min/max constraints.

    Parameters:
        values (list or np.ndarray): Sequence of angles.
        unit (str): 'DEGREES' or 'RADIANS'. Defaults to 'DEGREES'.
        min_val (float or None): Minimum allowed output value (same unit).
        max_val (float or None): Maximum allowed output value (same unit).

    Returns:
        np.ndarray: Smoothed angle sequence satisfying continuity and limits when possible.
    """
    vals = np.asarray(values, dtype=float)
    if unit == MeasurementUnits.DEGREE_UNIT:
        period = 360.0
    elif unit == MeasurementUnits.RADIAN_UNIT:
        period = 2 * np.pi
    else:
        raise ValueError("Unit must be 'DEGREES' or 'RADIANS'")

    smoothed = [vals[0]]
    for raw in vals[1:]:
        prev = smoothed[-1]
        # generate candidate offsets n * period near prev
        # estimate n so raw + n*period is close to prev
        n0 = int(np.round((prev - raw) / period))
        candidates = []
        # consider n0 and its neighbors
        for dn in (-1, 0, 1):
            candidate = raw + (n0 + dn) * period
            # if limits provided, test
            if min_val is not None and candidate < min_val:
                continue
            if max_val is not None and candidate > max_val:
                continue
            candidates.append(candidate)
        if candidates:
            # choose candidate closest to prev
            best = min(candidates, key=lambda c: abs(c - prev))
            smoothed.append(best)
        else:
            # fallback to unconstrained unwrap
            delta = raw - vals[np.where(vals == raw)[0][0] - 1] if False else raw - raw  # dummy
            # simpler: use numpy unwrap logic
            delta = raw - (raw - (prev - raw))  # dummy fallback to itself
            # actually just use short wrap
            dmod = (raw - (raw) + period/2) % period - period/2
            smoothed.append(prev + dmod)
    return np.array(smoothed)



def split_segments_at_events(combined_segments_with_atomic, recording: RecordingData):
    """
    Split motion segments whenever:
        - FORCED_SPLIT flag appears (general flags)
        - IO event appears (flags_io)
        - LED event appears (flags_led)

    Input segments:
        (start_frame, end_frame, mgroup, waypoint_list)

    Output segments:
        Split segments with waypoint_list trimmed to segment bounds.
    """

    # -------------------------------------------------------------
    # Build list of frames that force a segment split
    # -------------------------------------------------------------
    trigger_frames = set()

    # From general flags
    for frame, codes in recording.flags_general.items():
        for c in codes:
            if c in (FlagTypes.IO_CONTROL.value, FlagTypes.FORCED_SPLIT.value):
                trigger_frames.add(frame)

    # From dedicated IO table
    trigger_frames.update(recording.flags_io.keys())

    # From dedicated LED table
    trigger_frames.update(recording.flags_led.keys())

    # to keep original flag unchaged
    trigger_frames = {
        int(frame) - recording.frame_start
        for frame in trigger_frames
        if recording.frame_start <= int(frame) <= recording.frame_end
    }
    
    # No flags → no splitting
    if not trigger_frames:
        # also trim WPs here for safety
        trimmed = []
        for (start, end, mg, wps) in combined_segments_with_atomic:
            trimmed.append((start, end, mg, [wp for wp in wps if start <= wp <= end]))
        return trimmed

    trigger_frames = sorted(trigger_frames)

    # -------------------------------------------------------------
    # Apply splitting
    # -------------------------------------------------------------
    new_segments = []

    for (start, end, mgroup, wps) in combined_segments_with_atomic:

        # enforce waypoint trimming BEFORE splits
        wps = [wp for wp in wps if start <= wp <= end]

        # find flags inside this segment
        split_points = [f for f in trigger_frames if start < f < end]

        if not split_points:
            # no splits → trim and pass through
            new_segments.append(
                (start, end, mgroup, wps)
            )
            continue

        # sequential splitting
        current_start = start

        for sp in split_points:
            # waypoints for this subsegment
            sub_wps = [wp for wp in wps if current_start <= wp <= sp]
            new_segments.append((current_start, sp, mgroup, sub_wps))
            current_start = sp + 1

        # final tail segment
        current_start = min(current_start, end)
        sub_wps = [wp for wp in wps if current_start <= wp <= end]
        new_segments.append((current_start, end, mgroup, sub_wps))

    return new_segments


def make_waypoint_signature(plangroup, atomic_positions):
    key = [plangroup]

    for k in sorted(atomic_positions.keys()):
        vals = atomic_positions[k]

        if isinstance(vals, (list, tuple)):
            key.append(",".join(f"{float(v):.6f}" for v in vals))
        else:
            key.append(f"{float(vals):.6f}")

    return "|".join(key)


def make_trajectory_signature(plangroup, waypoint_names):
    return plangroup + "|" + "|".join(waypoint_names)
