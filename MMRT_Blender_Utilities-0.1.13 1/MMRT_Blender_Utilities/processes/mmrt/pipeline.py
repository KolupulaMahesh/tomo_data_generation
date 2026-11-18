from argparse import ArgumentParser
import os
import numpy as np
from pathlib import Path
import logging
import json
import common.recording_data as recording_data
import common.robot_description as robot_description
import common.motion_analysis as motion_analysis
from common.simulation import Simulation
from common.visualization import Visualization
from common.behavior_tree_pipeline import PipelineAnalysis, build_and_export_behavior_tree
from common.logging_setup import trace, setup_logger
from definitions import (
    MMRTPlannerConfig,
    MMRTProcessDefaults,
    MoveTomoTrajectoryConfig,
    LoggingVerbosity,
    PipelineCLIOptions,
    RobotJointTypes,
    WaypointMergeMethod,
    MeasurementUnits,
)
from common.console_effects import (
    boot_sequence,
    effect_data_mapping, effect_signal_processing,
    effect_segmentation, effect_waypoint_learning,
    effect_simulation
)

import sys


logger = logging.getLogger(__name__)
setup_logger(verbosity=LoggingVerbosity.DEFAULT)


def _write_progress(progress_file, progress, message):
    """Atomically publish machine-readable pipeline progress for Blender."""
    if not progress_file:
        return
    path = Path(progress_file)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary_path.write_text(
            json.dumps({"progress": float(progress), "message": str(message)}),
            encoding="utf-8",
        )
        os.replace(temporary_path, path)
    except OSError as exc:
        logger.debug("Could not update progress file %s: %s", path, exc)


def _episode_for_range(episodes, start_frame, end_frame):
    """Return an episode only when the complete inclusive range is inside it."""
    for episode in episodes:
        if episode["start_frame"] <= start_frame and end_frame <= episode["end_frame"]:
            return episode["id"]
    return None


def _range_overlaps_episode(episodes, start_frame, end_frame):
    return any(
        start_frame <= episode["end_frame"] and episode["start_frame"] <= end_frame
        for episode in episodes
    )

@robot_description.cleanup_resolved_temp_files
@trace
def run_mmrt_pipeline(
        record_path: str,
        urdf_path: str|None=None,
        behavior_tree_models_path: str|None=None,
        behavior_tree_output_path: str|None=None,
        srdf_path: str|None=None,
        env_path: str|None=None,
        mmrt_speed_thr: float=MMRTProcessDefaults.MOTION_SPEED_THRESHOLD_RAD_PER_SEC,
        mmrt_min_segment_time: float=MMRTProcessDefaults.MINIMUM_SEGMENT_DURATION_SEC,
        mmrt_segment_combine_thr: float=MMRTProcessDefaults.SEGMENT_MERGE_GAP_SEC,
        mmrt_sleep_thr: int=MMRTProcessDefaults.BEHAVIOR_TREE_SLEEP_GAP_MS,
        mmrt_wp_decimation_window: int=MMRTProcessDefaults.WAYPOINT_MERGE_WINDOW,
        mmrt_wp_decimation_iter: int=MMRTProcessDefaults.WAYPOINT_MERGE_ITERATIONS,
        mmrt_wp_decimation_method: str=MMRTProcessDefaults.WAYPOINT_MERGE_METHOD,
        mmrt_collision_thr: float=MMRTProcessDefaults.COLLISION_DISTANCE_THRESHOLD_M,
        bt_vel_scale: float=recording_data.BlackboardDefaults.VELOCITY_SCALE,
        bt_acc_scale: float=recording_data.BlackboardDefaults.ACCELERATION_SCALE,
        bt_collision_checks: int=recording_data.BlackboardDefaults.COLLISION_CHECKS,
        bt_blend_radius: float=recording_data.BlackboardDefaults.BLEND_RADIUS,
        bt_save_plan: bool=recording_data.BlackboardDefaults.SAVE_PLAN,
        bt_plan_pipeline: str=None,
        bt_plan_id: str=None,
        bt_enable_log: bool=None,
        bt_enable_log_to_file: bool=None,
        bt_debug_topic_name: str=None,
        bt_trajectory_strategy: str=MoveTomoTrajectoryConfig.DEFAULT_STRATEGY,
        bt_trajectory_resample: str=MoveTomoTrajectoryConfig.DEFAULT_RESAMPLE,
        bt_trajectory_time_parameterization: str=MoveTomoTrajectoryConfig.DEFAULT_TIME_PARAMETERIZATION,
        bt_trajectory_visualize: bool=MoveTomoTrajectoryConfig.DEFAULT_VISUALIZE,
        bt_trajectory_resolution: float=MoveTomoTrajectoryConfig.DEFAULT_RESOLUTION,
        mmrt_is_sim: bool=False, 
        mmrt_is_plot: bool=False,
        mmrt_is_save_plot: bool=False,
        mmrt_is_export_bt: bool=True,
        mmrt_show_console: bool=False,
        app_dir: str|None=None,
        robot_name: str|None=None,
        tree_id: str=MMRTProcessDefaults.TREE_ID,
        configure_tree_name: str=MMRTProcessDefaults.CONFIGURE_TREE_NAME,
        scene_setup_tree_name: str=MMRTProcessDefaults.SCENE_SETUP_TREE_NAME,
        move_segment_tree_name: str=MMRTProcessDefaults.MOVE_SEGMENT_TREE_NAME,
        progress_file: str | None = None,
):
    """
        Runs the full pipeline from loading data to exporting a Behavior Tree XML file.

        Args:
        record_path: Path to the MMRT YAML motion recording file.
        urdf_path: Optional legacy URDF/XACRO override. The recording metadata is used by default.
        behavior_tree_models_path: Path to extra behavior-tree model definitions.
        behavior_tree_output_path: Path to save the generated Behavior Tree XML file.
        bt_vel_scale: Scaling factor for velocity in motion evaluation.
        bt_acc_scale: Scaling factor for acceleration in motion evaluation.
        mmrt_is_sim: Whether to run a PyBullet simulation of the motion.
        mmrt_is_plot: Whether to plot the motion segmentation results.
        mmrt_is_export_bt: Whether to export the Behavior Tree XML file.
    """
    logger.info("▶ Starting MMRT pipeline...")
    _write_progress(progress_file, 1, "Starting MMRT pipeline")
    logger.info("=== PHASE 1/2: data analysis and simulation ===")
    if not behavior_tree_models_path or not behavior_tree_output_path:
        raise ValueError("Behavior-tree models and output paths are required.")
    if mmrt_show_console: boot_sequence()

    # i) Load & map
    logger.info("→ Loading motion data and mappings...")
    if mmrt_show_console: effect_data_mapping()
    mmrt_data = recording_data.load_recording_data(record_path)
    episodes = mmrt_data.episodes
    normalized_tree_id = (tree_id or "").strip() or MMRTProcessDefaults.TREE_ID
    configure_tree_name = (configure_tree_name or "").strip() or MMRTProcessDefaults.CONFIGURE_TREE_NAME
    scene_setup_tree_name = (scene_setup_tree_name or "").strip() or MMRTProcessDefaults.SCENE_SETUP_TREE_NAME
    move_segment_tree_name = (move_segment_tree_name or "").strip() or MMRTProcessDefaults.MOVE_SEGMENT_TREE_NAME
    main_tree_names = [
        normalized_tree_id,
        configure_tree_name,
        scene_setup_tree_name,
        move_segment_tree_name,
    ]
    if len({name.casefold() for name in main_tree_names}) != len(main_tree_names):
        raise ValueError("Main tree and support subtree names must be unique.")
    reserved_tree_names = {name.casefold() for name in main_tree_names}
    for episode in episodes:
        if episode["name"].casefold() in reserved_tree_names:
            raise ValueError(
                f"Episode name '{episode['name']}' is reserved by the generated behavior tree."
            )
    if episodes:
        logger.info("Loaded episodes: %s", ", ".join(
            f"{episode['name']} [{episode['id']}]={episode['start_frame']}->{episode['end_frame']}"
            for episode in episodes
        ))
    urdf_path = mmrt_data.resolve_bundle_path(urdf_path) if urdf_path else mmrt_data.urdf_path
    srdf_path = mmrt_data.resolve_bundle_path(srdf_path) if srdf_path else mmrt_data.srdf_path
    env_path = mmrt_data.resolve_bundle_path(env_path) if env_path else mmrt_data.environment_path
    robot_name = robot_name or mmrt_data.robot_name
    logger.debug(
        f"Paths:\n  Record={record_path}\n  URDF={urdf_path}\n"
        f"  SRDF={srdf_path}\n  Environment={env_path}"
    )
    if not urdf_path or not os.path.isfile(urdf_path):
        raise FileNotFoundError(
            "The recording bundle URDF is missing. Re-record the scene to export "
            "URDF/SRDF and robot meshes into the recording folder."
        )
    if not srdf_path or not os.path.isfile(srdf_path):
        raise FileNotFoundError(
            "The recording bundle SRDF is missing. Re-record the scene to export "
            "URDF/SRDF and robot meshes into the recording folder."
        )
    resolved_urdf_path = robot_description.resolve_urdf_source(urdf_path)
    urdf_data = robot_description.load_urdf_description(resolved_urdf_path)
    
    collision_links_pair = None
    if srdf_path:
        resolved_srdf_path = robot_description.resolve_srdf_source(srdf_path)
        collision_links_pair = robot_description.load_srdf_collision_pairs(resolved_srdf_path)
        joint_group_definition = recording_data.JointGroupDefinition(
            robot_description.load_srdf_joint_groups(resolved_srdf_path, urdf_data)
        )
    else:
        raise ValueError("SRDF is required to define MMRT joint groups.")
    logger.info("✓ Data loaded successfully.")
    _write_progress(progress_file, 12, "Motion data and robot descriptions loaded")
    
    # Load environment data if provided
    env_data = recording_data.EnvironmentData(env_path) if env_path else None
    if env_data and env_data.get_objects():
        logger.info(f"✓ Loaded environment with {len(env_data.get_objects())} objects. Units: {env_data.get_units()}")

    # ii) Preprocess
    logger.info("→ Preprocessing and smoothing joint data...")
    if mmrt_show_console: effect_signal_processing()
    active_joint_data = {}
    movable_joints = [
        joint['name'] for joint in urdf_data['joints']
        if joint.get('name') and joint.get('type', RobotJointTypes.FIXED) not in RobotJointTypes.NON_MOVING and not joint.get('mimic')
    ]
    
    angle_units = str(mmrt_data.angle_units).upper()
    
    if angle_units not in {MeasurementUnits.RADIAN_UNIT, MeasurementUnits.DEGREE_UNIT}:
        raise ValueError(f"Unsupported recording angle units: {angle_units}")
    joint_types = {
        joint["name"]: joint.get("type", RobotJointTypes.FIXED)
        for joint in urdf_data["joints"]
    }
    
    for joint_name in movable_joints:
        joint_values = mmrt_data.labeled_data.get(joint_name, None)
        if joint_values is None:
            logger.warning(f"Missing recorded URDF joint: {joint_name}")
            continue
        # check for degree joint units and convert to radian
        values = np.asarray(joint_values, dtype=float).copy()
        if (
            joint_types[joint_name] in {RobotJointTypes.REVOLUTE, "continuous"}
            and angle_units == MeasurementUnits.DEGREE_UNIT
        ):
            values = np.deg2rad(values)
        active_joint_data[joint_name] = values

    logger.info(f"✓ {len(active_joint_data)} joints processed.")
    _write_progress(progress_file, 20, "Joint data preprocessed")

    # iii) Segmentation
    # a) Segment per atomic group
    # atomic_segments example: {'arm': [(0, 10), (15, 30)], 'gripper': [(5, 20)]}
    logger.info("→ Segmenting motions by atomic groups...")
    if mmrt_show_console: effect_segmentation()
    atomic_speeds, atomic_segments = {}, {}
    for group, joint_names in joint_group_definition.atomic_groups.items():
        joints_angles = []
        for joint in joint_names:
            if joint not in list(active_joint_data.keys()):
                logger.warning(f"Joint {joint} not found in recorded data for atomic group {group}!")
                continue
            joints_angles.append(active_joint_data[joint])
        if not joints_angles:
            logger.warning(f"No valid joints in atomic group {group}")
            continue
        joint_data_stack = np.stack(joints_angles, axis=1)
        # stacked shape: (num_frames, num_joints_in_group)
        group_velocity = motion_analysis.calculate_joint_velocity(joint_data_stack, mmrt_data.fps)
        group_speed = np.linalg.norm(group_velocity, axis=1)
        group_segments = motion_analysis.find_motion_segments(
            group_speed, 
            threshold=mmrt_speed_thr, 
            min_duration_frames=int(mmrt_min_segment_time * mmrt_data.fps)
        )
        atomic_speeds[group] = group_speed
        atomic_segments[group] = group_segments
    _write_progress(progress_file, 32, "Motion segments detected")

    logger.info("Detected action segments:")
    for group, segments in atomic_segments.items():
        for start, end in segments:
            logger.info(f"  {group}: {start + mmrt_data.frame_start}->{end + mmrt_data.frame_start} ({(end-start)/mmrt_data.fps:.2f}s)")

    # b) Merge atomic segments into single timeline with active atomic groups
    # timeline_segments example: [(0, 5, ['arm']), (5, 10, ['arm', 'gripper'])]
    timeline_segments_with_atomic = motion_analysis.merge_group_segments(atomic_segments)
    logger.info("Merged segments with active move groups:")
    for start, end, groups in timeline_segments_with_atomic:
        logger.info(f"  {start + mmrt_data.frame_start}->{end + mmrt_data.frame_start} ({(end-start)/mmrt_data.fps:.2f}s): {groups}")

    # c1) Combine nearby segments
    # segments_with_movegroup example: [(0, 10, ['arm', 'gripper']), (15, 30, ['gripper'])]
    combined_segments_with_atomic = motion_analysis.merge_nearby_segments(
        timeline_segments_with_atomic,
        threshold_frames=int(mmrt_segment_combine_thr * mmrt_data.fps)
    )
    logger.info("Combined nearby segments:")
    for start, end, groups, _ in combined_segments_with_atomic:
        logger.info(f"  {start + mmrt_data.frame_start}->{end + mmrt_data.frame_start} ({(end-start)/mmrt_data.fps:.2f}s): {groups}")

    
    # c2) Remove flagged ignore segments

    # c3) Split segments at intervening events (e.g., IO events, Forced Splits)
    combined_segments_with_atomic = motion_analysis.split_segments_at_events(combined_segments_with_atomic, mmrt_data)

    # d) Find waypoints within each segment based on local minima in joints speeds
    timeline_segments_with_waypoints = []

    # Special mode: disable waypoints completely
    if mmrt_wp_decimation_iter == -1:
        for start, end, atomics, _existing_wps in combined_segments_with_atomic:
            timeline_segments_with_waypoints.append((start, end, atomics, []))
    else:
        joint_speed_profiles = {}
        for joint, angles in active_joint_data.items():
            # get absolute velocity  for finding minima
            velocity = motion_analysis.calculate_joint_velocity(angles, mmrt_data.fps)
            joint_speed_profiles[joint] = np.abs(velocity)

        for start, end, atomics, existing_wps in combined_segments_with_atomic:
            mgroup_minimums = set()  # local minimums for all movegroups in this segment

            for atomic_group in atomics:
                joint_list = joint_group_definition.get_atomic_group(atomic_group)
                for joint in joint_list:
                    if joint not in joint_speed_profiles:
                        logger.warning(
                            "Skipping joint '%s' in atomic group '%s': no recorded motion profile.",
                            joint, atomic_group,
                        )
                        continue
                    joint_minima = motion_analysis.find_local_speed_minima(
                        joint_speed_profiles[joint],
                        start=start,
                        end=end,
                    )
                    mgroup_minimums.update(joint_minima)

            mgroup_minimums = sorted(mgroup_minimums)
            for _ in range(mmrt_wp_decimation_iter):
                mgroup_minimums = motion_analysis.merge_nearby_indices(
                    mgroup_minimums,
                    window=mmrt_wp_decimation_window,
                    method=mmrt_wp_decimation_method,
                )

            all_wps = sorted(set(existing_wps + mgroup_minimums))
            timeline_segments_with_waypoints.append((start, end, atomics, all_wps))


    # Build atomic waypoints (will be empty if disabled above)
    atomic_waypoints = {}
    for group in joint_group_definition.atomic_groups.keys():
        wp_list = []
        for (start, end, atomics, wps) in timeline_segments_with_waypoints:
            if group in atomics:
                wp_list.extend(wps)
        atomic_waypoints[group] = sorted(set(wp_list))


    # e) Map segments into move groups instead of atomic groups
    # binned_segments example: [(0, 10, 'arm_gripper', [wpnts]), (15, 30, 'arm', [wpnts])]
    final_segments = motion_analysis.map_atomic_segments_to_move_groups(
        timeline_segments_with_waypoints,
        joint_group_definition.move_groups,
    )
    logger.info("Mapped segments to move groups:")
    if mmrt_show_console: effect_waypoint_learning()
    for start, end, mgroup, waypoints in final_segments:
        logger.info(f"  {start + mmrt_data.frame_start}->{end + mmrt_data.frame_start} ({(end-start)/mmrt_data.fps:.2f}s): {mgroup} with waypoints {waypoints}")
    _write_progress(progress_file, 48, "Waypoints and move groups prepared")

    # An action may only be put in an episode subtree when its complete motion
    # range is contained by that episode. Crossing an episode boundary keeps
    # the action in the outer movesegment sequence and raises a clear alert.
    motion_episode_by_frame = {}
    for start, end, _mgroup, waypoints in final_segments:
        absolute_start = start + mmrt_data.frame_start
        absolute_end = end + mmrt_data.frame_start
        action_episode = _episode_for_range(episodes, absolute_start, absolute_end)
        if episodes and action_episode is None and _range_overlaps_episode(
            episodes, absolute_start, absolute_end
        ):
            logger.error(
                "EPISODE ALERT: action %s-%s crosses an episode boundary; it will remain in the outer movesegment sequence.",
                absolute_start, absolute_end,
            )
        if action_episode:
            motion_episode_by_frame[absolute_start] = action_episode
            motion_episode_by_frame[absolute_end] = action_episode
            for waypoint in waypoints:
                motion_episode_by_frame[waypoint + mmrt_data.frame_start] = action_episode

    # iv) Motion evaluation and generate keypoints
    # a) Build a dictionary of flag keypoints FIRST
    logger.info("→ Evaluating flags and IO events...")
    flag_keypoints = []
    for frame in mmrt_data.flags_io.keys():
        flag_keypoints.append(
            recording_data.Keypoint(
                type=recording_data.KeypointType.IO,
                frame=frame
            )
        )
    
    for frame in mmrt_data.flags_led.keys():
        flag_keypoints.append(
            recording_data.Keypoint(
                type=recording_data.KeypointType.LED,
                frame=frame
            )
        )
    
    for frame, codes in mmrt_data.flags_general.items():
        for c in codes:
            flag_keypoints.append(
                recording_data.Keypoint(
                    type=recording_data.KeypointType.FLAG,
                    frame=frame,
                    flag_code=c
                )
        )

    for frame, names in mmrt_data.flags_custom.items():
        for name in names:
            flag_keypoints.append(
                recording_data.Keypoint(
                    type=recording_data.KeypointType.FLAG,
                    frame=frame,
                    flag_code=name,
                )
            )

    _write_progress(progress_file, 54, "Recording events evaluated")

    # b) Simulate and check collisions at each keypoint
    logger.info("→ Simulating and checking collisions...")
    if mmrt_show_console: effect_simulation()
    sim_env = Simulation(gui=mmrt_is_sim)
    sim_env.set_physics_parameters()
    robot_id = sim_env.load_robot(resolved_urdf_path)
    sim_env.disable_adjacent_link_collision(robot_id)
    if collision_links_pair is not None:
        for link1, link2 in collision_links_pair:
            sim_env.disable_collision_between_links(robot_id, link1, link2)
    sim_env.set_collision_filter_threshold(mmrt_collision_thr)

    if env_data:
        logger.info("Loading environment meshes into simulation...")
        environment_objects = env_data.get_objects()
        for object_index, obj in enumerate(environment_objects):
            resolved_mesh = env_data.resolve_mesh_path(obj.mesh_path)
            logger.debug(f"Loading environment object: {obj.name} from {resolved_mesh}")
            sim_env.load_mesh(
                mesh_path=resolved_mesh,
                position=obj.position,
                orientation=obj.rotation_euler,
                scale=obj.scale
            )
            _write_progress(
                progress_file,
                55 + 4 * (object_index + 1) / max(1, len(environment_objects)),
                f"Loading environment object {object_index + 1}/{len(environment_objects)}",
            )

    all_active_joint_ids = sim_env.get_ids_from_joint_names(robot_id, list(active_joint_data.keys()))
    keyframes = []
    collision_frames = []
    collision_frames_by_atomic_group = {
        group: [] for group in joint_group_definition.atomic_groups
    }

    def record_collisions(frame):
        contacts = sim_env.get_collision(robot_id)
        if not contacts:
            return
        logger.warning(f"Collision detected at frame {frame}!")
        collision_frames.append(frame)
        for group in sim_env.get_atomic_groups_for_contacts(
            robot_id, contacts, joint_group_definition.atomic_groups
        ):
            collision_frames_by_atomic_group[group].append(frame)
        sim_env.print_contact_points(robot_id)

    collision_checks_total = sum(
        len(waypoints) + (2 if waypoints else 1)
        for _start, _end, _mgroup, waypoints in final_segments
    )
    collision_checks_done = 0
    _write_progress(progress_file, 60, "Checking motion collisions")

    for start, end, mgroup, waypoints in final_segments:
        # get all joints in the movegroup
        atomic_groups = joint_group_definition.get_move_group(mgroup)
        active_joints = set()
        for atomic in atomic_groups:
            joints_in_atomic = joint_group_definition.get_atomic_group(atomic)
            active_joints.update(joints_in_atomic)
        active_joints = list(active_joints)

        if waypoints:
            start_frame_idx = start + mmrt_data.frame_start
            joint_angles = [active_joint_data[joint][start] for joint in active_joint_data.keys()]
            sim_env.set_joint_positions(robot_id, all_active_joint_ids, joint_angles, True)

            # Collision check
            record_collisions(start_frame_idx)
            collision_checks_done += 1
            _write_progress(
                progress_file,
                60 + 28 * collision_checks_done / max(1, collision_checks_total),
                f"Collision checks {collision_checks_done}/{collision_checks_total}",
            )

            # Create keyframe for the start of segment
            keyframes.append(recording_data.Keypoint(
                type=recording_data.KeypointType.MOTION,
                frame=start_frame_idx,
                movegroup=mgroup,
                joint_positions={joint: active_joint_data[joint][start] for joint in active_joints},
                is_waypoint=True
            ))

            for wp in waypoints:
                frame_idx = wp + mmrt_data.frame_start
                joint_angles = [active_joint_data[joint][wp] for joint in active_joint_data.keys()]
                sim_env.set_joint_positions(robot_id, all_active_joint_ids, joint_angles, True)

                record_collisions(frame_idx)
                collision_checks_done += 1
                _write_progress(
                    progress_file,
                    60 + 28 * collision_checks_done / max(1, collision_checks_total),
                    f"Collision checks {collision_checks_done}/{collision_checks_total}",
                )

                keyframes.append(recording_data.Keypoint(
                    type=recording_data.KeypointType.MOTION,
                    frame=frame_idx,
                    movegroup=mgroup,
                    joint_positions={joint: active_joint_data[joint][wp] for joint in active_joints},
                    is_waypoint=True
                ))

        # Final keypoint at segment end
        joint_angles = [active_joint_data[joint][end] for joint in active_joint_data.keys()]
        sim_env.set_joint_positions(robot_id, all_active_joint_ids, joint_angles, True)
        frame_idx = end + mmrt_data.frame_start

        record_collisions(frame_idx)
        collision_checks_done += 1
        _write_progress(
            progress_file,
            60 + 28 * collision_checks_done / max(1, collision_checks_total),
            f"Collision checks {collision_checks_done}/{collision_checks_total}",
        )

        keyframes.append(recording_data.Keypoint(
            type=recording_data.KeypointType.MOTION,
            frame=frame_idx,
            movegroup=mgroup,
            joint_positions={joint: active_joint_data[joint][end] for joint in active_joints},
            )
        )
    
    # sim_env.disconnect()
    
    if collision_frames:
        collision_frames = sorted(set(collision_frames))
        logger.error(f"❌ Total collision frames: {len(collision_frames)}")
        logger.error(f"Frames: {collision_frames}")
    else:
        logger.info("✓ No collisions detected in all segments.")

    all_keyframes = sorted(keyframes + flag_keypoints, key=lambda k: k.frame)

    # Package all phase-1 outputs before plotting or deployment export.
    analysis = PipelineAnalysis(
        mmrt_data=mmrt_data,
        episodes=episodes,
        env_data=env_data,
        active_joint_data=active_joint_data,
        joint_group_definition=joint_group_definition,
        atomic_speeds=atomic_speeds,
        atomic_segments=atomic_segments,
        atomic_waypoints=atomic_waypoints,
        final_segments=final_segments,
        all_keyframes=all_keyframes,
        motion_episode_by_frame=motion_episode_by_frame,
        collision_frames=collision_frames,
        collision_frames_by_atomic_group=collision_frames_by_atomic_group,
        sim_env=sim_env,
    )
    _write_progress(progress_file, 90, "Collision analysis complete")
            
    # v) Plotting
    if mmrt_is_plot or mmrt_is_save_plot:
        _write_progress(progress_file, 92, "Generating motion plots")
        save_path = None
        if mmrt_is_save_plot:
            out_dir = Path(behavior_tree_output_path).parent
            os.makedirs(out_dir, exist_ok=True)
            save_path = out_dir / (Path(behavior_tree_output_path).stem + "_plot.svg")

        Visualization(
            analysis.atomic_speeds,
            analysis.atomic_segments,
            analysis.final_segments,
            atomic_waypoints=analysis.atomic_waypoints,
            mmrt_data=analysis.mmrt_data,
            episodes=analysis.episodes,
            collision_frames=analysis.collision_frames,
            collision_frames_by_atomic_group=analysis.collision_frames_by_atomic_group,
        ).plot(save_path=str(save_path) if save_path else None, show=mmrt_is_plot)
        _write_progress(progress_file, 95, "Motion plots complete")

    # PHASE 2/2: behavior-tree construction and deployment export
    if mmrt_is_export_bt:
        _write_progress(progress_file, 96, "Exporting behavior tree and blackboards")
        logger.info("=== PHASE 2/2: behavior-tree building and export ===")
        export_result = build_and_export_behavior_tree(
            analysis,
            behavior_tree_models_path=behavior_tree_models_path,
            behavior_tree_output_path=behavior_tree_output_path,
            app_dir=app_dir,
            robot_name=robot_name,
            normalized_tree_id=normalized_tree_id,
            configure_tree_name=configure_tree_name,
            scene_setup_tree_name=scene_setup_tree_name,
            move_segment_tree_name=move_segment_tree_name,
            mmrt_sleep_thr=mmrt_sleep_thr,
            bt_vel_scale=bt_vel_scale,
            bt_acc_scale=bt_acc_scale,
            bt_collision_checks=bt_collision_checks,
            bt_blend_radius=bt_blend_radius,
            bt_save_plan=bt_save_plan,
            bt_plan_pipeline=bt_plan_pipeline,
            bt_plan_id=bt_plan_id,
            bt_enable_log=bt_enable_log,
            bt_enable_log_to_file=bt_enable_log_to_file,
            bt_debug_topic_name=bt_debug_topic_name,
            bt_trajectory_strategy=bt_trajectory_strategy,
            bt_trajectory_resample=bt_trajectory_resample,
            bt_trajectory_time_parameterization=bt_trajectory_time_parameterization,
            bt_trajectory_visualize=bt_trajectory_visualize,
            bt_trajectory_resolution=bt_trajectory_resolution,
            mmrt_show_console=mmrt_show_console,
        )
        logger.debug("BT export result: %s", export_result)
    logger.info("✓ MMRT pipeline complete.")
    _write_progress(progress_file, 100, "MMRT pipeline complete")
    
    # Hold simulation window open if running in GUI mode
    if mmrt_is_sim:
        logger.info("Simulation GUI active. Keeping window open... Close the PyBullet window or press Ctrl+C to exit.")
        try:
            while sim_env.is_connected():
                sim_env.step()
        except KeyboardInterrupt:
            logger.info("Simulation loop interrupted by user.")
        finally:
            sim_env.disconnect()
    else:
        sim_env.disconnect()


def main():
    parser = ArgumentParser(description="Process motion and export Behavior Tree XML")
    parser.add_argument(PipelineCLIOptions.RECORD, required=True, help="Input YAML motion file")
    parser.add_argument(PipelineCLIOptions.PROGRESS_FILE, default=None, help="Optional JSON progress file for the Blender UI")
    parser.add_argument(PipelineCLIOptions.URDF, default=None, help="Legacy URDF override relative to the recording folder; defaults to metadata")
    parser.add_argument(
        PipelineCLIOptions.BEHAVIOR_TREE_MODELS_PATH,
        dest="behavior_tree_models_path", required=True,
        help="Extra behavior-tree model definitions (built-in defaults load automatically)",
    )
    parser.add_argument(PipelineCLIOptions.OUTPUT, required=True, help="Output BT XML file")
    parser.add_argument(PipelineCLIOptions.APP_DIR, default=None, help="Deployable MMRT app directory")
    parser.add_argument(PipelineCLIOptions.TREE_ID, default=MMRTProcessDefaults.TREE_ID, help="Main behavior tree XML ID")
    parser.add_argument(
        PipelineCLIOptions.CONFIGURE_TREE_NAME,
        default=MMRTProcessDefaults.CONFIGURE_TREE_NAME,
        help="BehaviorTree XML ID for the configuration-loading subtree",
    )
    parser.add_argument(
        PipelineCLIOptions.SCENE_SETUP_TREE_NAME,
        default=MMRTProcessDefaults.SCENE_SETUP_TREE_NAME,
        help="BehaviorTree XML ID for the environment setup subtree",
    )
    parser.add_argument(
        PipelineCLIOptions.MOVE_SEGMENT_TREE_NAME,
        default=MMRTProcessDefaults.MOVE_SEGMENT_TREE_NAME,
        help="BehaviorTree XML ID for the motion execution subtree",
    )
    parser.add_argument(PipelineCLIOptions.SRDF, default=None, help="Legacy SRDF override relative to the recording folder; defaults to metadata")
    parser.add_argument(PipelineCLIOptions.ENV, default=None, help="Legacy environment override relative to the recording folder; defaults to metadata")
    parser.add_argument(PipelineCLIOptions.ROBOT_NAME, default=None, help="Legacy robot-name override; defaults to recording metadata")

    # --- Motion Segmentation Parameters ---
    parser.add_argument(PipelineCLIOptions.MMRT_SPEED_THR, type=float, default=MMRTProcessDefaults.MOTION_SPEED_THRESHOLD_RAD_PER_SEC, help="Speed threshold for motion segmentation")
    parser.add_argument(PipelineCLIOptions.MMRT_MIN_SEGMENT_TIME, type=float, default=MMRTProcessDefaults.MINIMUM_SEGMENT_DURATION_SEC, help="Minimum time (s) for a valid segment")
    parser.add_argument(PipelineCLIOptions.MMRT_SEGMENT_COMBINE_THR, type=float, default=MMRTProcessDefaults.SEGMENT_MERGE_GAP_SEC, help="Combine segments shorter than this threshold")

    # --- Behavior Tree Parameters ---
    parser.add_argument(PipelineCLIOptions.MMRT_SLEEP_THR, type=int, default=MMRTProcessDefaults.BEHAVIOR_TREE_SLEEP_GAP_MS, help="Sleep threshold in ms used in BT generation")

    # --- Waypoint Decimation ---
    parser.add_argument(PipelineCLIOptions.MMRT_WP_DECIMATION_WINDOW, type=int, default=MMRTProcessDefaults.WAYPOINT_MERGE_WINDOW, help="Waypoint decimation window size")
    parser.add_argument(PipelineCLIOptions.MMRT_WP_DECIMATION_ITER, type=int, default=MMRTProcessDefaults.WAYPOINT_MERGE_ITERATIONS, help="Number of waypoint decimation iterations")
    parser.add_argument(PipelineCLIOptions.MMRT_WP_DECIMATION_METHOD, choices=WaypointMergeMethod.CHOICES, default=MMRTProcessDefaults.WAYPOINT_MERGE_METHOD, help="Waypoint decimation method")

    # --- Collision Check ---
    parser.add_argument(PipelineCLIOptions.MMRT_COLLISION_THR, type=float, default=MMRTProcessDefaults.COLLISION_DISTANCE_THRESHOLD_M, help="Distance threshold for collision checking")

    # --- Velocity / Acceleration ---
    parser.add_argument(PipelineCLIOptions.BT_VEL_SCALE, type=float, default=recording_data.BlackboardDefaults.VELOCITY_SCALE, help="Velocity scaling factor")
    parser.add_argument(PipelineCLIOptions.BT_ACC_SCALE, type=float, default=recording_data.BlackboardDefaults.ACCELERATION_SCALE, help="Acceleration scaling factor")
    parser.add_argument(PipelineCLIOptions.BT_COLLISION_CHECKS, type=int, default=recording_data.BlackboardDefaults.COLLISION_CHECKS, help="Number of collision checks during motion execution")
    parser.add_argument(PipelineCLIOptions.BT_BLEND_RADIUS, type=float, default=recording_data.BlackboardDefaults.BLEND_RADIUS, help="Blend radius for motion transitions")
    parser.add_argument(
        PipelineCLIOptions.BT_SAVE_PLAN, nargs="?", const=True,
        type=lambda value: value.lower() == "true",
        default=recording_data.BlackboardDefaults.SAVE_PLAN,
        help="Save motion plans to blackboard (true/false; bare flag enables saving)",
    )
    parser.add_argument(PipelineCLIOptions.BT_PLAN_PIPELINE, choices=MMRTPlannerConfig.PIPELINES, default=None)
    parser.add_argument(PipelineCLIOptions.BT_PLAN_ID, default=None)
    parser.add_argument(PipelineCLIOptions.BT_ENABLE_LOG, type=lambda value: value.lower() == "true", default=None)
    parser.add_argument(PipelineCLIOptions.BT_ENABLE_LOG_TO_FILE, type=lambda value: value.lower() == "true", default=None)
    parser.add_argument(PipelineCLIOptions.BT_DEBUG_TOPIC_NAME, default=None)
    parser.add_argument(PipelineCLIOptions.BT_TRAJECTORY_STRATEGY, choices=MoveTomoTrajectoryConfig.STRATEGIES, default=MoveTomoTrajectoryConfig.DEFAULT_STRATEGY)
    parser.add_argument(PipelineCLIOptions.BT_TRAJECTORY_RESAMPLE, choices=MoveTomoTrajectoryConfig.RESAMPLE_METHODS, default=MoveTomoTrajectoryConfig.DEFAULT_RESAMPLE)
    parser.add_argument(PipelineCLIOptions.BT_TRAJECTORY_TIME_PARAMETERIZATION, choices=MoveTomoTrajectoryConfig.TIME_PARAMETERIZATIONS, default=MoveTomoTrajectoryConfig.DEFAULT_TIME_PARAMETERIZATION)
    parser.add_argument(
        PipelineCLIOptions.BT_TRAJECTORY_VISUALIZE,
        type=lambda value: value.lower() == "true",
        default=MoveTomoTrajectoryConfig.DEFAULT_VISUALIZE,
    )

    parser.add_argument(
        PipelineCLIOptions.BT_TRAJECTORY_RESOLUTION, type=float,
        default=MoveTomoTrajectoryConfig.DEFAULT_RESOLUTION,
        help="Trajectory resampling resolution in radians",
    )

    # --- Simulation / Plot / Export ---
    parser.add_argument(PipelineCLIOptions.SIM, action="store_true", help="Run PyBullet simulation")
    parser.add_argument(PipelineCLIOptions.PLOT, action="store_true", help="Plot motion segmentation")
    parser.add_argument(PipelineCLIOptions.SAVE_PLOT, action="store_true", help="Save the plot to the same output directory")
    parser.add_argument(PipelineCLIOptions.NO_EXPORT, action="store_true", help="Disable BT XML export")

    # --- Logging / UI ---
    parser.add_argument(PipelineCLIOptions.VERBOSITY, choices=LoggingVerbosity.CHOICES, default=LoggingVerbosity.DEFAULT, help="Logging verbosity level")
    parser.add_argument(PipelineCLIOptions.MMRT_SHOW_CONSOLE, action="store_true", help="Enable cinematic console visualization")

    args = parser.parse_args()

    run_mmrt_pipeline(
        record_path=args.record,
        urdf_path=args.urdf,
        behavior_tree_models_path=args.behavior_tree_models_path,
        behavior_tree_output_path=args.output,
        srdf_path=args.srdf,
        env_path=args.env,

        mmrt_speed_thr=args.mmrt_speed_thr,
        mmrt_min_segment_time=args.mmrt_min_segment_time,
        mmrt_segment_combine_thr=args.mmrt_segment_combine_thr,
        mmrt_sleep_thr=args.mmrt_sleep_thr,
        mmrt_wp_decimation_window=args.mmrt_wp_decimation_window,
        mmrt_wp_decimation_iter=args.mmrt_wp_decimation_iter,
        mmrt_wp_decimation_method=args.mmrt_wp_decimation_method,
        mmrt_collision_thr=args.mmrt_collision_thr,

        bt_vel_scale=args.bt_vel_scale,
        bt_acc_scale=args.bt_acc_scale,
        bt_collision_checks=args.bt_collision_checks,
        bt_blend_radius=args.bt_blend_radius,
        bt_save_plan=args.bt_save_plan,
        bt_plan_pipeline=args.bt_plan_pipeline,
        bt_plan_id=args.bt_plan_id,
        bt_enable_log=args.bt_enable_log,
        bt_enable_log_to_file=args.bt_enable_log_to_file,
        bt_debug_topic_name=args.bt_debug_topic_name,
        bt_trajectory_strategy=args.bt_trajectory_strategy,
        bt_trajectory_resample=args.bt_trajectory_resample,
        bt_trajectory_time_parameterization=args.bt_trajectory_time_parameterization,
        bt_trajectory_visualize=args.bt_trajectory_visualize,
        bt_trajectory_resolution=args.bt_trajectory_resolution,

        mmrt_is_sim=args.sim,
        mmrt_is_plot=args.plot,
        mmrt_is_save_plot=args.save_plot,
        mmrt_is_export_bt=not args.no_export,
        mmrt_show_console=args.mmrt_show_console,
        app_dir=args.app_dir,
        robot_name=args.robot_name,
        tree_id=args.tree_id,
        configure_tree_name=args.configure_tree_name,
        scene_setup_tree_name=args.scene_setup_tree_name,
        move_segment_tree_name=args.move_segment_tree_name,
        progress_file=args.progress_file,
    )

    sys.exit(0)


if __name__ == "__main__":
    main()
