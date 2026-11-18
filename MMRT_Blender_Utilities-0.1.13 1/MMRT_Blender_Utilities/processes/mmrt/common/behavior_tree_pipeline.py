"""Behavior-tree construction and deployment export phase.

This module intentionally owns phase 2 of the MMRT process. Phase 1 produces
PipelineAnalysis; this module consumes it and returns the generated artifact
paths so the orchestration layer can remain easy to follow.
"""

import logging
import json
import os
import shutil
import yaml
from dataclasses import dataclass, field
from pathlib import Path

import common.motion_analysis as motion_analysis
import common.recording_data as recording_data
from common.behavior_tree import (
    BehaviorTreeBuilder,
    BlackboardManager,
    BEHAVIOR_TREE_WAYPOINT_ATOMIC_GROUP_SUFFIX,
    translate_atomic_group_positions,
)
from common.console_effects import effect_bt_crafting_hybrid
from definitions import EnvironmentSchema, EpisodeConfig, GeneratedAppDirectories, MMRTProcessDefaults, MoveTomoTrajectoryConfig


logger = logging.getLogger(__name__)
MDataLabel = recording_data.MDataLabel


def _episode_for_frame(episodes, frame):
    """Return an episode ID for an inclusive absolute frame."""
    for episode in episodes:
        if episode[EpisodeConfig.START_FRAME_KEY] <= frame <= episode[EpisodeConfig.END_FRAME_KEY]:
            return episode[EpisodeConfig.ID_KEY]
    return None


def _collapse_single_child_sequences(node):
    """Remove redundant Sequence wrappers around exactly one child."""
    if not isinstance(node, dict):
        return node

    children = node.get("children")
    if not isinstance(children, list):
        return node

    node["children"] = [
        _collapse_single_child_sequences(child)
        for child in children
    ]
    if node.get("id") == "Sequence" and len(node["children"]) == 1:
        return node["children"][0]
    return node


def _unbrace_reference(value):
    """Return the key name from a BehaviorTree blackboard reference."""
    if isinstance(value, str) and value.startswith("{") and value.endswith("}"):
        return value[1:-1]
    return value


def _collect_referenced_motion_data(
    node,
    trajectories,
    referenced_trajectories,
    referenced_waypoints,
):
    """Collect motion blackboard entries reachable from emitted actions."""
    if not isinstance(node, dict):
        return

    node_id = node.get("id")
    params = node.get("params", {}) or {}
    if node_id == "MoveTomoWaypoint":
        waypoint_name = _unbrace_reference(params.get("waypoint"))
        if waypoint_name:
            referenced_waypoints.add(waypoint_name)
    elif node_id == "MoveTomoTrajectory":
        trajectory_name = _unbrace_reference(params.get("trajectory"))
        trajectory = trajectories.get(trajectory_name)
        if trajectory:
            referenced_trajectories.add(trajectory_name)
            referenced_waypoints.update(
                trajectory.get(MDataLabel.WAYPOINTS_KEY, [])
            )

    for child in node.get("children", []) or []:
        _collect_referenced_motion_data(
            child,
            trajectories,
            referenced_trajectories,
            referenced_waypoints,
        )


@dataclass
class PipelineAnalysis:
    """Outputs from phase 1: motion analysis, event mapping, and simulation.

    Example values:
        final_segments=[(0, 20, "arm", [0, 10, 20])]
        collision_frames=[12, 37]
        all_keyframes=[Keypoint(...), ...]
    """

    mmrt_data: object
    episodes: list
    env_data: object
    active_joint_data: dict
    joint_group_definition: object
    atomic_speeds: dict
    atomic_segments: dict
    atomic_waypoints: dict
    final_segments: list
    all_keyframes: list
    motion_episode_by_frame: dict
    collision_frames: list
    sim_env: object
    collision_frames_by_atomic_group: dict = field(default_factory=dict)


@dataclass
class BehaviorTreeExportResult:
    """Files produced by the BT export phase.

    Example return value:
        BehaviorTreeExportResult(
            main_tree_path=Path("app/trees/behavior_tree.bt.xml"),
            episode_tree_path=Path("app/trees/episode_subtrees.bt.xml"),
            blackboard_paths=[Path("app/blackboards/motion_waypoints.btbb.yaml"), ...],
            app_config_path=Path("app/config.apcfg.yaml"),
        )
    """

    main_tree_path: Path
    episode_tree_path: Path | None
    blackboard_paths: list
    app_config_path: Path


def build_and_export_behavior_tree(
        analysis: PipelineAnalysis,
        *,
        behavior_tree_models_path: str,
        behavior_tree_output_path: str,
        app_dir: str | None,
        robot_name: str | None,
        normalized_tree_id: str,
        mmrt_sleep_thr: int,
        bt_vel_scale: float,
        bt_acc_scale: float,
        bt_collision_checks: int,
        bt_blend_radius: float,
        bt_save_plan: bool,
        mmrt_show_console: bool,
        bt_plan_pipeline: str = None,
        bt_plan_id: str = None,
        bt_enable_log: bool = None,
        bt_enable_log_to_file: bool = None,
        bt_debug_topic_name: str = None,
        bt_trajectory_strategy: str = MoveTomoTrajectoryConfig.DEFAULT_STRATEGY,
        bt_trajectory_resample: str = MoveTomoTrajectoryConfig.DEFAULT_RESAMPLE,
        bt_trajectory_time_parameterization: str = MoveTomoTrajectoryConfig.DEFAULT_TIME_PARAMETERIZATION,
        bt_trajectory_visualize: bool = MoveTomoTrajectoryConfig.DEFAULT_VISUALIZE,
        bt_trajectory_resolution: float = MoveTomoTrajectoryConfig.DEFAULT_RESOLUTION,
        configure_tree_name: str = MMRTProcessDefaults.CONFIGURE_TREE_NAME,
        scene_setup_tree_name: str = MMRTProcessDefaults.SCENE_SETUP_TREE_NAME,
        move_segment_tree_name: str = MMRTProcessDefaults.MOVE_SEGMENT_TREE_NAME,
):
    """Build and export the deployable behavior-tree application.

    Major inputs:
        analysis.final_segments -> [(start, end, movegroup, waypoint_frames), ...]
        analysis.all_keyframes -> [motion/event keypoints ordered by frame]
        analysis.episodes -> [{"name": ..., "id": ..., "logic": ...}, ...]
    """
    mmrt_data = analysis.mmrt_data
    episodes = analysis.episodes
    env_data = analysis.env_data
    active_joint_data = analysis.active_joint_data
    joint_group_definition = analysis.joint_group_definition
    all_keyframes = analysis.all_keyframes
    motion_episode_by_frame = analysis.motion_episode_by_frame

    # 1) Prepare the deployable application layout.
    # Example values:
    #   app_root=Path("app"), tree_dir=Path("app/trees"), blackboard_dir=Path("app/blackboards")
    logger.info("→ Exporting Behavior Tree and Blackboard...")
    configure_tree_name = (configure_tree_name or "").strip() or MMRTProcessDefaults.CONFIGURE_TREE_NAME
    scene_setup_tree_name = (scene_setup_tree_name or "").strip() or MMRTProcessDefaults.SCENE_SETUP_TREE_NAME
    move_segment_tree_name = (move_segment_tree_name or "").strip() or MMRTProcessDefaults.MOVE_SEGMENT_TREE_NAME
    if mmrt_show_console:
        effect_bt_crafting_hybrid(
            configure_tree_name=configure_tree_name,
            move_segment_tree_name=move_segment_tree_name,
        )

    # The app directory is the deployable unit. Keep generated files in the
    # prescribed relative directories and leave only the app config at root.
    app_root = Path(app_dir).expanduser() if app_dir else Path(behavior_tree_output_path).parent
    tree_dir = app_root / recording_data.GeneratedAppDirectories.TREE_DIR
    blackboard_dir = app_root / recording_data.GeneratedAppDirectories.BLACKBOARD_DIR
    mesh_dir = app_root / recording_data.GeneratedAppDirectories.MESH_DIR
    tree_dir.mkdir(parents=True, exist_ok=True)
    blackboard_dir.mkdir(parents=True, exist_ok=True)
    mesh_dir.mkdir(parents=True, exist_ok=True)

    output_bt = tree_dir / Path(behavior_tree_output_path).name
    episode_output_bt = tree_dir / recording_data.GeneratedAppFilenames.EPISODE_BEHAVIOR_TREE_FILE
    system_blackboard_path = blackboard_dir / recording_data.GeneratedAppFilenames.SYSTEM_BLACKBOARD_FILE
    motion_blackboard_path = blackboard_dir / recording_data.GeneratedAppFilenames.MOTION_BLACKBOARD_FILE
    environment_blackboard_path = blackboard_dir / recording_data.GeneratedAppFilenames.ENVIRONMENT_BLACKBOARD_FILE
    episode_settings_path = blackboard_dir / recording_data.GeneratedAppFilenames.EPISODE_SETTINGS_BLACKBOARD_FILE
    tree_id = normalized_tree_id
    episode_tree_names = [episode[EpisodeConfig.NAME_KEY] for episode in episodes]
    main_tree_names = [
        tree_id,
        configure_tree_name,
        scene_setup_tree_name,
        move_segment_tree_name,
    ]
    all_tree_names = [*main_tree_names, *episode_tree_names]
    if len({name.casefold() for name in all_tree_names}) != len(all_tree_names):
        raise ValueError("Main tree, support subtree, and episode tree names must be unique.")
    builder = BehaviorTreeBuilder(main_tree_name=tree_id)
    builder.load_models(behavior_tree_models_path)
    blackboard_defaults = recording_data.BlackboardDefaults
    # 2) Prepare runtime blackboards and episode parameters.
    # Example values:
    #   episode_settings_available=True
    #   {"pick_repeat_count": 3, "pick_delay_msec": 250}
    system_settings_bb = BlackboardManager(blackboard_defaults.system_settings(
        bt_vel_scale, bt_acc_scale, bt_blend_radius, bt_collision_checks, bt_save_plan,
        bt_plan_pipeline, bt_plan_id, bt_enable_log, bt_enable_log_to_file,
        bt_debug_topic_name, bt_trajectory_strategy, bt_trajectory_resample,
        bt_trajectory_time_parameterization, bt_trajectory_visualize,
        trajectory_resolution=bt_trajectory_resolution,
    ))
    motion_bb = BlackboardManager()
    environment_bb = BlackboardManager()
    episode_settings_bb = BlackboardManager()

    # The decorator type remains in the episode XML. Parameterized values
    # are loaded from a separate blackboard so they can be changed without
    # regenerating the tree definitions.
    episode_settings_available = any(
        episode[EpisodeConfig.LOGIC_KEY] in {
            EpisodeConfig.LOGIC_REPEAT,
            EpisodeConfig.LOGIC_RETRY_UNTIL_SUCCESSFUL,
        } or
        (episode[EpisodeConfig.DELAY_ENABLED_KEY] and episode[EpisodeConfig.DELAY_MSEC_KEY] > 0)
        for episode in episodes
    )
    if episode_settings_available:
        for episode in episodes:
            episode_id = episode[EpisodeConfig.ID_KEY]
            if episode[EpisodeConfig.LOGIC_KEY] == EpisodeConfig.LOGIC_REPEAT:
                episode_settings_bb.add_value_to_blackboard(
                    f"{episode_id}_repeat_count", episode[EpisodeConfig.REPEAT_COUNT_KEY]
                )
            elif episode[EpisodeConfig.LOGIC_KEY] == EpisodeConfig.LOGIC_RETRY_UNTIL_SUCCESSFUL:
                episode_settings_bb.add_value_to_blackboard(
                    f"{episode_id}_retry_count", episode[EpisodeConfig.RETRY_COUNT_KEY]
                )
            if episode[EpisodeConfig.DELAY_ENABLED_KEY] and episode[EpisodeConfig.DELAY_MSEC_KEY] > 0:
                episode_settings_bb.add_value_to_blackboard(
                    f"{episode_id}_delay_msec", episode[EpisodeConfig.DELAY_MSEC_KEY]
                )

    # 3) Build support trees and environment setup.
    # Example: main_seq={"type": "control", "id": "Sequence", "children": [...]}
    main_seq = builder.add_sequence(tree_id)
    builder.add_subtree(main_seq, configure_tree_name)

    deployed_mesh_paths = {}
    environment_settings = []
    if env_data:
        # The recorder declares environment meshes relative to the env file
        # (normally <env-dir>/meshes). Copy the declared environment mesh
        # tree, but leave the recorder's robot archive in the record folder.
        declared_mesh_root = Path(env_data.base_dir) / recording_data.GeneratedAppDirectories.MESH_DIR
        if declared_mesh_root.is_dir():
            for source_mesh in declared_mesh_root.rglob("*"):
                if source_mesh.is_file():
                    relative_mesh = source_mesh.relative_to(declared_mesh_root)
                    if relative_mesh.parts[0].casefold() == GeneratedAppDirectories.ROBOT_MESH_DIR:
                        continue
                    destination_mesh = mesh_dir / relative_mesh
                    destination_mesh.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_mesh, destination_mesh)

        for obj in env_data.get_objects():
            source_mesh = env_data.resolve_mesh_path(obj.mesh_path)
            if not os.path.isfile(source_mesh):
                logger.warning(f"Environment mesh not found, skipping copy: {source_mesh}")
                continue
            mesh_name = Path(obj.mesh_path).name or Path(source_mesh).name
            destination_mesh = mesh_dir / mesh_name
            destination_mesh.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_mesh, destination_mesh)
            deployed_mesh_paths[obj.mesh_path] = mesh_name
            logger.info(f"Copied environment mesh to {destination_mesh}")

        if env_data.get_objects():
            logger.info("Creating SceneSetup subtree with environment objects...")
            builder.create_tree(scene_setup_tree_name)
            setup_seq = builder.add_sequence(scene_setup_tree_name)
            builder.add_SceneReset(setup_seq)

            for obj in env_data.get_objects():
                deployed_mesh = deployed_mesh_paths.get(obj.mesh_path, obj.mesh_path)
                environment_settings.append({
                    "name": obj.name,
                    "mesh": Path(deployed_mesh).stem,
                    "transform": {
                        "position": obj.position,
                        "rotation": obj.quaternion,
                        "scale": obj.scale,
                    },
                })
                builder.add_AddObject(
                    setup_seq,
                    object_name=recording_data.BlackboardDefaults.environment_object_key(
                        len(environment_settings) - 1, "name"
                    ),
                    mesh_name=recording_data.BlackboardDefaults.environment_object_key(
                        len(environment_settings) - 1, "mesh"
                    ),
                    position=recording_data.BlackboardDefaults.environment_object_key(
                        len(environment_settings) - 1, "pose"
                    )
                )
                follows_selected_robot = (
                    not obj.follow_robot or
                    not robot_name or
                    obj.follow_robot == robot_name
                )
                if obj.follow_robot and not follows_selected_robot:
                    logger.warning(
                        "Skipping follow/collision rules for environment object '%s': "
                        "declared robot '%s' does not match '%s'.",
                        obj.name, obj.follow_robot, robot_name,
                    )
                if obj.follow_link and follows_selected_robot:
                    builder.add_AttachObject(
                        setup_seq,
                        object_name=recording_data.BlackboardDefaults.environment_object_key(
                            len(environment_settings) - 1, "name"
                        ),
                        link_name=obj.follow_link,
                    )
                if obj.collision_movegroups and follows_selected_robot:
                    for movegroup in obj.collision_movegroups:
                        builder.add_SetCollisionObjectMovegroup(
                            setup_seq,
                        object_name=recording_data.BlackboardDefaults.environment_object_key(
                                len(environment_settings) - 1, "name"
                            ),
                            movegroup=movegroup,
                            allow_collision=obj.allow_collision,
                        )
                if obj.collision_links and follows_selected_robot:
                    builder.add_SetCollisionObjectLink(
                        setup_seq,
                        object=recording_data.BlackboardDefaults.environment_object_key(
                            len(environment_settings) - 1, "name"
                        ),
                        links=",".join(obj.collision_links),
                        allow=obj.allow_collision,
                    )
                logger.debug(f"Added object '{obj.name}' to SceneSetup with pose: {obj.serialized_pose}")

            for rule in env_data.get_collision_rules():
                rule_type = str(rule.get("type", "")).strip().lower()
                allow_collision = rule.get(EnvironmentSchema.ALLOW_COLLISION, rule.get("allow", False))
                if isinstance(allow_collision, str):
                    allow_collision = allow_collision.strip().lower() in {"1", "true", "yes", "on"}
                else:
                    allow_collision = bool(allow_collision)
                if rule_type in {"movegroups", "movegroup"}:
                    builder.add_SetCollisionMovegroups(
                        setup_seq,
                        movegroup_1=rule.get("movegroup_1", ""),
                        movegroup_2=rule.get("movegroup_2", ""),
                        allow_collision=allow_collision,
                    )
                elif rule_type in {"links", "link"}:
                    builder.add_SetCollisionLinks(
                        setup_seq,
                        links_1=rule.get("links_1", ""),
                        links_2=rule.get("links_2", ""),
                        allow_collision=allow_collision,
                    )

            builder.add_subtree(main_seq, scene_setup_tree_name)

    builder.add_subtree(main_seq, move_segment_tree_name)

    builder.create_tree(configure_tree_name)
    cfg_seq = builder.add_sequence(configure_tree_name)
    blackboard_paths = [
        system_blackboard_path,
        motion_blackboard_path,
        environment_blackboard_path,
    ]
    if episode_settings_available:
        blackboard_paths.append(episode_settings_path)
    for blackboard_path in blackboard_paths:
        builder.add_LoadBlackboard(cfg_seq, file_name=blackboard_path.name.removesuffix(".yaml"))
    builder.add_SetVelAcc(
        cfg_seq,
        velocity=blackboard_defaults.reference(blackboard_defaults.VELOCITY_SCALE_KEY),
        acceleration=blackboard_defaults.reference(blackboard_defaults.ACCELERATION_SCALE_KEY),
    )
    builder.add_SetPlanner(
        cfg_seq,
        pipeline=blackboard_defaults.reference(blackboard_defaults.PLAN_PIPELINE_KEY),
        id=blackboard_defaults.reference(blackboard_defaults.PLAN_ID_KEY),
    )

    # 4) Build the movement tree, episode subtrees, and motion actions.
    # Example: episode_sequences={"pick": <Sequence node>, "place": <Sequence node>}
    builder.create_tree(move_segment_tree_name)
    move_seq = builder.add_sequence(move_segment_tree_name)

    is_traj = False
    wp_list = []
    traj_start_kp = None
    waypoint_hash_map = {}
    trajectory_hash_map = {}
    forced_waypoint_names = set(mmrt_data.flags_wp_names.values())
    exported_traj_count = 0
    previous_frame = all_keyframes[0].frame if all_keyframes else None

    # Episode subtrees receive one SubTree reference each. The motion subtree
    # sequence keeps unassigned actions in timeline order, while its episode
    # references are reordered to match the explicit episode list.
    # and also retains every unassigned action/event directly.
    episode_by_id = {episode[EpisodeConfig.ID_KEY]: episode for episode in episodes}
    episode_sequences = {}
    emitted_episode_refs = set()
    episode_subtree_nodes = {}
    episode_waypoint_hash_maps = {episode[EpisodeConfig.ID_KEY]: {} for episode in episodes}
    waypoint_counts = {}
    episode_trajectory_hash_maps = {episode[EpisodeConfig.ID_KEY]: {} for episode in episodes}
    episode_trajectory_counts = {episode[EpisodeConfig.ID_KEY]: 0 for episode in episodes}
    episode_first_trajectory_seen = {episode[EpisodeConfig.ID_KEY]: False for episode in episodes}

    def create_episode_sequence(episode):
        """Create an episode subtree and wrap its sequence with its logic."""
        tree_name = episode[EpisodeConfig.NAME_KEY]
        builder.create_tree(tree_name)
        parent = tree_name
        if episode[EpisodeConfig.DELAY_ENABLED_KEY] and episode[EpisodeConfig.DELAY_MSEC_KEY] > 0:
            parent = builder.add_decorator(
                parent,
                "Delay",
                {EpisodeConfig.DELAY_MSEC_KEY: f"{{{episode[EpisodeConfig.ID_KEY]}_delay_msec}}"},
            )
        logic = episode[EpisodeConfig.LOGIC_KEY]
        if logic != EpisodeConfig.LOGIC_NONE:
            decorator_id = {
                EpisodeConfig.LOGIC_REPEAT: "Repeat",
                EpisodeConfig.LOGIC_RETRY_UNTIL_SUCCESSFUL: "RetryUntilSuccessful",
                EpisodeConfig.LOGIC_KEEP_RUNNING_UNTIL_FAILURE: "KeepRunningUntilFailure",
            }[logic]
            params = {
                "num_cycles": f"{{{episode[EpisodeConfig.ID_KEY]}_repeat_count}}"
            } if logic == EpisodeConfig.LOGIC_REPEAT else ({
                "num_attempts": f"{{{episode[EpisodeConfig.ID_KEY]}_retry_count}}"
            } if logic == EpisodeConfig.LOGIC_RETRY_UNTIL_SUCCESSFUL else {})
            parent = builder.add_decorator(parent, decorator_id, params)
        return builder.add_sequence(parent)

    for episode in episodes:
        episode_sequences[episode[EpisodeConfig.ID_KEY]] = create_episode_sequence(episode)

    def sequence_for_episode(episode_id):
        if not episode_id:
            return move_seq
        if episode_id not in emitted_episode_refs:
            episode_subtree_nodes[episode_id] = builder.add_subtree(
                move_seq,
                episode_by_id[episode_id][EpisodeConfig.NAME_KEY],
            )
            emitted_episode_refs.add(episode_id)
        return episode_sequences[episode_id]

    def episode_for_motion_keypoint(keypoint):
        return motion_episode_by_frame.get(keypoint.frame)

    def waypoint_name_for(keypoint, episode_id, fallback_name, forced):
        atomic_positions = keypoint.get_atomic_joint_positions(joint_group_definition)
        signature = motion_analysis.make_waypoint_signature(keypoint.movegroup, atomic_positions)
        hash_map = episode_waypoint_hash_maps.get(episode_id, waypoint_hash_map)
        if signature in hash_map and not forced:
            return hash_map[signature], atomic_positions

        if episode_id:
            waypoint_index = waypoint_counts.get(episode_id, 0)
            canonical_name = f"{episode_id}_wp{waypoint_index}"
            waypoint_counts[episode_id] = waypoint_index + 1
        else:
            canonical_name = fallback_name
        hash_map[signature] = canonical_name
        return canonical_name, atomic_positions

    tuning_waypoints = {}

    def track_tuning_waypoint(name, keypoint, atomic_positions):
        """Map an exported waypoint back to its Blender sample frame/joints."""
        joint_keys = {}
        for atomic_group in atomic_positions:
            joint_keys[
                f"{atomic_group}{BEHAVIOR_TREE_WAYPOINT_ATOMIC_GROUP_SUFFIX}"
            ] = [
                joint_name
                for joint_name in joint_group_definition.get_atomic_group(atomic_group)
                if joint_name in keypoint.joint_positions
            ]

        waypoint = tuning_waypoints.setdefault(name, {
            MDataLabel.MOVEGROUP_KEY: keypoint.movegroup,
            MDataLabel.SAMPLE_FRAMES_KEY: [],
            MDataLabel.JOINT_KEYS_KEY: joint_keys,
        })
        if (
            waypoint[MDataLabel.MOVEGROUP_KEY] != keypoint.movegroup
            or waypoint[MDataLabel.JOINT_KEYS_KEY] != joint_keys
        ):
            waypoint[MDataLabel.TRACKING_ERROR_KEY] = (
                "This generated name is shared by keypoints with different movegroups or joint layouts."
            )
        sample_frames = waypoint[MDataLabel.SAMPLE_FRAMES_KEY]
        if keypoint.frame not in sample_frames:
            sample_frames.append(int(keypoint.frame))

    # --------------------------------------------------
    # MAIN LOOP
    # --------------------------------------------------
    active_motion_episode = None
    for idx, kp in enumerate(all_keyframes):

        # ---------- choose where to put nodes ----------
        point_episode = episode_for_motion_keypoint(kp) if kp.type == recording_data.KeypointType.MOTION else _episode_for_frame(episodes, kp.frame)
        target_seq = sequence_for_episode(point_episode)

        # ---------- time gap Sleep ----------
        idle_ms = (kp.frame - previous_frame) / mmrt_data.fps * 1000
        if idle_ms > mmrt_sleep_thr:
            builder.add_Sleep(target_seq, msec=int(idle_ms))

        if kp.type == recording_data.KeypointType.IO:
            builder.add_IOController(target_seq, command="set", data=recording_data.format_io_byte(mmrt_data.flags_io[kp.frame][0]))
            delay_ms = mmrt_data.flags_io[kp.frame][1]
            if delay_ms > 0: builder.add_Sleep(target_seq, msec=delay_ms)
            previous_frame = kp.frame
            continue

        if kp.type == recording_data.KeypointType.LED:
            leds = mmrt_data.flags_led.get(kp.frame)
            if not leds:
                previous_frame = kp.frame
                continue
            kwargs = {}
            for led_id, cfg in leds.items():
                mode, color, value = cfg["mode"].lower(), cfg["color"], cfg["value"]
                if led_id == 1:
                    kwargs.update(center_mode=mode, center_color=color, center_progress=value)
                elif led_id == 2:
                    kwargs.update(outer_mode=mode, outer_color=color, outer_progress=value)
            if kwargs: builder.add_Iheart(target_seq, **kwargs)
            previous_frame = kp.frame
            continue

        if kp.type == recording_data.KeypointType.FLAG:
            previous_frame = kp.frame
            continue

        if kp.type == recording_data.KeypointType.MOTION:
            if kp.is_waypoint and not is_traj:
                active_motion_episode = episode_for_motion_keypoint(kp)
            motion_episode = active_motion_episode or point_episode
            wp_name_flag = mmrt_data.flags_wp_names.get(kp.frame, None)
            wp_name = wp_name_flag if wp_name_flag else f"w{kp.frame}"

            # ----- waypoint in trajectory -----
            if kp.is_waypoint:
                if not is_traj:
                    is_traj = True
                    traj_start_kp = kp
                    wp_list = []

                atomic_positions = kp.get_atomic_joint_positions(joint_group_definition)
                translated_positions = translate_atomic_group_positions(atomic_positions)
                forced = wp_name in forced_waypoint_names
                canonical_wp, _ = waypoint_name_for(kp, motion_episode, wp_name, forced)
                track_tuning_waypoint(canonical_wp, kp, atomic_positions)
                if canonical_wp not in motion_bb.waypoints:
                    motion_bb.add_waypoint(canonical_wp, kp.movegroup, **translated_positions)

                wp_list.append(canonical_wp)
                previous_frame = kp.frame
                continue

            # ----- finalize trajectory -----
            if is_traj:
                atomic_positions = kp.get_atomic_joint_positions(joint_group_definition)
                translated_positions = translate_atomic_group_positions(atomic_positions)
                forced = wp_name in forced_waypoint_names
                canonical_wp, _ = waypoint_name_for(kp, motion_episode, wp_name, forced)
                track_tuning_waypoint(canonical_wp, kp, atomic_positions)
                if canonical_wp not in motion_bb.waypoints:
                    motion_bb.add_waypoint(canonical_wp, kp.movegroup, **translated_positions)

                wp_list.append(canonical_wp)
                traj_hash = motion_analysis.make_trajectory_signature(traj_start_kp.movegroup, wp_list)
                trajectory_hashes = episode_trajectory_hash_maps.get(
                    motion_episode, trajectory_hash_map
                )
                if traj_hash in trajectory_hashes:
                    canonical_traj = trajectory_hashes[traj_hash]
                else:
                    if motion_episode:
                        trajectory_id = episode_trajectory_counts[motion_episode]
                        canonical_traj = f"{motion_episode}_trajectory{trajectory_id}"
                        episode_trajectory_counts[motion_episode] += 1
                    else:
                        canonical_traj = f"t{traj_start_kp.frame}_{kp.frame}"
                    trajectory_hashes[traj_hash] = canonical_traj
                    if motion_bb.add_trajectory(canonical_traj, f"{exported_traj_count}", traj_start_kp.movegroup, wp_list):
                        exported_traj_count += 1

                trajectory_seq = (
                    sequence_for_episode(motion_episode)
                    if motion_episode
                    else target_seq
                )
                episode_start_with_waypoint = (
                    motion_episode
                    and not episode_first_trajectory_seen[motion_episode]
                    and episode_by_id[motion_episode].get(EpisodeConfig.START_WITH_WAYPOINT_KEY, False)
                )
                start_waypoint_mode = (
                    episode_by_id[motion_episode].get(
                        EpisodeConfig.START_WAYPOINT_MODE_KEY, EpisodeConfig.DEFAULT_START_WAYPOINT_MODE
                    )
                    if episode_start_with_waypoint
                    else EpisodeConfig.DEFAULT_START_WAYPOINT_MODE
                )
                if episode_start_with_waypoint:
                    waypoint_name = (
                        wp_list[-1]
                        if start_waypoint_mode == EpisodeConfig.START_WAYPOINT_LAST
                        else wp_list[0]
                    )
                    builder.add_MoveTomoWaypoint(
                        trajectory_seq,
                        waypoint_name=waypoint_name,
                        waypoint=f"{{{waypoint_name}}}",
                        plan_pipeline=blackboard_defaults.reference(
                            blackboard_defaults.PLAN_PIPELINE_KEY
                        ),
                        plan_id=blackboard_defaults.reference(
                            blackboard_defaults.PLAN_ID_KEY
                        ),
                        save_plan=blackboard_defaults.reference(
                            blackboard_defaults.SAVE_PLAN_KEY
                        ),
                    )
                if not (
                    episode_start_with_waypoint
                    and start_waypoint_mode == EpisodeConfig.START_WAYPOINT_LAST
                ):
                    builder.add_MoveTomoTrajectory(
                        trajectory_seq,
                        trajectory_name=canonical_traj,
                        trajectory=f"{{{canonical_traj}}}",
                        resolution=blackboard_defaults.reference(
                            blackboard_defaults.TRAJECTORY_RESOLUTION_KEY
                        ),
                        strategy=blackboard_defaults.reference(
                            blackboard_defaults.TRAJECTORY_STRATEGY_KEY
                        ),
                        resample=blackboard_defaults.reference(
                            blackboard_defaults.TRAJECTORY_RESAMPLE_KEY
                        ),
                        time_parameterization=blackboard_defaults.reference(
                            blackboard_defaults.TRAJECTORY_TIME_PARAMETERIZATION_KEY
                        ),
                        visualize=blackboard_defaults.reference(
                            blackboard_defaults.TRAJECTORY_VISUALIZE_KEY
                        ),
                        velocity_scale=blackboard_defaults.reference(
                            blackboard_defaults.VELOCITY_SCALE_KEY
                        ),
                        acceleration_scale=blackboard_defaults.reference(
                            blackboard_defaults.ACCELERATION_SCALE_KEY
                        ),
                        blend_radius=blackboard_defaults.reference(
                            blackboard_defaults.BLEND_RADIUS_KEY
                        ),
                        collision_checks=blackboard_defaults.reference(
                            blackboard_defaults.COLLISION_CHECKS_KEY
                        ),
                        save_plan=blackboard_defaults.reference(
                            blackboard_defaults.SAVE_PLAN_KEY
                        ),
                    )
                if motion_episode:
                    episode_first_trajectory_seen[motion_episode] = True
                is_traj = False
                active_motion_episode = None
                wp_list = []
                previous_frame = kp.frame
                continue

            # ----- single waypoint -----
            atomic_positions = kp.get_atomic_joint_positions(joint_group_definition)
            translated_positions = translate_atomic_group_positions(atomic_positions)
            forced = wp_name in forced_waypoint_names
            canonical_wp, _ = waypoint_name_for(kp, motion_episode, wp_name, forced)
            track_tuning_waypoint(canonical_wp, kp, atomic_positions)
            if canonical_wp not in motion_bb.waypoints:
                motion_bb.add_waypoint(canonical_wp, kp.movegroup, **translated_positions)

            builder.add_MoveTomoWaypoint(
                target_seq, waypoint_name=canonical_wp, waypoint=f"{{{canonical_wp}}}",
                plan_pipeline=blackboard_defaults.reference(blackboard_defaults.PLAN_PIPELINE_KEY),
                plan_id=blackboard_defaults.reference(blackboard_defaults.PLAN_ID_KEY),
                save_plan=blackboard_defaults.reference(blackboard_defaults.SAVE_PLAN_KEY),
            )
            previous_frame = kp.frame

    # The timeline still determines where unassigned motion/events are placed,
    # but episode subtree references follow the explicit episode list order.
    ordered_episode_nodes = [
        episode_subtree_nodes[episode[EpisodeConfig.ID_KEY]]
        for episode in episodes
        if episode[EpisodeConfig.ID_KEY] in episode_subtree_nodes
    ]
    if ordered_episode_nodes:
        episode_node_ids = {id(node) for node in ordered_episode_nodes}
        ordered_nodes = iter(ordered_episode_nodes)
        move_seq["children"] = [
            next(ordered_nodes) if id(node) in episode_node_ids else node
            for node in move_seq["children"]
        ]

    # A subtree with one generated action does not need a Sequence wrapper.
    for tree_name, tree in builder.trees.items():
        builder.trees[tree_name] = _collapse_single_child_sequences(tree)

    referenced_trajectories = set()
    referenced_waypoints = set()
    for tree in builder.trees.values():
        _collect_referenced_motion_data(
            tree,
            motion_bb.trajectories,
            referenced_trajectories,
            referenced_waypoints,
        )
    motion_bb.trajectories = {
        name: trajectory
        for name, trajectory in motion_bb.trajectories.items()
        if name in referenced_trajectories
    }
    motion_bb.waypoints = {
        name: waypoint
        for name, waypoint in motion_bb.waypoints.items()
        if name in referenced_waypoints
    }
    tuning_waypoints = {
        name: waypoint
        for name, waypoint in tuning_waypoints.items()
        if name in referenced_waypoints
    }

    # 5) Export separate XML files and blackboards.
    # Example outputs:
    #   main_tree_names=[tree_id, configure_tree_name, move_segment_tree_name]
    #   episode_tree_names=["Pick", "Place"]
    #   blackboard_paths=[Path("app/blackboards/system_settings.btbb.yaml"), ...]
    # --------------------------------------------------
    episode_tree_names = [episode[EpisodeConfig.NAME_KEY] for episode in episodes]
    episode_tree_name_set = set(episode_tree_names)
    main_tree_names = [
        name for name in builder.trees if name not in episode_tree_name_set
    ]
    builder.export_bt(
        file_path=str(output_bt),
        main_tree_name=tree_id,
        tree_names=main_tree_names,
    )
    if episode_tree_names:
        builder.export_bt(
            file_path=str(episode_output_bt),
            tree_names=episode_tree_names,
        )
    elif episode_output_bt.is_file():
        # Do not leave episode definitions from an earlier export in a
        # reused app directory when the current recording has no episodes.
        episode_output_bt.unlink()
    system_settings_bb.export_blackboard(
        file_path=str(system_blackboard_path),
    )
    motion_bb.export_blackboard(
        file_path=str(motion_blackboard_path),
        keys=[],
        include_waypoints=True,
        include_trajectories=True,
        waypoint_comments={
            name: min(waypoint[MDataLabel.SAMPLE_FRAMES_KEY])
            for name, waypoint in tuning_waypoints.items()
            if waypoint.get(MDataLabel.SAMPLE_FRAMES_KEY)
        },
        required_keys=[],
    )
    frame_tracking = dict(getattr(mmrt_data, "addon_frame_tracking", {}) or {})
    frame_tracking.setdefault(MDataLabel.SCHEMA_VERSION_KEY, MDataLabel.SCHEMA_VERSION)
    frame_tracking.setdefault(
        MDataLabel.SCENE_FRAME_START_KEY,
        mmrt_data.source_frame_start,
    )
    frame_tracking.setdefault(
        MDataLabel.SCENE_FRAME_END_KEY,
        mmrt_data.source_frame_end,
    )
    frame_tracking.setdefault(MDataLabel.SAMPLE_FRAME_START_KEY, mmrt_data.frame_start)
    frame_tracking.setdefault(MDataLabel.SAMPLE_FRAME_END_KEY, mmrt_data.frame_end)
    frame_tracking.setdefault(MDataLabel.FRAME_MAP_OLD_KEY, mmrt_data.frame_map_old)
    frame_tracking.setdefault(MDataLabel.FRAME_MAP_NEW_KEY, mmrt_data.frame_map_new)
    frame_tracking.setdefault(MDataLabel.SAMPLE_FRAME_DOMAIN_KEY, MDataLabel.SAMPLE_FRAME_DOMAIN_SCENE_FRAME_SET)
    if (
        int(frame_tracking[MDataLabel.FRAME_MAP_OLD_KEY])
        == int(frame_tracking[MDataLabel.FRAME_MAP_NEW_KEY])
        and frame_tracking[MDataLabel.SCENE_FRAME_START_KEY] is None
        and frame_tracking[MDataLabel.SCENE_FRAME_END_KEY] is None
    ):
        # Older recordings without source bounds are still unambiguous when
        # they did not apply a frame-map stretch.
        frame_tracking[MDataLabel.SCENE_FRAME_START_KEY] = mmrt_data.frame_start
        frame_tracking[MDataLabel.SCENE_FRAME_END_KEY] = mmrt_data.frame_end
    tuning_metadata = {
        MDataLabel.SCHEMA_VERSION_KEY: MDataLabel.SCHEMA_VERSION,
        MDataLabel.RECORDING_KEY: {
            **frame_tracking,
            MDataLabel.FPS_KEY: mmrt_data.fps,
            MDataLabel.ANGLE_UNITS_KEY: recording_data.MeasurementUnits.RADIAN_UNIT,
            MDataLabel.ROBOT_NAME_KEY: mmrt_data.robot_name,
        },
        MDataLabel.WAYPOINTS_KEY: tuning_waypoints,
        MDataLabel.TRAJECTORIES_KEY: {
            name: {
                MDataLabel.MOVEGROUP_KEY: trajectory.get(MDataLabel.PLANGROUP_KEY),
                MDataLabel.WAYPOINTS_KEY: trajectory.get(MDataLabel.WAYPOINTS_KEY, []),
            }
            for name, trajectory in motion_bb.trajectories.items()
        },
    }
    tuning_metadata_path = app_root / recording_data.GeneratedAppFilenames.ADDON_TUNING_METADATA_FILE
    tuning_metadata_tmp_path = tuning_metadata_path.with_suffix(
        tuning_metadata_path.suffix + ".tmp"
    )
    with open(tuning_metadata_tmp_path, "w", encoding="utf-8") as metadata_file:
        json.dump(tuning_metadata, metadata_file, indent=2, sort_keys=True)
        metadata_file.write("\n")
    os.replace(tuning_metadata_tmp_path, tuning_metadata_path)
    environment_bb.add_values(**recording_data.BlackboardDefaults.environment_object_values(environment_settings))
    environment_bb.export_blackboard(
        file_path=str(environment_blackboard_path),
        required_keys=[],
    )
    if episode_settings_available:
        episode_settings_bb.export_blackboard(
            file_path=str(episode_settings_path),
            required_keys=[],
        )
    elif episode_settings_path.is_file():
        # Avoid leaving parameter values from an earlier export in a
        # reused app directory when the current recording has none.
        episode_settings_path.unlink()
    app_config = recording_data.GeneratedAppConfig.create(
        tree_id,
        builder.packages,
        bt_vel_scale,
        bt_acc_scale,
        plan_pipeline=bt_plan_pipeline,
        plan_id=bt_plan_id,
        tree_files=[
            output_bt.name,
            *([episode_output_bt.name] if episode_tree_names else []),
        ],
        blackboard_files=[
            system_blackboard_path.name,
            motion_blackboard_path.name,
            environment_blackboard_path.name,
            *([episode_settings_path.name] if episode_settings_available else []),
        ],
    )
    app_config_path = app_root / (recording_data.GeneratedAppFilenames.APP_CONFIG_FILE)
    with open(app_config_path, "w", encoding="utf-8") as config_file:
        yaml.safe_dump(app_config, config_file, sort_keys=False)
    logger.info(f"✓ Main behavior tree written to {output_bt}")
    if episode_tree_names:
        logger.info(f"✓ Episode subtrees written to {episode_output_bt}")
    logger.info(f"✓ Blackboards written to {blackboard_dir}")
    logger.info(f"✓ App config written to {app_config_path}")

    return BehaviorTreeExportResult(
        main_tree_path=output_bt,
        episode_tree_path=episode_output_bt if episode_tree_names else None,
        blackboard_paths=blackboard_paths,
        app_config_path=app_config_path,
    )


