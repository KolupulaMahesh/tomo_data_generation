import bpy
from ..common.blender import OPS, draw_collapsible_box, get_bl_idname
from ..common.config import (
    EnvironmentExportMode,
    ObjectOriginMode,
    RecordTransformSpace,
    RecorderOperatorOptions,
    SceneRecognitionConfig,
    UIConfig,
)
from ..common.marker_utils import FLAG_ICONS, FlagTypes, HANDLER_REGISTRY
from ..processes.mmrt.definitions import EpisodeConfig
from ..common.recording.frame_domain import (
    EpisodeRangeStatus,
    episode_range_status,
    get_frame_domain,
    remap_frame,
)


def _stretched_frame(scene, frame):
    domain = get_frame_domain(scene)
    return remap_frame(
        frame,
        stretch_factor=domain["stretch_factor"],
    )

class MMRT_UL_ObjectList(bpy.types.UIList):
    """UI list for recorded scene objects."""

    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        obj = item.obj
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            row = layout.row(align=True)

            if obj:
                if obj.override_library:
                    icon_id = 'LIBRARY_DATA_OVERRIDE'
                elif obj.library:
                    icon_id = 'LINKED'
                else:
                    icon_id = 'OBJECT_DATA'

                row.label(text=obj.name, icon=icon_id)

            else:
                row.label(text="(None)", icon='ERROR')

        elif self.layout_type == 'GRID':
            layout.alignment = 'CENTER'
            if obj and obj.library:
                layout.label(text="", icon='LINKED')
            else:
                layout.label(text="", icon='OBJECT_DATA')


class MMRT_UL_MarkerList(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)

        row.prop(item, "marker_name", text="", icon='MARKER')
        row.label(text=f"{_stretched_frame(context.scene, item.frame):4d}")

        icons = row.row(align=True)
        for event in item.events:
            icons.label(text="", icon=FLAG_ICONS.get(event.flag_type, "BOOKMARKS"))


class MMRT_UL_MarkerEventList(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        row.label(
            text=(
                item.custom_flag_name
                if item.flag_type == FlagTypes.CUSTOM.name
                else item.flag_type.replace("_", " ").title()
            ),
            icon=FLAG_ICONS.get(item.flag_type, "BOOKMARKS"),
        )


class MMRT_UL_EpisodeList(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        row.prop(item, "batch_selected", text="")
        row.prop(item, "episode_name", text="")
        row.label(text=f"[{item.episode_id}]")
        row.label(
            text=(
                f"{_stretched_frame(context.scene, item.start_frame)}:"
                f"{_stretched_frame(context.scene, item.end_frame)}"
            )
        )

# === Recorder Panel ===
class MMRT_PT_RecorderPanel(bpy.types.Panel):
    bl_idname = "MMRT_PT_RecorderPanel"
    bl_label = UIConfig.PANEL_RECORDER_LABEL
    bl_category = UIConfig.ADDON_CATEGORY
    bl_space_type = UIConfig.ADDON_SPACE
    bl_region_type = UIConfig.ADDON_REGION
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        rec = scene.mmrt_recorder

        draw_collapsible_box(
            layout,
            scene,
            rec,
            "show_scene_content",
            self.draw_scene_content,
            label="Scene Contents & Discovery",
        )
        draw_collapsible_box(
            layout,
            scene,
            rec,
            "show_recording_tools",
            self.draw_recording_tools,
            label="Recording Timeline & Tools",
        )
        draw_collapsible_box(
            layout,
            scene,
            rec,
            "show_advanced",
            self.draw_recording_input_output,
            label="Recording Input / Output",
        )

        # Keep recording as the final action in the panel.
        layout.separator()
        if rec.is_running:
            layout.operator(OPS.idname("stop_recording"), icon='CANCEL')
        else:
            layout.operator(OPS.idname("start_recording"), icon='PLAY')

    def draw_scene_content(self, box, scene):
        rec = scene.mmrt_recorder
        box.operator(OPS.idname("auto_identify_scene"), icon='VIEWZOOM')
        draw_collapsible_box(box, scene, rec, "show_robot_box", self.draw_robot_section, label="Robots")
        draw_collapsible_box(box, scene, rec, "show_ul_box", self.draw_object_section, label="Active Objects")
        draw_collapsible_box(box, scene, rec, "show_env_box", self.draw_env_section, label=UIConfig.REC_ENV_LIST_LABEL)

    def draw_recording_tools(self, box, scene):
        self.draw_frame_domain(box, scene)
        rec = scene.mmrt_recorder
        draw_collapsible_box(box, scene, rec, "show_episode_box", self.draw_episode_section, label="Episodes")
        draw_collapsible_box(box, scene, rec, "show_marker_box", self.draw_marker_section, label=UIConfig.REC_MARKER_LIST_LABEL)
        draw_collapsible_box(
            box,
            scene,
            rec,
            "show_transform_box",
            self.draw_transform_section,
            label="Diagnostics & Transform Tools",
        )

    @staticmethod
    def draw_frame_domain(layout, scene):
        domain = get_frame_domain(scene)
        box = layout.box()
        box.alert = domain["map_old"] != domain["map_new"]
        box.label(text="Recorder Time Stretch", icon='TIME')

        row = box.row(align=True)
        row.prop(scene.render, "frame_map_old", text="Old")
        row.prop(scene.render, "frame_map_new", text="New")

        row = box.row()
        row.label(
            text=(
                f"Original: {domain['source_start']}–{domain['source_end']}"
                f"  •  Recorder: {domain['stretched_start']}–{domain['stretched_end']}"
            )
        )
        box.label(
            text=f"Scale: {domain['stretch_factor']:.4g}x (recorder domain)",
            icon='INFO',
        )
    

    def draw_recording_input_output(self, box, scene):
        rec = scene.mmrt_recorder
        cfg = rec.config

        box.use_property_split = True
        box.use_property_decorate = False 
        
        box.prop(cfg, "env_export_mode", text="Environment Mode")
        if cfg.env_export_mode == EnvironmentExportMode.MANUAL:
            box.label(text="Using existing environment file", icon='INFO')
        elif cfg.env_export_mode == EnvironmentExportMode.AUTO:
            box.label(text="Environment will be exported on record", icon='FILE_REFRESH')

        box.prop(rec, "is_fast_mode", text="Fast Mode (No Screen Updates)")
        box.prop(cfg, "export_binary", text="Export as Binary (.bin)")
        box.prop(cfg, "include_hidden", text="Include Hidden Objects")
        box.prop(cfg, "rotation_format", text="Rotation Format")
        box.prop(cfg, "angle_unit", text="Angle Units")
        box.prop(cfg, "length_unit", text="Location Units")
        box.prop(cfg, "output_dir")
        box.prop(cfg, "base_filename")


    @staticmethod
    def draw_transform_section(box, scene):
        box.operator(
            OPS.idname("print_current_frame"),
            text="Print Current Frame Data",
            icon='COPYDOWN',
        )
        box.operator(
            OPS.idname("print_relative_transforms"),
            text="Print Relative Transforms",
            icon='ORIENTATION_GLOBAL',
        )
        box.operator(
            OPS.idname("print_relative_transform_array"),
            text="Print Flat Transform Arrays",
            icon='COPYDOWN',
        )

    
    def draw_object_section(self, box, scene):
        # === Objects UIList ===
        # box = layout.box()
        rec = scene.mmrt_recorder
        row = box.row()

        row.template_list(
            "MMRT_UL_ObjectList", "",
            rec, "objs",
            rec, "objs_index",
            rows=3
        )

        # + / - buttons
        col = row.column(align=True)
        col.operator(OPS.idname("add_object"), icon='ADD', text="")
        op_remove = col.operator(OPS.idname("remove_object"), icon='REMOVE', text="")
        op_remove.index = rec.objs_index
        col.separator()
        col.operator(OPS.idname("clear_objects"), icon='TRASH', text="")

        op = row.operator(
            OPS.idname("set_all_transform_space"),
            text="Set All Transform",
            icon='ORIENTATION_GLOBAL'
        )
        op.target = RecorderOperatorOptions.TRANSFORM_RECORD

        # === Active Object Settings ===
        if 0 <= rec.objs_index < len(rec.objs):
            item = rec.objs[rec.objs_index]
            self.draw_object_slot(box, item, rec.objs_index)


    # === Object Slot (details below list) ===
    def draw_object_slot(self, layout, item, idx):
        box = layout.box()
        row = box.row(align=True)

        icon = 'TRIA_DOWN' if item.show_joints else 'TRIA_RIGHT'
        row.prop(item, "show_joints", text="", icon=icon, emboss=False)
        row.prop(item, "obj", text="Object")

        # Remove button per-item
        op_rm = row.operator(OPS.idname("remove_object"), text="", icon='X')
        op_rm.index = idx

        if not item.show_joints:
            return

        # === ACTIVE ARMATURE SETTINGS ===
        if item.obj and item.obj.type == SceneRecognitionConfig.ROBOT_TYPE:
            if item.joints:
                # Keep the header and each bone row in one shared grid so the
                # compact checkboxes stay under their Location/Rotation
                # headings regardless of the bone name lengths.
                grid = box.grid_flow(
                    row_major=True,
                    columns=4,
                    even_columns=True,
                    align=True,
                )
                grid.label(text="", icon='BLANK1')
                grid.label(text="Bone Name", icon='BLANK1')
                grid.label(text="Location", icon='BLANK1')
                grid.label(text="Rotation", icon='BLANK1')

                for bone_item in item.joints:
                    grid.prop(bone_item, "record", text="")
                    grid.label(text=bone_item.name)
                    grid.prop(bone_item, "record_location", text="")
                    grid.prop(bone_item, "record_rotation", text="")

        # === TRANSFORM TRACKING ===
        row3 = box.row(align=True)
        row3.prop(item, "space", text="Space")
        if item.space == RecordTransformSpace.CUSTOM:
            row3.prop(item, "custom_ref", text="")

        row4 = box.row(align=True)
        row4.prop(item, "record_location", text="Location")
        row4.prop(item, "record_rotation", text="Rotation")


    def draw_robot_section(self, box, scene):
        rec = scene.mmrt_recorder
        row = box.row()
        row.template_list("MMRT_UL_ObjectList", "robots", rec, "robots", rec, "robots_index", rows=3)
        col = row.column(align=True)
        col.operator(OPS.idname("add_robot"), icon='ADD', text="")
        op_remove = col.operator(OPS.idname("remove_robot"), icon='REMOVE', text="")
        op_remove.index = rec.robots_index
        col.separator()
        col.operator(OPS.idname("clear_robots"), icon='TRASH', text="")

        if 0 <= rec.robots_index < len(rec.robots):
            self.draw_robot_slot(box, rec.robots[rec.robots_index], rec.robots_index)


    def draw_robot_slot(self, layout, item, idx):
        box = layout.box()
        row = box.row(align=True)
        icon = 'TRIA_DOWN' if item.show_joints else 'TRIA_RIGHT'
        row.prop(item, "show_joints", text="", icon=icon, emboss=False)
        row.prop(item, "obj", text="Robot Armature")
        op_rm = row.operator(OPS.idname("remove_robot"), text="", icon='X')
        op_rm.index = idx
        if not item.show_joints:
            return

        box.prop(item, "urdf_text", text="URDF")
        box.prop(item, "srdf_text", text="SRDF")
        box.prop(item, "meshes_collection", text="Original Meshes")
        row_ops = box.row(align=True)
        op_ref = row_ops.operator(OPS.idname("refresh_robot_joints"), text="Refresh Joints", icon='FILE_REFRESH')
        op_ref.robot_index = idx
        op_ad = row_ops.operator(OPS.idname("auto_detect_robot_axes"), text="Auto-Detect Axes", icon='AXIS_SIDE')
        op_ad.robot_index = idx

        header = box.row(align=True)
        header.label(text="Record")
        header.label(text="URDF Joint")
        header.label(text="Bone")
        header.label(text="Axis")
        header.label(text="Inv")
        header.label(text="Offset")
        for joint in item.joints:
            row_j = box.row(align=True)
            row_j.prop(joint, "record", text="")
            row_j.label(text=joint.joint_name)
            if item.obj and item.obj.type == SceneRecognitionConfig.ROBOT_TYPE:
                row_j.prop_search(joint, "bone_name", item.obj.data, "bones", text="")
            else:
                row_j.prop(joint, "bone_name", text="")
            row_j.prop(joint, "rot_axis", text="")
            row_j.prop(joint, "direction", text="")
            row_j.prop(joint, "offset", text="")


    def draw_env_section(self, box, scene):
        rec = scene.mmrt_recorder
        box.operator(OPS.idname("export_environment"), icon='EXPORT')
        row = box.row()
        row.template_list(
            "MMRT_UL_ObjectList", "",
            rec, "env_objs",
            rec, "env_objs_index",
            rows=3
        )

        col = row.column(align=True)
        col.operator(OPS.idname("add_env_object"), icon='ADD', text="")

        op_remove = col.operator(OPS.idname("remove_env_object"), icon='REMOVE', text="")
        op_remove.index = rec.env_objs_index

        col.separator()
        col.operator(OPS.idname("clear_env_objects"), icon='TRASH', text="")

        op = row.operator(
            OPS.idname("set_all_transform_space"),
            text="Set All Transform",
            icon='ORIENTATION_GLOBAL'
        )
        op.target = RecorderOperatorOptions.TRANSFORM_ENV

        if 0 <= rec.env_objs_index < len(rec.env_objs):
            item = rec.env_objs[rec.env_objs_index]
            self.draw_env_object_slot(box, item, rec.env_objs_index)
    

    def draw_env_object_slot(self, layout, item, idx):
        box = layout.box()
        row = box.row(align=True)
        icon = 'TRIA_DOWN' if item.show_settings else 'TRIA_RIGHT'
        row.prop(item, "show_settings", text="", icon=icon, emboss=False)
        row.prop(item, "obj", text="Env Object")
        if item.obj and item.obj.library:
            row.label(text="", icon='LINKED')
        op_rm = row.operator(OPS.idname("remove_env_object"), text="", icon='X')
        op_rm.index = idx

        if not item.show_settings:
            return

        row2 = box.row(align=True)
        row2.prop(item, "origin_mode", text="Origin")
        if item.origin_mode == ObjectOriginMode.CUSTOM:
            row2.prop(item, "custom_ref", text="")
        follow_row = box.row(align=True)
        follow_row.prop(item, "follow_robot", text="Follow Robot")
        if item.follow_robot:
            follow_row.prop(item, "follow_link", text="Link")
        collision_row = box.row(align=True)
        if item.follow_robot:
            collision_row.prop(item, "collision_link_choice", text="Link")
            add_link = collision_row.operator(OPS.idname("add_env_collision_link"), text="", icon='ADD')
            add_link.env_index = idx
            collision_row.prop(item, "collision_movegroup_choice", text="Movegroup")
            add_group = collision_row.operator(OPS.idname("add_env_collision_movegroup"), text="", icon='ADD')
            add_group.env_index = idx
        else:
            collision_row.label(text="Select a follow robot for link/movegroup rules", icon='INFO')

        if item.follow_robot:
            if item.collision_links:
                box.label(text="Collision Links")
                for rule_index, rule in enumerate(item.collision_links):
                    rule_row = box.row(align=True)
                    rule_row.label(text=rule.name, icon='BONE_DATA')
                    remove_link = rule_row.operator(OPS.idname("remove_env_collision_link"), text="", icon='X')
                    remove_link.env_index = idx
                    remove_link.rule_index = rule_index
            if item.collision_movegroups:
                box.label(text="Collision Movegroups")
                for rule_index, rule in enumerate(item.collision_movegroups):
                    rule_row = box.row(align=True)
                    rule_row.label(text=rule.name, icon='GROUP_BONE')
                    remove_group = rule_row.operator(OPS.idname("remove_env_collision_movegroup"), text="", icon='X')
                    remove_group.env_index = idx
                    remove_group.rule_index = rule_index
        box.prop(item, "allow_collision")
        row3 = box.row(align=True)
        row3.prop(item, "apply_modifiers", text="Apply Modifiers")
        row3.prop(item, "include_for_export", text="Export")


    def draw_marker_section(self, box, scene):
        row1 = box.row()
        row1.operator(OPS.idname("refresh_marker_list"), icon='FILE_REFRESH', text="Refresh Marker List")
        row1.operator(OPS.idname("add_flag_popup"), icon='MARKER', text="Add Marker")
        row2 = box.row()
        row2.template_list(
            "MMRT_UL_MarkerList",
            "mmrt_marker_list",
            scene,
            "mmrt_markers",
            scene,
            "mmrt_markers_index",
            rows=3
        )

        # Right panel buttons
        col = row2.column(align=True)
        if scene.mmrt_markers and 0 <= scene.mmrt_markers_index < len(scene.mmrt_markers):
            active = scene.mmrt_markers[scene.mmrt_markers_index]

            opJ = col.operator(OPS.idname("marker_jump"), text="", icon='VIEWZOOM')
            opJ.frame = active.frame

            opD = col.operator(OPS.idname("marker_delete"), text="", icon='X')
            opD.frame = active.frame
            opD.marker_name = active.name
            opD.marker_uid = active.uid

            detail = box.box()
            detail.label(
                text=(
                    f"{active.marker_name or active.name}  •  Recorder Frame "
                    f"{_stretched_frame(scene, active.frame)}"
                ),
                icon='MARKER',
            )

            event_header = detail.row(align=True)
            event_header.label(text="Events")
            add_event = event_header.operator(OPS.idname("add_flag_popup"), text="", icon='ADD')
            add_event.marker_uid = active.uid
            add_event.use_selected_marker = True
            remove_event = event_header.operator(OPS.idname("marker_event_delete"), text="", icon='X')
            remove_event.marker_uid = active.uid
            remove_event.event_index = active.events_index

            detail.template_list(
                "MMRT_UL_MarkerEventList",
                "mmrt_marker_event_list",
                active,
                "events",
                active,
                "events_index",
                rows=max(1, min(4, len(active.events))),
            )

            if active.events and 0 <= active.events_index < len(active.events):
                event = active.events[active.events_index]
                event_box = detail.box()
                event_box.label(
                    text=(
                        event.custom_flag_name
                        if event.flag_type == FlagTypes.CUSTOM.name
                        else event.flag_type.replace("_", " ").title()
                    ),
                    icon=FLAG_ICONS.get(event.flag_type, "BOOKMARKS"),
                )
                handler = HANDLER_REGISTRY.get(FlagTypes.from_name(event.flag_type))
                if handler:
                    handler.draw_ui(event_box, event_box.row(align=True), event)
                else:
                    event_box.label(text="This event has no editable details.", icon='INFO')


    def draw_episode_section(self, box, scene):
        rec = scene.mmrt_recorder
        domain = get_frame_domain(scene)
        box.label(
            text=(
                "Episode ranges are entered in original Blender frames; "
                "the recorder uses the stretched domain."
            ),
            icon='INFO',
        )
        box.label(
            text="Episode list order is used for behavior-tree sequence order.",
            icon='SORT_ASC',
        )

        row = box.row()
        row.template_list(
            "MMRT_UL_EpisodeList", "mmrt_episode_list",
            rec, "episodes", rec, "episodes_index", rows=3,
        )
        col = row.column(align=True)
        col.operator(OPS.idname("add_episode"), icon='ADD', text="")
        remove = col.operator(OPS.idname("remove_episode"), icon='REMOVE', text="")
        remove.index = rec.episodes_index
        col.separator()
        up_row = col.row(align=True)
        up_row.enabled = 0 < rec.episodes_index < len(rec.episodes)
        move_up = up_row.operator(OPS.idname("move_episode"), icon='TRIA_UP', text="")
        move_up.direction = RecorderOperatorOptions.MOVE_UP
        down_row = col.row(align=True)
        down_row.enabled = 0 <= rec.episodes_index < len(rec.episodes) - 1
        move_down = down_row.operator(OPS.idname("move_episode"), icon='TRIA_DOWN', text="")
        move_down.direction = RecorderOperatorOptions.MOVE_DOWN

        if rec.episodes:
            ordered = sorted(rec.episodes, key=lambda episode: episode.start_frame)
            seen_names = set()
            seen_ids = set()
            previous_end = None
            for episode in ordered:
                episode_name = episode.episode_name.strip() or episode.episode_id.strip()
                episode_id = episode.episode_id.strip()
                problems = []
                folded_name = episode_name.casefold()
                folded_id = episode_id.casefold()
                range_status = episode_range_status(
                    episode.start_frame,
                    episode.end_frame,
                    record_start=domain["source_start"],
                    record_end=domain["source_end"],
                )
                if not episode_name:
                    problems.append("name is empty")
                if not episode_id:
                    problems.append("ID is empty")
                if episode.start_frame > episode.end_frame:
                    problems.append("start is after end")
                elif range_status == EpisodeRangeStatus.OUTSIDE:
                    # This episode is intentionally ignored for this
                    # recording and must not affect active episode checks.
                    continue
                else:
                    if range_status == EpisodeRangeStatus.PARTIAL:
                        problems.append("partially overlaps recorder range")
                    if range_status == EpisodeRangeStatus.INSIDE:
                        if folded_name in seen_names:
                            problems.append("duplicate name")
                        if folded_id in seen_ids:
                            problems.append("duplicate ID")
                        seen_names.add(folded_name)
                        seen_ids.add(folded_id)
                        if previous_end is not None and episode.start_frame <= previous_end:
                            problems.append("overlaps another episode")
                        previous_end = max(previous_end, episode.end_frame) if previous_end is not None else episode.end_frame
                if problems:
                    warning = box.row()
                    warning.alert = True
                    warning.label(text=f"{episode_id or '<unnamed>'}: {', '.join(problems)}", icon='ERROR')

        draw_collapsible_box(
            box,
            scene,
            rec,
            "show_episode_batch_box",
            self.draw_episode_batch_section,
            label="Batch Episode Operations",
        )

        if 0 <= rec.episodes_index < len(rec.episodes):
            episode = rec.episodes[rec.episodes_index]
            detail = box.box()
            detail.prop(episode, "episode_name")
            detail.prop(episode, "episode_id")
            range_row = detail.row(align=True)
            range_row.label(text="Recorder Start")
            range_row.label(text=str(_stretched_frame(scene, episode.start_frame)))
            range_row.label(text="Recorder End")
            range_row.label(text=str(_stretched_frame(scene, episode.end_frame)))
            source_row = detail.row(align=True)
            start_row = source_row.row(align=True)
            start_row.prop(episode, "start_frame", text="Original Start")
            set_start = start_row.operator(
                OPS.idname("set_episode_frame_to_current"),
                text="",
                icon='KEYFRAME',
            )
            set_start.boundary = RecorderOperatorOptions.FRAME_START
            end_row = source_row.row(align=True)
            end_row.prop(episode, "end_frame", text="Original End")
            set_end = end_row.operator(
                OPS.idname("set_episode_frame_to_current"),
                text="",
                icon='KEYFRAME',
            )
            set_end.boundary = RecorderOperatorOptions.FRAME_END
            range_status = episode_range_status(
                episode.start_frame,
                episode.end_frame,
                record_start=domain["source_start"],
                record_end=domain["source_end"],
            )
            if range_status == EpisodeRangeStatus.OUTSIDE:
                detail.label(text="Outside recorder range; ignored", icon='INFO')
            elif range_status == EpisodeRangeStatus.PARTIAL:
                detail.label(text="Partial overlap will block recording", icon='ERROR')
            detail.prop(episode, "logic")
            if episode.logic == EpisodeConfig.LOGIC_REPEAT:
                detail.prop(episode, "repeat_count")
            elif episode.logic == EpisodeConfig.LOGIC_RETRY_UNTIL_SUCCESSFUL:
                detail.prop(episode, "retry_count")
            detail.prop(episode, "delay_enabled")
            if episode.delay_enabled:
                detail.prop(episode, "delay_msec")
            detail.separator()
            detail.label(text="Episode Start", icon='PLAY')
            detail.prop(episode, "start_with_waypoint")
            if episode.start_with_waypoint:
                detail.prop(episode, "start_waypoint_mode")

    def draw_episode_batch_section(self, box, scene):
        rec = scene.mmrt_recorder
        marked_count = sum(1 for episode in rec.episodes if episode.batch_selected)

        offset_box = box.box()
        offset_box.label(text="Shift episode ranges", icon='FORWARD')
        offset_box.prop(rec, "episode_offset", text="Frames")
        target_row = offset_box.row(align=True)
        active_offset = target_row.operator(OPS.idname("offset_episodes"), text="Active")
        active_offset.target = RecorderOperatorOptions.TARGET_ACTIVE
        batch_offset = target_row.operator(OPS.idname("offset_episodes"), text="Batch")
        batch_offset.target = RecorderOperatorOptions.TARGET_BATCH
        all_offset = target_row.operator(OPS.idname("offset_episodes"), text="All")
        all_offset.target = "ALL"
        offset_box.label(
            text=(
                "Check episodes in the list for Batch, or use All. "
                f"{marked_count} marked."
            ),
            icon='INFO',
        )

        batch_box = box.box()
        batch_box.label(text="Batch episode start settings", icon='SETTINGS')
        batch_box.prop(rec, "episode_batch_start_with_waypoint")
        if rec.episode_batch_start_with_waypoint:
            batch_box.prop(rec, "episode_batch_waypoint_mode")
        batch_row = batch_box.row(align=True)
        apply_batch = batch_row.operator(
            OPS.idname("configure_episode_batch"),
            text="Apply Marked",
        )
        apply_batch.target = RecorderOperatorOptions.TARGET_BATCH
        apply_all = batch_row.operator(
            OPS.idname("configure_episode_batch"),
            text="Apply All",
        )
        apply_all.target = "ALL"


# Register/unregister
classes = (
    MMRT_PT_RecorderPanel,
    MMRT_UL_ObjectList,
    MMRT_UL_MarkerList,
    MMRT_UL_MarkerEventList,
    MMRT_UL_EpisodeList,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
