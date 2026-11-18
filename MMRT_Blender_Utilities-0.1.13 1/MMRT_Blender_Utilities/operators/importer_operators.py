import os
import xml.etree.ElementTree as ET

import bpy
import math
import mathutils

from ..common.blender import get_bl_idname
from ..common.config import ObjectMode, ObjectParentType, URDFImportConfig
from ..processes.mmrt.definitions import MDataLabel, OutputSuffixes, RobotDescriptionKind, RobotJointTypes, RotationFormat
from ..common.robot_description.kinematics import (
    add_mimic_constraints,
    compute_bone_from_joint,
    get_joint_transform,
)
from ..common.robot_description.storage import (
    MESH_EXPORT_SOURCE_PROP,
    MESH_RELATIVE_PATH_PROP,
    MESH_SOURCE_MATRIX_PROP,
    MESH_SOURCE_PATH_PROP,
    ROBOT_MESH_COLLECTION_PROP,
    ROBOT_SRDF_TEXT_PROP,
    ROBOT_URDF_TEXT_PROP,
    get_text_mesh_path_base,
    migrate_legacy_description,
)
from ..common.robot_description.xml import (
    parse_urdf,
    read_robot_xml,
    resolve_mesh_path,
    save_element_to_temp_file,
    store_optional_srdf_from_path,
    store_robot_xml,
    store_robot_xml_from_path,
)


class MMRT_OT_AddEndLink(bpy.types.Operator):
    bl_idname = get_bl_idname("add_end_link")
    bl_label = "Add End Link"

    def execute(self, context):
        context.scene.robot_importer.end_links.add()
        return {'FINISHED'}


class MMRT_OT_RemoveEndLink(bpy.types.Operator):
    bl_idname = get_bl_idname("remove_end_link")
    bl_label = "Remove Last End Link"

    def execute(self, context):
        links = context.scene.robot_importer.end_links
        if links:
            links.remove(len(links) - 1)
        return {'FINISHED'}


class MMRT_OT_AddCameraLink(bpy.types.Operator):
    bl_idname = get_bl_idname("add_camera_link")
    bl_label = "Add Camera Link"

    def execute(self, context):
        context.scene.robot_importer.camera_links.add()
        return {'FINISHED'}


class MMRT_OT_RemoveCameraLink(bpy.types.Operator):
    bl_idname = get_bl_idname("remove_camera_link")
    bl_label = "Remove Last Camera Link"

    def execute(self, context):
        links = context.scene.robot_importer.camera_links
        if links:
            links.remove(len(links) - 1)
        return {'FINISHED'}


class MMRT_OT_StoreRobotDescriptions(bpy.types.Operator):
    bl_idname = get_bl_idname("store_robot_descriptions")
    bl_label = "Store Robot Descriptions"
    bl_description = "Store URDF and optional SRDF in the Blender file without changing the scene"

    def execute(self, context):
        import_props = context.scene.robot_importer
        urdf_source = bpy.path.abspath(import_props.filepath)

        try:
            stored_urdf = store_robot_xml_from_path(
                RobotDescriptionKind.URDF,
                urdf_source,
                resolve_mesh_paths=True,
            )
        except Exception as e:
            self.report({'ERROR'}, f"URDF read failed: {e}")
            return {'CANCELLED'}

        try:
            text_base = stored_urdf.name.removesuffix("_URDF")
            stored_srdf = store_optional_srdf_from_path(
                import_props.srdf_filepath,
                text_name=f"{text_base}_SRDF",
            )
        except Exception as e:
            self.report({'ERROR'}, f"SRDF read failed: {e}")
            return {'CANCELLED'}

        migrate_legacy_description(RobotDescriptionKind.URDF, stored_urdf)
        if stored_srdf:
            migrate_legacy_description(RobotDescriptionKind.SRDF, stored_srdf)
        else:
            migrate_legacy_description(RobotDescriptionKind.SRDF)

        stored_names = [stored_urdf.name]
        if stored_srdf:
            stored_names.append(stored_srdf.name)
        self.report({'INFO'}, f"Stored robot descriptions: {', '.join(stored_names)}")
        return {'FINISHED'}


class MMRT_OT_ImportXACROURDF(bpy.types.Operator):
    bl_idname = get_bl_idname("import_xacro_urdf")
    bl_label = "Import XACRO/URDF"

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")

    def execute(self, context):
        scene = context.scene
        import_props = scene.robot_importer

        source_filepath = bpy.path.abspath(import_props.filepath)
        # Explicit IK link selections from the importer panel
        end_links = [s.name.strip() for s in import_props.end_links if s.name.strip()]
        end_links = end_links if end_links else None
        camera_links = [s.name.strip() for s in import_props.camera_links if s.name.strip()]
        camera_links = camera_links if camera_links else None

        try:
            merged_root, package_dirs = read_robot_xml(source_filepath, resolve_mesh_paths=True)
        except Exception as e:
            self.report({'ERROR'}, f"URDF read failed: {e}")
            return {'CANCELLED'}

        srdf_root = None
        srdf_source = import_props.srdf_filepath.strip()
        srdf_package_dirs = {}
        try:
            if srdf_source:
                srdf_source = bpy.path.abspath(srdf_source)
                srdf_root, srdf_package_dirs = read_robot_xml(
                    srdf_source,
                    resolve_mesh_paths=True,
                )
        except Exception as e:
            self.report({'ERROR'}, f"SRDF read failed: {e}")
            return {'CANCELLED'}

        bone_length = URDFImportConfig.BONE_LENGTH
        search_paths = list(package_dirs.values())
        filepath = ""
        try:
            filepath = save_element_to_temp_file(merged_root)
            urdf_data = parse_urdf(filepath)
        except Exception as e:
            self.report({'ERROR'}, f"URDF parse failed: {e}")
            return {'CANCELLED'}
        finally:
            try:
                if filepath:
                    os.remove(filepath)
            except OSError:
                pass
        armature_obj = self.create_armature(context, urdf_data, visual_bone_length=bone_length)
        armature_urdf = store_robot_xml(
            RobotDescriptionKind.URDF,
            merged_root,
            source_filepath,
            search_paths=list(package_dirs.values()),
            text_name=f"{armature_obj.name}_URDF",
        )
        armature_srdf = None
        if srdf_root is not None:
            armature_srdf = store_robot_xml(
                RobotDescriptionKind.SRDF,
                srdf_root,
                srdf_source,
                search_paths=list(srdf_package_dirs.values()),
                text_name=f"{armature_obj.name}_SRDF",
            )

        mesh_collection = self._create_robot_mesh_collection(context, armature_obj)
        mesh_reference_map = self._mesh_reference_map(merged_root, armature_urdf)
        if import_props.import_meshes:
            self.import_meshes(
                context,
                urdf_data,
                armature_obj,
                source_filepath,
                search_paths,
                offset_bone_length=bone_length,
                mesh_collection=mesh_collection,
                mesh_reference_map=mesh_reference_map,
            )
        self.archive_remaining_meshes(
            context,
            merged_root,
            armature_obj,
            source_filepath,
            search_paths,
            mesh_collection,
            mesh_reference_map,
        )
        if armature_srdf:
            self.archive_text_meshes(
                context,
                armature_srdf,
                armature_obj,
                mesh_collection,
        )
        armature_obj[ROBOT_MESH_COLLECTION_PROP] = mesh_collection.name
        armature_obj[ROBOT_URDF_TEXT_PROP] = armature_urdf.name
        armature_obj.mmrt_robot_mesh_collection = mesh_collection
        armature_obj.mmrt_robot_urdf_text = armature_urdf
        if armature_srdf:
            armature_obj.mmrt_robot_srdf_text = armature_srdf
        armature_urdf[ROBOT_MESH_COLLECTION_PROP] = mesh_collection.name
        if armature_srdf:
            armature_obj[ROBOT_SRDF_TEXT_PROP] = armature_srdf.name
            armature_srdf[ROBOT_MESH_COLLECTION_PROP] = mesh_collection.name

        migrate_legacy_description(RobotDescriptionKind.URDF, armature_urdf)
        if armature_srdf:
            migrate_legacy_description(RobotDescriptionKind.SRDF, armature_srdf)
        else:
            migrate_legacy_description(RobotDescriptionKind.SRDF)

        self.add_ik_to_end_bones(
            context,
            urdf_data,
            armature_obj,
            end_links=end_links,
            camera_links=camera_links,
        )
        self.add_camera_tracking_with_arrows(
            context,
            urdf_data,
            armature_obj,
            camera_links=camera_links,
            camera_link_keyword=URDFImportConfig.CAMERA_LINK_KEYWORD,
        )

        self.report({'INFO'}, "Successfully imported and stored URDF in the Blender file")
        return {'FINISHED'}

    def create_armature(self, context, urdf_data, visual_bone_length):
        self._ensure_object_mode(context)
        obj = self._new_armature(context, urdf_data.get(MDataLabel.ROBOT_NAME_KEY, "Robot"))

        joint_map = {j['child']: j for j in urdf_data['joints']}
        bones_by_link = self._create_edit_bones(obj.data, urdf_data['joints'], joint_map, visual_bone_length)
        self._parent_edit_bones(bones_by_link, urdf_data['joints'])

        bpy.ops.object.mode_set(mode='POSE')
        self._configure_pose_bones(obj, urdf_data['joints'])

        add_mimic_constraints(obj.pose.bones, obj, urdf_data['joints'])
        return obj

    def _ensure_object_mode(self, context):
        if context.active_object is not None:
            try:
                bpy.ops.object.mode_set(mode=ObjectMode.OBJECT)
            except RuntimeError:
                pass

    def _activate_in_object_mode(self, context, obj):
        """Activate an importer-owned object before using a mode operator."""
        context.view_layer.objects.active = obj
        obj.select_set(True)
        if obj.mode != ObjectMode.OBJECT:
            bpy.ops.object.mode_set(mode=ObjectMode.OBJECT)

    def _new_armature(self, context, robot_name):
        bpy.ops.object.armature_add(enter_editmode=True)
        obj = context.active_object
        arm_obj = obj.data

        arm_obj.name = f"{robot_name}_Armature"
        obj.name = robot_name if robot_name.endswith("_Object") else f"{robot_name}_Object"

        for bone in arm_obj.edit_bones:
            arm_obj.edit_bones.remove(bone)

        return obj

    def _create_edit_bones(self, arm_obj, joints, joint_map, visual_bone_length):
        bones_by_link = {}
        parent_set = {j['parent'] for j in joints}
        child_set = {j['child'] for j in joints}
        bpy.ops.object.mode_set(mode='EDIT')

        root_links = parent_set - child_set
        if root_links:
            base_name = root_links.pop()
            base_bone = arm_obj.edit_bones.new(base_name)
            base_bone.head = (0.0, 0.0, 0.0)
            base_bone.tail = (0.0, visual_bone_length, 0.0)
            bones_by_link[base_name] = base_bone

        for joint in joints:
            head, tail = compute_bone_from_joint(joint, joint_map, visual_bone_length)
            bone = arm_obj.edit_bones.new(joint['name'])
            bones_by_link[joint['child']] = bone
            bone.head = head
            bone.tail = tail
            self._align_bone_roll(bone, joint, joint_map, head, tail)

        return bones_by_link

    def _align_bone_roll(self, bone, joint, joint_map, head, tail):
        axis = self._valid_axis(joint.get('axis'))
        y_axis = tail - head
        if y_axis.length < 1e-4:
            return

        y_axis.normalize()
        global_tf = get_joint_transform(joint, joint_map, {})
        world_axis = (global_tf.to_3x3() @ axis).normalized()
        current_z = bone.matrix.to_3x3() @ mathutils.Vector((0, 0, 1))
        proj_des = world_axis - y_axis * world_axis.dot(y_axis)
        proj_cur = current_z - y_axis * current_z.dot(y_axis)
        if proj_des.length < 1e-4 or proj_cur.length < 1e-4:
            return

        proj_des.normalize()
        proj_cur.normalize()
        bone.roll = math.atan2(
            y_axis.dot(proj_cur.cross(proj_des)),
            proj_cur.dot(proj_des),
        )

    def _parent_edit_bones(self, bones_by_link, joints):
        for joint in joints:
            child = joint['child']
            parent = joint['parent']
            if child in bones_by_link and parent in bones_by_link:
                bones_by_link[child].parent = bones_by_link[parent]

    def _configure_pose_bones(self, armature_obj, joints):
        for joint in joints:
            pb = armature_obj.pose.bones.get(joint['name'])
            if not pb:
                continue

            joint_type = joint.get('type', RobotJointTypes.FIXED)
            axis = joint.get('axis', mathutils.Vector((0, 0, 0)))
            pb.lock_scale = (True, True, True)

            if joint_type == RobotJointTypes.FIXED:
                self._configure_fixed_joint(pb)
                continue

            if joint_type == RobotJointTypes.PRISMATIC:
                self._configure_prismatic_joint(pb, joint, axis)
                continue

            if joint_type == RobotJointTypes.REVOLUTE:
                self._configure_revolute_joint(pb, joint, axis)

    def _configure_fixed_joint(self, pb):
        pb.lock_location = (True, True, True)
        pb.lock_rotation = (True, True, True)
        if pb.rotation_mode == RotationFormat.QUATERNION:
            pb.lock_rotation_w = True
        pb.lock_ik_x = pb.lock_ik_y = pb.lock_ik_z = True

    def _configure_prismatic_joint(self, pb, joint, axis):
        pb.lock_location = (not axis.x, not axis.y, not axis.z)
        pb.lock_rotation = (True, True, True)
        if pb.rotation_mode == RotationFormat.QUATERNION:
            pb.lock_rotation_w = True
        pb.lock_ik_x = pb.lock_ik_y = pb.lock_ik_z = True

        lower, upper = self._joint_limits(joint, 0.0, 0.0)
        con = pb.constraints.new('LIMIT_LOCATION')
        con.use_min_x = con.use_max_x = bool(axis.x)
        con.use_min_y = con.use_max_y = bool(axis.y)
        con.use_min_z = con.use_max_z = bool(axis.z)
        self._set_axis_range(con, axis, lower, upper, "min_", "max_")
        con.owner_space = 'LOCAL'
        con.use_transform_limit = True

    def _configure_revolute_joint(self, pb, joint, axis):
        pb.lock_location = (True, True, True)
        pb.lock_rotation = (not axis.x, not axis.y, not axis.z)
        if pb.rotation_mode == RotationFormat.QUATERNION:
            pb.lock_rotation_w = not any((axis.x, axis.y, axis.z))

        pb.lock_ik_x = pb.lock_ik_y = pb.lock_ik_z = True
        pb.ik_stiffness_x = pb.ik_stiffness_y = pb.ik_stiffness_z = 0.0
        pb.use_ik_limit_x = pb.use_ik_limit_y = pb.use_ik_limit_z = False

        lower, upper = self._joint_limits(joint, -math.pi, math.pi)
        self._set_ik_rotation_limits(pb, axis, lower, upper)

        con = pb.constraints.new('LIMIT_ROTATION')
        con.use_limit_x = bool(axis.x)
        con.use_limit_y = bool(axis.y)
        con.use_limit_z = bool(axis.z)
        self._set_axis_range(con, axis, lower, upper, "min_", "max_")
        con.owner_space = 'LOCAL'
        con.use_transform_limit = True

    def _set_ik_rotation_limits(self, pb, axis, lower, upper):
        for axis_name in ("x", "y", "z"):
            if not getattr(axis, axis_name):
                continue
            setattr(pb, f"lock_ik_{axis_name}", False)
            setattr(pb, f"use_ik_limit_{axis_name}", True)
            setattr(pb, f"ik_min_{axis_name}", lower)
            setattr(pb, f"ik_max_{axis_name}", upper)

    def _set_axis_range(self, constraint, axis, min_value, max_value, min_prefix, max_prefix):
        for axis_name in ("x", "y", "z"):
            if not getattr(axis, axis_name):
                continue
            setattr(constraint, f"{min_prefix}{axis_name}", min_value)
            setattr(constraint, f"{max_prefix}{axis_name}", max_value)

    def _joint_limits(self, joint, default_lower, default_upper):
        limit = joint.get('limit', {})
        return (
            float(limit.get('lower', default_lower)),
            float(limit.get('upper', default_upper)),
        )

    def _valid_axis(self, axis):
        if axis is not None and axis.length >= 1e-4:
            return axis
        return mathutils.Vector((0, 0, 1))

    def _create_robot_mesh_collection(self, context, armature_obj):
        collection = bpy.data.collections.new(f"{armature_obj.name}_Original_Meshes")
        parent_collection = context.collection or context.scene.collection
        parent_collection.children.link(collection)
        return collection

    def _mesh_reference_map(self, resolved_root, stored_text):
        try:
            stored_root = ET.fromstring(stored_text.as_string())
        except (AttributeError, ET.ParseError):
            return {}

        references = {}
        resolved_meshes = resolved_root.findall(".//mesh")
        stored_meshes = stored_root.findall(".//mesh")
        for resolved_mesh, stored_mesh in zip(resolved_meshes, stored_meshes):
            source_path = resolved_mesh.get("filename", "")
            stored_path = stored_mesh.get("filename", "")
            if source_path and stored_path:
                references[os.path.normcase(os.path.normpath(source_path))] = stored_path
        return references

    def _import_mesh_file(self, context, mesh_path):
        if context.active_object is not None and context.active_object.mode != ObjectMode.OBJECT:
            try:
                bpy.ops.object.mode_set(mode=ObjectMode.OBJECT)
            except RuntimeError:
                pass
        existing_objects = set(bpy.data.objects)
        extension = os.path.splitext(mesh_path)[1].lower()
        try:
            if extension == '.obj':
                bpy.ops.wm.obj_import(filepath=mesh_path)
            elif extension == '.stl':
                bpy.ops.wm.stl_import(filepath=mesh_path)
            else:
                self.report({'WARNING'}, f"Unsupported mesh format: {extension}")
                return []
        except (RuntimeError, OSError) as exc:
            self.report({'WARNING'}, f"Could not import mesh '{mesh_path}': {exc}")
            return []
        return [obj for obj in bpy.data.objects if obj not in existing_objects]

    def _move_to_mesh_collection(self, obj, mesh_collection):
        if mesh_collection.objects.get(obj.name) is None:
            mesh_collection.objects.link(obj)
        for collection in tuple(obj.users_collection):
            if collection != mesh_collection:
                collection.objects.unlink(obj)

    def _tag_source_mesh(self, obj, source_path, relative_path, export_source=True):
        obj[MESH_SOURCE_PATH_PROP] = os.path.abspath(os.path.normpath(source_path))
        obj[MESH_RELATIVE_PATH_PROP] = relative_path.replace("\\", "/")
        obj[MESH_EXPORT_SOURCE_PROP] = export_source
        if export_source or MESH_SOURCE_MATRIX_PROP not in obj:
            obj[MESH_SOURCE_MATRIX_PROP] = [
                value
                for row in obj.matrix_world
                for value in row
            ]

    def _linked_mesh_instances(self, source_objects):
        """Create independently parentable objects sharing the source Mesh data."""
        instances = []
        for source_obj in source_objects:
            obj = source_obj.copy()
            obj.data = source_obj.data
            instances.append(obj)
        return instances

    def import_meshes(
        self,
        context,
        urdf_data,
        arm_obj,
        urdf_filepath,
        search_paths,
        offset_bone_length,
        mesh_collection,
        mesh_reference_map,
    ):
        self._activate_in_object_mode(context, arm_obj)
        parent_set = {j['parent'] for j in urdf_data['joints']}
        child_set = {j['child'] for j in urdf_data['joints']}
        base_links = parent_set - child_set
        base_link_name = next(iter(base_links)) if base_links else None
        mesh_instances_by_source = {}

        for link in urdf_data.get('links', []):
            link_name = link.get('name')
            visual = link.get('visual')

            if not visual or not visual.get('filename'):
                continue

            joint_for_link = next((j for j in urdf_data['joints'] if j['child'] == link_name), None)
            bone_name = joint_for_link['name'] if joint_for_link else base_link_name
            axis = joint_for_link['axis'] if joint_for_link else mathutils.Vector((0, 0, 1))
            if not bone_name:
                self.report({'WARNING'}, f"No bone found to attach mesh: {link_name}")
                continue
            if axis.length < 1e-4:
                axis = mathutils.Vector((0, 0, 1))
            axis = axis.normalized()

            mesh_path = resolve_mesh_path(visual['filename'], urdf_filepath, search_paths)
            if not mesh_path or not os.path.exists(mesh_path):
                self.report({'WARNING'}, f"Mesh not found: {visual['filename']} → {mesh_path}")
                continue

            source_key = os.path.normcase(
                os.path.abspath(os.path.normpath(mesh_path))
            )
            source_objects = mesh_instances_by_source.get(source_key)
            reuses_mesh_data = source_objects is not None
            if reuses_mesh_data:
                imported_objs = self._linked_mesh_instances(source_objects)
            else:
                imported_objs = self._import_mesh_file(context, mesh_path)
            if not imported_objs:
                continue
            if not reuses_mesh_data:
                mesh_instances_by_source[source_key] = tuple(imported_objs)

            for obj in imported_objs:
                relative_path = mesh_reference_map.get(
                    source_key,
                    os.path.basename(mesh_path),
                )
                self._tag_source_mesh(
                    obj,
                    mesh_path,
                    relative_path,
                    export_source=not reuses_mesh_data,
                )
                obj["mmrt_urdf_link"] = link_name
                self._move_to_mesh_collection(obj, mesh_collection)
                obj.location = (0, -offset_bone_length, 0)

                # Handle axis flip
                if axis.z < 0:
                    obj.rotation_euler = (math.pi, 0, 0)
                else:
                    obj.rotation_euler = (0, 0, 0)
                obj.parent = arm_obj
                obj.parent_bone = bone_name
                obj.parent_type = 'BONE'
                obj.matrix_parent_inverse = arm_obj.matrix_world.inverted()

    def archive_remaining_meshes(
        self,
        context,
        resolved_root,
        armature_obj,
        urdf_filepath,
        search_paths,
        mesh_collection,
        mesh_reference_map,
    ):
        archived_sources = {
            os.path.normcase(os.path.normpath(obj.get(MESH_SOURCE_PATH_PROP, "")))
            for obj in mesh_collection.objects
            if obj.get(MESH_SOURCE_PATH_PROP)
        }

        for mesh in resolved_root.findall(".//mesh"):
            source_path = resolve_mesh_path(
                mesh.get("filename", ""),
                urdf_filepath,
                search_paths,
            )
            if not source_path:
                self.report({'WARNING'}, f"Could not archive robot mesh: {mesh.get('filename', '')}")
                continue
            source_key = os.path.normcase(os.path.normpath(source_path))
            if source_key in archived_sources:
                continue

            imported_objs = self._import_mesh_file(context, source_path)
            if not imported_objs:
                continue
            relative_path = mesh_reference_map.get(source_key, os.path.basename(source_path))
            for obj in imported_objs:
                self._tag_source_mesh(obj, source_path, relative_path)
                obj["mmrt_mesh_archive_only"] = True
                self._move_to_mesh_collection(obj, mesh_collection)
                obj.parent = armature_obj
                obj.hide_viewport = True
                obj.hide_render = True
                obj.hide_select = True
            archived_sources.add(source_key)

    def archive_text_meshes(self, context, text_block, armature_obj, mesh_collection):
        """Archive mesh references from an additional stored description, such as SRDF."""
        try:
            root = ET.fromstring(text_block.as_string())
        except (AttributeError, ET.ParseError):
            return

        base_dir = get_text_mesh_path_base(text_block)
        description_path = os.path.join(base_dir, f"description{OutputSuffixes.SRDF}")
        archived_sources = {
            os.path.normcase(os.path.normpath(obj.get(MESH_SOURCE_PATH_PROP, "")))
            for obj in mesh_collection.objects
            if obj.get(MESH_SOURCE_PATH_PROP)
        }
        for mesh in root.findall(".//mesh"):
            relative_path = mesh.get("filename", "")
            source_path = resolve_mesh_path(relative_path, description_path, ())
            if not source_path:
                self.report({'WARNING'}, f"Could not archive SRDF mesh: {relative_path}")
                continue
            source_key = os.path.normcase(os.path.normpath(source_path))
            if source_key in archived_sources:
                continue
            imported_objs = self._import_mesh_file(context, source_path)
            for obj in imported_objs:
                self._tag_source_mesh(obj, source_path, relative_path)
                obj["mmrt_mesh_archive_only"] = True
                self._move_to_mesh_collection(obj, mesh_collection)
                obj.parent = armature_obj
                obj.hide_viewport = True
                obj.hide_render = True
                obj.hide_select = True
            if imported_objs:
                archived_sources.add(source_key)
    
    def add_ik_to_end_bones(
        self,
        context,
        urdf_data,
        armature_obj,
        end_links=None,
        camera_links=None,
    ):
        self._activate_in_object_mode(context, armature_obj)
        arm = armature_obj.data
        pose_bones = armature_obj.pose.bones

        parent_map = {j['child']: j['parent'] for j in urdf_data['joints']}
        name_map = {j['child']: j['name'] for j in urdf_data['joints']}
        all_parents = set(parent_map.values())

        # Determine target end links
        if end_links:
            targets = [link for link in end_links if link in name_map]
        else:
            explicit_camera_links = set(camera_links or ())
            targets = [
                child
                for child in name_map
                if child not in all_parents
                and child not in explicit_camera_links
                and (
                    bool(explicit_camera_links)
                    or URDFImportConfig.CAMERA_LINK_KEYWORD.lower() not in child.lower()
                )
            ]

        for end_link in targets:
            joint_name = name_map.get(end_link)
            if not joint_name:
                continue
            bone = arm.bones.get(joint_name)
            pb = pose_bones.get(joint_name)
            if not bone or not pb:
                continue

            bone_matrix_world = armature_obj.matrix_world @ bone.matrix_local
            tail_world_pos = armature_obj.matrix_world @ bone.tail_local
            tail_matrix_world = bone_matrix_world.copy()
            tail_matrix_world.translation = tail_world_pos

            empty_arrows = bpy.data.objects.new(f"{joint_name}_ik_control", None)
            empty_arrows.empty_display_type = 'ARROWS'
            empty_arrows.rotation_mode = RotationFormat.QUATERNION
            empty_arrows.empty_display_size = 0.1
            empty_arrows.parent = armature_obj
            context.collection.objects.link(empty_arrows)
            empty_arrows.matrix_world = bone_matrix_world

            empty_axes = bpy.data.objects.new(f"{joint_name}_ik_target", None)
            empty_axes.empty_display_type = 'PLAIN_AXES'
            empty_axes.rotation_mode = RotationFormat.QUATERNION
            empty_axes.empty_display_size = 0.01

            context.collection.objects.link(empty_axes)
            empty_axes.parent = empty_arrows
            empty_axes.parent_type = ObjectParentType.OBJECT
            empty_axes.matrix_world = tail_matrix_world

            empty_axes.lock_location = (True, True, True)
            empty_axes.lock_rotation_w = True
            empty_axes.lock_rotation = (True, True, True)
            empty_axes.lock_scale = (True, True, True)
            empty_axes.hide_select = True

            chain_len = 1
            current = end_link
            while current in parent_map:
                current = parent_map[current]
                if current not in parent_map:
                    break
                chain_len += 1

            ik = pb.constraints.new('IK')
            ik.name = "IK_Target"
            ik.target = empty_axes
            ik.chain_count = chain_len
            ik.use_rotation = True
            ik.use_location = True
            ik.target_space = 'WORLD'
            ik.owner_space = 'WORLD'
    

    def add_camera_tracking_with_arrows(
        self,
        context,
        urdf_data,
        armature_obj,
        camera_links=None,
        camera_link_keyword="camera",
    ):
        self._activate_in_object_mode(context, armature_obj)
        arm = armature_obj.data
        pose_bones = armature_obj.pose.bones
        mw = armature_obj.matrix_world

        # Build URDF maps
        parent_map  = {j['child']: j['parent'] for j in urdf_data['joints']}
        name_map    = {j['child']: j['name']   for j in urdf_data['joints']}

        if camera_links:
            camera_child_names = list(dict.fromkeys(camera_links))
        else:
            camera_child_names = [
                name
                for name in name_map
                if camera_link_keyword.lower() in name.lower()
            ][:1]

        if not camera_child_names:
            self.report({'INFO'}, "No camera link found in the URDF data.")
            return

        tracking_controls = []
        for camera_child_name in camera_child_names:
            if camera_child_name not in name_map:
                self.report({'WARNING'}, f"Camera link '{camera_child_name}' not found in the URDF data.")
                continue
            controls = self._add_camera_tracking_for_link(
                context,
                armature_obj,
                arm,
                pose_bones,
                mw,
                parent_map,
                name_map,
                camera_child_name,
            )
            if controls:
                tracking_controls.append(controls)

        return tracking_controls

    def _add_camera_tracking_for_link(
        self,
        context,
        armature_obj,
        arm,
        pose_bones,
        mw,
        parent_map,
        name_map,
        camera_child_name,
    ):
        cam_real_name = name_map.get(camera_child_name)
        cam_pb = pose_bones.get(cam_real_name)
        cam_bone = arm.bones.get(cam_real_name)
        
        if not cam_bone or not cam_pb:
            self.report({'WARNING'}, f"Camera bone for '{camera_child_name}' not found.")
            return

        # 1) Arrows empty at camera bone head
        bone_mat_ws = mw @ cam_bone.matrix_local
        head_ws = mw @ cam_bone.head_local

        arrows = bpy.data.objects.new(f"{camera_child_name}_camera_arrows", None)
        arrows.empty_display_type = 'ARROWS'
        arrows.empty_display_size = 0.1
        arrows.location = head_ws
        arrows.rotation_mode = RotationFormat.QUATERNION
        arrows.rotation_quaternion = bone_mat_ws.to_quaternion()
        context.collection.objects.link(arrows)
        # parent to the armature object
        arrows.parent = armature_obj
        arrows.parent_type = ObjectParentType.OBJECT
        arrows.hide_select = True

        # 2) Look-target sphere
        target = bpy.data.objects.new(f"{camera_child_name}_camera_target", None)
        target.empty_display_type = 'SPHERE'
        target.empty_display_size = 0.05
        target.location = head_ws
        context.collection.objects.link(target)
        # parent to the armature object
        target.parent = armature_obj
        target.parent_type = ObjectParentType.OBJECT

        # 3) Damped track on arrows
        damp = arrows.constraints.new('DAMPED_TRACK')
        damp.name         = "Arrows_DampedTrack"
        damp.target       = target
        damp.track_axis   = 'TRACK_Z'
        damp.owner_space  = 'WORLD'
        damp.target_space = 'WORLD'

        # 4) Build IK chain for camera (stop before root)
        chain_bones = []
        cur = camera_child_name
        while cur in parent_map:
            real = name_map.get(cur)
            if not real:
                break
            chain_bones.append(real)
            parent = parent_map[cur]
            if parent not in parent_map:
                break
            cur = parent

        # 5) Add IK on camera pose-bone (rotation only) targeting arrows
        ik = cam_pb.constraints.new('IK')
        ik.name         = "Camera_IK"
        ik.target       = arrows
        ik.chain_count  = len(chain_bones)
        ik.use_location = False
        ik.use_rotation = True
        ik.owner_space  = 'WORLD'
        ik.target_space = 'WORLD'

        return arrows, target


classes = (
    MMRT_OT_AddEndLink,
    MMRT_OT_RemoveEndLink,
    MMRT_OT_AddCameraLink,
    MMRT_OT_RemoveCameraLink,
    MMRT_OT_StoreRobotDescriptions,
    MMRT_OT_ImportXACROURDF,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
