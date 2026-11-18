import mathutils
import math
from ...processes.mmrt.definitions import RobotJointTypes


def get_joint_transform(j, jm, cache=None):
    if cache is None: cache = {}
    if j['name'] in cache: return cache[j['name']]
    origin = j.get('origin', {})
    t = [float(x) for x in origin.get('xyz', '0 0 0').split()]
    r = [float(r) for r in origin.get('rpy', '0 0 0').split()]
    mat = mathutils.Matrix.Translation(t) @ mathutils.Euler(r, 'XYZ').to_matrix().to_4x4()
    parent = jm.get(j['parent'])
    global_mat = (get_joint_transform(parent, jm, cache) @ mat) if parent else mat
    cache[j['name']] = global_mat
    return global_mat


def compute_bone_from_joint(j, jm, length):
    axis = j['axis']
    if axis.length < 1e-4:
        axis = mathutils.Vector([0, 0, 1])  # fallback

    g = get_joint_transform(j, jm)
    head = (g @ mathutils.Vector([0, 0, 0, 1])).to_3d()
    # Tail along bone local Y
    rot_corr = mathutils.Euler((-math.radians(90),0,0), 'XYZ').to_matrix()
    local_tail = (rot_corr @ axis).normalized() * length
    tail = (g @ mathutils.Vector([local_tail.x, local_tail.y, local_tail.z, 1])).to_3d()
    return head, tail


def map_range_with_axis(min_val, max_val, multiplier, offset, axis_value):
    # Apply multiplier first
    a = multiplier * min_val + offset
    b = multiplier * max_val + offset

    # Flip if axis is negative
    if axis_value < 0:
        return b, a
    return a, b


def add_mimic_constraints(pose_bones, armature_obj, joint_map):
    for joint in joint_map:
        mimic = joint.get('mimic')
        if not mimic:
            continue

        target_joint_name = mimic.get('joint')
        multiplier = float(mimic.get('multiplier', 1.0))
        offset = float(mimic.get('offset', 0.0))
        target_joint = next((j for j in joint_map if j['name'] == target_joint_name), None)
        if not target_joint:
            continue

        source_axis = target_joint.get('axis', mathutils.Vector((0, 0, 1)))
        dest_axis   = joint.get('axis', mathutils.Vector((0, 0, 1)))
        joint_type = joint.get('type', RobotJointTypes.REVOLUTE)

        pb = pose_bones.get(joint['name'])
        target_pb = pose_bones.get(target_joint_name)

        if not (pb and target_pb):
            continue

        con = pb.constraints.new('TRANSFORM')
        con.name = "Mimic_" + target_joint_name
        con.target = armature_obj
        con.subtarget = target_joint_name

        con.owner_space = 'LOCAL'
        con.target_space = 'LOCAL'
        con.use_motion_extrapolate = True

        # =========================
        # REVOLUTE
        # =========================
        if joint_type == RobotJointTypes.REVOLUTE:
            con.map_from = 'ROTATION'
            con.map_to = 'ROTATION'
            con.from_rotation_mode = 'XYZ'
            con.to_euler_order = 'XYZ'

            min_val = -math.pi
            max_val = math.pi

            # FROM (target joint axis)
            if abs(source_axis.x) > 0.5:
                con.from_min_x_rot = min_val
                con.from_max_x_rot = max_val
                src_axis_val = source_axis.x

                # TO (current joint axis)
                if abs(dest_axis.x) > 0.5:
                    to_min, to_max = map_range_with_axis(min_val, max_val, multiplier, offset, dest_axis.x * src_axis_val)
                    con.to_min_x_rot = to_min
                    con.to_max_x_rot = to_max
                elif abs(dest_axis.y) > 0.5:
                    to_min, to_max = map_range_with_axis(min_val, max_val, multiplier, offset, dest_axis.y * src_axis_val)
                    con.to_min_y_rot = to_min
                    con.to_max_y_rot = to_max
                elif abs(dest_axis.z) > 0.5:
                    to_min, to_max = map_range_with_axis(min_val, max_val, multiplier, offset, dest_axis.z * src_axis_val)
                    con.to_min_z_rot = to_min
                    con.to_max_z_rot = to_max

            elif abs(source_axis.y) > 0.5:
                con.from_min_y_rot = min_val
                con.from_max_y_rot = max_val
                src_axis_val = source_axis.y

                if abs(dest_axis.x) > 0.5:
                    to_min, to_max = map_range_with_axis(min_val, max_val, multiplier, offset, dest_axis.x * src_axis_val)
                    con.to_min_x_rot = to_min
                    con.to_max_x_rot = to_max
                elif abs(dest_axis.y) > 0.5:
                    to_min, to_max = map_range_with_axis(min_val, max_val, multiplier, offset, dest_axis.y * src_axis_val)
                    con.to_min_y_rot = to_min
                    con.to_max_y_rot = to_max
                elif abs(dest_axis.z) > 0.5:
                    to_min, to_max = map_range_with_axis(min_val, max_val, multiplier, offset, dest_axis.z * src_axis_val)
                    con.to_min_z_rot = to_min
                    con.to_max_z_rot = to_max

            elif abs(source_axis.z) > 0.5:
                con.from_min_z_rot = min_val
                con.from_max_z_rot = max_val
                src_axis_val = source_axis.z

                if abs(dest_axis.x) > 0.5:
                    to_min, to_max = map_range_with_axis(min_val, max_val, multiplier, offset, dest_axis.x * src_axis_val)
                    con.to_min_x_rot = to_min
                    con.to_max_x_rot = to_max
                elif abs(dest_axis.y) > 0.5:
                    to_min, to_max = map_range_with_axis(min_val, max_val, multiplier, offset, dest_axis.y * src_axis_val)
                    con.to_min_y_rot = to_min
                    con.to_max_y_rot = to_max
                elif abs(dest_axis.z) > 0.5:
                    to_min, to_max = map_range_with_axis(min_val, max_val, multiplier, offset, dest_axis.z * src_axis_val)
                    con.to_min_z_rot = to_min
                    con.to_max_z_rot = to_max

        # =========================
        # PRISMATIC
        # =========================
        elif joint_type == RobotJointTypes.PRISMATIC:
            con.map_from = 'LOCATION'
            con.map_to = 'LOCATION'

            # Use actual limits if available (better than fixed -1~1)
            limit = joint.get('limit', {})
            min_val = float(limit.get('lower', -1.0))
            max_val = float(limit.get('upper', 1.0))

            # =========================
            # FROM (target joint axis)
            # =========================
            if abs(source_axis.x) > 0.5:
                con.from_min_x = min_val
                con.from_max_x = max_val
                src_axis_val = source_axis.x

                # =========================
                # TO (current joint axis)
                # =========================
                if abs(dest_axis.x) > 0.5:
                    to_min, to_max = map_range_with_axis(min_val, max_val, multiplier, offset, dest_axis.x * src_axis_val)
                    con.to_min_x = to_min
                    con.to_max_x = to_max

                elif abs(dest_axis.y) > 0.5:
                    to_min, to_max = map_range_with_axis(min_val, max_val, multiplier, offset, dest_axis.y * src_axis_val)
                    con.to_min_y = to_min
                    con.to_max_y = to_max

                elif abs(dest_axis.z) > 0.5:
                    to_min, to_max = map_range_with_axis(min_val, max_val, multiplier, offset, dest_axis.z * src_axis_val)
                    con.to_min_z = to_min
                    con.to_max_z = to_max

            elif abs(source_axis.y) > 0.5:
                con.from_min_y = min_val
                con.from_max_y = max_val
                src_axis_val = source_axis.y

                if abs(dest_axis.x) > 0.5:
                    to_min, to_max = map_range_with_axis(min_val, max_val, multiplier, offset, dest_axis.x * src_axis_val)
                    con.to_min_x = to_min
                    con.to_max_x = to_max

                elif abs(dest_axis.y) > 0.5:
                    to_min, to_max = map_range_with_axis(min_val, max_val, multiplier, offset, dest_axis.y * src_axis_val)
                    con.to_min_y = to_min
                    con.to_max_y = to_max

                elif abs(dest_axis.z) > 0.5:
                    to_min, to_max = map_range_with_axis(min_val, max_val, multiplier, offset, dest_axis.z * src_axis_val)
                    con.to_min_z = to_min
                    con.to_max_z = to_max

            elif abs(source_axis.z) > 0.5:
                con.from_min_z = min_val
                con.from_max_z = max_val
                src_axis_val = source_axis.z

                if abs(dest_axis.x) > 0.5:
                    to_min, to_max = map_range_with_axis(min_val, max_val, multiplier, offset, dest_axis.x * src_axis_val)
                    con.to_min_x = to_min
                    con.to_max_x = to_max

                elif abs(dest_axis.y) > 0.5:
                    to_min, to_max = map_range_with_axis(min_val, max_val, multiplier, offset, dest_axis.y * src_axis_val)
                    con.to_min_y = to_min
                    con.to_max_y = to_max

                elif abs(dest_axis.z) > 0.5:
                    to_min, to_max = map_range_with_axis(min_val, max_val, multiplier, offset, dest_axis.z * src_axis_val)
                    con.to_min_z = to_min
                    con.to_max_z = to_max
