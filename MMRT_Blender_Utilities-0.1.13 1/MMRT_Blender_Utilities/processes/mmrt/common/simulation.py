import pybullet as p
import pybullet_data
import numpy as np
from collections import defaultdict
import time
import logging
from common.collision_meshes import prepare_collision_urdf
from common.logging_setup import trace
logger = logging.getLogger("Simulation")

class Simulation:
    def __init__(self, gui: bool = False):
        """Initialize the PyBullet simulation."""
        if p.isConnected():
            p.disconnect()

        if gui:
            p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        else:
            p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.gui = gui

        if gui:
            p.resetDebugVisualizerCamera(
                cameraDistance=1.5,
                cameraYaw=45,
                cameraPitch=-30,
                cameraTargetPosition=[0, 0, 0]
            )
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
            p.setRealTimeSimulation(1)
        else:
            p.setRealTimeSimulation(0)
        
        self.contact_threshold = None

    # ---------------------------
    # Core utilities
    # ---------------------------
    def disconnect(self):
        if p.isConnected():
            p.disconnect()

    def is_connected(self):
        return p.isConnected()

    def step(self):
        """Step simulation (with sleep if GUI)."""
        p.stepSimulation()
        if self.gui:
            time.sleep(1.0 / 240.0)

    def load_robot(self, urdf_path: str, fixed_base=True):
        """Load a robot URDF into the environment."""
        logger.info(f"Loading URDF: {urdf_path}")
        start_pos = [0, 0, 0]
        start_ori = p.getQuaternionFromEuler([0, 0, 0])
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        try:
            with prepare_collision_urdf(urdf_path) as prepared_urdf_path:
                robot_id = p.loadURDF(
                    prepared_urdf_path,
                    start_pos,
                    start_ori,
                    useFixedBase=fixed_base,
                    flags=p.URDF_USE_SELF_COLLISION |
                        #   p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS |
                        p.URDF_MAINTAIN_LINK_ORDER |
                        #   p.URDF_USE_IMPLICIT_CYLINDER |
                        p.URDF_USE_INERTIA_FROM_FILE |
                        p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES |
                        p.URDF_ENABLE_SLEEPING
                        #   p.URDF_MERGE_FIXED_LINKS
                )
            p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 1)
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
            logger.info(f"URDF loaded successfully with ID: {robot_id}")
        except Exception as e:
            logger.error(f"Failed to load URDF '{urdf_path}': {e}")
            raise e
        return robot_id

    def set_physics_parameters(self):
        p.setGravity(0, 0, 0)
        p.setRealTimeSimulation(0)
        p.setTimeStep(1.0 / 240.0)
        p.setPhysicsEngineParameter(
            numSolverIterations=10,
            numSubSteps=10,
            enableConeFriction=1,
            contactBreakingThreshold=0.001,
            contactSlop=0.001
        )
    
    def disable_adjacent_link_collision(self, robot_id):
        """Disable collision between adjacent links of the robot."""
        joint_num = p.getNumJoints(robot_id)
        for j in range(joint_num):
            parent_link = p.getJointInfo(robot_id, j)[16]
            if parent_link != -1:
                p.setCollisionFilterPair(
                    robot_id, robot_id,
                    parent_link, j,
                    enableCollision=0
                )

    @trace
    def disable_collision_between_links(self, robot_id, link_name1: str, link_name2: str):
        """Disable collision between two specified links by name."""
        link_id1 = self.get_id_from_link_name(robot_id, link_name1)
        link_id2 = self.get_id_from_link_name(robot_id, link_name2)
        if link_id1 is not None and link_id2 is not None:
            p.setCollisionFilterPair(
                robot_id, robot_id,
                link_id1, link_id2,
                enableCollision=0
            )
            logger.info(f"Disabled collision between links '{link_name1}' and '{link_name2}'.")
        else:
            logger.warning(f"Cannot disable collision: Link(s) '{link_name1}' or '{link_name2}' not found.")

    
    def set_collision_filter_threshold(self, threshold: float):
        """Set the global collision filter threshold."""
        self.contact_threshold = threshold

    # ---------------------------
    # Robot state
    # ---------------------------
    def set_joint_positions(self, robot_id, joint_indices, joint_angles, use_reset=False):
        for j, angle in zip(joint_indices, joint_angles):
            if use_reset:
                p.resetJointState(robot_id, j, angle)
            else:
                p.setJointMotorControl2(
                    robot_id, j,
                    p.POSITION_CONTROL,
                    targetPosition=angle,
                    force=500.0
                )

    @trace
    def set_robot_pose(self, robot_id, joint_ids, target_positions):
        """Stop motors and snap joints to target pose."""
        logger.warning("Recovering robot to target pose...")
        for jid, target in zip(joint_ids, target_positions):
            p.setJointMotorControl2(robot_id, jid, p.VELOCITY_CONTROL, force=0)
            p.resetJointState(robot_id, jid, target)
        self.step()

    def set_robot_pose_default(self, robot_id):
        """Reset all joints to 0 with motors disabled."""
        joint_num = p.getNumJoints(robot_id)
        for j in range(joint_num):
            p.setJointMotorControl2(robot_id, j, p.VELOCITY_CONTROL, force=0)
            p.resetJointState(robot_id, j, 0.0)

    # ---------------------------
    # Robot info
    # ---------------------------
    def get_joint_limits_and_indices(self, robot_id):
        limits, indices = [], []
        for j in range(p.getNumJoints(robot_id)):
            info = p.getJointInfo(robot_id, j)
            if info[2] in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                low, high = info[8], info[9]
                if low >= high or low <= -2 * np.pi or high >= 2 * np.pi:
                    low, high = -np.pi, np.pi
                limits.append([low, high])
                indices.append(j)
        return limits, indices

    def get_joint_limits(self, robot_id, joint_ids):
        lower, upper, ranges = [], [], []
        for j in joint_ids:
            info = p.getJointInfo(robot_id, j)
            ll, ul = info[8], info[9]
            lower.append(ll)
            upper.append(ul)
            ranges.append(max(0, ul - ll))  # handle continuous joints
        return lower, upper, ranges

    def get_joint_name(self, robot_id, joint_id: int) -> str:
        try:
            if joint_id == -1:
                # Base link
                base_name = p.getBodyInfo(robot_id)[0].decode("utf-8")
                return f"{base_name} (base)"

            # Valid joint ID
            joint_info = p.getJointInfo(robot_id, joint_id)
            joint_name = joint_info[1].decode("utf-8")
            return joint_name if joint_name else f"joint_{joint_id}"

        except Exception:
            # No joint info → return fallback
            return f"link_{joint_id}"

    @trace
    def get_id_from_joint_name(self, robot_id, joint_name: str):
        """
        Find the joint index from its name.
        Accepts joint_name as a string.
        Raises ValueError if not found.
        """
        for i in range(p.getNumJoints(robot_id)):
            if joint_name == p.getJointInfo(robot_id, i)[1].decode():
                return i

        logger.warning(f"URDF joint '{joint_name}' not found.")
        return None
    
    def get_id_from_link_name(self, robot_id, link_name: str):
        """
        Find the link index from its name.
        Returns joint index (0..N-1), -1 for the base link, or None if not found.
        """
        # 1. Check if it matches the base link (-1)
        try:
            base_name = p.getBodyInfo(robot_id)[0].decode("utf-8")
            if link_name == base_name:
                return -1
        except Exception:
            pass

        # 2. Check remaining child links (0..N-1)
        for i in range(p.getNumJoints(robot_id)):
            if link_name == p.getJointInfo(robot_id, i)[12].decode("utf-8"):
                return i

        logger.warning(f"URDF link '{link_name}' not found.")
        return None

    @trace
    def get_ids_from_joint_names(self, robot_id: int, joint_names: list):
        """Return joint indices for the given names, or None if not found."""
        name_to_idx = {p.getJointInfo(robot_id, j)[1].decode(): j for j in range(p.getNumJoints(robot_id))}
        indices = []
        for name in joint_names:
            if name not in name_to_idx:
                logger.warning(f"URDF joint '{name}' not found. Skipping.")
                indices.append(None)
            else:
                indices.append(name_to_idx[name])
        return indices

    def get_parent_id_from_joint_name(self, robot_id, joint_name: str):
        """Find the joint index by its child link name (returns None if not found)."""
        for j in range(p.getNumJoints(robot_id)):
            if p.getJointInfo(robot_id, j)[12].decode("utf-8") == joint_name:
                return j
        return None

    def get_end_effector_pose(self, robot_id, link_id):
        """Return EE pose (position, orientation quaternion)."""
        state = p.getLinkState(robot_id, link_id, computeForwardKinematics=True)
        return np.array(state[0]), np.array(state[1])

    def get_end_effectors_from_movegroup(self, robot_id, joint_ids):
        """Heuristic: child link of the last joint(s) = EE(s)."""
        ee_links = []
        for j in joint_ids:
            child_link = p.getJointInfo(robot_id, j)[12].decode("utf-8")
            link_id = self.get_ids_from_joint_names(robot_id, [child_link])[0]
            if link_id is not None:
                ee_links.append(link_id)
        return list(set(ee_links))  # unique EEs

    # ---------------------------
    # Collision & Jacobian
    # ---------------------------
    def get_collision(self, robot_id):
        self.step()
        return p.getContactPoints(robot_id)

    def get_atomic_groups_for_contacts(self, robot_id, contacts, atomic_groups):
        """Map contacted robot links to the closest ancestor with a grouped joint.

        This includes fixed tool links, without assigning environment link IDs
        or propagating a contact into every group of a compound move group.
        """
        joint_groups = defaultdict(set)
        for group, joints in atomic_groups.items():
            for joint in joints:
                joint_groups[joint].add(group)
        joint_info = {
            index: p.getJointInfo(robot_id, index)
            for index in range(p.getNumJoints(robot_id))
        }
        groups = set()
        for contact in contacts:
            for body, link in ((contact[1], contact[3]), (contact[2], contact[4])):
                if body != robot_id:
                    continue
                while link in joint_info:
                    info = joint_info[link]
                    owners = joint_groups.get(info[1].decode("utf-8"))
                    if owners:
                        groups.update(owners)
                        break
                    link = info[16]
        return groups

    def get_jacobian(self, robot_id, ee_index, joint_indices, joint_angles):
        zero_vec = [0.0] * len(joint_angles)
        J = p.calculateJacobian(robot_id, ee_index, [0, 0, 0],
                                joint_angles, zero_vec, zero_vec)
        return np.vstack((np.array(J[0]), np.array(J[1])))

    @trace
    def print_contact_points(self, robot_id, max_display=3):
        contacts = self.get_collision(robot_id)

        if not contacts:
            logger.info("✓ No contact points detected.")
            return

        grouped = defaultdict(list)

        for c in contacts:
            if self.contact_threshold is not None and c[8] > self.contact_threshold:
                continue

            key = (c[1], c[3], c[2], c[4])  # bodyA, linkA, bodyB, linkB
            grouped[key].append(c)

        logger.warning(f"🔹 {len(grouped)} unique collision pairs found.")

        items = list(grouped.items())[:max_display]

        for i, ((bA, lA, bB, lB), group) in enumerate(items):
            linkA = self.get_joint_name(bA, lA)
            linkB = self.get_joint_name(bB, lB)

            # Min distance of group for readability
            min_dist = min(c[8] for c in group)

            logger.warning(
                f"[{i}] {linkA} ↔ {linkB} | pairs={len(group)} | min_dist={min_dist:.5f}"
            )

        if len(grouped) > max_display:
            logger.warning(f"… and {len(grouped) - max_display} more groups not shown.")

    def load_mesh(self, mesh_path: str, position: list, orientation: list, scale: list = None):
        """
        Load a raw mesh file (.obj, .stl, etc.) into the simulation scene 
        for collision checking and visual rendering.
        
        Args:
            mesh_path: Path to the mesh file (e.g., .obj, .stl)
            position: [x, y, z] position
            orientation: [roll, pitch, yaw] orientation in radians
            scale: [x, y, z] scale factors (default: [1.0, 1.0, 1.0])
        """
        if scale is None:
            scale = [1.0, 1.0, 1.0]
            
        try:
            # Convert orientation from Euler (RPY) to a quaternion
            quat = p.getQuaternionFromEuler(orientation)
            
            # 1. Create the collision shape geometry backend
            collision_shape_id = p.createCollisionShape(
                shapeType=p.GEOM_MESH,
                fileName=mesh_path,
                meshScale=scale
            )
            
            # 2. Create the visual shape matching backend (so it renders in the GUI)
            visual_shape_id = p.createVisualShape(
                shapeType=p.GEOM_MESH,
                fileName=mesh_path,
                meshScale=scale
            )
            
            # 3. Combine shapes into a fixed, static environment body (mass = 0.0)
            mesh_id = p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=collision_shape_id,
                baseVisualShapeIndex=visual_shape_id,
                basePosition=position,
                baseOrientation=quat
            )
            
            logger.info(f"Loaded mesh '{mesh_path}' as static MultiBody with ID: {mesh_id}")
            return mesh_id
            
        except Exception as e:
            logger.error(f"Failed to load mesh '{mesh_path}': {e}")
            return None


    # ---------------------------
    # Kinematics
    # ---------------------------
    def get_fk_pose(self, robot_id, joint_state: dict, end_effector: str):
        """
        Compute forward kinematics pose for an end-effector given a (possibly partial) joint_state.
        joint_state: {joint_name: value}
        end_effector: name of the end-effector link
        Returns: np.array [x, y, z, qx, qy, qz, qw]
        """
        # Apply given joint state
        for jn, val in joint_state.items():
            j_id = self.get_id_from_joint_name(robot_id, jn)
            if j_id is not None:
                p.resetJointState(robot_id, j_id, val)

        # Get pose of end effector
        ee_id = self.get_id_from_joint_name(robot_id, end_effector)
        if ee_id is None:
            raise ValueError(f"End effector '{end_effector}' not found")

        link_state = p.getLinkState(robot_id, ee_id, computeForwardKinematics=True)
        pos, quat = link_state[4], link_state[5]

        return np.array([*pos, *quat])

    def get_alternative_fk(
        self,
        robot_id,
        fk_targets: dict[tuple[str, str], np.ndarray],
        joint_definitions: dict,
    ):
        """
        Solve IK for multiple move groups, supporting multi-end-effector groups.
        fk_targets: {(move_group, end_effector): fk_pose}
        Returns: list of joint dictionaries {joint_name: angle}
        """

        alt_solutions = []

        # Group fk_targets by move group
        grouped_targets = {}
        for (mg, ee), pose in fk_targets.items():
            if mg not in grouped_targets:
                grouped_targets[mg] = []
            grouped_targets[mg].append((ee, pose))

        # Solve IK per move group (can have multiple EE targets)
        for mg, ee_targets in grouped_targets.items():
            atomic_groups = joint_definitions["move_groups"].get(mg, [])
            mg_joints = []
            for ag in atomic_groups:
                mg_joints.extend(joint_definitions["atomic_groups"].get(ag, []))

            joint_ids = self.get_ids_from_joint_names(robot_id, mg_joints)

            # If group has multiple EE → use IK2 (if compiled in PyBullet)
            if len(ee_targets) > 1 and hasattr(p, "calculateInverseKinematics2"):
                link_ids = []
                target_positions = []
                target_orientations = []

                for ee, pose in ee_targets:
                    link_ids.append(self.get_id_from_joint_name(robot_id, ee))
                    target_positions.append(pose[:3])
                    target_orientations.append(pose[3:])

                ik_solution = p.calculateInverseKinematics2(
                    bodyUniqueId=robot_id,
                    endEffectorLinkIndices=link_ids,
                    targetPositions=target_positions,
                    targetOrientations=target_orientations,
                    jointDamping=[0.01] * len(joint_ids),
                )
            else:
                # Solve separately for each EE (fallback if IK2 not available)
                ee, pose = ee_targets[0]
                ik_solution = p.calculateInverseKinematics(
                    bodyUniqueId=robot_id,
                    endEffectorLinkIndex=self.get_id_from_joint_name(robot_id, ee),
                    targetPosition=pose[:3],
                    targetOrientation=pose[3:],
                    jointDamping=[0.01] * len(joint_ids),
                )

            # Build solution dict
            alt_solution = {jn: ik_solution[i] for i, jn in enumerate(mg_joints)}
            alt_solutions.append(alt_solution)

        return alt_solutions

    # ---------------------------
    # Motion execution
    # ---------------------------
    def execute_move_robot_segments(self, robot_id, joint_indices, angle_data, segments):
        for seg_idx, (start, end) in enumerate(segments):
            logger.info(f"Running segment {seg_idx + 1}: frames {start}->{end}")
            for frame in range(start, end):
                self.set_joint_positions(robot_id, joint_indices, angle_data[frame])
                self.step()

    def execute_stream_robot(self, robot_id, joint_indices, angle_data):
        for frame in angle_data:
            self.set_joint_positions(robot_id, joint_indices, frame)
            self.step()

    def execute_move_robot_to_position(self, robot_id, joint_ids, target_positions,
                                   tolerance=1e-3, timeout=2.0):
        """
        Move robot joints toward target using POSITION_CONTROL.
        Returns:
            (success: bool, exec_time: float in simulation seconds)
        """
        timestep = 1.0 / 240.0  # default PyBullet step size
        joint_limits = []
        for jid in joint_ids:
            info = p.getJointInfo(robot_id, jid)
            vmax = info[11] if info[11] > 0 else 1.0
            fmax = info[10] if info[10] > 0 else 200.0
            joint_limits.append((vmax, fmax))

        sim_steps = 0
        success = False

        while sim_steps * timestep < timeout:
            all_reached = True
            for jid, target, (vmax, fmax) in zip(joint_ids, target_positions, joint_limits):
                pos, _, _, _ = p.getJointState(robot_id, jid)
                if abs(target - pos) > tolerance:
                    all_reached = False
                p.setJointMotorControl2(
                    robot_id, jid,
                    p.POSITION_CONTROL,
                    targetPosition=target,
                    maxVelocity=vmax,
                    force=fmax
                )
            self.step()
            sim_steps += 1

            if all_reached:
                success = True
                break

        exec_time = sim_steps * timestep
        return success, exec_time
