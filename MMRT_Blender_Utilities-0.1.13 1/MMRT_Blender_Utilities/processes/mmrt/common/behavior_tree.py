import xml.etree.ElementTree as ET
from copy import deepcopy
from types import MethodType
import yaml
import numpy as np
from common.logging_setup import trace
import logging
from definitions import BehaviorTreeModelConfig, BlackboardDefaults, MDataLabel, MMRTProcessDefaults


logger = logging.getLogger(__name__)


BEHAVIOR_TREE_WAYPOINT_ATOMIC_GROUP_SUFFIX = "_joint"


def translate_atomic_group_positions(atomic_positions: dict) -> dict:
    """Translate raw atomic group joint keys to waypoint port-compatible keys."""
    translated = {}
    for group_name, values in atomic_positions.items():
        new_key = f"{group_name}{BEHAVIOR_TREE_WAYPOINT_ATOMIC_GROUP_SUFFIX}"
        translated[new_key] = values
    return translated


def format_xml_parameter(value):
    """Format a value to the string representation used in XML attributes."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        out = []
        for v in value:
            if isinstance(v, (int, float)):
                out.append(f"{float(v):.6f}")
            else:
                out.append(str(v))
        return ",".join(out)
    if isinstance(value, (int, float)):
        return f"{float(value):.6f}"
    return str(value)


class BlackboardManager:
    def __init__(self, initial_values=None):
        """Manage one deployable blackboard.

        MMRT-specific defaults are supplied by the caller so separate
        blackboards cannot accidentally inherit values from one another.
        """
        self.blackboard_content = dict(initial_values or {})
        self.waypoints = {}
        self.trajectories = {}
    
    def set(self, key: str, value):
        self.blackboard_content[key] = value

    @staticmethod
    def _to_csv_string(values):
        if isinstance(values, (list, tuple, np.ndarray)):
            return ', '.join(f"{v:.6f}" if isinstance(v, (int, float)) else str(v) for v in values)
        elif isinstance(values, (int, float)):
            return f"{values:.6f}"
        else:
            return str(values)
    
    @trace
    def add_value_to_blackboard(self, key: str, value):
        # add more general values to blackboard extra dict
        if key in self.blackboard_content:
            logger.warning(f"Conflict detected for key '{key}'. Updated with new value.")
        self.blackboard_content[key] = value
    
    @trace
    def add_values(self, **kwargs):
        for key, value in kwargs.items():
            self.add_value_to_blackboard(key, value)

    @trace
    def add_pose_to_blackboard(self, pose_name: str, pose_values):
        if pose_name in self.blackboard_content:
            logger.warning(f"Conflict detected for key '{pose_name}'. Updated with new values.")
        self.blackboard_content[pose_name] = self._to_csv_string(pose_values)

    def add_poses(self, poses: dict):
        for name, values in poses.items():
            self.add_pose_to_blackboard(name, values)

    @trace
    def add_waypoint(self, name: str, plangroup: str, **kwargs):
        """
        Example:
            add_waypoint("test_pose_1", "robot_move_group",
                         head_joint=[-0.7, -0.48],
                         larm_pose=[0.46, 0.27, 0.16, -0.707, 0, 0, 0.707])
        """
        if name in self.waypoints:
            logger.warning(f"Waypoint '{name}' already exists. Overwriting.")

        wp_data = {
            MDataLabel.PLANGROUP_KEY: plangroup
        }

        for k, v in kwargs.items():
            wp_data[k] = self._to_csv_string(v)

        self.waypoints[name] = wp_data

    @trace
    def add_trajectory(self, name: str, traj_id: str, plangroup: str, waypoints: list):
        """
        Accepts either:
        New format:
            ["wp1", "wp2", "wp3"]
        Old format:
            [
                {"wp1": {...}},
                {"wp2": {...}}
            ]
        """
        traj_wp = []
        for wp in waypoints:
            # -------------------------
            # New format
            # -------------------------
            if isinstance(wp, str):
                if wp not in self.waypoints:
                    raise ValueError(f"Waypoint '{wp}' not found in blackboard.")

                traj_wp.append(wp)
                continue

            # -------------------------
            # Old format
            # -------------------------
            if isinstance(wp, dict):
                for wp_name, wp_content in wp.items():
                    if wp_name not in self.waypoints:
                        wp_data = {
                            MDataLabel.PLANGROUP_KEY: wp_content[MDataLabel.PLANGROUP_KEY]
                        }
                        for k, v in wp_content.items():
                            if k == MDataLabel.PLANGROUP_KEY:
                                continue
                            wp_data[k] = self._to_csv_string(v)

                        self.waypoints[wp_name] = wp_data
                        logger.debug(f"Added waypoint '{wp_name}' from trajectory.")

                    traj_wp.append(wp_name)

        # -------------------------
        # print duplicated waypoints
        # -------------------------
        duplicates = [x for x in set(traj_wp) if traj_wp.count(x) > 1]
        if duplicates:
            logger.warning(
                f"Trajectory '{name}' contains duplicated waypoint references: {duplicates}"
            )

        traj_data = {
            "id": traj_id,
            MDataLabel.PLANGROUP_KEY: plangroup,
            MDataLabel.WAYPOINTS_KEY: traj_wp
        }

        # -------------------------
        # overwrite protection
        # -------------------------
        if name in self.trajectories:
            existing = self.trajectories[name]
            if (
                existing[MDataLabel.PLANGROUP_KEY] == plangroup
                and existing[MDataLabel.WAYPOINTS_KEY] == traj_wp
            ):
                logger.debug(f"Trajectory '{name}' identical. Skipping overwrite.")
                return False

            logger.warning(f"Trajectory '{name}' overwritten.")

        self.trajectories[name] = traj_data
        logger.debug(
            f"Trajectory '{name}' stored with {len(traj_wp)} waypoints."
        )
        return True

    # ─────────────────────────────────────────────
    # Export to YAML
    # ─────────────────────────────────────────────
    @trace
    def export_blackboard(
        self,
        file_path: str,
        keys=None,
        include_waypoints=False,
        include_trajectories=False,
        extra_dict=None,
        required_keys=None,
        waypoint_comments=None,
        **kwargs,
    ):
        """Export selected values and optional motion data to a YAML file."""
        required_keys = [BlackboardDefaults.ENABLE_LOG_KEY, BlackboardDefaults.ENABLE_LOG_TO_FILE_KEY] if required_keys is None else required_keys
        for key in required_keys:
            if key not in kwargs and key not in self.blackboard_content and (extra_dict is None or key not in extra_dict):
                raise ValueError(f"Missing required blackboard argument: {key}")

        output = {}
        if keys is None:
            output.update(self.blackboard_content)
        else:
            output.update({key: self.blackboard_content[key] for key in keys if key in self.blackboard_content})

        for key, value in kwargs.items():
            if key in output:
                logger.warning(f"Overriding key '{key}' with value from kwargs")
            output[key] = value

        if extra_dict:
            output.update(extra_dict)
        if include_waypoints:
            output[MDataLabel.WAYPOINTS_KEY] = self.waypoints
        if include_trajectories:
            output["trajectories"] = self.trajectories

        serialized = yaml.safe_dump(output, sort_keys=False)
        for waypoint_name, frame_id in (waypoint_comments or {}).items():
            key_line = f"  {waypoint_name}:\n"
            commented_key_line = f"  {waypoint_name}:  # frame {frame_id}\n"
            serialized = serialized.replace(key_line, commented_key_line, 1)

        with open(file_path, "w") as f:
            f.write(serialized)

        logger.info(f"Blackboard export: {len(self.waypoints)} waypoints, {len(self.trajectories)} trajectories")
        return output


class BehaviorTreeBuilder:
    def __init__(self, main_tree_name=MMRTProcessDefaults.TREE_ID):
        self.models = {}  # action_id -> { "type": "action", "params": {name: {"default":..., "required":bool}}}
        self.trees = {}   # built trees
        self.packages = set()
        self.main_tree_name = main_tree_name
        # Built-in actions (Sleep, Script) are shared by all model catalogs.
        self.load_models(BehaviorTreeModelConfig.DEFAULT_MODELS_PATH)
        self.create_tree(main_tree_name)

    # -------------------- model loading --------------------
    def load_models(self, xml_path):
        """
        Load a model catalog and register its action methods on this builder.
        Later catalogs replace definitions with the same action ID.
        Build `self.models[action_id]["params"]` like:
        { param_name: {"default": <str>, "required": <bool>} }
        Includes both <input_port> and <output_port> elements as parameters.
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()

        for model in root.findall(".//Action") + root.findall(".//Decorator"):
            package = (model.get("package") or "").strip()
            if package:
                self.packages.add(package)

        for action in root.findall(".//Action"):
            action_id = action.get("ID")
            params = {}

            # Collect input_port and output_port elements
            ports = list(action.findall("input_port")) + list(action.findall("output_port"))

            # If neither input_port nor output_port exists, fall back to Param
            if not ports:
                ports = action.findall("Param")

            for p in ports:
                name = p.get("name")
                if not name:
                    continue  # skip nameless ports

                default = p.get("default", "")
                required = (p.get("required", "false").lower() == "true")

                # If already exists (e.g., same name in input & output), prefer non-empty default
                if name in params:
                    if default and not params[name]["default"]:
                        params[name]["default"] = default
                    params[name]["required"] = params[name]["required"] or required
                else:
                    params[name] = {"default": default, "required": required}

            self.models[action_id] = {"type": "action", "params": params}
            self._create_action_wrapper(action_id)

    # -------------------- action wrapper --------------------
    def _create_action_wrapper(self, action_id: str):
        """
        Creates methods named ``add_<ActionID>(parent, **kwargs)``.
        Ensures every param declared in the model is present in node['params']
        (uses provided value -> model default -> ""), and raises if a required
        param has no provided value and no default.
        """
        def action_method(self, parent, **kwargs):
            model_params = self.models.get(action_id, {}).get("params", {})
            final_params = {}

            # Fill parameters declared in the model (preserve order)
            for pname, meta in model_params.items():
                if pname in kwargs:
                    final_params[pname] = kwargs.pop(pname)
                elif meta.get("default", "") != "":
                    final_params[pname] = meta["default"]
                else:
                    # Not provided and no default
                    if meta.get("required", False):
                        raise ValueError(
                            f"Action '{action_id}' missing required parameter '{pname}'"
                        )
                    # Optional -> explicitly set empty string so attribute appears in XML
                    # final_params[pname] = ""

            # Include any extra kwargs that are not declared in model (pass-through)
            for extra_k, extra_v in kwargs.items():
                final_params[extra_k] = extra_v

            node = {
                "type": "action",
                "id": action_id,
                "params": final_params,
                "children": []
            }
            self._attach_node_to_parent(parent, node)
            return node

        setattr(self, f"add_{action_id}", MethodType(action_method, self))


    # -------------------- attach helper --------------------
    def _attach_node_to_parent(self, parent, node: dict):
        """
        Attach node to parent. parent can be:
          - tree name (str) -> append to self.trees[parent]['children']
          - node dict -> append into parent['children']
        """
        if isinstance(parent, str):
            if parent not in self.trees:
                raise ValueError(f"Tree '{parent}' does not exist")
            self.trees[parent].setdefault("children", []).append(node)
        elif isinstance(parent, dict):
            parent.setdefault("children", []).append(node)
        else:
            raise TypeError(f"Parent must be tree name (str) or node dict, got {type(parent)}")

    # -------------------- tree creation / control nodes --------------------
    def create_tree(self, tree_name):
        if tree_name in self.trees:
            return
        self.trees[tree_name] = {"type": "root", "children": []}

    def add_sequence(self, parent):
        seq = {"type": "control", "id": "Sequence", "children": []}
        self._attach_node_to_parent(parent, seq)
        return seq

    def add_parallel(self, parent, success_threshold=None, failure_threshold=None):
        node = {"type": "control", "id": "Parallel", "children": []}
        if success_threshold is not None or failure_threshold is not None:
            node["params"] = {}
            if success_threshold is not None:
                node["params"]["success_threshold"] = success_threshold
            if failure_threshold is not None:
                node["params"]["failure_threshold"] = failure_threshold
        self._attach_node_to_parent(parent, node)
        return node

    def add_fallback(self, parent):
        node = {"type": "control", "id": "Fallback", "children": []}
        self._attach_node_to_parent(parent, node)
        return node

    def add_repeater(self, parent, num_cycles="1"):
        node = {
            "type": "control",
            "id": "Repeat",
            "params": {"num_cycles": str(num_cycles)},
            "children": []
        }
        self._attach_node_to_parent(parent, node)
        return node

    def add_decorator(self, parent, decorator_name, params=None):
        node = {"type": "control", "id": decorator_name, "params": params or {}, "children": []}
        self._attach_node_to_parent(parent, node)
        return node

    def add_subtree(self, parent, subtree_id, autoremap=True):
        node = {
            "type": "action",
            "id": "SubTree",
            "params": {"ID": subtree_id, "_autoremap": str(autoremap).lower()},
            "children": []
        }
        self._attach_node_to_parent(parent, node)
        return node

    # -------------------- XML building/export --------------------
    def _append_nodes(self, parent_elem: ET.Element, nodes):
        if nodes is None:
            ET.SubElement(parent_elem, "AlwaysSuccess")
            return

        if isinstance(nodes, dict):
            nodes_list = [nodes]
        else:
            nodes_list = nodes

        if not isinstance(nodes_list, (list, tuple)):
            nodes_list = [nodes_list]

        if len(nodes_list) == 0:
            ET.SubElement(parent_elem, "AlwaysSuccess")
            return

        for node in nodes_list:
            self._append_node(parent_elem, node)

    def _append_node(self, parent_elem: ET.Element, node):
        if isinstance(node, ET.Element):
            parent_elem.append(deepcopy(node))
            return

        if isinstance(node, str):
            ET.SubElement(parent_elem, node)
            return

        if not isinstance(node, dict):
            raise TypeError(f"Unsupported node type: {type(node)} (value={node})")

        node_id = node.get("id") or node.get("ID") or node.get("tag")
        if not node_id:
            ntype = node.get("type")
            if isinstance(ntype, str):
                mapping = {
                    "sequence": "Sequence",
                    "control": "Sequence",
                    "parallel": "Parallel",
                    "fallback": "Fallback",
                    "selector": "Selector"
                }
                node_id = mapping.get(ntype.lower(), None)
            if not node_id:
                raise ValueError(f"Node dict missing 'id'/'ID'/'tag': {node}")

        params = node.get("params", {}) or {}
        attrs = {}
        for k, v in params.items():
            attrs[k] = format_xml_parameter(v)

        elem = ET.SubElement(parent_elem, node_id, attrs)

        children = node.get("children", None)
        if children:
            self._append_nodes(elem, children)
        else:
            if node_id in ("Sequence", "Parallel", "Fallback", "Selector"):
                ET.SubElement(elem, "AlwaysSuccess")

    def export_bt(self, file_path: str, main_tree_name: str = None, tree_names=None):
        """Export selected trees to a BehaviorTree.CPP XML file.

        ``main_tree_name`` is optional so a file can contain only subtree
        definitions.  When ``tree_names`` is omitted, all builder trees are
        exported for backwards compatibility.  If a main tree is supplied it
        is emitted first, followed by the remaining selected trees.
        """
        if main_tree_name is None and tree_names is None:
            main_tree_name = self.main_tree_name

        if tree_names is None:
            selected_tree_names = list(self.trees)
        else:
            selected_tree_names = list(dict.fromkeys(tree_names))

        missing_tree_names = [name for name in selected_tree_names if name not in self.trees]
        if missing_tree_names:
            raise ValueError(
                "Tree(s) not found in builder.trees: " + ", ".join(missing_tree_names)
            )

        if main_tree_name is not None:
            if main_tree_name not in self.trees:
                raise ValueError(f"Main tree '{main_tree_name}' not found in builder.trees")
            if main_tree_name not in selected_tree_names:
                raise ValueError(
                    f"Main tree '{main_tree_name}' must be included in tree_names"
                )
            ordered_tree_names = [main_tree_name] + [
                name for name in selected_tree_names if name != main_tree_name
            ]
        else:
            ordered_tree_names = selected_tree_names

        if not ordered_tree_names:
            raise ValueError("Cannot export an empty BehaviorTree XML file")

        root = ET.Element("root", {"BTCPP_format": "4"})

        def tree_to_children(tree_entry):
            if tree_entry is None:
                return []
            if isinstance(tree_entry, dict):
                if "children" in tree_entry and isinstance(tree_entry["children"], (list, tuple)):
                    return tree_entry["children"]
                return [tree_entry]
            elif isinstance(tree_entry, (list, tuple)):
                return list(tree_entry)
            else:
                return [tree_entry]

        for name in ordered_tree_names:
            entry = self.trees[name]
            sub_children = tree_to_children(entry)
            sub_elem = ET.Element("BehaviorTree", {"ID": name})
            if not sub_children:
                if name == main_tree_name:
                    raise ValueError(
                        f"Main tree '{main_tree_name}' is empty. "
                        "Add actions/control nodes before exporting."
                    )
                seq = ET.SubElement(sub_elem, "Sequence")
                ET.SubElement(seq, "AlwaysSuccess")
            else:
                self._append_nodes(sub_elem, sub_children)
            root.append(sub_elem)
        self._indent(root, level=0, indent_str="  ")
        ET.ElementTree(root).write(file_path, encoding="utf-8", xml_declaration=True)
    
    def _indent(self, elem: ET.Element, level: int = 0, indent_str: str = "  "):
        """
        In-place pretty-print indent for ElementTree elements.
        Adds newlines and indentation so ET.write(...) produces readable XML.
        """
        i = "\n" + (level * indent_str)
        if len(elem):
            if elem.text is None or not elem.text.strip():
                elem.text = i + indent_str
            for child in elem:
                self._indent(child, level + 1, indent_str)
            if elem.tail is None or not elem.tail.strip():
                elem.tail = i
        else:
            # no children
            if level and (elem.tail is None or not elem.tail.strip()):
                elem.tail = i
