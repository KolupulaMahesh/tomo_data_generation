"""Protocol constants for the MMRT-to-robot ROS bridge.

Keep the wire contract in this module. The Blender operators and bridge helper
should not contain duplicated ROS service names, command numbers, or
rosbridge message shapes.
"""

from __future__ import annotations

from typing import Any


class RobotServerConfig:
    """Configuration shared by the Blender client and the bridge process."""

    # Local UDP hop between Blender and ros2_bridge.py.
    BRIDGE_LOCAL_HOST = "127.0.0.1"
    BRIDGE_REQUEST_PORT = 5006
    BRIDGE_EVENT_PORT = 5005
    BRIDGE_MAX_DATAGRAM_BYTES = 65507
    BRIDGE_CONNECTION_TIMEOUT_SECONDS = 30.0
    BRIDGE_SERVICE_RESPONSE_TIMEOUT_SECONDS = 60.0

    # Remote rosbridge WebSocket endpoint.
    ROSBRIDGE_SCHEME = "ws"
    ROSBRIDGE_DEFAULT_PORT = 9090
    BRIDGE_REMOTE_IP_ARGUMENT = "--remote_ip"
    BRIDGE_REMOTE_PORT_ARGUMENT = "--remote_port"
    BRIDGE_STATUS_FILE_ARGUMENT = "--status_file"
    # rosbridge operation names.
    ROSBRIDGE_CALL_SERVICE_OP = "call_service"
    ROSBRIDGE_SERVICE_RESPONSE_OP = "service_response"
    ROSBRIDGE_PUBLISH_OP = "publish"
    ROSBRIDGE_SUBSCRIBE_OP = "subscribe"

    # Robot console service contract.
    CONSOLE_SERVICE_ROOT = "/console_service"
    CONSOLE_SERVICES = {
        "load_config": f"{CONSOLE_SERVICE_ROOT}/load_config",
        "load_application": f"{CONSOLE_SERVICE_ROOT}/load_app",
        "control": f"{CONSOLE_SERVICE_ROOT}/control",
        "get_state": f"{CONSOLE_SERVICE_ROOT}/get_status",
    }

    # Values from console_interfaces/srv/Control.srv.
    CONSOLE_COMMANDS = {
        "run": 0,
        "pause": 1,
        "stop": 2,
        "configure": 3,
        "generate_tree": 4,
        "home_pose": 5,
        "reset_scene": 6,
    }

    # Request field names from console_interfaces/srv/*.srv.
    LOAD_CONFIG_ARGUMENT = "file_path"
    LOAD_APPLICATION_ARGUMENT = "app_dir"
    CONTROL_ARGUMENT = "command"

    # Topics emitted by the robot-side BT logger and ROS 2 logging system.
    BT_LOG_TOPIC = "/BTLog"
    BT_LOG_TOPIC_TYPE = "std_msgs/msg/String"
    ROS_LOG_TOPIC = "/rosout"
    ROS_LOG_TOPIC_TYPE = "rcl_interfaces/msg/Log"

    # Robot-server defaults, rather than Blender-only UI preferences.
    DEFAULT_REMOTE_HOST = "localhost"
    DEFAULT_REMOTE_APP_DIR = "/var/lib/mmrt/bt/apps/untitled_app/"

    @classmethod
    def console_service(cls, name: str) -> str:
        try:
            return cls.CONSOLE_SERVICES[name]
        except KeyError as exc:
            raise ValueError(f"Unknown console service: {name}") from exc

    @classmethod
    def console_command(cls, name: str) -> int:
        try:
            return cls.CONSOLE_COMMANDS[name]
        except KeyError as exc:
            raise ValueError(f"Unknown console command: {name}") from exc


def bridge_service_request(
    service: str,
    args: dict[str, Any] | list[Any] | None = None,
    request_id: str | None = None,
) -> dict[str, Any]:
    """Build the rosbridge ``call_service`` envelope."""
    request: dict[str, Any] = {
        "op": RobotServerConfig.ROSBRIDGE_CALL_SERVICE_OP,
        "service": service,
        "args": args or {},
    }
    if request_id:
        request["id"] = request_id
    return request


def load_config_request(file_path: str, request_id: str | None = None) -> dict[str, Any]:
    """Build a LoadConfig request for a path visible on the robot server."""
    return bridge_service_request(
        RobotServerConfig.console_service("load_config"),
        {RobotServerConfig.LOAD_CONFIG_ARGUMENT: file_path},
        request_id,
    )


def load_application_request(app_dir: str, request_id: str | None = None) -> dict[str, Any]:
    """Build a LoadApplication request for a path visible on the robot server."""
    # rosbridge also accepts positional request values. LoadApplication has a
    # single directory argument, so this avoids coupling the client to the
    # robot service's spelling for that field (for example app_dir vs
    # app_directory).
    return bridge_service_request(
        RobotServerConfig.console_service("load_application"),
        [app_dir],
        request_id,
    )


def control_request(command: str | int, request_id: str | None = None) -> dict[str, Any]:
    """Build a Control request using a symbolic or numeric command."""
    command_value = (
        RobotServerConfig.console_command(command)
        if isinstance(command, str)
        else int(command)
    )
    return bridge_service_request(
        RobotServerConfig.console_service("control"),
        {RobotServerConfig.CONTROL_ARGUMENT: command_value},
        request_id,
    )


def get_state_request(request_id: str | None = None) -> dict[str, Any]:
    """Build a GetState request."""
    return bridge_service_request(
        RobotServerConfig.console_service("get_state"),
        {},
        request_id,
    )


ROSBRIDGE_SUBSCRIPTIONS = (
    {
        "id": "mmrt_bt_log_subscription",
        "topic": RobotServerConfig.BT_LOG_TOPIC,
        "type": RobotServerConfig.BT_LOG_TOPIC_TYPE,
        # A best-effort ROS 2 subscription is compatible with both
        # best-effort and reliable publishers. This avoids rosbridge's
        # reliable-default mismatch with some BTLog publishers.
        "qos": {
            "history": "keep_last",
            "depth": 100,
            "reliability": "best_effort",
            "durability": "volatile",
        },
    },
    {
        "id": "mmrt_ros_log_subscription",
        "topic": RobotServerConfig.ROS_LOG_TOPIC,
        "type": RobotServerConfig.ROS_LOG_TOPIC_TYPE,
    },
)
