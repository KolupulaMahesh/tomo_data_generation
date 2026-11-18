"""Forward local Blender requests to a remote rosbridge WebSocket."""

import argparse
import asyncio
import importlib.util
import json
import os
import socket
import sys
import threading
import time

try:
    from ...common.robot_server_config import (
        ROSBRIDGE_SUBSCRIPTIONS,
        RobotServerConfig,
    )
except ImportError:
    # This file is launched directly by Blender's configured external Python,
    # so the addon package is not necessarily on sys.path yet.
    try:
        # The PyInstaller spec packages the add-on's standalone protocol
        # module as a top-level module. This branch keeps the bridge usable
        # when launched from the source tree as well.
        from robot_server_config import (  # type: ignore
            ROSBRIDGE_SUBSCRIPTIONS,
            RobotServerConfig,
        )
    except ImportError:
        addon_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        config_path = os.path.join(addon_root, "common", "robot_server_config.py")
        config_spec = importlib.util.spec_from_file_location("robot_server_config", config_path)
        if config_spec is None or config_spec.loader is None:
            raise ImportError(f"Cannot load robot server config: {config_path}")
        config_module = importlib.util.module_from_spec(config_spec)
        config_spec.loader.exec_module(config_module)
        ROSBRIDGE_SUBSCRIPTIONS = config_module.ROSBRIDGE_SUBSCRIPTIONS
        RobotServerConfig = config_module.RobotServerConfig

import websockets


def fail(msg):
    print(f"[ERROR] {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


async def ros_task(uri, status_file):
    # The SSH command can report that rosbridge launched a little before the
    # ROS node has opened its WebSocket port. Retry refused connections during
    # that startup window so starting the Blender client immediately is safe.
    deadline = time.monotonic() + RobotServerConfig.BRIDGE_CONNECTION_TIMEOUT_SECONDS
    last_error = None
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            fail(
                "Cannot connect to ROS bridge within "
                f"{RobotServerConfig.BRIDGE_CONNECTION_TIMEOUT_SECONDS:g} seconds: "
                f"{last_error}"
            )
        try:
            ws = await websockets.connect(
                uri,
                open_timeout=min(3.0, remaining),
            )
            break
        except (OSError, asyncio.TimeoutError) as exc:
            last_error = exc
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                continue
            await asyncio.sleep(min(0.5, remaining))

    # rosbridge does not forward topic messages until the client subscribes.
    for subscription in ROSBRIDGE_SUBSCRIPTIONS:
        await ws.send(json.dumps({
            "op": RobotServerConfig.ROSBRIDGE_SUBSCRIBE_OP,
            **subscription,
        }))
    print("[BRIDGE] WebSocket connected and topic subscriptions sent.", flush=True)

    # UDP socket -> Blender
    sock_out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    loop = asyncio.get_running_loop()

    # Bind the request channel before reporting a successful connection. This
    # makes Blender's Connected state mean both the WebSocket and local UDP
    # request path are ready.
    sock_in = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock_in.bind((
            RobotServerConfig.BRIDGE_LOCAL_HOST,
            RobotServerConfig.BRIDGE_REQUEST_PORT,
        ))
    except OSError as exc:
        sock_in.close()
        fail(f"Cannot bind local bridge request port: {exc}")

    btlog_count = 0

    async def listen_ros():
        nonlocal btlog_count
        while True:
            try:
                msg = await ws.recv()
            except Exception as exc:
                fail(f"Lost connection to ROS: {exc}")
            payload = msg if isinstance(msg, bytes) else msg.encode("utf-8")
            try:
                event = json.loads(payload.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                event = None
            if isinstance(event, dict):
                if event.get("op") == RobotServerConfig.ROSBRIDGE_SERVICE_RESPONSE_OP:
                    print(
                        "[BRIDGE] Received service response "
                        f"id={event.get('id', '<missing>')}",
                        flush=True,
                    )
                elif event.get("op") == "status" and event.get("id"):
                    print(
                        "[BRIDGE] Received request status "
                        f"id={event.get('id')}: {event.get('msg', event.get('message', ''))}",
                        flush=True,
                    )
                elif (
                    event.get("op") == RobotServerConfig.ROSBRIDGE_PUBLISH_OP
                    and event.get("topic") == RobotServerConfig.BT_LOG_TOPIC
                ):
                    btlog_count += 1
                    if btlog_count == 1 or btlog_count % 100 == 0:
                        print(f"[BRIDGE] Received {btlog_count} BTLog messages from rosbridge.", flush=True)
            try:
                sock_out.sendto(
                    payload,
                    (
                        RobotServerConfig.BRIDGE_LOCAL_HOST,
                        RobotServerConfig.BRIDGE_EVENT_PORT,
                    ),
                )
            except OSError as exc:
                fail(f"Cannot forward ROS event to Blender: {exc}")

    def listen_blender():
        while True:
            try:
                data, _ = sock_in.recvfrom(RobotServerConfig.BRIDGE_MAX_DATAGRAM_BYTES)
                req = json.loads(data.decode("utf-8"))
                if not isinstance(req, dict):
                    print("[WARN] Request must be a JSON object", flush=True)
                    continue
                req.setdefault(
                    "op",
                    RobotServerConfig.ROSBRIDGE_CALL_SERVICE_OP,
                )
                if req["op"] != RobotServerConfig.ROSBRIDGE_CALL_SERVICE_OP:
                    print("[WARN] Unsupported bridge operation", flush=True)
                    continue
                if not req.get("service"):
                    print("[WARN] Service name is missing", flush=True)
                    continue

                print(
                    "[BRIDGE] Forwarding service request "
                    f"{req.get('service')} id={req.get('id', '<missing>')}",
                    flush=True,
                )
                fut = asyncio.run_coroutine_threadsafe(
                    ws.send(json.dumps(req)), loop
                )
                fut.result()
            except json.JSONDecodeError:
                print("[WARN] Bad JSON", flush=True)
            except OSError:
                return
            except Exception as exc:
                print(f"[WARN] Failed to send service call: {exc}", flush=True)

    threading.Thread(target=listen_blender, daemon=True).start()
    if status_file:
        temporary_file = f"{status_file}.{os.getpid()}.tmp"
        try:
            with open(temporary_file, "w", encoding="utf-8") as marker:
                json.dump({"state": "connected", "pid": os.getpid()}, marker)
                marker.flush()
                os.fsync(marker.fileno())
            os.replace(temporary_file, status_file)
        except OSError as exc:
            try:
                os.remove(temporary_file)
            except OSError:
                pass
            fail(f"Cannot report the ROS bridge connection to Blender: {exc}")
    await listen_ros()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        RobotServerConfig.BRIDGE_REMOTE_IP_ARGUMENT,
        required=True,
    )
    parser.add_argument(
        RobotServerConfig.BRIDGE_REMOTE_PORT_ARGUMENT,
        type=int,
        default=RobotServerConfig.ROSBRIDGE_DEFAULT_PORT,
    )
    parser.add_argument(
        RobotServerConfig.BRIDGE_STATUS_FILE_ARGUMENT,
        required=True,
    )
    args = parser.parse_args()

    uri = (
        f"{RobotServerConfig.ROSBRIDGE_SCHEME}://"
        f"{args.remote_ip}:{args.remote_port}"
    )
    print(f"[BRIDGE] Connecting to {uri}", flush=True)

    try:
        asyncio.run(ros_task(uri, args.status_file))
    except KeyboardInterrupt:
        fail("Stopped")


if __name__ == "__main__":
    main()
