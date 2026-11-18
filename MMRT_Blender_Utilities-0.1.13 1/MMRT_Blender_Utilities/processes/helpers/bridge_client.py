"""Non-blocking local client for the Blender side of ros2_bridge.py."""

from __future__ import annotations

import json
import queue
import socket
import threading
import uuid
from collections import deque
from collections.abc import Callable
from typing import Any

from ...common.robot_server_config import RobotServerConfig


BridgeCallback = Callable[[dict[str, Any]], None]


class LocalBridgeClient:
    """Send requests over UDP and queue bridge events for Blender's main thread."""

    def __init__(self) -> None:
        self._events: queue.Queue[dict[str, Any]] = queue.Queue()
        self._callbacks: dict[str, BridgeCallback] = {}
        self._callback_order: deque[str] = deque()
        self._callbacks_lock = threading.Lock()
        self._socket: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._btlog_event_count = 0

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.running:
            return

        self._stop_event.clear()
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            # On Windows, SO_REUSEADDR can let multiple Blender processes
            # bind this UDP port and receive bridge events unpredictably.
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        else:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((RobotServerConfig.BRIDGE_LOCAL_HOST, RobotServerConfig.BRIDGE_EVENT_PORT))
        sock.settimeout(0.2)
        self._socket = sock
        self._thread = threading.Thread(
            target=self._receive_events,
            name="mmrt-bridge-events",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        sock = self._socket
        self._socket = None
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass

        thread = self._thread
        self._thread = None
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)

        with self._callbacks_lock:
            self._callbacks.clear()
            self._callback_order.clear()

    def send(self, request: dict[str, Any], callback: BridgeCallback | None = None) -> str:
        """Send one request to the local bridge and return its request ID."""
        if not self.running:
            raise RuntimeError("Local ROS bridge event listener is not running")

        request_to_send = dict(request)
        request_id = str(request_to_send.get("id") or f"mmrt_{uuid.uuid4().hex}")
        request_to_send["id"] = request_id

        if callback is not None:
            with self._callbacks_lock:
                self._callbacks[request_id] = callback
                self._callback_order.append(request_id)

        payload = json.dumps(request_to_send).encode("utf-8")
        if len(payload) > RobotServerConfig.BRIDGE_MAX_DATAGRAM_BYTES:
            with self._callbacks_lock:
                self._callbacks.pop(request_id, None)
                try:
                    self._callback_order.remove(request_id)
                except ValueError:
                    pass
            raise ValueError("Bridge request is too large for the local UDP transport")

        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.sendto(
                payload,
                (
                    RobotServerConfig.BRIDGE_LOCAL_HOST,
                    RobotServerConfig.BRIDGE_REQUEST_PORT,
                ),
            )
        print(
            "[mmrt] Sent local bridge request "
            f"service={request_to_send.get('service', '<missing>')} id={request_id}"
        )
        return request_id

    def poll(self, max_events: int = 100) -> list[dict[str, Any]]:
        """Dispatch service responses and return non-service events."""
        other_events: list[dict[str, Any]] = []
        for _ in range(max_events):
            try:
                event = self._events.get_nowait()
            except queue.Empty:
                break

            if event.get("op") == RobotServerConfig.ROSBRIDGE_SERVICE_RESPONSE_OP:
                request_id = event.get("id")
                callback = None
                with self._callbacks_lock:
                    if request_id:
                        callback = self._callbacks.pop(str(request_id), None)
                        if callback is not None:
                            try:
                                self._callback_order.remove(str(request_id))
                            except ValueError:
                                pass

                    # Some rosbridge/server combinations omit the response
                    # ID or return a different ID. Requests are serialized
                    # through the local UDP hop, so the oldest pending
                    # callback is the safe compatibility fallback.
                    if callback is None:
                        while self._callback_order:
                            fallback_id = self._callback_order.popleft()
                            callback = self._callbacks.pop(fallback_id, None)
                            if callback is not None:
                                break
                if callback is not None:
                    try:
                        callback(event)
                    except Exception as exc:
                        # A response callback must not stop future events.
                        print(f"[MMRT] Bridge response handling error: {exc}")
                else:
                    # Keep responses without a matching callback visible to
                    # Blender so missing or rewritten request IDs are clear.
                    other_events.append(event)
            elif event.get("op") == "status" and event.get("id"):
                # rosbridge reports failed call_service requests as status
                # events that carry the request ID, rather than as a
                # service_response. Route those failures to the waiting call.
                request_id = str(event["id"])
                with self._callbacks_lock:
                    callback = self._callbacks.pop(request_id, None)
                    if callback is not None:
                        try:
                            self._callback_order.remove(request_id)
                        except ValueError:
                            pass
                if callback is not None:
                    try:
                        print(
                            "[mmrt] Dispatching service response "
                            f"id={event.get('id', '<missing>')} to its request callback."
                        )
                        callback(event)
                    except Exception as exc:
                        print(f"[MMRT] Bridge response handling error: {exc}")
                else:
                    other_events.append(event)
            else:
                other_events.append(event)
        return other_events

    def cancel_request(self, request_id: str) -> None:
        """Forget a pending response callback after its request times out."""
        with self._callbacks_lock:
            self._callbacks.pop(request_id, None)
            try:
                self._callback_order.remove(request_id)
            except ValueError:
                pass

    def _receive_events(self) -> None:
        while not self._stop_event.is_set():
            sock = self._socket
            if sock is None:
                return
            try:
                payload, _ = sock.recvfrom(RobotServerConfig.BRIDGE_MAX_DATAGRAM_BYTES)
            except socket.timeout:
                continue
            except OSError:
                return

            try:
                event = json.loads(payload.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if isinstance(event, dict):
                operation = event.get("op")
                if operation == RobotServerConfig.ROSBRIDGE_SERVICE_RESPONSE_OP:
                    print(
                        "[mmrt] Received ROS service response "
                        f"id={event.get('id', '<missing>')}"
                    )
                elif operation == "status" and event.get("id"):
                    print(
                        "[mmrt] Received rosbridge request status "
                        f"id={event.get('id')}: {event.get('msg', event.get('message', ''))}"
                    )
                elif (
                    operation == RobotServerConfig.ROSBRIDGE_PUBLISH_OP
                    and event.get("topic") == RobotServerConfig.BT_LOG_TOPIC
                ):
                    self._btlog_event_count += 1
                    if self._btlog_event_count == 1 or self._btlog_event_count % 100 == 0:
                        print(
                            "[mmrt] Received "
                            f"{self._btlog_event_count} BTLog messages from the local bridge."
                        )
                self._events.put(event)
