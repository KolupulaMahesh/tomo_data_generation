"""Keep an SSH session open while rosbridge_server runs on the robot."""

from __future__ import annotations

import argparse
import codecs
import paramiko
import shlex
import signal
import sys
import threading
import time


def _remote_start_script(port: int, domain_id: int, extra_commands: str) -> str:
    lines = ["set -e", "shopt -s expand_aliases"]
    setup_commands = extra_commands.strip()
    if setup_commands:
        lines.extend((setup_commands, ""))

    launch_command = f"ros2 launch rosbridge_server rosbridge_websocket_launch.xml port:={port}"
    lines.extend(
        (
            f"export ROS_DOMAIN_ID={domain_id}",
            "if ! command -v ros2 >/dev/null 2>&1; then",
            '  echo "ROS 2 is unavailable after running the extra commands." >&2',
            "  exit 20",
            "fi",
            f"if ! eval {shlex.quote('ros2 pkg prefix rosbridge_server')} >/dev/null 2>&1; then",
            '  echo "rosbridge_server is not installed or is not available in the sourced ROS environment." >&2',
            "  exit 21",
            "fi",
            "ignored_pids=\" \"",
            "current_pid=$$",
            "while [ -n \"$current_pid\" ] && [ \"$current_pid\" -gt 1 ] 2>/dev/null; do",
            "  ignored_pids=\"$ignored_pids$current_pid \"",
            "  current_pid=$(ps -o ppid= -p \"$current_pid\" 2>/dev/null | tr -d ' ')",
            "done",
            "for cmdline in /proc/[0-9]*/cmdline; do",
            "  [ -r \"$cmdline\" ] || continue",
            "  process_pid=${cmdline#/proc/}",
            "  process_pid=${process_pid%/cmdline}",
            "  case \"$ignored_pids\" in *\" $process_pid \"*) continue ;; esac",
            "  process_args=$(tr '\\0' ' ' < \"$cmdline\" 2>/dev/null || true)",
            "  case \"$process_args\" in",
            "    *rosbridge_websocket*)",
            '      echo "ROS Bridge server is already running (PID $process_pid)." >&2',
            "      exit 22",
            "      ;;",
            "  esac",
            "done",
            f"if (echo > /dev/tcp/127.0.0.1/{port}) >/dev/null 2>&1; then",
            f'  echo "Port {port} is already accepting connections; ROS Bridge appears to be running." >&2',
            "  exit 22",
            "fi",
            f"echo 'Starting ROS Bridge on port {port}; SSH output will remain attached.'",
            # eval occurs after setup so aliases from .bashrc and setup commands resolve.
            f"eval {shlex.quote(launch_command)}",
        )
    )
    return "\n".join(lines)


def _stream_channel(channel, stop_requested: threading.Event) -> int:
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    shutdown_started = None

    while not channel.exit_status_ready():
        if channel.recv_ready():
            chunk = channel.recv(4096)
            if not chunk:
                break
            sys.stdout.write(decoder.decode(chunk))
            sys.stdout.flush()
            continue

        if stop_requested.is_set():
            if shutdown_started is None:
                shutdown_started = time.monotonic()
            elif time.monotonic() - shutdown_started > 8.0:
                channel.close()
                break
        time.sleep(0.05)

    while channel.recv_ready():
        chunk = channel.recv(4096)
        if not chunk:
            break
        sys.stdout.write(decoder.decode(chunk))
    tail = decoder.decode(b"", final=True)
    if tail:
        sys.stdout.write(tail)
    sys.stdout.flush()
    return channel.recv_exit_status()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run rosbridge_server through a persistent remote SSH session."
    )
    parser.add_argument("--ip", required=True)
    parser.add_argument("--user", required=True)
    parser.add_argument("--password", default="")
    parser.add_argument("--port", type=int, required=True, help="ROS Bridge WebSocket port")
    parser.add_argument("--domain-id", type=int, required=True)
    parser.add_argument("--extra-commands", default="")
    args = parser.parse_args()

    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if not 0 <= args.domain_id <= 232:
        parser.error("--domain-id must be between 0 and 232")

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    stop_requested = threading.Event()
    channel = None
    previous_handlers = {}

    def request_remote_shutdown(signum, _frame):
        stop_requested.set()
        if channel is not None and not channel.closed:
            try:
                # With a PTY, this sends Ctrl-C to ros2 launch and lets ROS
                # shut down its child nodes before the SSH channel closes.
                channel.send(b"\x03")
            except (OSError, paramiko.SSHException):
                pass

    handled_signals = [signal.SIGINT, signal.SIGTERM]
    if hasattr(signal, "SIGBREAK"):
        handled_signals.append(signal.SIGBREAK)
    for signum in handled_signals:
        try:
            previous_handlers[signum] = signal.signal(signum, request_remote_shutdown)
        except (ValueError, OSError):
            pass

    try:
        print(f"Connecting to {args.user}@{args.ip} over SSH", flush=True)
        client.connect(
            hostname=args.ip,
            username=args.user,
            password=args.password or None,
            timeout=15,
            banner_timeout=15,
            auth_timeout=15,
            allow_agent=True,
            look_for_keys=True,
        )
        remote_script = _remote_start_script(
            args.port,
            args.domain_id,
            args.extra_commands,
        )
        command = "bash -ic " + shlex.quote(remote_script)
        _stdin, stdout, _stderr = client.exec_command(command, get_pty=True)
        channel = stdout.channel
        channel.set_combine_stderr(True)
        return _stream_channel(channel, stop_requested)
    except (OSError, paramiko.SSHException, TimeoutError) as exc:
        print(f"Remote ROS Bridge SSH session failed: {exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        if channel is not None and not channel.closed:
            channel.close()
        client.close()
        for signum, handler in previous_handlers.items():
            try:
                signal.signal(signum, handler)
            except (ValueError, OSError):
                pass


if __name__ == "__main__":
    raise SystemExit(main())
