import argparse
import paramiko
import os
import sys
import posixpath
import logging
from common_utils.logging_utils import setup_logger, trace, log_exceptions

setup_logger("normal")
logger = logging.getLogger("SENDER")

@trace
def fail(msg):
    logger.error(f"[ERROR] {msg}")
    sys.exit(1)


@trace
def _mkdir_p(sftp, path):
    """Create a remote directory and its parents if they do not exist."""
    parts = [part for part in path.split("/") if part]
    current = "/" if path.startswith("/") else ""
    for part in parts:
        current = posixpath.join(current, part)
        try:
            sftp.stat(current)
        except IOError:
            sftp.mkdir(current)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", required=True)
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument("--user", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--local", required=True)
    parser.add_argument("--dest", required=True)
    args = parser.parse_args()

    if not os.path.isfile(args.local) and not os.path.isdir(args.local):
        fail("Local file or directory not found")

    logger.info("Connecting...")
    try:
        transport = paramiko.Transport((args.ip, args.port))
    except:
        fail("Cannot connect to host")

    logger.info("Logging in...")
    try:
        transport.connect(username=args.user, password=args.password)
    except:
        fail("Login failed")

    logger.info("Starting SFTP...")
    try:
        sftp = paramiko.SFTPClient.from_transport(transport)
    except:
        transport.close()
        fail("SFTP init failed")

    try:
        if os.path.isdir(args.local):
            local_root = os.path.abspath(args.local)
            for root, _, names in os.walk(local_root):
                rel_root = os.path.relpath(root, local_root)
                remote_root = args.dest if rel_root == "." else posixpath.join(args.dest, *rel_root.split(os.sep))
                _mkdir_p(sftp, remote_root)
                for name in names:
                    local_path = os.path.join(root, name)
                    remote_path = posixpath.join(remote_root, name)
                    logger.info(f"Uploading to: {remote_path}")
                    sftp.put(local_path, remote_path)
        else:
            remote_path = posixpath.join(args.dest, os.path.basename(args.local))
            logger.info(f"Uploading to: {remote_path}")
            _mkdir_p(sftp, args.dest)
            sftp.put(args.local, remote_path)
    except FileNotFoundError:
        sftp.close()
        transport.close()
        fail("Remote directory missing")
    except:
        sftp.close()
        transport.close()
        fail("Upload error")

    logger.info("Done.")
    try:
        sftp.close()
        transport.close()
    except:
        pass

    sys.exit(0)


if __name__ == "__main__":
    main()
# To run: python send_file.py --ip <IP> --port <PORT> --user <USER> --password <PASSWORD> --local <LOCAL_FILE_PATH> --dest <REMOTE_DEST_PATH>
