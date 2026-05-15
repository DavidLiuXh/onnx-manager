import json
import os
import signal
import sys
from pathlib import Path

import uvicorn

import onnx_manager.config as config
from onnx_manager.daemon.app import create_app
from onnx_manager.pool.manager import ModelPool
from onnx_manager.store.registry import ModelRegistry


def write_daemon_info(pid: int, host: str, port: int) -> None:
    info_file = config.DAEMON_INFO_FILE
    info_file.parent.mkdir(parents=True, exist_ok=True)
    info_file.write_text(json.dumps({"pid": pid, "host": host, "port": port}))


def read_daemon_info() -> dict | None:
    info_file = config.DAEMON_INFO_FILE
    if not info_file.exists():
        return None
    try:
        return json.loads(info_file.read_text())
    except (ValueError, OSError):
        return None


def is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # process exists but we don't own it


def cleanup_stale_pid(pid_file: Path = None) -> None:
    info = read_daemon_info()
    if info is not None and not is_running(info["pid"]):
        config.DAEMON_INFO_FILE.unlink(missing_ok=True)
    # also clean up legacy pid file if present
    if pid_file is None:
        pid_file = config.PID_FILE
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
            if not is_running(pid):
                pid_file.unlink(missing_ok=True)
        except (ValueError, OSError):
            pid_file.unlink(missing_ok=True)


def stop_daemon() -> bool:
    info = read_daemon_info()
    if info is None or not is_running(info["pid"]):
        return False
    os.kill(info["pid"], signal.SIGTERM)
    return True


def run_daemon(host: str = None, port: int = None) -> None:
    if host is None:
        host = config.DEFAULT_HOST
    if port is None:
        port = config.DEFAULT_PORT

    cleanup_stale_pid()
    info = read_daemon_info()
    if info is not None and is_running(info["pid"]):
        print(f"Daemon is already running (PID {info['pid']}) on {info['host']}:{info['port']}. Use 'onnx stop' first.")
        sys.exit(1)

    pool = ModelPool()
    registry = ModelRegistry()
    app = create_app(pool=pool, registry=registry)

    write_daemon_info(pid=os.getpid(), host=host, port=port)

    def _cleanup(signum, frame):
        config.DAEMON_INFO_FILE.unlink(missing_ok=True)
        sys.exit(0)

    signal.signal(signal.SIGTERM, _cleanup)

    try:
        uvicorn.run(app, host=host, port=port, log_level="info")
    finally:
        config.DAEMON_INFO_FILE.unlink(missing_ok=True)
