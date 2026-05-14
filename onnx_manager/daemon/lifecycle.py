import os
import signal
import sys
from pathlib import Path

import uvicorn

import onnx_manager.config as config
from onnx_manager.daemon.app import create_app
from onnx_manager.pool.manager import ModelPool
from onnx_manager.store.registry import ModelRegistry


def write_pid(pid_file: Path = None) -> None:
    if pid_file is None:
        pid_file = config.PID_FILE
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(str(os.getpid()))


def read_pid(pid_file: Path = None) -> int | None:
    if pid_file is None:
        pid_file = config.PID_FILE
    if not pid_file.exists():
        return None
    try:
        return int(pid_file.read_text().strip())
    except ValueError:
        return None


def is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def cleanup_stale_pid(pid_file: Path = None) -> None:
    if pid_file is None:
        pid_file = config.PID_FILE
    pid = read_pid(pid_file)
    if pid is not None and not is_running(pid):
        pid_file.unlink(missing_ok=True)


def stop_daemon(pid_file: Path = None) -> bool:
    if pid_file is None:
        pid_file = config.PID_FILE
    pid = read_pid(pid_file)
    if pid is None or not is_running(pid):
        return False
    os.kill(pid, signal.SIGTERM)
    return True


def run_daemon(host: str = None, port: int = None) -> None:
    if host is None:
        host = config.DEFAULT_HOST
    if port is None:
        port = config.DEFAULT_PORT
    pid_file = config.PID_FILE

    cleanup_stale_pid(pid_file)
    existing_pid = read_pid(pid_file)
    if existing_pid is not None and is_running(existing_pid):
        print(f"Daemon is already running (PID {existing_pid}). Use 'onnx stop' first.")
        sys.exit(1)

    pool = ModelPool()
    registry = ModelRegistry()
    app = create_app(pool=pool, registry=registry)

    write_pid(pid_file)

    def _cleanup(signum, frame):
        pid_file.unlink(missing_ok=True)
        sys.exit(0)

    signal.signal(signal.SIGTERM, _cleanup)

    try:
        uvicorn.run(app, host=host, port=port, log_level="info")
    finally:
        pid_file.unlink(missing_ok=True)
