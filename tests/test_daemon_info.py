import json
import pytest
from unittest.mock import patch, MagicMock
from onnx_manager.daemon.lifecycle import write_daemon_info, read_daemon_info, cleanup_stale_pid
from onnx_manager.cli.client import DaemonClient


def test_write_and_read_daemon_info(tmp_onnx_home):
    write_daemon_info(pid=12345, host="127.0.0.1", port=7700)
    info = read_daemon_info()
    assert info is not None
    assert info["pid"] == 12345
    assert info["host"] == "127.0.0.1"
    assert info["port"] == 7700


def test_read_daemon_info_missing(tmp_onnx_home):
    assert read_daemon_info() is None


def test_daemon_client_reads_port_from_info_file(tmp_onnx_home):
    write_daemon_info(pid=12345, host="127.0.0.1", port=7700)
    # port not specified → should read from daemon.json
    client = DaemonClient()
    assert client.base_url == "http://127.0.0.1:7700"


def test_daemon_client_explicit_port_overrides_info_file(tmp_onnx_home):
    write_daemon_info(pid=12345, host="127.0.0.1", port=7700)
    # explicit port → should override daemon.json
    client = DaemonClient(port=9999)
    assert client.base_url == "http://127.0.0.1:9999"


def test_daemon_client_falls_back_to_default_when_no_info_file(tmp_onnx_home):
    # no daemon.json → fall back to DEFAULT_PORT
    client = DaemonClient()
    assert "11434" in client.base_url


def test_cleanup_stale_pid_removes_daemon_json(tmp_onnx_home):
    write_daemon_info(pid=999999, host="127.0.0.1", port=7700)
    info_file = tmp_onnx_home / "daemon.json"
    assert info_file.exists()
    # PID 999999 does not exist → stale
    cleanup_stale_pid()
    assert not info_file.exists()
