import click
import subprocess
import sys
from onnx_manager.daemon.lifecycle import stop_daemon, run_daemon
import onnx_manager.config as config


@click.command("serve")
@click.option("--host", default=None, help="Host to bind (default: 127.0.0.1)")
@click.option("--port", default=None, type=int, help="Port to listen on (default: 11434)")
@click.option("--background", is_flag=True, default=False, help="Run in background")
def serve_cmd(host, port, background):
    """Start the ONNX daemon."""
    _host = host or config.DEFAULT_HOST
    _port = port or config.DEFAULT_PORT
    if background:
        subprocess.Popen(
            [sys.executable, "-m", "onnx_manager.daemon._run", "--host", _host, "--port", str(_port)],
            start_new_session=True,
        )
        click.echo(f"Daemon started in background on {_host}:{_port}")
    else:
        run_daemon(host=_host, port=_port)


@click.command("stop")
def stop_cmd():
    """Stop the background ONNX daemon."""
    stopped = stop_daemon()
    if stopped:
        click.echo("Daemon stopped.")
    else:
        click.echo("Daemon is not running.")
