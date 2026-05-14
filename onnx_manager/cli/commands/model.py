import shutil
import click
from onnx_manager.store.registry import ModelRegistry
from onnx_manager.cli.client import DaemonClient
import onnx_manager.config as config


@click.command("list")
def list_cmd():
    """List all downloaded models."""
    registry = ModelRegistry()
    records = registry.list_all()
    if not records:
        click.echo("No models downloaded. Run: onnx pull <model>")
        return
    click.echo(f"{'ID':<45} {'TASK':<18} {'SIZE':>10}  PULLED AT")
    click.echo("-" * 90)
    for r in records:
        size = f"{(r.size_bytes or 0) / 1_000_000:.1f} MB"
        click.echo(f"{r.id:<45} {r.task:<18} {size:>10}  {r.pulled_at}")


@click.command("rm")
@click.argument("model_id")
def rm_cmd(model_id):
    """Remove a downloaded model."""
    registry = ModelRegistry()
    record = registry.get(model_id)
    if record is None:
        click.echo(f"Model {model_id!r} not found.", err=True)
        raise SystemExit(1)
    shutil.rmtree(record.local_path, ignore_errors=True)
    registry.delete(model_id)
    click.echo(f"Removed {model_id}")


@click.command("show")
@click.argument("model_id")
def show_cmd(model_id):
    """Show metadata for a model."""
    registry = ModelRegistry()
    record = registry.get(model_id)
    if record is None:
        click.echo(f"Model {model_id!r} not found.", err=True)
        raise SystemExit(1)
    click.echo(f"ID:         {record.id}")
    click.echo(f"Task:       {record.task}")
    click.echo(f"Source:     {record.source}")
    click.echo(f"Path:       {record.local_path}")
    click.echo(f"Size:       {(record.size_bytes or 0) / 1_000_000:.1f} MB")
    click.echo(f"Pulled at:  {record.pulled_at}")


@click.command("ps")
@click.option("--port", default=None, type=int)
def ps_cmd(port):
    """Show models currently loaded in memory."""
    client = DaemonClient(port=port)
    if not client.is_alive():
        click.echo("Daemon is not running. Run: onnx serve")
        return
    loaded = client.list_loaded()
    if not loaded:
        click.echo("No models loaded.")
        return
    for m in loaded:
        click.echo(f"{m['id']}  ({m['task']})")


@click.command("load")
@click.argument("model_id")
@click.option("--port", default=None, type=int)
def load_cmd(model_id, port):
    """Load a model into memory."""
    client = DaemonClient(port=port)
    if not client.is_alive():
        click.echo("Daemon is not running. Run: onnx serve", err=True)
        raise SystemExit(1)
    client.load_model(model_id)
    click.echo(f"Loaded {model_id}")


@click.command("unload")
@click.argument("model_id")
@click.option("--port", default=None, type=int)
def unload_cmd(model_id, port):
    """Unload a model from memory."""
    client = DaemonClient(port=port)
    if not client.is_alive():
        click.echo("Daemon is not running. Run: onnx serve", err=True)
        raise SystemExit(1)
    client.unload_model(model_id)
    click.echo(f"Unloaded {model_id}")
