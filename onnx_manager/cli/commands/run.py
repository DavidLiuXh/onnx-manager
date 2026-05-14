import click
from onnx_manager.cli.client import DaemonClient
import onnx_manager.config as config


@click.command("run")
@click.argument("model_id")
@click.argument("text")
@click.option("--port", default=None, type=int)
def run_cmd(model_id, text, port):
    """Run a single inference (embedding shown as first 8 dims)."""
    client = DaemonClient(port=port)
    if not client.is_alive():
        click.echo("Daemon is not running. Run: onnx serve", err=True)
        raise SystemExit(1)
    vec = client.embed(model_id=model_id, text=text)
    preview = vec[:8]
    click.echo(f"[{', '.join(f'{v:.4f}' for v in preview)}, ...]  (dim={len(vec)})")
