import click
from pathlib import Path
from onnx_manager.store.registry import ModelRegistry
from onnx_manager.store.downloader import pull_from_huggingface, import_local_model
import onnx_manager.config as config


@click.command("pull")
@click.argument("source")
@click.option("--name", default=None, help="Model name (required for local files)")
@click.option("--task", type=click.Choice(config.TASK_TYPES), default=None,
              help="Task type (required for local files)")
def pull_cmd(source, name, task):
    """Download a model from HuggingFace or import a local ONNX file."""
    registry = ModelRegistry()
    src = Path(source)
    if src.exists() and src.suffix == ".onnx":
        if not name:
            raise click.UsageError("--name is required when importing a local file")
        if not task:
            raise click.UsageError("--task is required when importing a local file")
        click.echo(f"Importing local model {src} as {name!r}...")
        record = import_local_model(onnx_path=src, name=name, task=task, registry=registry)
    else:
        click.echo(f"Pulling {source} from HuggingFace...")
        record = pull_from_huggingface(model_id=source, registry=registry)

    size_mb = (record.size_bytes or 0) / 1_000_000
    click.echo(f"Done. {record.id} ({record.task}, {size_mb:.1f} MB)")
