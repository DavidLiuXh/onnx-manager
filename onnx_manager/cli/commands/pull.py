import click
from pathlib import Path
from onnx_manager.store.registry import ModelRegistry
from onnx_manager.store.downloader import (
    pull_from_huggingface,
    import_local_model,
    convert_and_import,
    NoOnnxExportError,
)
import onnx_manager.config as config


@click.command("pull")
@click.argument("source")
@click.option("--name", default=None, help="Model name (required for local files)")
@click.option("--task", type=click.Choice(config.TASK_TYPES), default=None,
              help="Task type (required for local files)")
@click.option("--convert", "force_convert", is_flag=True, default=False,
              help="Convert from PyTorch to ONNX via optimum-cli without prompting")
def pull_cmd(source, name, task, force_convert):
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
        try:
            record = pull_from_huggingface(model_id=source, registry=registry)
        except NoOnnxExportError as e:
            if not e.has_pytorch:
                raise click.ClickException(str(e))
            click.echo(str(e))
            optimum_task = config.PIPELINE_TAG_TO_OPTIMUM_TASK.get(e.pipeline_tag, "feature-extraction")
            onnx_task = config.PIPELINE_TAG_MAP.get(e.pipeline_tag, "embedding")
            trust_remote_code = "custom_code" in e.tags
            if not force_convert:
                extra = " --trust-remote-code" if trust_remote_code else ""
                if not click.confirm(
                    f"\nConvert {source} to ONNX using optimum-cli?{extra}\n"
                    f"(requires: pip install optimum[onnxruntime])",
                    default=False,
                ):
                    raise click.Abort()
            if trust_remote_code:
                click.echo("Note: model uses custom code, passing --trust-remote-code to optimum-cli.")
            click.echo(f"Converting {source} → ONNX (task={optimum_task})...")
            click.echo("This may take several minutes for large models.")
            try:
                record = convert_and_import(
                    model_id=source,
                    task=onnx_task,
                    optimum_task=optimum_task,
                    registry=registry,
                    trust_remote_code=trust_remote_code,
                )
            except RuntimeError as conv_err:
                raise click.ClickException(str(conv_err))

    size_mb = (record.size_bytes or 0) / 1_000_000
    click.echo(f"Done. {record.id} ({record.task}, {size_mb:.1f} MB)")
