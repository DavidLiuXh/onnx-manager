import click
from onnx_manager.cli.client import DaemonClient
from onnx_manager.store.registry import ModelRegistry


@click.command("run")
@click.argument("model_id")
@click.argument("text")
@click.option("--port", default=None, type=int)
@click.option("--query", default=None, help="Query text for rerank models (TEXT becomes the document)")
@click.option("--max-tokens", default=100, show_default=True, help="Max tokens for text-generation")
def run_cmd(model_id, text, port, query, max_tokens):
    """Run a single inference against a loaded model.

    For embedding: shows first 8 dimensions of the vector.
    For rerank: --query is the query, TEXT is the document to score.
    For text-generation: TEXT is the prompt.
    """
    client = DaemonClient(port=port)
    if not client.is_alive():
        click.echo("Daemon is not running. Run: onnx serve", err=True)
        raise SystemExit(1)

    # Look up task type from registry
    registry = ModelRegistry()
    record = registry.get(model_id)
    if record is None:
        click.echo(f"Model {model_id!r} not found locally. Run: onnx pull {model_id}", err=True)
        raise SystemExit(1)

    task = record.task

    if task == "embedding":
        vec = client.embed(model_id=model_id, text=text)
        preview = vec[:8]
        click.echo(f"[{', '.join(f'{v:.4f}' for v in preview)}, ...]  (dim={len(vec)})")

    elif task == "rerank":
        if query is None:
            click.echo("--query is required for rerank models", err=True)
            raise SystemExit(1)
        results = client.rerank(model_id=model_id, query=query, documents=[text])
        score = results[0]["score"]
        click.echo(f"Score: {score:.4f}")

    elif task == "text-generation":
        output = client.complete(model_id=model_id, prompt=text, max_tokens=max_tokens)
        click.echo(output)

    else:
        click.echo(f"Unknown task type {task!r} for model {model_id!r}", err=True)
        raise SystemExit(1)
