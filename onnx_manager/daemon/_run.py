import click
from onnx_manager.daemon.lifecycle import run_daemon


@click.command()
@click.option("--host", default=None)
@click.option("--port", default=None, type=int)
def main(host, port):
    run_daemon(host=host, port=port)


if __name__ == "__main__":
    main()
