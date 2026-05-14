import click
from onnx_manager.cli.commands.pull import pull_cmd
from onnx_manager.cli.commands.serve import serve_cmd, stop_cmd
from onnx_manager.cli.commands.model import list_cmd, rm_cmd, show_cmd, ps_cmd, load_cmd, unload_cmd
from onnx_manager.cli.commands.run import run_cmd


@click.group()
def cli():
    """ONNX Manager — local ONNX model management and serving."""
    pass


cli.add_command(pull_cmd, name="pull")
cli.add_command(serve_cmd, name="serve")
cli.add_command(stop_cmd, name="stop")
cli.add_command(list_cmd, name="list")
cli.add_command(rm_cmd, name="rm")
cli.add_command(show_cmd, name="show")
cli.add_command(ps_cmd, name="ps")
cli.add_command(load_cmd, name="load")
cli.add_command(unload_cmd, name="unload")
cli.add_command(run_cmd, name="run")
