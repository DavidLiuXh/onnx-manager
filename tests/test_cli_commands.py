import pytest
from click.testing import CliRunner
from unittest.mock import patch, MagicMock
from onnx_manager.cli.main import cli


def test_pull_huggingface():
    runner = CliRunner()
    mock_record = MagicMock()
    mock_record.id = "BAAI/bge-small-en-v1.5"
    mock_record.task = "embedding"
    mock_record.size_bytes = 23_000_000

    with patch("onnx_manager.cli.commands.pull.pull_from_huggingface", return_value=mock_record), \
         patch("onnx_manager.cli.commands.pull.ModelRegistry"):
        result = runner.invoke(cli, ["pull", "BAAI/bge-small-en-v1.5"])

    assert result.exit_code == 0
    assert "BAAI/bge-small-en-v1.5" in result.output


def test_pull_local_file(tmp_path):
    runner = CliRunner()
    fake_onnx = tmp_path / "model.onnx"
    fake_onnx.write_bytes(b"fake")

    mock_record = MagicMock()
    mock_record.id = "mymodel"
    mock_record.task = "embedding"
    mock_record.size_bytes = 4

    with patch("onnx_manager.cli.commands.pull.import_local_model", return_value=mock_record), \
         patch("onnx_manager.cli.commands.pull.ModelRegistry"):
        result = runner.invoke(cli, [
            "pull", str(fake_onnx), "--name", "mymodel", "--task", "embedding"
        ])

    assert result.exit_code == 0


def test_list_models():
    runner = CliRunner()
    mock_records = [
        MagicMock(id="BAAI/bge-small-en-v1.5", task="embedding", size_bytes=23_000_000, pulled_at="2026-05-14T10:00:00"),
    ]
    with patch("onnx_manager.cli.commands.model.ModelRegistry") as MockReg:
        MockReg.return_value.list_all.return_value = mock_records
        result = runner.invoke(cli, ["list"])

    assert result.exit_code == 0
    assert "bge-small-en-v1.5" in result.output


def test_rm_model():
    runner = CliRunner()
    mock_record = MagicMock()
    mock_record.local_path = "/tmp/fake-model"

    with patch("onnx_manager.cli.commands.model.ModelRegistry") as MockReg, \
         patch("onnx_manager.cli.commands.model.shutil") as mock_shutil:
        MockReg.return_value.get.return_value = mock_record
        result = runner.invoke(cli, ["rm", "BAAI/bge-small-en-v1.5"])

    assert result.exit_code == 0


def test_rm_nonexistent_model():
    runner = CliRunner()
    with patch("onnx_manager.cli.commands.model.ModelRegistry") as MockReg:
        MockReg.return_value.get.return_value = None
        result = runner.invoke(cli, ["rm", "nonexistent"])

    assert result.exit_code != 0
    assert "not found" in result.output.lower()


def test_stop_no_daemon():
    runner = CliRunner()
    with patch("onnx_manager.cli.commands.serve.stop_daemon", return_value=False):
        result = runner.invoke(cli, ["stop"])
    assert result.exit_code == 0
    assert "not running" in result.output.lower()
