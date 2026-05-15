import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, call
from onnx_manager.store.downloader import convert_and_import, NoOnnxExportError
from onnx_manager.store.registry import ModelRegistry


def test_no_onnx_export_error_raised_for_safetensors_repo(tmp_onnx_home):
    mock_files = ["config.json", "model.safetensors", "tokenizer.json"]
    mock_info = MagicMock()
    mock_info.pipeline_tag = "text-ranking"

    import onnx_manager.store.downloader as dl
    with patch.object(dl, "list_repo_files", return_value=mock_files):
        import huggingface_hub
        orig = huggingface_hub.model_info
        huggingface_hub.model_info = lambda *a, **k: mock_info
        try:
            with pytest.raises(NoOnnxExportError) as exc_info:
                dl.pull_from_huggingface(
                    "Alibaba-NLP/gte-multilingual-reranker-base", ModelRegistry()
                )
            assert "safetensors" in str(exc_info.value).lower() or "pytorch" in str(exc_info.value).lower()
        finally:
            huggingface_hub.model_info = orig


def test_convert_and_import_calls_optimum_cli(tmp_onnx_home, tmp_path):
    # Simulate optimum-cli creating model.onnx in the output dir
    def fake_run(cmd, check, **kwargs):
        out_dir = Path(cmd[-1])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "model.onnx").write_bytes(b"fake-onnx")
        (out_dir / "tokenizer.json").write_text('{"version":"1.0"}')

    with patch("onnx_manager.store.downloader.subprocess.run", side_effect=fake_run) as mock_run, \
         patch("onnx_manager.store.downloader.onnx.checker.check_model"):
        reg = ModelRegistry()
        record = convert_and_import(
            model_id="Alibaba-NLP/gte-multilingual-reranker-base",
            task="rerank",
            optimum_task="text-classification",
            registry=reg,
        )

    assert record.id == "Alibaba-NLP/gte-multilingual-reranker-base"
    assert record.task == "rerank"
    assert record.source == "huggingface"
    # verify optimum-cli was called with correct arguments
    cmd = mock_run.call_args[0][0]
    assert "optimum-cli" in cmd
    assert "export" in cmd
    assert "onnx" in cmd
    assert "--model" in cmd
    assert "Alibaba-NLP/gte-multilingual-reranker-base" in cmd
    assert "--task" in cmd
    assert "text-classification" in cmd
    # model.onnx should be in dest_dir
    dest_dir = tmp_onnx_home / "models" / "Alibaba-NLP--gte-multilingual-reranker-base"
    assert (dest_dir / "model.onnx").exists()


def test_convert_and_import_fails_if_optimum_not_installed(tmp_onnx_home):
    import subprocess
    def fake_run(cmd, check, **kwargs):
        raise FileNotFoundError("optimum-cli not found")

    with patch("onnx_manager.store.downloader.subprocess.run", side_effect=fake_run):
        with pytest.raises(RuntimeError, match="optimum"):
            convert_and_import(
                model_id="Alibaba-NLP/gte-multilingual-reranker-base",
                task="rerank",
                optimum_task="text-classification",
                registry=ModelRegistry(),
            )
