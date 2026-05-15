import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import onnx
from huggingface_hub import hf_hub_download, list_repo_files
from huggingface_hub.utils import EntryNotFoundError

import onnx_manager.config as config
from onnx_manager.store.registry import ModelRecord, ModelRegistry


class NoOnnxExportError(ValueError):
    """Raised when a HuggingFace repo has no ONNX export."""
    def __init__(self, model_id: str, pipeline_tag: str, has_pytorch: bool, tags: list[str] | None = None):
        self.model_id = model_id
        self.pipeline_tag = pipeline_tag
        self.has_pytorch = has_pytorch
        self.tags = tags or []
        optimum_task = config.PIPELINE_TAG_TO_OPTIMUM_TASK.get(pipeline_tag, "feature-extraction")
        name = model_id.split("/")[-1]
        task = config.PIPELINE_TAG_MAP.get(pipeline_tag, "embedding")
        msg = f"No ONNX export found in repo {model_id!r}."
        if has_pytorch:
            msg += (
                f"\nThe repo contains PyTorch weights (pipeline_tag={pipeline_tag!r})."
                f"\nRun `onnx pull {model_id} --convert` to convert automatically,"
                f"\nor convert manually:\n"
                f"  pip install optimum[onnxruntime]\n"
                f"  optimum-cli export onnx --model {model_id} --task {optimum_task} ./onnx_output/\n"
                f"  onnx pull ./onnx_output/model.onnx --name {name} --task {task}"
            )
        super().__init__(msg)


def model_id_to_dirname(model_id: str) -> str:
    return model_id.replace("/", "--")


def detect_task_from_pipeline_tag(pipeline_tag: str) -> Optional[str]:
    return config.PIPELINE_TAG_MAP.get(pipeline_tag)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def pull_from_huggingface(model_id: str, registry: ModelRegistry) -> ModelRecord:
    from huggingface_hub import model_info as hf_model_info

    info = hf_model_info(model_id)
    pipeline_tag = getattr(info, "pipeline_tag", None) or ""
    task = detect_task_from_pipeline_tag(pipeline_tag)
    if task is None:
        raise ValueError(
            f"Cannot detect task for pipeline_tag={pipeline_tag!r}. "
            f"Supported: {list(config.PIPELINE_TAG_MAP.keys())}"
        )

    # Fast pre-check: model card tags tell us if an ONNX export exists,
    # avoiding a full list_repo_files() round-trip for non-ONNX repos.
    tags = getattr(info, "tags", []) or []
    has_pytorch = any(
        t in tags for t in ("safetensors", "pytorch")
    )
    if "onnx" not in tags:
        raise NoOnnxExportError(model_id, pipeline_tag, has_pytorch, tags)

    dirname = model_id_to_dirname(model_id)
    dest_dir = config.MODELS_DIR / dirname
    dest_dir.mkdir(parents=True, exist_ok=True)

    # List all repo files to find model.onnx and any external data files
    all_files = list(list_repo_files(repo_id=model_id))

    # Prefer onnx/ subfolder layout; fall back to root
    onnx_files = [f for f in all_files if f.startswith("onnx/model.onnx")]
    if not onnx_files:
        onnx_files = [f for f in all_files if Path(f).name.startswith("model.onnx")]
    if not onnx_files:
        has_pytorch = any(f.endswith(".safetensors") or f.endswith(".bin") for f in all_files)
        raise NoOnnxExportError(model_id, pipeline_tag, has_pytorch)

    in_subdir = any(f.startswith("onnx/") for f in onnx_files)
    tmp_subdir = dest_dir / "onnx" if in_subdir else None

    for filename in onnx_files:
        local_path = hf_hub_download(
            repo_id=model_id, filename=filename, local_dir=str(dest_dir)
        )
        # Flatten onnx/ subdirectory into dest_dir
        src = Path(local_path)
        dst = dest_dir / src.name
        if src != dst:
            shutil.move(str(src), str(dst))

    if tmp_subdir and tmp_subdir.exists():
        shutil.rmtree(str(tmp_subdir), ignore_errors=True)

    # Download tokenizer files (best effort)
    for fname in ("tokenizer.json", "tokenizer_config.json"):
        try:
            hf_hub_download(repo_id=model_id, filename=fname, local_dir=str(dest_dir))
        except EntryNotFoundError:
            pass

    size_bytes = (dest_dir / "model.onnx").stat().st_size
    record = ModelRecord(
        id=model_id,
        name=model_id.split("/")[-1],
        task=task,
        source="huggingface",
        local_path=str(dest_dir),
        size_bytes=size_bytes,
        pulled_at=_now_iso(),
    )
    registry.add(record)
    return record


def import_local_model(
    onnx_path: Path,
    name: str,
    task: str,
    registry: ModelRegistry,
) -> ModelRecord:
    onnx.checker.check_model(str(onnx_path))

    dest_dir = config.MODELS_DIR / name
    dest_dir.mkdir(parents=True, exist_ok=True)

    dest_onnx = dest_dir / "model.onnx"
    shutil.copy2(str(onnx_path), str(dest_onnx))

    # Copy external data files (model.onnx_data, model.onnx_data_0, etc.)
    src_dir = onnx_path.parent
    for data_file in src_dir.glob("model.onnx*"):
        if data_file != onnx_path:
            shutil.copy2(str(data_file), str(dest_dir / data_file.name))

    # Copy tokenizer files from same directory if present
    for fname in ("tokenizer.json", "tokenizer_config.json"):
        src = src_dir / fname
        if src.exists():
            shutil.copy2(str(src), str(dest_dir / fname))

    size_bytes = dest_onnx.stat().st_size
    record = ModelRecord(
        id=name,
        name=name,
        task=task,
        source="local",
        local_path=str(dest_dir),
        size_bytes=size_bytes,
        pulled_at=_now_iso(),
    )
    registry.add(record)
    return record


def convert_and_import(
    model_id: str,
    task: str,
    optimum_task: str,
    registry: ModelRegistry,
    trust_remote_code: bool = False,
) -> ModelRecord:
    """Convert a HuggingFace model to ONNX via optimum-cli, then import it."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        cmd = [
            "optimum-cli", "export", "onnx",
            "--model", model_id,
            "--task", optimum_task,
        ]
        if trust_remote_code:
            cmd.append("--trust-remote-code")
        cmd.append(tmp_dir)

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        except FileNotFoundError:
            raise RuntimeError(
                "optimum-cli not found. Install it with: pip install optimum[onnxruntime]"
            )
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.strip() if e.stderr else "(no stderr)"
            raise RuntimeError(
                f"optimum-cli failed (exit {e.returncode}):\n{stderr}"
            )

        onnx_path = Path(tmp_dir) / "model.onnx"
        if not onnx_path.exists():
            # some exports place files in a subfolder
            candidates = list(Path(tmp_dir).glob("**/model.onnx"))
            if not candidates:
                raise RuntimeError(
                    f"optimum-cli ran but produced no model.onnx in {tmp_dir}"
                )
            onnx_path = candidates[0]

        # Use dirname as the local name so slashes don't create subdirectories,
        # then fix the registry id back to the original model_id.
        dirname = model_id_to_dirname(model_id)
        record = import_local_model(
            onnx_path=onnx_path,
            name=dirname,
            task=task,
            registry=registry,
        )
        # Re-register under the canonical HuggingFace model_id
        registry.delete(dirname)
        record.id = model_id
        record.name = model_id.split("/")[-1]
        record.source = "huggingface"
        registry.add(record)
        return record
