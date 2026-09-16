import os
from pathlib import Path

from huggingface_hub import snapshot_download

from pretrained_models.registry import MODELS, local_path


def _token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def download_model(name: str, token: str | None = None) -> Path:
    if name not in MODELS:
        known = ", ".join(MODELS)
        raise KeyError(f"Unknown model {name!r}. Known models: {known}")

    spec = MODELS[name]
    dest = local_path(name)
    dest.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=spec.hf_id,
        local_dir=dest,
        token=token if token is not None else _token(),
        ignore_patterns=list(spec.ignore_patterns),
    )
    return dest


def download_all(token: str | None = None) -> dict[str, Path]:
    return {name: download_model(name, token=token) for name in MODELS}
