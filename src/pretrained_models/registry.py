from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    hf_id: str
    license: str
    pipeline: str
    default_steps: int
    default_guidance: float
    ignore_patterns: tuple[str, ...]


MODELS = {
    "FLUX.1-schnell": ModelSpec(
        name="FLUX.1-schnell",
        hf_id="black-forest-labs/FLUX.1-schnell",
        license="Apache-2.0",
        pipeline="flux",
        default_steps=4,
        default_guidance=0.0,
        ignore_patterns=("flux1-schnell.safetensors", "ae.safetensors"),
    ),
    "FLUX.2-klein-4B": ModelSpec(
        name="FLUX.2-klein-4B",
        hf_id="black-forest-labs/FLUX.2-klein-4B",
        license="Apache-2.0",
        pipeline="flux2_klein",
        default_steps=4,
        default_guidance=1.0,
        ignore_patterns=("flux-2-klein-4b.safetensors",),
    ),
    "FLUX.2-klein-base-4B": ModelSpec(
        name="FLUX.2-klein-base-4B",
        hf_id="black-forest-labs/FLUX.2-klein-base-4B",
        license="Apache-2.0",
        pipeline="flux2_klein",
        default_steps=50,
        default_guidance=4.0,
        ignore_patterns=("flux-2-klein-base-4b.safetensors",),
    ),
}


def local_path(name: str) -> Path:
    if name not in MODELS:
        known = ", ".join(MODELS)
        raise KeyError(f"Unknown model {name!r}. Known models: {known}")
    return MODELS_DIR / name
