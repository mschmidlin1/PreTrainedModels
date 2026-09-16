from pathlib import Path

import torch
from diffusers import Flux2KleinPipeline, FluxPipeline
from PIL import Image

from pretrained_models.registry import MODELS, local_path

PIPELINES = {
    "flux": FluxPipeline,
    "flux2_klein": Flux2KleinPipeline,
}


def _dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def run_inference(
    name: str,
    prompt: str,
    lora_path: str | Path | None = None,
    steps: int | None = None,
    guidance: float | None = None,
    height: int = 1024,
    width: int = 1024,
    seed: int = 0,
) -> Image.Image:
    if name not in MODELS:
        known = ", ".join(MODELS)
        raise KeyError(f"Unknown model {name!r}. Known models: {known}")

    spec = MODELS[name]
    model_dir = local_path(name)
    if not (model_dir / "model_index.json").is_file():
        raise FileNotFoundError(
            f"No local snapshot at {model_dir}. "
            f"Download it first: python scripts/download_models.py --model {name}"
        )

    pipe_cls = PIPELINES[spec.pipeline]
    pipe = pipe_cls.from_pretrained(model_dir, torch_dtype=_dtype())
    if torch.cuda.is_available():
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cpu")

    if lora_path:
        pipe.load_lora_weights(str(lora_path))

    generator = torch.Generator(device="cpu").manual_seed(seed)
    image = pipe(
        prompt,
        num_inference_steps=spec.default_steps if steps is None else steps,
        guidance_scale=spec.default_guidance if guidance is None else guidance,
        height=height,
        width=width,
        generator=generator,
    ).images[0]
    return image
