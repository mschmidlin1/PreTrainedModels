# PreTrainedModels

A local catalog of Apache-2.0 licensed, open-weight image models. Download the trained parameters into `models/`, generate images from those local files, and later refine a checkpoint with LoRA.

This repo is glue: download + inference. The neural-net class definitions come from libraries installed into `.venv`. The trained numbers come from Hugging Face and land in `models/`.

## Structure vs weights

A runnable ML model is still two things:

1. **Structure** — the `nn.Module` graph: which layers exist, how they connect, what `forward()` does.
2. **Weights** — the trained parameter tensors that fill those layers.

You need both on disk. They are stored in **two different places**.

### Structure lives in `.venv`

The Python class definitions are installed by `pip` into the virtualenv. They are local on this machine after setup. They are not checked into git, because `.venv/` is gitignored (normal Python practice).

On Windows, after `pip install -r requirements.txt`:

| Piece | Class | File on disk |
| --- | --- | --- |
| FLUX.1 [schnell] denoiser | `FluxTransformer2DModel` | `.venv/Lib/site-packages/diffusers/models/transformers/transformer_flux.py` |
| FLUX.2 [klein] denoiser | `Flux2Transformer2DModel` | `.venv/Lib/site-packages/diffusers/models/transformers/transformer_flux2.py` |
| Schnell text-to-image stack | `FluxPipeline` | `.venv/Lib/site-packages/diffusers/pipelines/flux/pipeline_flux.py` |
| Klein text-to-image stack | `Flux2KleinPipeline` | `.venv/Lib/site-packages/diffusers/pipelines/flux2/pipeline_flux2_klein.py` |
| Text encoders (CLIP, T5, Qwen3) | various | `.venv/Lib/site-packages/transformers/` |

**Diffusers** is the library that owns the Flux `nn.Module` classes and the generate loop. **Transformers** is the library that owns the text-encoder classes.

`models/<name>/transformer/config.json` is **not** the structure. It is a hyperparameter recipe: a class name plus sizes (`num_layers`, head count, hidden dim). Diffusers reads it, constructs the class from site-packages, then loads weights into that object.

### Weights live in `models/`

After you run the download script, trained parameters sit under `models/` as **`.safetensors`** files.

**Safetensors** is the current default weight format on Hugging Face. Same job as a classic `torch.save(...)` `.pth` / `.pt` dump: a named collection of tensors. The official Flux checkpoints do not ship `.pth` files.

These catalogs are **pipelines**, not a single `nn.Module`. A Hub “model” here is three networks wired together:

- **Transformer** — the denoiser that generates latents
- **Text encoder(s)** — turn the prompt into embeddings
- **VAE** — decode latents to pixels

**FLUX.1-schnell** (12B, distilled, fast):

- Transformer weights: `models/FLUX.1-schnell/transformer/diffusion_pytorch_model-00001-of-00003.safetensors` (and shards `00002`, `00003`) plus `diffusion_pytorch_model.safetensors.index.json`
- CLIP text encoder: `models/FLUX.1-schnell/text_encoder/model.safetensors`
- T5 text encoder: `models/FLUX.1-schnell/text_encoder_2/model-00001-of-00002.safetensors` and `model-00002-of-00002.safetensors`
- VAE: `models/FLUX.1-schnell/vae/diffusion_pytorch_model.safetensors`
- Recipe files: `models/FLUX.1-schnell/model_index.json`, `transformer/config.json`

**FLUX.2-klein-4B** and **FLUX.2-klein-base-4B** share the same 4B structure (`Flux2Transformer2DModel`). Only the trained numbers and default sampler settings differ. Base is undistilled and is the one to fine-tune.

For `NAME` in `FLUX.2-klein-4B` or `FLUX.2-klein-base-4B`:

- Transformer weights: `models/NAME/transformer/diffusion_pytorch_model.safetensors`
- Qwen3 text encoder: `models/NAME/text_encoder/model-00001-of-00002.safetensors` and `model-00002-of-00002.safetensors`
- VAE: `models/NAME/vae/diffusion_pytorch_model.safetensors`
- Recipe files: `models/NAME/model_index.json`, `transformer/config.json`

### How load maps onto the old pattern

Classic PyTorch:

```python
model = MyNet()
model.load_state_dict(torch.load("weights.pth"))
```

Current Diffusers:

```python
from diffusers import Flux2KleinPipeline
pipe = Flux2KleinPipeline.from_pretrained("models/FLUX.2-klein-4B")
```

**`from_pretrained`** means: build the modules from config, then load weights from this folder. Concretely it:

1. Reads `model_index.json` — which pipeline class to assemble
2. Reads each subfolder’s `config.json` — constructor kwargs / layer sizes
3. Instantiates those `nn.Module` classes from `.venv/.../site-packages`
4. Loads the `.safetensors` into those modules

No Hub access is required at generate time if the snapshot is already in `models/`.

## Catalog

All three checkpoints are Apache-2.0. This repo’s own code is also Apache-2.0 (see `LICENSE`). The model licenses still apply to the weights.

| Local name | Hugging Face repo | Notes |
| --- | --- | --- |
| `FLUX.1-schnell` | [black-forest-labs/FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell) | Gated: accept the Hub terms, then use `HF_TOKEN`. ~12B, 4-step inference. |
| `FLUX.2-klein-4B` | [black-forest-labs/FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) | Distilled 4B, 4-step, ~13GB VRAM. |
| `FLUX.2-klein-base-4B` | [black-forest-labs/FLUX.2-klein-base-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B) | Same structure as klein 4B; undistilled; prefer this for LoRA / fine-tuning. |

The **Hugging Face Hub** is the registry we download from. A **snapshot** is a full copy of one Hub repo on disk (`models/<name>/`). FLUX.1-schnell is **gated**: the Hub requires you to accept terms in the browser; `HF_TOKEN` is just the login.

A **LoRA** is a small extra weight file trained later. Structure stays the same; you add low-rank adapters on top of the base tensors.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Copy `.example.env` to `.env` if needed and set `HF_TOKEN` to a Hugging Face token. For schnell, also open [FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell) while logged in and accept the license.

`diffusers>=0.37.0` is required so `Flux2KleinPipeline` exists.

## Download weights

Expect tens of gigabytes. Schnell’s Hub repo is ~58 GB before we skip the duplicate single-file weights; each klein snapshot is on the order of ~24 GB. The script skips ComfyUI-style duplicates (`flux1-schnell.safetensors`, `flux-2-klein-4b.safetensors`, …) and keeps the Diffusers folder layout.

```powershell
python scripts/download_models.py
python scripts/download_models.py --model FLUX.2-klein-base-4B
```

Weights are gitignored. Do not commit `models/`.

## Inference

Loads **from `models/`**, not from the Hub:

```powershell
python scripts/infer.py --model FLUX.1-schnell --prompt "a cat holding a sign that says hello world"
python scripts/infer.py --model FLUX.2-klein-4B --prompt "a cat holding a sign that says hello world"
python scripts/infer.py --model FLUX.2-klein-base-4B --prompt "a cat holding a sign that says hello world" --output klein.png
```

Optional LoRA path (no `loras/` folder in this repo):

```powershell
python scripts/infer.py --model FLUX.2-klein-base-4B --prompt "..." --lora path\to\adapter.safetensors --output klein.png
```

`--output` defaults to `output.png` in the current directory. CPU offload is used when CUDA is available. Schnell is 12B and still wants a lot of RAM/VRAM. Klein 4B is meant to fit around 13GB VRAM.

Defaults: schnell uses 4 steps and guidance `0.0`; distilled klein uses 4 steps and guidance `1.0`; klein base uses 50 steps and guidance `4.0`.

## LoRA / fine-tuning later

Use **`FLUX.2-klein-base-4B`** as the training base. This repo does not include a trainer. Hugging Face’s DreamBooth example is [`train_dreambooth_lora_flux2_klein.py`](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth_lora_flux2_klein.py); Black Forest Labs also has a [klein LoRA guide](https://huggingface.co/blog/black-forest-labs/flux-2-klein-lora).
