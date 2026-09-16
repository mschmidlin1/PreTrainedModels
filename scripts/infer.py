import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv

from pretrained_models.infer import run_inference
from pretrained_models.registry import MODELS


def main() -> None:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(
        description="Run text-to-image inference from a local models/ snapshot."
    )
    parser.add_argument("--model", required=True, choices=list(MODELS))
    parser.add_argument("--prompt", required=True)
    parser.add_argument(
        "--output",
        default="output.png",
        help="PNG path. Default: output.png in the current directory.",
    )
    parser.add_argument("--lora", help="Optional path to a LoRA adapter (.safetensors).")
    parser.add_argument("--steps", type=int, help="Override the model's default step count.")
    parser.add_argument("--guidance", type=float, help="Override the model's default guidance.")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    image = run_inference(
        name=args.model,
        prompt=args.prompt,
        lora_path=args.lora,
        steps=args.steps,
        guidance=args.guidance,
        height=args.height,
        width=args.width,
        seed=args.seed,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    print(f"Wrote {output.resolve()}")


if __name__ == "__main__":
    main()
