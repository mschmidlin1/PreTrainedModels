import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv

from pretrained_models.download import download_all, download_model
from pretrained_models.registry import MODELS


def main() -> None:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(
        description="Download Diffusers snapshots of the catalog models from Hugging Face."
    )
    parser.add_argument(
        "--model",
        choices=list(MODELS),
        help="Download one model. Default: download every model in the catalog.",
    )
    args = parser.parse_args()

    if args.model:
        dest = download_model(args.model)
        print(f"Downloaded {args.model} -> {dest}")
        return

    for name, dest in download_all().items():
        print(f"Downloaded {name} -> {dest}")


if __name__ == "__main__":
    main()
