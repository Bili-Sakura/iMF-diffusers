import argparse
import json
import sys
from pathlib import Path

import torch

try:
    from src.diffusers.models.transformers.transformer_imf import IMF_PRESET_CONFIGS, IMFTransformer2DModel
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from src.diffusers.models.transformers.transformer_imf import IMF_PRESET_CONFIGS, IMFTransformer2DModel


def get_args():
    parser = argparse.ArgumentParser(description="Convert legacy iMF PyTorch checkpoint to Diffusers layout.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to legacy checkpoint (.pth).")
    parser.add_argument("--output_dir", type=str, required=True, help="Output Diffusers model directory.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        choices=list(IMF_PRESET_CONFIGS.keys()),
        help="Model preset. Inferred from checkpoint args when omitted.",
    )
    parser.add_argument(
        "--safe_serialization",
        action="store_true",
        help="Save weights with safetensors.",
    )
    return parser.parse_args()


def main():
    args = get_args()
    model, metadata = IMFTransformer2DModel.from_imf_checkpoint(
        args.checkpoint_path,
        model_type=args.model_type,
        map_location="cpu",
        strict=True,
        eval_mode=True,
    )

    output_dir = Path(args.output_dir)
    transformer_dir = output_dir / "transformer"
    transformer_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(transformer_dir), safe_serialization=args.safe_serialization)

    metadata_path = output_dir / "conversion_metadata.json"
    metadata["imf_args"] = {
        "model_type": metadata.get("model_type"),
        "model_str": metadata.get("model_type"),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    print(f"Saved Diffusers model to: {output_dir}")


if __name__ == "__main__":
    main()
