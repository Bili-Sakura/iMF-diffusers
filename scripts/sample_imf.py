import argparse
import sys
from pathlib import Path

import torch

try:
    from src.diffusers import IMFPipeline
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from src.diffusers import IMFPipeline


def get_args():
    parser = argparse.ArgumentParser(description="Sample images with an iMF Diffusers pipeline.")
    parser.add_argument("--model", type=str, required=True, help="Path to Diffusers model directory.")
    parser.add_argument("--class-label", type=int, default=207, help="ImageNet class id.")
    parser.add_argument("--num-inference-steps", type=int, default=1, help="Number of sampling steps.")
    parser.add_argument("--guidance-scale", type=float, default=2.7, help="CFG scale (omega).")
    parser.add_argument("--guidance-interval-start", type=float, default=0.1, help="CFG interval start.")
    parser.add_argument("--guidance-interval-end", type=float, default=0.9, help="CFG interval end.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--output", type=str, default="sample.pt", help="Output path for latent tensor.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = get_args()
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    pipeline = IMFPipeline.from_pretrained(args.model)
    pipeline = pipeline.to(args.device)

    output = pipeline(
        class_labels=args.class_label,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        guidance_interval_start=args.guidance_interval_start,
        guidance_interval_end=args.guidance_interval_end,
        generator=generator,
        output_type="latent",
    )
    latents = output.images[0] if isinstance(output.images, list) else output.images
    torch.save(latents.cpu(), args.output)
    print(f"Saved latent sample to: {args.output}")


if __name__ == "__main__":
    main()
