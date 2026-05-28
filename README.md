# iMF Diffusers

PyTorch [Improved Mean Flows (iMF)](https://arxiv.org/abs/2512.02012) packaged in a native [Diffusers](https://github.com/huggingface/diffusers) layout, following the same migration pattern as [JiT-diffusers](https://github.com/Bili-Sakura/JiT-diffusers).

This repository no longer depends on the original JAX training codebase. All model, scheduler, and pipeline code lives under `src/diffusers`.

## Package layout

- `src/diffusers/models/transformers/transformer_imf.py` — `IMFTransformer2DModel` (`ModelMixin` / `ConfigMixin`) with dual u/v heads.
- `src/diffusers/schedulers/scheduling_imf.py` — `IMFScheduler` for mean-flow updates from `t=1` to `t=0`.
- `src/diffusers/pipelines/imf/pipeline_imf.py` — `IMFPipeline` with classifier-free guidance and latent sampling.
- `scripts/convert_imf_to_diffusers.py` — convert legacy PyTorch checkpoints to Diffusers directories.
- `scripts/convert_diffusers_to_imf.py` — convert Diffusers models back to legacy checkpoint format.
- `scripts/sample_imf.py` — minimal latent sampling script.

## Install

```bash
conda env create -f environment.yaml
conda activate imf-diffusers
```

Or with pip:

```bash
pip install "diffusers>=0.35.1" "torch>=2.5" "accelerate" "safetensors"
```

## Convert a checkpoint

Legacy PyTorch inference checkpoints (from the upstream [torch branch](https://github.com/Lyy-iiis/imeanflow/tree/torch)) use `net.*` parameter names and can be converted as follows:

```bash
python scripts/convert_imf_to_diffusers.py \
  --checkpoint_path path/to/checkpoint.pth \
  --output_dir imf-diffusers \
  --model_type "iMF-B/2" \
  --safe_serialization
```

Supported presets: `iMF-B/2`, `iMF-M/2`, `iMF-L/2`, `iMF-XL/2`.

## Convert back to legacy format

```bash
python scripts/convert_diffusers_to_imf.py \
  --model_path imf-diffusers \
  --output_path checkpoint-converted.pth
```

## Sample latents

```bash
python scripts/sample_imf.py \
  --model imf-diffusers \
  --class-label 207 \
  --num-inference-steps 1 \
  --guidance-scale 2.7 \
  --output sample.pt
```

Decode latents with your VAE (for example Stable Diffusion VAE on 4-channel 32×32 latents).

## Use in Python

```python
from src.diffusers import IMFPipeline

pipeline = IMFPipeline.from_pretrained("imf-diffusers")
pipeline = pipeline.to("cuda")

output = pipeline(
    class_labels=207,
    num_inference_steps=1,
    guidance_scale=2.7,
    guidance_interval_start=0.1,
    guidance_interval_end=0.9,
    output_type="latent",
)
latents = output.images
```

## Upstreaming to Hugging Face Diffusers

To upstream these components, copy files under `src/diffusers` into matching paths in `huggingface/diffusers` and register lazy imports there.

## Pre-trained checkpoints

| Model | Inference checkpoint | FID (paper) |
|-------|---------------------|-------------|
| iMF-B/2 | [download](https://huggingface.co/Lyy0725/iMF/blob/main/iMF-B-2.zip) | 3.39 |
| iMF-M/2 | [download](https://huggingface.co/Lyy0725/iMF/blob/main/iMF-M-2.zip) | 2.27 |
| iMF-L/2 | [download](https://huggingface.co/Lyy0725/iMF/blob/main/iMF-L-2.zip) | 1.86 |
| iMF-XL/2 | [download](https://huggingface.co/Lyy0725/iMF/blob/main/iMF-XL-2.zip) | 1.72 |

## License

MIT — see [LICENSE](./LICENSE).

## Citation

```bibtex
@article{imeanflow,
  title={Improved Mean Flows: On the Challenges of Fastforward Generative Models},
  author={Geng, Zhengyang and Lu, Yiyang and Wu, Zongze and Shechtman, Eli and Kolter, J Zico and He, Kaiming},
  journal={arXiv preprint arXiv:2512.02012},
  year={2025}
}
```
