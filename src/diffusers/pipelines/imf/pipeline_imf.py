from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.utils.torch_utils import randn_tensor

from ...models.transformers.transformer_imf import IMFTransformer2DModel
from ...schedulers.scheduling_imf import IMFScheduler


class IMFPipeline(DiffusionPipeline):
    model_cpu_offload_seq = "transformer"

    def __init__(
        self,
        transformer: IMFTransformer2DModel,
        scheduler: IMFScheduler | None = None,
        id2label: Optional[Dict[Union[int, str], str]] = None,
    ):
        super().__init__()
        self.register_modules(transformer=transformer, scheduler=scheduler or IMFScheduler())
        self._id2label = self._normalize_id2label(id2label)
        self.labels = self._build_label2id(self._id2label)
        self._labels_loaded_from_model_index = bool(self._id2label)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        model_kwargs = dict(kwargs)
        transformer_subfolder = model_kwargs.pop("transformer_subfolder", None)
        scheduler_subfolder = model_kwargs.pop("scheduler_subfolder", None)
        scheduler_kwargs = model_kwargs.pop("scheduler_kwargs", {})
        base_path = Path(pretrained_model_name_or_path)

        if transformer_subfolder is None and (base_path / "transformer").exists():
            transformer_subfolder = "transformer"
        if scheduler_subfolder is None and (base_path / "scheduler").exists():
            scheduler_subfolder = "scheduler"

        try:
            return super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        except Exception:
            if transformer_subfolder is not None:
                transformer_path = str(base_path / transformer_subfolder)
            else:
                transformer_path = pretrained_model_name_or_path

            transformer = IMFTransformer2DModel.from_pretrained(transformer_path, **model_kwargs)
            try:
                scheduler = IMFScheduler.from_pretrained(
                    pretrained_model_name_or_path,
                    subfolder=scheduler_subfolder,
                    **scheduler_kwargs,
                )
            except Exception:
                scheduler = IMFScheduler(**scheduler_kwargs)

            id2label = cls._read_id2label_from_model_index(str(base_path))
            return cls(transformer=transformer, scheduler=scheduler, id2label=id2label)

    def _ensure_labels_loaded(self) -> None:
        if self._labels_loaded_from_model_index:
            return
        loaded = self._read_id2label_from_model_index(getattr(self.config, "_name_or_path", None))
        if loaded:
            self._id2label = loaded
            self.labels = self._build_label2id(self._id2label)
        self._labels_loaded_from_model_index = True

    @staticmethod
    def _normalize_id2label(id2label: Optional[Dict[Union[int, str], str]]) -> Dict[int, str]:
        if not id2label:
            return {}
        return {int(key): value for key, value in id2label.items()}

    @staticmethod
    def _read_id2label_from_model_index(variant_path: Optional[str]) -> Dict[int, str]:
        if not variant_path:
            return {}
        model_index_path = Path(variant_path).resolve() / "model_index.json"
        if not model_index_path.exists():
            return {}
        raw = json.loads(model_index_path.read_text(encoding="utf-8"))
        id2label = raw.get("id2label")
        if not isinstance(id2label, dict):
            return {}
        return {int(key): value for key, value in id2label.items()}

    @staticmethod
    def _build_label2id(id2label: Dict[int, str]) -> Dict[str, int]:
        label2id: Dict[str, int] = {}
        for class_id, value in id2label.items():
            for synonym in value.split(","):
                synonym = synonym.strip()
                if synonym:
                    label2id[synonym] = int(class_id)
        return dict(sorted(label2id.items()))

    @property
    def id2label(self) -> Dict[int, str]:
        self._ensure_labels_loaded()
        return self._id2label

    def get_label_ids(self, label: Union[str, List[str]]) -> List[int]:
        self._ensure_labels_loaded()
        if not self.labels:
            raise ValueError("No labels loaded. Ensure `id2label` exists in model_index.json.")
        if isinstance(label, str):
            label = [label]
        missing = [item for item in label if item not in self.labels]
        if missing:
            preview = ", ".join(list(self.labels.keys())[:8])
            raise ValueError(f"Unknown label(s): {missing}. Example valid labels: {preview}, ...")
        return [self.labels[item] for item in label]

    def _normalize_class_labels(self, class_labels: Union[int, str, List[Union[int, str]]]) -> List[int]:
        if isinstance(class_labels, int):
            return [class_labels]
        if isinstance(class_labels, str):
            return self.get_label_ids(class_labels)
        if class_labels and isinstance(class_labels[0], str):
            return self.get_label_ids(class_labels)
        return list(class_labels)

    def _predict_velocity_u(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        time_gap: torch.Tensor,
        class_labels: torch.Tensor,
        class_null: torch.Tensor,
        guidance_scale: float,
        guidance_interval_start: float,
        guidance_interval_end: float,
        do_classifier_free_guidance: bool,
    ) -> torch.Tensor:
        if do_classifier_free_guidance:
            latents_in = torch.cat([latents, latents], dim=0)
            labels = torch.cat([class_labels, class_null], dim=0)
            omega = torch.tensor([guidance_scale, 1.0], device=latents.device, dtype=latents.dtype)
            t_min = torch.tensor([guidance_interval_start, 0.0], device=latents.device, dtype=latents.dtype)
            t_max = torch.tensor([guidance_interval_end, 1.0], device=latents.device, dtype=latents.dtype)
            batch = latents.shape[0]
            timestep_in = timestep.reshape(1).repeat(2 * batch)
            time_gap_in = time_gap.reshape(1).repeat(2 * batch)
            omega = omega.repeat(batch)
            t_min = t_min.repeat(batch)
            t_max = t_max.repeat(batch)
        else:
            latents_in = latents
            labels = class_labels
            batch = latents.shape[0]
            timestep_in = timestep.reshape(1).repeat(batch)
            time_gap_in = time_gap.reshape(1).repeat(batch)
            omega = torch.full((batch,), guidance_scale, device=latents.device, dtype=latents.dtype)
            t_min = torch.full((batch,), guidance_interval_start, device=latents.device, dtype=latents.dtype)
            t_max = torch.full((batch,), guidance_interval_end, device=latents.device, dtype=latents.dtype)

        outputs = self.transformer(
            sample=latents_in,
            timestep=timestep_in,
            class_labels=labels,
            time_gap=time_gap_in,
            guidance_scale=omega,
            guidance_interval_start=t_min,
            guidance_interval_end=t_max,
            return_dict=True,
        )
        velocity_u = outputs.velocity_u

        if not do_classifier_free_guidance:
            return velocity_u

        u_cond, u_uncond = velocity_u.chunk(2, dim=0)
        return u_uncond + guidance_scale * (u_cond - u_uncond)

    @torch.inference_mode()
    def __call__(
        self,
        class_labels: Union[int, str, List[Union[int, str]]],
        num_inference_steps: int = 1,
        guidance_scale: float = 2.7,
        guidance_interval_start: float = 0.1,
        guidance_interval_end: float = 0.9,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pt",
        return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:
        if output_type not in {"pil", "np", "pt", "latent"}:
            raise ValueError("output_type must be one of: 'pil', 'np', 'pt', 'latent'.")

        class_label_ids = self._normalize_class_labels(class_labels)
        do_classifier_free_guidance = guidance_scale > 1.0
        batch_size = len(class_label_ids)

        image_size = int(self.transformer.config.sample_size)
        channels = int(self.transformer.config.in_channels)
        null_class_val = int(getattr(self.transformer.config, "num_classes", 1000))

        if latents is None:
            latents = randn_tensor(
                shape=(batch_size, channels, image_size, image_size),
                generator=generator,
                device=self._execution_device,
                dtype=self.transformer.dtype,
            )

        class_labels_t = torch.tensor(class_label_ids, device=latents.device, dtype=torch.long).reshape(-1)
        class_labels_t = class_labels_t.clamp(0, null_class_val - 1)
        class_null = torch.full_like(class_labels_t, null_class_val)

        self.scheduler.set_timesteps(num_inference_steps, device=latents.device)
        timesteps = self.scheduler.timesteps

        for i in self.progress_bar(range(num_inference_steps)):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            time_gap = t - t_next
            velocity_u = self._predict_velocity_u(
                latents,
                t,
                time_gap,
                class_labels_t,
                class_null,
                guidance_scale,
                guidance_interval_start,
                guidance_interval_end,
                do_classifier_free_guidance,
            )
            latents = self.scheduler.step(velocity_u, t, latents).prev_sample

        if output_type == "latent":
            images = latents
        else:
            images_pt = latents.float().clamp(-4, 4)
            if output_type == "pt":
                images = images_pt
            elif output_type == "np":
                images = images_pt.cpu().permute(0, 2, 3, 1).numpy()
            else:
                images = self.numpy_to_pil(images_pt.cpu().permute(0, 2, 3, 1).numpy())

        self.maybe_free_model_hooks()

        if not return_dict:
            return (images,)
        return ImagePipelineOutput(images=images)


IMFPipelineOutput = ImagePipelineOutput
