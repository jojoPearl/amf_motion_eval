#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import gc
import json
import math
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision.io import read_video


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WAN_REPO = PROJECT_ROOT / "Wan2.2"

if str(DEFAULT_WAN_REPO) not in sys.path:
    sys.path.insert(0, str(DEFAULT_WAN_REPO))

import wan
from wan.configs import WAN_CONFIGS
from wan.utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from wan.utils.utils import masks_like, save_video

# Import modularized guidance components
sys.path.insert(0, str(PROJECT_ROOT / "amf_motion_eval"))
from guidance_utils import (
    ProcessorMode,
    register_wan_amf_processors,
    WanFeatureStorage,
    compute_wan_motion_flow,
    compute_wan_motion_flow_loss,
)


def parse_size(size):
    """Parse size string in format 'WIDTH*HEIGHT' or 'WIDTHxHEIGHT'.
    
    Args:
        size: Size string (e.g., '1280*704') or tuple
        
    Returns:
        Tuple (width, height)
        
    Raises:
        ValueError: If format is invalid
    """
    if isinstance(size, tuple):
        return size
    try:
        parts = size.lower().replace("x", "*").split("*")
        if len(parts) != 2:
            raise ValueError("Size must contain exactly one '*' or 'x' separator")
        width, height = int(parts[0]), int(parts[1])
        if width <= 0 or height <= 0:
            raise ValueError("Width and height must be positive")
        return width, height
    except (ValueError, AttributeError) as e:
        raise ValueError(
            f"Invalid size format '{size}'. Expected 'WIDTH*HEIGHT' or "
            f"'WIDTHxHEIGHT', got error: {e}"
        )


def ceil_div(a, b):
    return (int(a) + int(b) - 1) // int(b)


def sorted_frame_paths(frame_dir):
    paths = []
    for suffix in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
        paths.extend(Path(frame_dir).glob(suffix))

    def sort_key(path):
        digits = "".join(ch for ch in path.stem if ch.isdigit())
        return (int(digits) if digits else 0, path.name)

    return sorted(paths, key=sort_key)


def load_reference_video(video_path, frame_num, size, device):
    """Load reference video and convert to normalized latent space.
    
    Args:
        video_path: Path to video file or directory of frames
        frame_num: Number of frames to load
        size: (width, height)
        device: torch device
        
    Returns:
        Video tensor [C, F, H, W] normalized to [-1, 1]
    """
    width, height = size

    if os.path.isdir(video_path):
        frames = []
        for path in sorted_frame_paths(video_path)[:frame_num]:
            img = Image.open(path).convert("RGB").resize((width, height))
            frame = torch.from_numpy(np.array(img)).float() / 255.0
            frames.append(frame)
        if len(frames) == 0:
            raise FileNotFoundError(f"No frames found in {video_path}")

        video = torch.stack(frames, dim=0)
    else:
        video = read_video(video_path, pts_unit="sec")[0]
        if video.numel() == 0:
            raise ValueError(f"Could not read video: {video_path}")
        video = video[:frame_num].float() / 255.0

    if video.shape[0] < frame_num:
        raise ValueError(
            f"Reference has {video.shape[0]} frames, but frame_num={frame_num}"
        )

    video = video.permute(0, 3, 1, 2)
    video = F.interpolate(
        video,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    video = video.mul_(2.0).sub_(1.0)

    return video.permute(1, 0, 2, 3).contiguous().to(device)





# ============================================================
# Main extractor / generator
# ============================================================

class WanAMFExtractor:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(f"cuda:{config.device_id}")
        self.guidance_logs = []
        self.latest_kv_snapshot = {}
        self.resolved_seed = None

        if config.frame_num <= 0 or (config.frame_num - 1) % 4 != 0:
            raise ValueError("Wan video frame_num must be 4n+1, e.g. 81 or 121")

        print("Loading Wan video model")
        cfg = WAN_CONFIGS[config.task]

        pipe_cls = wan.WanT2V if config.task.startswith("t2v") else wan.WanTI2V

        self.wan_pipe = pipe_cls(
            config=cfg,
            checkpoint_dir=config.ckpt_dir,
            device_id=config.device_id,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_sp=False,
            t5_cpu=config.t5_cpu,
            init_on_cpu=True,
            convert_model_dtype=config.convert_model_dtype,
        )

        self.dtype = getattr(self.wan_pipe, "param_dtype", cfg.param_dtype)

        self.model = self.wan_pipe.model
        self.model.eval().requires_grad_(False)

        self.vae = self.wan_pipe.vae
        self.text_encoder = self.wan_pipe.text_encoder

        self.patch_size = self.wan_pipe.patch_size
        self.vae_stride = self.wan_pipe.vae_stride

        self.size = parse_size(config.size)

        full_patch_w = self.vae_stride[2] * self.patch_size[2]
        full_patch_h = self.vae_stride[1] * self.patch_size[1]

        if self.size[0] % full_patch_w != 0 or self.size[1] % full_patch_h != 0:
            print(
                "[Warning] input size is not divisible by vae_stride * patch_size. "
                "This is allowed. Wan token grid will use ceil-style patch grid from mask/grid_sizes. "
                f"size={self.size}, vae_stride={self.vae_stride}, patch_size={self.patch_size}, "
                f"full_patch=({full_patch_w}, {full_patch_h})"
            )

        self.output_path = config.output_path
        os.makedirs(self.output_path, exist_ok=True)

        print("Wan video model loaded")

        # Initialize feature storage
        self.feature_storage = WanFeatureStorage(self.output_path)

        # Register attention processors for AMF capture
        self.attn_processors = register_wan_amf_processors(
            self.model.blocks,
            self.config.guidance_blocks,
        )

        # Apply memory optimizations
        self._apply_memory_optimizations()

        print("Loading reference video latent")
        self.motion_latent = self.load_latent()

        _, self.latent_num_frames, self.latent_height, self.latent_width = self.motion_latent.shape

        self.patches_height = ceil_div(self.latent_height, self.patch_size[1])
        self.patches_width = ceil_div(self.latent_width, self.patch_size[2])

        self.seq_len = self.compute_seq_len(
            self.latent_num_frames,
            self.latent_height,
            self.latent_width,
        )

        if max(self.config.guidance_blocks) >= len(self.model.blocks):
            raise ValueError(
                f"guidance block out of range: model has {len(self.model.blocks)} blocks, "
                f"got {self.config.guidance_blocks}"
            )

        print(
            f"Wan latent grid: frames={self.latent_num_frames}, "
            f"latent={self.latent_height}x{self.latent_width}, "
            f"patch={self.patches_height}x{self.patches_width}, seq_len={self.seq_len}"
        )

    def compute_seq_len(self, latent_frames, latent_height, latent_width):
        """Compute Wan token seq_len using ceil-style spatial patch grid.

        This matches mask2[0][0][:, ::patch_h, ::patch_w].flatten().
        Example: latent_width=45, patch_w=2 -> grid_width=23.
        """
        sp_size = getattr(self.wan_pipe, "sp_size", 1)

        grid_f = int(latent_frames)
        grid_h = ceil_div(latent_height, self.patch_size[1])
        grid_w = ceil_div(latent_width, self.patch_size[2])

        tokens = grid_f * grid_h * grid_w
        return math.ceil(tokens / sp_size) * sp_size

    def _apply_memory_optimizations(self):
        """Apply memory optimization techniques for large video generation.
        
        Includes:
        - VAE tiling: Encode/decode video in spatial tiles
        - Gradient checkpointing: Trade compute for memory during backprop
        """
        # VAE tiling for large resolution videos
        try:
            if hasattr(self.vae, 'enable_tiling'):
                self.vae.enable_tiling()
                print("[Memory] Enabled VAE tiling")
        except Exception as e:
            print(f"[Warning] Could not enable VAE tiling: {e}")

        # Model gradient checkpointing
        try:
            if hasattr(self.model, 'enable_gradient_checkpointing'):
                self.model.enable_gradient_checkpointing()
                print("[Memory] Enabled gradient checkpointing")
        except Exception as e:
            print(f"[Warning] Could not enable gradient checkpointing: {e}")

    def _set_processor_capture_mode(self, with_grad=False):
        """Set specified processors to capture mode, idle others."""
        for processor in self.attn_processors.values():
            processor.set_mode(ProcessorMode.IDLE)

        for block_id in self.config.guidance_blocks:
            if block_id in self.attn_processors:
                processor = self.attn_processors[block_id]
                mode = ProcessorMode.CAPTURE_FOR_GRAD if with_grad else ProcessorMode.CAPTURE
                processor.set_mode(mode)

    def _reset_all_processors(self):
        """Reset all processors to idle state."""
        for processor in self.attn_processors.values():
            processor.set_mode(ProcessorMode.IDLE)

    @torch.no_grad()
    def encode_prompt(self):
        return self.encode_prompt_text(self.config.reference_prompt)

    @torch.no_grad()
    def encode_prompt_text(self, prompt):
        if not self.config.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([prompt], self.device)

            if self.config.offload_t5:
                self.text_encoder.model.cpu()
                torch.cuda.empty_cache()
        else:
            context = self.text_encoder([prompt], torch.device("cpu"))
            context = [u.to(self.device) for u in context]

        return context

    @torch.no_grad()
    def load_latent(self):
        video = load_reference_video(
            self.config.video_path,
            self.config.frame_num,
            self.size,
            self.device,
        )

        latent = self.vae.encode([video])[0]
        return latent.to(self.device)

    def resolve_reference_timestep(self):
        ref_t = self.config.reference_timestep

        if isinstance(ref_t, str) and ref_t.lower() in {"first", "start", "high", "auto"}:
            _, timesteps = self.build_scheduler()
            return timesteps[0].to(self.device)

        return torch.tensor(float(ref_t), device=self.device)

    @torch.no_grad()
    def load_attn_features(self):
        """Extract reference AMF."""
        self._reset_all_processors()
        self._set_processor_capture_mode(with_grad=False)

        context = self.encode_prompt()
        self.model.to(self.device)

        _, mask2 = masks_like([self.motion_latent], zero=False)

        ref_t = self.resolve_reference_timestep()
        timestep = self.make_timestep(ref_t, mask2)

        with torch.autocast(device_type="cuda", dtype=self.dtype):
            self.model(
                [self.motion_latent],
                t=timestep,
                context=context,
                seq_len=self.seq_len,
            )

        attn_features = {}

        for block_id in self.config.guidance_blocks:
            processor = self.attn_processors[block_id]

            if processor.query is None or processor.key is None:
                raise RuntimeError(f"No Q/K captured for block {block_id}")

            grid_size = tuple(int(v) for v in processor.grid_size.tolist())
            expected_seq = grid_size[0] * grid_size[1] * grid_size[2]

            if expected_seq > processor.query.shape[1]:
                raise RuntimeError(
                    f"Captured Q too short for block {block_id}: "
                    f"expected at least {expected_seq}, got {processor.query.shape[1]}"
                )

            amf = compute_wan_motion_flow(
                processor.query,
                processor.key,
                grid_size=grid_size,
                temp=self.config.motion_temp,
                argmax=self.config.argmax_motion_flow,
                head_reduce=self.config.head_reduce,
            )

            attn_features[processor.block_name] = amf.detach().cpu()

            # Use feature storage for AMF management
            config_dict = {
                "video_path": self.config.video_path,
                "prompt": self.config.reference_prompt,
                "target_prompt": self.config.target_prompt,
                "reference_timestep": self.config.reference_timestep,
                "reference_timestep_resolved": float(ref_t.detach().cpu()),
                "motion_temp": self.config.motion_temp,
                "head_reduce": self.config.head_reduce,
                "argmax_motion_flow": self.config.argmax_motion_flow,
            }

            amf_path = self.feature_storage.save_amf(
                amf=amf.detach().cpu(),
                block_id=block_id,
                block_name=processor.block_name,
                grid_size=grid_size,
                config_dict=config_dict,
            )

            print("[Saved Wan AMF]", amf_path, tuple(amf.shape))

        self._reset_all_processors()
        return attn_features

    def make_timestep(self, t, mask2):
        timestep = torch.stack([t]) if t.ndim == 0 else t

        temp_ts = (
            mask2[0][0][:, ::self.patch_size[1], ::self.patch_size[2]] * timestep
        ).flatten()

        if temp_ts.size(0) > self.seq_len:
            print(
                "[Warning] mask timestep length exceeds seq_len; expanding seq_len dynamically. "
                f"mask_len={temp_ts.size(0)}, old_seq_len={self.seq_len}"
            )
            self.seq_len = int(temp_ts.size(0))

        temp_ts = torch.cat([
            temp_ts,
            temp_ts.new_ones(self.seq_len - temp_ts.size(0)) * timestep,
        ])

        return temp_ts.unsqueeze(0)

    def build_scheduler(self):
        if self.config.sample_solver == "unipc":
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.wan_pipe.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False,
            )
            sample_scheduler.set_timesteps(
                self.config.sample_steps,
                device=self.device,
                shift=self.config.sample_shift,
            )
            return sample_scheduler, sample_scheduler.timesteps

        if self.config.sample_solver == "dpm++":
            sample_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=self.wan_pipe.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False,
            )
            sampling_sigmas = get_sampling_sigmas(
                self.config.sample_steps,
                self.config.sample_shift,
            )
            timesteps, _ = retrieve_timesteps(
                sample_scheduler,
                device=self.device,
                sigmas=sampling_sigmas,
            )
            return sample_scheduler, timesteps

        raise NotImplementedError(f"Unsupported solver: {self.config.sample_solver}")

    def should_apply_guidance(self, step_index):
        end = self.guidance_end_step()

        if step_index < self.config.guidance_start_step or step_index >= end:
            return False

        offset = step_index - self.config.guidance_start_step
        return offset % self.config.guidance_every == 0

    def guidance_end_step(self):
        if self.config.guidance_end_step >= 0:
            return self.config.guidance_end_step

        return max(
            1,
            math.ceil(self.config.sample_steps * self.config.guidance_fraction),
        )

    def guidance_lr_for_step(self, step_index):
        start = self.config.guidance_start_step
        end = self.guidance_end_step()
        denom = max(end - start - 1, 1)
        ratio = min(max((step_index - start) / denom, 0.0), 1.0)

        return (
            self.config.guidance_lr
            + ratio * (self.config.guidance_lr_end - self.config.guidance_lr)
        )

    def compute_current_amf_loss(self, latent, timestep, context, ref_attn_features):
        self._reset_all_processors()
        self._set_processor_capture_mode(with_grad=True)

        with torch.autocast(device_type="cuda", dtype=self.dtype):
            self.model(
                [latent],
                t=timestep,
                context=context,
                seq_len=self.seq_len,
            )

        total_loss = latent.new_tensor(0.0)

        for block_id in self.config.guidance_blocks:
            processor = self.attn_processors[block_id]
            ref_amf = ref_attn_features[processor.block_name]

            if self.config.ref_amf_cpu:
                ref_amf = ref_amf.to(device=latent.device, dtype=torch.float32)

            if processor.query is None or processor.key is None:
                raise RuntimeError(f"No Q/K captured for block {block_id}")

            if not processor.query.requires_grad:
                raise RuntimeError(
                    f"Captured Q for block {block_id} has no grad. "
                    "AMF guidance requires save_for_grad=True and torch.enable_grad()."
                )

            grid_size = tuple(int(v) for v in processor.grid_size.tolist())
            expected_seq = grid_size[0] * grid_size[1] * grid_size[2]

            if expected_seq > processor.query.shape[1]:
                raise RuntimeError(
                    f"Captured Q too short for block {block_id}: "
                    f"expected at least {expected_seq}, got {processor.query.shape[1]}"
                )

            total_loss = total_loss + compute_wan_motion_flow_loss(
                processor.query,
                processor.key,
                ref_amf,
                grid_size=grid_size,
                temp=self.config.motion_temp,
                threshloss=self.config.threshloss,
                head_reduce=self.config.head_reduce,
            )

        self._reset_all_processors()
        return total_loss / max(len(self.config.guidance_blocks), 1)

    def optimize_latent_with_amf(self, latent, step_index, timestep, context, ref_attn_features):
        optimized_latent = latent.detach().float().requires_grad_(True)

        lr = self.guidance_lr_for_step(step_index)

        if self.config.guidance_optimizer == "adam":
            optimizer = torch.optim.Adam([optimized_latent], lr=lr)
        elif self.config.guidance_optimizer == "sgd":
            optimizer = torch.optim.SGD([optimized_latent], lr=lr)
        else:
            raise ValueError(f"Unsupported guidance optimizer: {self.config.guidance_optimizer}")

        scaler = GradScaler(enabled=bool(self.config.use_grad_scaler))

        for inner_idx in range(self.config.guidance_inner_steps):
            optimizer.zero_grad(set_to_none=True)

            raw_loss = self.compute_current_amf_loss(
                optimized_latent,
                timestep,
                context,
                ref_attn_features,
            )

            loss = raw_loss * self.config.guidance_loss_weight

            scaler.scale(loss).backward()

            if self.config.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [optimized_latent],
                    self.config.max_grad_norm,
                )

            scaler.step(optimizer)
            scaler.update()

            self.guidance_logs.append(
                {
                    "step_index": int(step_index),
                    "inner_step": int(inner_idx),
                    "lr": float(lr),
                    "loss": float(raw_loss.detach().cpu()),
                    "weighted_loss": float(loss.detach().cpu()),
                }
            )

        return optimized_latent.detach()

    def timestep_to_sigma(self, t):
        sigma = t.float() / float(self.wan_pipe.num_train_timesteps)
        return sigma.clamp(0.0, 1.0)

    def make_reference_noisy_latent(self, t, generator):
        if self.config.disable_reference_noising:
            return self.motion_latent

        noise = torch.randn(
            self.motion_latent.shape,
            dtype=torch.float32,
            device=self.device,
            generator=generator,
        )

        sigma = self.timestep_to_sigma(t).to(device=self.device, dtype=torch.float32)

        while sigma.ndim < self.motion_latent.ndim:
            sigma = sigma.view(*sigma.shape, *([1] * (self.motion_latent.ndim - sigma.ndim)))

        return (1.0 - sigma) * self.motion_latent.float() + sigma * noise

    @torch.no_grad()
    def capture_reference_kv_for_injection(self, t, timestep, reference_context, generator):
        self._reset_all_processors()
        self._set_processor_capture_mode(with_grad=False)

        noisy_motion = self.make_reference_noisy_latent(t, generator)

        with torch.autocast(device_type="cuda", dtype=self.dtype):
            self.model(
                [noisy_motion],
                t=timestep,
                context=reference_context,
                seq_len=self.seq_len,
            )

        for block_id in self.config.guidance_blocks:
            processor = self.attn_processors[block_id]
            processor.promote_capture_to_injection()
            self.latest_kv_snapshot[block_id] = {
                "key": processor.inject_key.detach().cpu(),
                "value": processor.inject_value.detach().cpu(),
                "seq_lens": (
                    processor.inject_seq_lens.detach().cpu()
                    if processor.inject_seq_lens is not None
                    else None
                ),
            }

    def clear_all_injections(self):
        for processor in self.attn_processors.values():
            processor.set_mode(ProcessorMode.IDLE)

    def prepare_ref_attn_features(self, ref_attn_features):
        if self.config.ref_amf_cpu:
            return {
                block_name: amf.cpu().float()
                for block_name, amf in ref_attn_features.items()
            }

        dtype = torch.float16 if self.config.ref_amf_fp16 else torch.float32

        return {
            block_name: amf.to(self.device, dtype=dtype)
            for block_name, amf in ref_attn_features.items()
        }

    def generate_with_amf_guidance(self):
        # Try to load K/V snapshot for zero-shot transfer
        loaded_snapshot = self._load_kv_snapshot_from_storage()
        use_loaded_kv_snapshot = loaded_snapshot is not None
        
        if use_loaded_kv_snapshot:
            # Zero-shot K/V injection mode: skip AMF extraction
            ref_attn_features = {}  # Not used in this mode
            print("[Mode] Zero-shot K/V injection (no AMF optimization)")
            if self.config.guidance_inner_steps > 0 and self.config.guidance_loss_weight > 0:
                print(
                    "[Info] Loaded K/V snapshot skips AMF latent optimization; "
                    "set --guidance_inner_steps 0 to silence this message."
                )
        else:
            # Standard mode: extract AMF and perform optimization
            ref_attn_features = self.prepare_ref_attn_features(self.load_attn_features())
            print("[Mode] AMF extraction + latent/embedding optimization")

        target_prompt = self.config.target_prompt or self.config.prompt
        negative_prompt = self.config.negative_prompt or self.wan_pipe.sample_neg_prompt

        context = self.encode_prompt_text(target_prompt)
        context_null = self.encode_prompt_text(negative_prompt)
        reference_context = (
            self.encode_prompt()
            if self.config.enable_kv_injection and not use_loaded_kv_snapshot
            else None
        )

        seed = self.config.base_seed
        if seed < 0:
            seed = random.randint(0, sys.maxsize)
        self.resolved_seed = seed

        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        target_shape = (
            self.vae.model.z_dim,
            self.latent_num_frames,
            self.latent_height,
            self.latent_width,
        )

        noise = [
            torch.randn(
                target_shape,
                dtype=torch.float32,
                device=self.device,
                generator=seed_g,
            )
        ]

        _, mask2 = masks_like(noise, zero=False)

        sample_scheduler, timesteps = self.build_scheduler()
        latents = noise

        self.model.to(self.device)

        for step_index, t in enumerate(tqdm(timesteps, desc="Wan AMF guidance")):
            timestep = self.make_timestep(t, mask2)

            apply_guidance = self.should_apply_guidance(step_index)

            if apply_guidance:
                if (
                    not use_loaded_kv_snapshot
                    and self.config.guidance_inner_steps > 0
                    and self.config.guidance_loss_weight > 0
                ):
                    with torch.enable_grad():
                        latents = [
                            self.optimize_latent_with_amf(
                                latents[0],
                                step_index,
                                timestep,
                                context,
                                ref_attn_features,
                            )
                        ]

                if self.config.enable_kv_injection and not use_loaded_kv_snapshot:
                    self.capture_reference_kv_for_injection(
                        t=t,
                        timestep=timestep,
                        reference_context=reference_context,
                        generator=seed_g,
                    )
                elif use_loaded_kv_snapshot:
                    self._inject_loaded_kv_to_processors(loaded_snapshot)

            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=self.dtype):
                noise_pred_cond = self.model(
                    latents,
                    t=timestep,
                    context=context,
                    seq_len=self.seq_len,
                )[0]

                noise_pred_uncond = self.model(
                    latents,
                    t=timestep,
                    context=context_null,
                    seq_len=self.seq_len,
                )[0]

                noise_pred = noise_pred_uncond + self.config.sample_guide_scale * (
                    noise_pred_cond - noise_pred_uncond
                )

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latents[0].unsqueeze(0),
                    return_dict=False,
                    generator=seed_g,
                )[0]

                latents = [temp_x0.squeeze(0).float()]

            self.clear_all_injections()

        with torch.no_grad():
            video = self.vae.decode(latents)[0]

        save_dir = os.path.dirname(self.config.save_file)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        save_video(
            tensor=video[None],
            save_file=self.config.save_file,
            fps=WAN_CONFIGS[self.config.task].sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1),
        )

        print("[Saved Wan AMF-guided video]", self.config.save_file)

        log_path = self.config.save_file + ".guidance.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(self.guidance_logs, f, indent=2)

        print("[Saved Wan AMF guidance log]", log_path)

        meta_path = self.config.save_file + ".metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "task": self.config.task,
                    "ckpt_dir": self.config.ckpt_dir,
                    "video_path": self.config.video_path,
                    "prompt": self.config.prompt,
                    "target_prompt": target_prompt,
                    "reference_prompt": self.config.reference_prompt,
                    "size": self.config.size,
                    "frame_num": self.config.frame_num,
                    "base_seed": seed,
                    "sample_solver": self.config.sample_solver,
                    "sample_steps": self.config.sample_steps,
                    "sample_shift": self.config.sample_shift,
                    "sample_guide_scale": self.config.sample_guide_scale,
                    "guidance_blocks": self.config.guidance_blocks,
                    "guidance_inner_steps": self.config.guidance_inner_steps,
                    "guidance_start_step": self.config.guidance_start_step,
                    "guidance_end_step": self.config.guidance_end_step,
                    "guidance_fraction": self.config.guidance_fraction,
                    "motion_temp": self.config.motion_temp,
                    "argmax_motion_flow": self.config.argmax_motion_flow,
                    "threshloss": self.config.threshloss,
                    "reference_timestep": self.config.reference_timestep,
                    "enable_kv_injection": self.config.enable_kv_injection,
                    "load_kv_snapshot": self.config.load_kv_snapshot,
                    "ref_amf_cpu": self.config.ref_amf_cpu,
                    "ref_amf_fp16": self.config.ref_amf_fp16,
                    "seq_len": self.seq_len,
                    "latent_num_frames": self.latent_num_frames,
                    "latent_height": self.latent_height,
                    "latent_width": self.latent_width,
                    "patches_height": self.patches_height,
                    "patches_width": self.patches_width,
                },
                f,
                indent=2,
            )

        print("[Saved Wan AMF metadata]", meta_path)

        # Optionally save K/V snapshot for zero-shot reuse
        if self.config.save_kv_snapshot and self.config.enable_kv_injection:
            self._save_kv_snapshot_to_storage()

        del noise, latents, sample_scheduler
        gc.collect()
        torch.cuda.empty_cache()

        return self.config.save_file

    def _save_kv_snapshot_to_storage(self):
        """Save captured K/V from injection mode to storage for zero-shot reuse."""
        kv_dict = {}
        snapshot_metadata = {
            "video_path": self.config.video_path,
            "reference_prompt": self.config.reference_prompt,
            "target_prompt": self.config.target_prompt or self.config.prompt,
            "seed": self.resolved_seed,
            "guidance_blocks": self.config.guidance_blocks,
            "motion_temp": self.config.motion_temp,
        }

        for block_id in self.config.guidance_blocks:
            if block_id in self.latest_kv_snapshot:
                kv_dict[block_id] = self.latest_kv_snapshot[block_id]

        if kv_dict:
            snapshot_name = (
                self.config.save_kv_snapshot or
                f"kv_snapshot_{self.config.base_seed}"
            )
            
            snapshot_path = self.feature_storage.save_kv_snapshot(
                kv_dict=kv_dict,
                snapshot_name=snapshot_name,
                metadata=snapshot_metadata,
            )
            
            print(f"[Saved K/V snapshot] {snapshot_path}")
        else:
            print("[Warning] No K/V injection captured to save")

    def _load_kv_snapshot_from_storage(self):
        """Load K/V snapshot for zero-shot motion transfer.
        
        Returns:
            dict: K/V snapshot or None if not found
        """
        if not self.config.load_kv_snapshot:
            return None

        snapshot = self.feature_storage.load_kv_snapshot(
            self.config.load_kv_snapshot
        )
        
        if snapshot is None:
            print(f"[Warning] Snapshot '{self.config.load_kv_snapshot}' not found")
            return None

        print(f"[Loaded K/V snapshot] {self.config.load_kv_snapshot}")
        return snapshot

    def _inject_loaded_kv_to_processors(self, snapshot):
        """Inject loaded K/V snapshot into processors.
        
        Args:
            snapshot: K/V snapshot dict from storage
        """
        for block_id, entry in snapshot.get("kv_dict", {}).items():
            block_id = int(block_id)
            if block_id in self.attn_processors:
                if isinstance(entry, dict):
                    k = entry["key"]
                    v = entry["value"]
                    seq_lens = entry.get("seq_lens")
                else:
                    if len(entry) == 3:
                        k, v, seq_lens = entry
                    else:
                        k, v = entry
                        seq_lens = None

                processor = self.attn_processors[block_id]
                processor.set_injection(
                    k.to(self.device),
                    v.to(self.device),
                    seq_lens.to(self.device) if seq_lens is not None else None,
                )


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser("Extract Wan AMF or generate AMF-guided Wan video")

    parser.add_argument("--mode", choices=["extract", "generate"], default="generate")

    parser.add_argument(
        "--task",
        default="ti2v-5B",
        help="Wan task key in WAN_CONFIGS, e.g. ti2v-5B.",
    )

    parser.add_argument(
        "--ckpt_dir",
        default=str(DEFAULT_WAN_REPO / "Wan2.2-TI2V-5B"),
        help="Wan checkpoint directory.",
    )

    parser.add_argument("--video_path", required=True)

    parser.add_argument(
        "--prompt",
        default="",
        help="Target prompt fallback; not used for reference AMF unless --reference_prompt is set.",
    )

    parser.add_argument(
        "--reference_prompt",
        default="",
        help="Prompt used when extracting reference AMF. Defaults to empty, following DiTFlow.",
    )

    parser.add_argument(
        "--target_prompt",
        default=None,
        help="Target generation prompt. Defaults to --prompt.",
    )

    parser.add_argument("--negative_prompt", default="")

    parser.add_argument(
        "--output_path",
        default=str(PROJECT_ROOT / "amf_motion_eval" / "outputs" / "wan_amf"),
    )

    parser.add_argument(
        "--save_file",
        default=str(
            PROJECT_ROOT
            / "amf_motion_eval"
            / "outputs"
            / "wan_amf"
            / "amf_guided.mp4"
        ),
    )

    parser.add_argument("--size", default="1280*704")
    parser.add_argument("--frame_num", type=int, default=81)

    parser.add_argument("--guidance_blocks", type=int, nargs="+", default=[15])

    parser.add_argument(
        "--reference_timestep",
        default="first",
        help=(
            "Reference timestep for AMF extraction. "
            "Use 'first' for scheduler.timesteps[0], or pass a numeric timestep."
        ),
    )

    parser.add_argument("--motion_temp", type=float, default=2.0)

    parser.add_argument(
        "--head_reduce",
        choices=["logits", "attn"],
        default="logits",
        help="AMF head aggregation. 'logits' matches the public DiTFlow repo.",
    )

    parser.add_argument(
        "--soft_reference_motion_flow",
        dest="argmax_motion_flow",
        action="store_false",
        default=True,
        help="Use soft expected displacement for reference AMF ablation.",
    )

    parser.add_argument("--sample_solver", choices=["unipc", "dpm++"], default="unipc")
    parser.add_argument("--sample_steps", type=int, default=50)
    parser.add_argument("--sample_shift", type=float, default=5.0)
    parser.add_argument("--sample_guide_scale", type=float, default=5.0)
    parser.add_argument("--base_seed", type=int, default=-1)

    parser.add_argument(
        "--guidance_optimizer",
        choices=["adam", "sgd"],
        default="adam",
    )

    parser.add_argument("--guidance_lr", type=float, default=0.002)
    parser.add_argument("--guidance_lr_end", type=float, default=0.001)
    parser.add_argument("--guidance_inner_steps", type=int, default=5)
    parser.add_argument("--guidance_loss_weight", type=float, default=1.0)

    parser.add_argument("--use_grad_scaler", action="store_true")

    parser.add_argument(
        "--enable_kv_injection",
        action="store_true",
        default=False,
        help="Enable DiTFlow-style reference K/V injection. Default is disabled.",
    )

    parser.add_argument(
        "--disable_reference_noising",
        action="store_true",
        help="Do not noise reference latent before capturing K/V.",
    )

    parser.add_argument(
        "--save_kv_snapshot",
        type=str,
        default=None,
        help="Save optimized K/V to snapshot with this name for zero-shot reuse.",
    )

    parser.add_argument(
        "--load_kv_snapshot",
        type=str,
        default=None,
        help="Load K/V from snapshot for zero-shot motion transfer (skips reference processing).",
    )

    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--guidance_start_step", type=int, default=0)

    parser.add_argument(
        "--guidance_end_step",
        type=int,
        default=-1,
        help="Exclusive end index for guidance. -1 means first --guidance_fraction of denoising steps.",
    )

    parser.add_argument("--guidance_fraction", type=float, default=0.2)
    parser.add_argument("--guidance_every", type=int, default=1)

    parser.add_argument("--threshloss", action="store_true", default=True)

    parser.add_argument(
        "--no_threshloss",
        dest="threshloss",
        action="store_false",
        help="Disable thresholded AMF loss mask.",
    )

    parser.add_argument("--ref_amf_cpu", action="store_true")
    parser.add_argument("--ref_amf_fp16", action="store_true")

    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--t5_cpu", action="store_true")
    parser.add_argument("--offload_t5", action="store_true")
    parser.add_argument("--convert_model_dtype", action="store_true")

    return parser.parse_args()


def main():
    config = parse_args()

    extractor = WanAMFExtractor(config)

    if config.mode == "extract":
        extractor.load_attn_features()
    else:
        extractor.generate_with_amf_guidance()


if __name__ == "__main__":
    main()
