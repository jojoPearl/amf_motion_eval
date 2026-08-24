from __future__ import annotations
import argparse
import copy
import inspect
import json
import logging
import math
import os

import numpy as np
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from diffusers import HunyuanVideoPipeline, __version__ as DIFFUSERS_VERSION
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.utils import export_to_video


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

Tensor = torch.Tensor
MODEL_ID = "hunyuanvideo-community/HunyuanVideo"


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

@dataclass
class AMFSearchConfig:
    beam_size: int = 2
    candidates_per_beam: int = 3
    lookahead_steps: int = 3

    # Equivalent to Wan steps [4, 6, 8, 11, 15] out of 50.
    select_step_ratios: Tuple[float, ...] = (0.08, 0.12, 0.16, 0.22, 0.30)

    # Sigma-aware branch perturbation.
    branch_noise_scale: float = 0.05
    temporal_smooth_noise: bool = True
    temporal_smooth_kernel: int = 5

    # AMF-QK
    amf_block_id: int = 15
    # Uniformly sample temporal triplets; 6 is a practical quality/compute compromise.
    max_amf_triplets: int = 6
    motion_temp: float = 2.0

    verbose: bool = True


@dataclass
class StatefulBeam:
    latents: Tensor
    scheduler: Any
    cumulative_reward: float = 0.0
    history: Optional[List[Dict[str, Any]]] = None

    def with_update(
        self,
        latents: Tensor,
        scheduler: Any,
        reward: float,
        info: Dict[str, Any],
    ) -> "StatefulBeam":
        history = [] if self.history is None else list(self.history)
        history.append(info)
        return StatefulBeam(
            latents=latents.detach(),
            scheduler=scheduler,
            cumulative_reward=self.cumulative_reward + float(reward),
            history=history,
        )


@dataclass
class QKCapture:
    query: Optional[Tensor] = None
    key: Optional[Tensor] = None
    grid_size: Optional[Tuple[int, int, int]] = None


class StopAfterCapture(RuntimeError):
    """Internal control-flow exception used to stop scoring forwards early."""


# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------

def _parse_ratios(value: str) -> Tuple[float, ...]:
    return tuple(float(x.strip()) for x in value.split(",") if x.strip())


def _selection_step_ids(num_steps: int, ratios: Sequence[float]) -> List[int]:
    if num_steps <= 1:
        return [0]
    ids = []
    for ratio in ratios:
        idx = int(round(float(ratio) * (num_steps - 1)))
        idx = max(0, min(num_steps - 1, idx))
        ids.append(idx)
    return sorted(set(ids))


def _scheduler_sigma(scheduler: Any, step_i: int) -> float:
    """Best-effort current sigma for FlowMatch schedulers; falls back to 1.0."""
    sigmas = getattr(scheduler, "sigmas", None)
    if sigmas is None or len(sigmas) == 0:
        return 1.0
    idx = max(0, min(int(step_i), len(sigmas) - 1))
    value = sigmas[idx]
    if torch.is_tensor(value):
        value = float(value.detach().float().cpu())
    return max(0.0, float(value))


def _temporal_smooth_noise(x: Tensor, kernel_size: int) -> Tensor:
    if kernel_size <= 1:
        return x
    if kernel_size % 2 == 0:
        kernel_size += 1
    if x.ndim != 5 or x.shape[2] < 2:
        return x

    pad = kernel_size // 2
    y = x.movedim(2, -1)
    original_shape = y.shape
    y = y.reshape(-1, 1, original_shape[-1])
    weight = torch.ones(1, 1, kernel_size, device=x.device, dtype=x.dtype) / float(kernel_size)
    y = F.pad(y, (pad, pad), mode="replicate")
    y = F.conv1d(y, weight)
    return y.reshape(original_shape).movedim(-1, 2)


def _make_candidate_inputs(
    latents: Tensor,
    cfg: AMFSearchConfig,
    generator: torch.Generator,
    scheduler: Any,
    step_i: int,
) -> List[Tensor]:
    """Create conservative, sigma-aware local branches around the valid x_t state.

    Candidate 0 is always the untouched Hunyuan trajectory. Perturbations shrink with
    scheduler sigma and are capped relative to latent std, avoiding the old 0.4*std jumps.
    """
    candidates = [latents.detach()]
    if cfg.candidates_per_beam <= 1 or cfg.branch_noise_scale <= 0:
        return candidates

    latent_std = latents.detach().float().std().clamp(min=1e-6)
    sigma = _scheduler_sigma(scheduler, step_i)
    # Keep perturbation tiny and naturally decay it as denoising progresses.
    scale = float(cfg.branch_noise_scale) * min(1.0, sigma) * latent_std

    for _ in range(cfg.candidates_per_beam - 1):
        noise = torch.randn(
            latents.shape, device=latents.device, dtype=latents.dtype, generator=generator
        )
        if cfg.temporal_smooth_noise:
            noise = _temporal_smooth_noise(noise, cfg.temporal_smooth_kernel)
        noise_std = noise.detach().float().std().clamp(min=1e-6)
        noise = noise / noise_std.to(noise.dtype)
        candidates.append((latents + scale.to(latents.dtype) * noise).detach())

    return candidates


# ---------------------------------------------------------------------
# AMF / AMF-TV: DiTFlow-style per-head QK logits + AMF-TV acceleration
# ---------------------------------------------------------------------

def _reshape_qk(
    q: Tensor,
    k: Tensor,
    grid_size: Tuple[int, int, int],
):
    frames, grid_h, grid_w = [int(x) for x in grid_size]
    hw = grid_h * grid_w
    seq_len = frames * hw

    if q.ndim != 4 or k.ndim != 4:
        raise ValueError(
            f"Expected q/k [B,S,H,D], got q={tuple(q.shape)} k={tuple(k.shape)}"
        )

    if q.shape[1] < seq_len or k.shape[1] < seq_len:
        raise ValueError(
            f"Q/K sequence too short for grid={grid_size}: "
            f"q={tuple(q.shape)} k={tuple(k.shape)}"
        )

    q = q[0, :seq_len].permute(1, 0, 2).reshape(
        q.shape[2], frames, hw, q.shape[-1]
    )
    k = k[0, :seq_len].permute(1, 0, 2).reshape(
        k.shape[2], frames, hw, k.shape[-1]
    )

    return q, k, frames, grid_h, grid_w


def _pair_attention(
    q_i: Tensor,
    k_j: Tensor,
    temp: float,
):
    """Frame-to-frame correspondence with DiTFlow-style head aggregation.

    q_i/k_j are [H, HW, D]. Compute QK^T independently inside each
    attention head, then average the logits across heads before softmax.
    This avoids artificial cross-head Q_i K_j interactions introduced by
    averaging Q and K separately before the dot product.
    """
    logits_per_head = torch.matmul(
        q_i, k_j.transpose(-1, -2)
    ) / math.sqrt(q_i.shape[-1])  # [H, HW, HW]
    logits = logits_per_head.mean(dim=0)  # [HW, HW]
    return torch.softmax(logits * temp, dim=-1)


def _patch_coords(grid_h: int, grid_w: int, device, dtype):
    rows, cols = torch.meshgrid(
        torch.arange(grid_h, device=device, dtype=dtype),
        torch.arange(grid_w, device=device, dtype=dtype),
        indexing="ij",
    )
    return torch.stack((rows, cols), dim=-1).reshape(grid_h * grid_w, 2)


def _attention_to_flow(attn: Tensor, grid_h: int, grid_w: int):
    coords = _patch_coords(grid_h, grid_w, attn.device, attn.dtype)
    expected = attn @ coords
    return expected - coords


def _amf_loss(
    q: Tensor,
    k: Tensor,
    grid_size: Tuple[int, int, int],
    cfg: AMFSearchConfig,
):
    q, k, frames, grid_h, grid_w = _reshape_qk(q, k, grid_size)

    if frames < 3:
        raise RuntimeError("AMF-TV requires at least 3 latent frames.")

    triplet_ids = list(range(frames - 2))

    if cfg.max_amf_triplets > 0 and len(triplet_ids) > cfg.max_amf_triplets:
        keep = (
            torch.linspace(
                0,
                len(triplet_ids) - 1,
                steps=cfg.max_amf_triplets,
            )
            .round()
            .long()
            .tolist()
        )
        triplet_ids = [triplet_ids[i] for i in keep]

    losses = []
    per_triplet = []

    for t in triplet_ids:
        attn_01 = _pair_attention(
            q[:, t],
            k[:, t + 1],
            cfg.motion_temp,
        )
        attn_12 = _pair_attention(
            q[:, t + 1],
            k[:, t + 2],
            cfg.motion_temp,
        )

        flow_01 = _attention_to_flow(attn_01, grid_h, grid_w)
        flow_12 = _attention_to_flow(attn_12, grid_h, grid_w)

        # L1 flow acceleration on DiTFlow-style aggregated attention correspondences.
        loss_t = (flow_12 - flow_01).abs().sum(dim=-1).mean()
        losses.append(loss_t)
        per_triplet.append(float(loss_t.detach().float().cpu()))

    loss = torch.stack(losses).mean()

    return loss, {
        "triplet_ids": triplet_ids,
        "per_triplet_loss": per_triplet,
        "grid_size": [frames, grid_h, grid_w],
    }



# ---------------------------------------------------------------------
# Hunyuan after-RoPE Q/K capture
# ---------------------------------------------------------------------

class HunyuanCaptureProcessor:
    """Version-compatible Hunyuan latent Q/K capture after RMSNorm + RoPE.

    Diffusers 0.32.x uses BHSD internally; newer releases use BSHD. Captured
    tensors are normalized to BSHD so downstream AMF code is version-independent.
    """

    def __init__(self, capture: QKCapture, grid_size: Tuple[int, int, int], layout: str):
        self.capture = capture
        self.grid_size = grid_size
        self.layout = layout

    def __call__(
        self,
        attn,
        hidden_states: Tensor,
        encoder_hidden_states: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        image_rotary_emb: Optional[Tensor] = None,
        **kwargs,
    ):
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)

        if self.layout == "BHSD":
            query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        else:
            query = query.unflatten(2, (attn.heads, -1))
            key = key.unflatten(2, (attn.heads, -1))

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if image_rotary_emb is None:
            raise RuntimeError("Hunyuan capture did not receive image_rotary_emb.")

        if self.layout == "BHSD":
            # diffusers 0.32.x apply_rotary_emb expects sequence dimension at dim=2.
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)
            query = query.transpose(1, 2)  # -> BSHD
            key = key.transpose(1, 2)
        else:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        query = query.contiguous()
        key = key.contiguous()
        expected_tokens = int(self.grid_size[0]) * int(self.grid_size[1]) * int(self.grid_size[2])
        if query.shape[1] != expected_tokens:
            raise RuntimeError(
                f"Hunyuan token-grid mismatch after layout normalization: "
                f"q={tuple(query.shape)}, grid={self.grid_size}, expected={expected_tokens}, "
                f"layout={self.layout}, diffusers={DIFFUSERS_VERSION}"
            )

        self.capture.query = query.detach()
        self.capture.key = key.detach()
        self.capture.grid_size = self.grid_size
        raise StopAfterCapture()


def _detect_attention_layout(attn) -> str:
    """Detect Hunyuan attention tensor convention without hardcoding version."""
    try:
        src = inspect.getsource(attn.processor.__class__.__call__)
        if ".unflatten(2, (attn.heads, -1)).transpose(1, 2)" in src:
            return "BHSD"
    except Exception:
        pass
    # 0.32.x is known BHSD; newer backend-refactored releases are BSHD.
    try:
        major, minor = [int(x) for x in DIFFUSERS_VERSION.split(".")[:2]]
        if (major, minor) <= (0, 32):
            return "BHSD"
    except Exception:
        pass
    return "BSHD"


def _token_grid_from_latents(
    transformer,
    latents: Tensor,
) -> Tuple[int, int, int]:
    if latents.ndim != 5:
        raise ValueError(
            f"Expected Hunyuan latents [B,C,T,H,W], got {tuple(latents.shape)}"
        )

    _, _, t, h, w = latents.shape

    patch_t = int(getattr(transformer.config, "patch_size_t", 1))
    patch_s = int(getattr(transformer.config, "patch_size", 2))

    if t % patch_t != 0 or h % patch_s != 0 or w % patch_s != 0:
        raise RuntimeError(
            f"Latent shape {tuple(latents.shape)} incompatible with "
            f"patch_size_t={patch_t}, patch_size={patch_s}"
        )

    return (
        t // patch_t,
        h // patch_s,
        w // patch_s,
    )


def _capture_hunyuan_qk(
    pipe: HunyuanVideoPipeline,
    latents: Tensor,
    timestep: Tensor,
    prompt_embeds: Tensor,
    prompt_attention_mask: Tensor,
    pooled_prompt_embeds: Tensor,
    guidance: Tensor,
    cfg: AMFSearchConfig,
) -> Dict[str, Any]:
    """
    Extra forward used only for scoring.

    The target dual-stream attention processor is replaced temporarily.
    The custom processor stops immediately after after-RoPE video Q/K are
    captured, so the remainder of the transformer does not run.
    """
    transformer = pipe.transformer

    if not hasattr(transformer, "transformer_blocks"):
        raise AttributeError("Hunyuan transformer has no transformer_blocks.")

    if cfg.amf_block_id < 0 or cfg.amf_block_id >= len(transformer.transformer_blocks):
        raise ValueError(
            f"amf_block_id={cfg.amf_block_id} out of range for "
            f"{len(transformer.transformer_blocks)} dual-stream blocks."
        )

    block = transformer.transformer_blocks[cfg.amf_block_id]
    attn = block.attn
    original_processor = attn.processor

    capture = QKCapture()
    grid_size = _token_grid_from_latents(transformer, latents)

    layout = _detect_attention_layout(attn)
    attn.set_processor(HunyuanCaptureProcessor(capture=capture, grid_size=grid_size, layout=layout))

    try:
        latent_model_input = latents.to(transformer.dtype)
        ts = timestep.expand(latents.shape[0]).to(latents.dtype)

        try:
            _ = transformer(
                hidden_states=latent_model_input,
                timestep=ts,
                encoder_hidden_states=prompt_embeds,
                encoder_attention_mask=prompt_attention_mask,
                pooled_projections=pooled_prompt_embeds,
                guidance=guidance,
                attention_kwargs=None,
                return_dict=False,
            )
        except StopAfterCapture:
            pass

        if capture.query is None or capture.key is None or capture.grid_size is None:
            raise RuntimeError("AMF-QK capture failed: no Q/K/grid was captured.")

        loss, debug = _amf_loss(
            capture.query,
            capture.key,
            capture.grid_size,
            cfg,
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"AMF-TV is non-finite: {loss}; debug={debug}"
            )

        return {
            "amf_tv": float(loss.detach().float().cpu()),
            "debug": debug,
            "q_shape": list(capture.query.shape),
            "k_shape": list(capture.key.shape),
            "grid_size": list(capture.grid_size),
        }

    finally:
        attn.set_processor(original_processor)
        capture.query = None
        capture.key = None


# ---------------------------------------------------------------------
# Diffusers compatibility helpers
# ---------------------------------------------------------------------

def _prepare_hunyuan_latents(
    pipe: HunyuanVideoPipeline,
    batch_size: int,
    height: int,
    width: int,
    num_frames: int,
    dtype: torch.dtype,
    device: torch.device,
    generator: torch.Generator,
) -> Tensor:
    """Match the installed HunyuanVideoPipeline's own latent-frame convention.

    Diffusers 0.32.x expects *latent frame count* in prepare_latents, while newer
    versions accept output frame count and compress internally.
    """
    num_channels = pipe.transformer.config.in_channels
    try:
        src = inspect.getsource(pipe.prepare_latents)
    except Exception:
        src = ""

    old_api = "num_latent_frames" in src or (
        "num_frames," in src and "vae_scale_factor_temporal" not in src
    )
    # Explicit version fallback when source inspection is unavailable.
    if not src:
        try:
            major, minor = [int(x) for x in DIFFUSERS_VERSION.split(".")[:2]]
            old_api = (major, minor) <= (0, 32)
        except Exception:
            old_api = False

    frames_arg = num_frames
    if old_api:
        temporal = int(getattr(pipe, "vae_scale_factor_temporal", 4))
        frames_arg = (num_frames - 1) // temporal + 1

    logging.info(
        "[hunyuan-amf] diffusers=%s prepare_latents frames_arg=%s output_frames=%s old_api=%s",
        DIFFUSERS_VERSION, frames_arg, num_frames, old_api,
    )
    return pipe.prepare_latents(
        batch_size=batch_size,
        num_channels_latents=num_channels,
        height=height,
        width=width,
        num_frames=frames_arg,
        dtype=dtype,
        device=device,
        generator=generator,
        latents=None,
    )


def _decode_hunyuan_video(pipe: HunyuanVideoPipeline, latents: Tensor, output_type: str):
    """Use the same VAE scaling/decode/postprocess contract as the official pipeline."""
    if output_type == "latent":
        return latents
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    decode_latents = latents.to(pipe.vae.dtype) / pipe.vae.config.scaling_factor
    video = pipe.vae.decode(decode_latents, return_dict=False)[0]
    return pipe.video_processor.postprocess_video(video, output_type=output_type)


# ---------------------------------------------------------------------
# Hunyuan generation + candidate selection
# ---------------------------------------------------------------------

@torch.no_grad()
def generate_hunyuan_amf(
    pipe: HunyuanVideoPipeline,
    prompt: str,
    seed: int,
    cfg: AMFSearchConfig,
    height: int = 320,
    width: int = 512,
    num_frames: int = 61,
    num_inference_steps: int = 30,
    guidance_scale: float = 6.0,
    output_type: str = "pil",
):
    device = pipe._execution_device
    transformer_dtype = pipe.transformer.dtype

    if num_frames < 3:
        raise ValueError("num_frames must be >= 3.")

    # ---------------------------------------------------------
    # 1. Prompt encoding
    # ---------------------------------------------------------
    prompt_embeds, pooled_prompt_embeds, prompt_attention_mask = pipe.encode_prompt(
        prompt=prompt,
        prompt_2=None,
        num_videos_per_prompt=1,
        device=device,
        max_sequence_length=256,
    )

    prompt_embeds = prompt_embeds.to(transformer_dtype)
    prompt_attention_mask = prompt_attention_mask.to(transformer_dtype)
    pooled_prompt_embeds = pooled_prompt_embeds.to(transformer_dtype)

    # ---------------------------------------------------------
    # 2. Timesteps
    # ---------------------------------------------------------
    # Match the official diffusers HunyuanVideoPipeline schedule exactly.
    # In diffusers 0.32.x the pipeline builds a linear sigma schedule first,
    # then calls scheduler.set_timesteps(sigmas=...). Calling set_timesteps(N)
    # directly produces a different trajectory even with identical initial noise.
    sigma_schedule = np.linspace(1.0, 0.0, num_inference_steps + 1)[:-1]
    set_timestep_params = inspect.signature(pipe.scheduler.set_timesteps).parameters
    if "sigmas" in set_timestep_params:
        pipe.scheduler.set_timesteps(
            sigmas=sigma_schedule,
            device=device,
        )
    else:
        # Compatibility fallback for schedulers that do not expose custom sigmas.
        # This branch cannot be bit-identical to a Hunyuan pipeline version whose
        # official __call__ uses custom sigmas, so emit an explicit warning.
        logging.warning(
            "[hunyuan-amf] scheduler %s has no sigmas= argument; falling back "
            "to set_timesteps(num_inference_steps). Exact baseline alignment is "
            "not guaranteed.",
            pipe.scheduler.__class__.__name__,
        )
        pipe.scheduler.set_timesteps(
            num_inference_steps,
            device=device,
        )
    timesteps = pipe.scheduler.timesteps
    num_inference_steps = len(timesteps)

    # ---------------------------------------------------------
    # 3. Initial latents
    # ---------------------------------------------------------
    init_generator = torch.Generator(device="cpu").manual_seed(seed)

    latents = _prepare_hunyuan_latents(
        pipe=pipe,
        batch_size=1,
        height=height,
        width=width,
        num_frames=num_frames,
        dtype=torch.float32,
        device=device,
        generator=init_generator,
    )

    # Candidate noise generator must live on the same device as latents.
    gen_device = latents.device.type
    branch_generator = torch.Generator(device=gen_device).manual_seed(seed + 100003)

    guidance = (
        torch.tensor(
            [guidance_scale] * latents.shape[0],
            dtype=transformer_dtype,
            device=device,
        )
        * 1000.0
    )

    selection_step_ids = (
        _selection_step_ids(len(timesteps), cfg.select_step_ratios)
        if cfg.beam_size > 0 and cfg.candidates_per_beam > 0
        else []
    )

    identity_mode = cfg.beam_size == 1 and cfg.candidates_per_beam == 1
    if identity_mode:
        logging.info(
            "[hunyuan-amf] identity sanity mode: beam=1 candidates=1; "
            "candidate 0 is the untouched official trajectory. QK scoring may run "
            "at selection steps but cannot change the chosen latent state."
        )

    logging.info(
        "[hunyuan-amf] selection steps=%s / total=%s",
        selection_step_ids,
        len(timesteps),
    )
    logging.info(
        "[hunyuan-amf] beam=%s candidates=%s lookahead=%s noise=%.3f "
        "block=%s",
        cfg.beam_size,
        cfg.candidates_per_beam,
        cfg.lookahead_steps,
        cfg.branch_noise_scale,
        cfg.amf_block_id,
    )

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------
    def scheduler_step(
        candidate_latents: Tensor,
        scheduler: Any,
        step_i: int,
    ) -> Tensor:
        t = timesteps[step_i]

        latent_model_input = candidate_latents.to(transformer_dtype)
        ts = t.expand(candidate_latents.shape[0]).to(candidate_latents.dtype)

        noise_pred = pipe.transformer(
            hidden_states=latent_model_input,
            timestep=ts,
            encoder_hidden_states=prompt_embeds,
            encoder_attention_mask=prompt_attention_mask,
            pooled_projections=pooled_prompt_embeds,
            guidance=guidance,
            attention_kwargs=None,
            return_dict=False,
        )[0]

        next_latents = scheduler.step(
            noise_pred,
            t,
            candidate_latents,
            return_dict=False,
        )[0]

        return next_latents.detach()

    def score_with_lookahead(
        candidate_latents: Tensor,
        scheduler: Any,
        start_step_i: int,
    ) -> Dict[str, Any]:
        preview_scheduler = copy.deepcopy(scheduler)
        z = candidate_latents.detach()

        max_j = min(
            cfg.lookahead_steps,
            max(0, len(timesteps) - start_step_i),
        )

        amf_scores: List[float] = []
        q_shapes: List[List[int]] = []
        grids: List[List[int]] = []
        debug_records: List[Dict[str, Any]] = []

        for j in range(max_j):
            idx = start_step_i + j
            t = timesteps[idx]

            rec = _capture_hunyuan_qk(
                pipe=pipe,
                latents=z,
                timestep=t,
                prompt_embeds=prompt_embeds,
                prompt_attention_mask=prompt_attention_mask,
                pooled_prompt_embeds=pooled_prompt_embeds,
                guidance=guidance,
                cfg=cfg,
            )

            amf_scores.append(float(rec["amf_tv"]))
            q_shapes.append(rec["q_shape"])
            grids.append(rec["grid_size"])
            debug_records.append(rec["debug"])

            if j + 1 < max_j:
                z = scheduler_step(
                    z,
                    preview_scheduler,
                    idx,
                )

        if not amf_scores:
            raise RuntimeError(
                "No AMF-QK records captured during Hunyuan lookahead."
            )

        return {
            "amf_tv": float(sum(amf_scores) / len(amf_scores)),
            "amf_qk_records": len(amf_scores),
            "q_shapes": q_shapes,
            "grids": grids,
            "amf_debug": debug_records,
        }

    # ---------------------------------------------------------
    # 4. Beam denoising loop
    # ---------------------------------------------------------
    beams: List[StatefulBeam] = [
        StatefulBeam(
            latents=latents.detach(),
            scheduler=copy.deepcopy(pipe.scheduler),
            cumulative_reward=0.0,
            history=[],
        )
    ]

    with pipe.progress_bar(total=len(timesteps)) as progress_bar:
        for step_i, t in enumerate(timesteps):
            pipe._current_timestep = t

            # Normal denoising step.
            if step_i not in selection_step_ids:
                next_beams: List[StatefulBeam] = []

                for beam in beams:
                    z_next = scheduler_step(
                        beam.latents,
                        beam.scheduler,
                        step_i,
                    )
                    next_beams.append(
                        StatefulBeam(
                            latents=z_next,
                            scheduler=beam.scheduler,
                            cumulative_reward=beam.cumulative_reward,
                            history=beam.history,
                        )
                    )

                beams = next_beams
                progress_bar.update()
                continue

            # Strict baseline sanity path. With beam=1 and candidates=1 there is
            # nothing to select, so do not run the extra Q/K scoring forward.
            # This makes the wrapper denoising path identical to the official
            # Hunyuan pipeline and isolates baseline-alignment bugs from scoring
            # side effects.
            if identity_mode:
                beam = beams[0]
                z_next = scheduler_step(
                    beam.latents,
                    beam.scheduler,
                    step_i,
                )
                beams = [
                    StatefulBeam(
                        latents=z_next,
                        scheduler=beam.scheduler,
                        cumulative_reward=beam.cumulative_reward,
                        history=beam.history,
                    )
                ]
                progress_bar.update()
                continue

            # -------------------------------------------------
            # Branch and score
            # -------------------------------------------------
            expanded: List[StatefulBeam] = []
            failed_fallbacks: List[StatefulBeam] = []

            for beam_id, beam in enumerate(beams):
                candidate_inputs = _make_candidate_inputs(
                    beam.latents, cfg, branch_generator, beam.scheduler, step_i
                )

                cand0 = candidate_inputs[0].detach()
                group_items: List[Dict[str, Any]] = []

                scoring_failed = None
                baseline_fallback = None

                for candidate_id, candidate_input in enumerate(candidate_inputs):
                    candidate_scheduler = copy.deepcopy(beam.scheduler)

                    # First perform the real current denoising step. Candidate 0 is
                    # exactly the official trajectory and is retained as a safe fallback.
                    candidate_next = scheduler_step(
                        candidate_input, candidate_scheduler, step_i
                    )
                    if candidate_id == 0:
                        baseline_fallback = (candidate_next, candidate_scheduler)

                    try:
                        score = score_with_lookahead(
                            candidate_next, candidate_scheduler, step_i + 1
                        )
                    except Exception as exc:
                        scoring_failed = exc
                        logging.exception(
                            "[hunyuan-amf] AMF scoring failed at step=%s beam=%s; "
                            "falling back to untouched Hunyuan trajectory for this selection step.",
                            step_i, beam_id,
                        )
                        break

                    latent_dist = (
                        0.0
                        if candidate_id == 0
                        else float(
                            (candidate_input.detach().float() - cand0.float())
                            .square().mean().sqrt().cpu()
                        )
                    )

                    group_items.append(
                        {
                            "beam": beam,
                            "beam_id": beam_id,
                            "candidate_id": candidate_id,
                            "candidate_next": candidate_next,
                            "candidate_scheduler": candidate_scheduler,
                            "amf_tv": float(score["amf_tv"]),
                            "reward": -float(score["amf_tv"]),
                            "latent_dist": latent_dist,
                            "amf_qk_records": int(score["amf_qk_records"]),
                            "q_shapes": score["q_shapes"],
                            "grids": score["grids"],
                            "amf_debug": score["amf_debug"],
                        }
                    )

                if scoring_failed is not None:
                    if baseline_fallback is None:
                        fallback_scheduler = copy.deepcopy(beam.scheduler)
                        fallback_latents = scheduler_step(
                            beam.latents, fallback_scheduler, step_i
                        )
                    else:
                        fallback_latents, fallback_scheduler = baseline_fallback

                    # IMPORTANT:
                    # A scoring failure must not receive reward=0.0 and compete with
                    # valid candidates, because normal reward=-AMF-TV is <= 0 and
                    # zero would therefore be artificially optimal.
                    #
                    # Keep the untouched candidate-0 trajectory only as an emergency
                    # fallback. It is excluded from normal beam ranking. If every
                    # beam fails at this selection step, we restore these fallbacks
                    # below without changing cumulative_reward.
                    info = {
                        "step_i": int(step_i),
                        "timestep": float(t.detach().float().cpu()),
                        "beam_id": int(beam_id),
                        "candidate_id": 0,
                        "reward": None,
                        "fallback": True,
                        "fallback_reason": repr(scoring_failed),
                    }
                    fallback_history = (
                        [] if beam.history is None else list(beam.history)
                    )
                    fallback_history.append(info)
                    failed_fallbacks.append(
                        StatefulBeam(
                            latents=fallback_latents.detach(),
                            scheduler=fallback_scheduler,
                            cumulative_reward=beam.cumulative_reward,
                            history=fallback_history,
                        )
                    )
                    continue


                group_items.sort(
                    key=lambda x: (
                        x["beam"].cumulative_reward + x["reward"]
                    ),
                    reverse=True,
                )

                if cfg.verbose:
                    logging.info(
                        "[hunyuan-amf] step=%03d beam=%s",
                        step_i,
                        beam_id,
                    )

                    for item in group_items:
                        logging.info(
                            "[hunyuan-amf]   cand=%s amf_tv=%.6f "
                            "latent_dist=%.6f "
                            "qk_records=%s reward=%.6f grid=%s q=%s",
                            item["candidate_id"],
                            item["amf_tv"],
                            item["latent_dist"],
                            item["amf_qk_records"],
                            item["reward"],
                            item["grids"][0] if item["grids"] else None,
                            item["q_shapes"][0] if item["q_shapes"] else None,
                        )

                for item in group_items:
                    info = {
                        "step_i": int(step_i),
                        "timestep": float(t.detach().float().cpu()),
                        "beam_id": int(item["beam_id"]),
                        "candidate_id": int(item["candidate_id"]),
                        "amf_tv": float(item["amf_tv"]),
                        "latent_dist": float(item["latent_dist"]),
                        "amf_qk_records": int(item["amf_qk_records"]),
                        "reward": float(item["reward"]),
                        "grid": item["grids"][0] if item["grids"] else None,
                        "q_shape": item["q_shapes"][0] if item["q_shapes"] else None,
                    }

                    expanded.append(
                        item["beam"].with_update(
                            latents=item["candidate_next"],
                            scheduler=item["candidate_scheduler"],
                            reward=item["reward"],
                            info=info,
                        )
                    )

            if not expanded:
                if not failed_fallbacks:
                    raise RuntimeError(
                        "AMF selection produced no valid candidates and no baseline fallbacks."
                    )
                logging.warning(
                    "[hunyuan-amf] all AMF scoring failed at selection step=%s; "
                    "restoring untouched candidate-0 trajectories without reward update.",
                    step_i,
                )
                beams = failed_fallbacks[: cfg.beam_size]
            else:
                expanded.sort(
                    key=lambda item: item.cumulative_reward,
                    reverse=True,
                )
                beams = expanded[: cfg.beam_size]

            if cfg.verbose:
                logging.info(
                    "[hunyuan-amf] keep top-%s cumulative_rewards=%s paths=%s",
                    cfg.beam_size,
                    [round(b.cumulative_reward, 6) for b in beams],
                    [
                        [int(x["candidate_id"]) for x in (b.history or [])]
                        for b in beams
                    ],
                )

            progress_bar.update()

    pipe._current_timestep = None

    # ---------------------------------------------------------
    # 5. Final beam + decode
    # ---------------------------------------------------------
    best_beam = max(
        beams,
        key=lambda item: item.cumulative_reward,
    )
    final_latents = best_beam.latents

    video = _decode_hunyuan_video(pipe, final_latents, output_type)

    pipe.maybe_free_model_hooks()

    history = {
        "seed": int(seed),
        "prompt": prompt,
        "selection_step_ids": selection_step_ids,
        "config": {
            "beam_size": cfg.beam_size,
            "candidates_per_beam": cfg.candidates_per_beam,
            "lookahead_steps": cfg.lookahead_steps,
            "select_step_ratios": list(cfg.select_step_ratios),
            "branch_noise_scale": cfg.branch_noise_scale,
            "temporal_smooth_noise": cfg.temporal_smooth_noise,
            "temporal_smooth_kernel": cfg.temporal_smooth_kernel,
            "amf_block_id": cfg.amf_block_id,
            "max_amf_triplets": cfg.max_amf_triplets,
            "motion_temp": cfg.motion_temp,
        },
        "best_cumulative_reward": best_beam.cumulative_reward,
        "best_history": best_beam.history or [],
        "remaining_beams": [
            {
                "cumulative_reward": b.cumulative_reward,
                "history": b.history or [],
            }
            for b in beams
        ],
    }

    return video, history


# ---------------------------------------------------------------------
# Baseline route
# ---------------------------------------------------------------------

@torch.no_grad()
def generate_baseline(
    pipe: HunyuanVideoPipeline,
    prompt: str,
    seed: int,
    height: int,
    width: int,
    num_frames: int,
    num_inference_steps: int,
    guidance_scale: float,
):
    generator = torch.Generator(device="cpu").manual_seed(seed)

    return pipe(
        prompt=prompt,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).frames[0]


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def case_name_and_seed(
    case: Dict[str, Any],
    index: int,
) -> Tuple[str, int]:
    baseline_stem = Path(case.get("baseline_video", "")).stem
    match = re.search(r"_seed(\d+)$", baseline_stem)
    seed = int(match.group(1)) if match else int(case.get("seed", 42 + index))
    name = baseline_stem or f"case_{index + 1:02d}_seed{seed}"
    return name, seed


def inspect_transformer(pipe: HunyuanVideoPipeline) -> None:
    """Print the model configuration and attention modules, then exit."""
    print("HUNYUAN TRANSFORMER CONFIG")
    print(pipe.transformer.config)
    print("\nHUNYUAN ATTENTION MODULES")
    for name, module in pipe.transformer.named_modules():
        lower_name = name.lower()
        if any(token in lower_name for token in ("attn", "attention", "to_q", "to_k")):
            print(f"{name:100s} {type(module)}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="HunyuanVideo generation with DiTFlow-style AMF-QK candidate selection."
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/human_motion_judge_goodcase.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=MODEL_ID,
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Print transformer/attention structure and exit without generation.",
    )

    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num-frames", type=int, default=61)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--fps", type=float, default=15.0)

    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Run plain Hunyuan baseline instead of AMF candidate selection.",
    )

    parser.add_argument("--beam-size", type=int, default=2)
    parser.add_argument("--candidates-per-beam", type=int, default=3)
    parser.add_argument("--lookahead-steps", type=int, default=3)
    parser.add_argument(
        "--selection-step-ratios",
        type=str,
        default="0.08,0.12,0.16,0.22,0.30",
    )
    parser.add_argument("--branch-noise-scale", type=float, default=0.05)

    parser.add_argument(
        "--temporal-smooth-noise",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-temporal-smooth-noise",
        dest="temporal_smooth_noise",
        action="store_false",
    )
    parser.add_argument("--temporal-smooth-kernel", type=int, default=5)

    parser.add_argument("--amf-block-id", type=int, default=15)
    parser.add_argument("--max-amf-triplets", type=int, default=6)
    parser.add_argument("--motion-temp", type=float, default=2.0)

    offload_group = parser.add_mutually_exclusive_group()
    offload_group.add_argument(
        "--cpu-offload",
        action="store_true",
        help=(
            "Enable model CPU offload. This is opt-in because AMF scoring "
            "stops transformer forwards early to capture Q/K."
        ),
    )
    offload_group.add_argument(
        "--no-cpu-offload",
        dest="cpu_offload",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(cpu_offload=False)
    parser.add_argument(
        "--quiet",
        action="store_true",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.height % 16 != 0 or args.width % 16 != 0:
        raise ValueError("Hunyuan height/width must be divisible by 16.")

    if (args.num_frames - 1) % 4 != 0:
        logging.warning(
            "HunyuanVideo usually expects num_frames = 4*k+1; got %s",
            args.num_frames,
        )

    logging.info("Loading HunyuanVideo from %s", args.model_id)

    # Keep the official model loading path. BF16 works on H100/A100; VAE decode
    # is handled using pipe.vae.dtype.
    pipe = HunyuanVideoPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
    )

    if args.inspect:
        inspect_transformer(pipe)
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    with args.config.open(encoding="utf-8") as file:
        cases = json.load(file)

    if not isinstance(cases, list) or not cases:
        raise ValueError(
            f"Expected a non-empty case list in {args.config}"
        )

    pipe.vae.enable_tiling()
    if hasattr(pipe.vae, "enable_slicing"):
        pipe.vae.enable_slicing()

    if args.cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")

    logging.info(
        "Transformer config: layers=%s single_layers=%s heads=%s head_dim=%s "
        "patch=%s patch_t=%s",
        getattr(pipe.transformer.config, "num_layers", None),
        getattr(pipe.transformer.config, "num_single_layers", None),
        getattr(pipe.transformer.config, "num_attention_heads", None),
        getattr(pipe.transformer.config, "attention_head_dim", None),
        getattr(pipe.transformer.config, "patch_size", None),
        getattr(pipe.transformer.config, "patch_size_t", None),
    )

    cfg = AMFSearchConfig(
        beam_size=args.beam_size,
        candidates_per_beam=args.candidates_per_beam,
        lookahead_steps=args.lookahead_steps,
        select_step_ratios=_parse_ratios(args.selection_step_ratios),
        branch_noise_scale=args.branch_noise_scale,
        temporal_smooth_noise=args.temporal_smooth_noise,
        temporal_smooth_kernel=args.temporal_smooth_kernel,
        amf_block_id=args.amf_block_id,
        max_amf_triplets=args.max_amf_triplets,
        motion_temp=args.motion_temp,
        verbose=not args.quiet,
    )

    for index, case in enumerate(cases):
        prompt = case.get("prompt")

        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(
                f"Case {index + 1} has no valid prompt."
            )

        name, seed = case_name_and_seed(case, index)

        suffix = "hunyuan_baseline" if args.baseline else "hunyuan_amf"
        output_path = args.output_dir / f"{name}_{suffix}.mp4"
        # history_path = args.output_dir / f"{name}_{suffix}_history.json"

        logging.info(
            "[%s/%s] seed=%s prompt=%s",
            index + 1,
            len(cases),
            seed,
            prompt,
        )

        if args.baseline:
            frames = generate_baseline(
                pipe=pipe,
                prompt=prompt,
                seed=seed,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
            )
            history = {
                "mode": "baseline",
                "seed": seed,
                "prompt": prompt,
            }
        else:
            video, history = generate_hunyuan_amf(
                pipe=pipe,
                prompt=prompt,
                seed=seed,
                cfg=cfg,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                output_type="pil",
            )
            # postprocess_video returns a batch; export_to_video needs one frame list.
            frames = video[0] if isinstance(video, (list, tuple)) and len(video) == 1 else video
            history["mode"] = "amf"

        print(
            f"[DEBUG] mode={'baseline' if args.baseline else 'amf'} "
            f"type={type(frames)} "
            f"num_frames={len(frames)} "
            f"fps={args.fps}"
        )

        export_to_video(
            frames,
            str(output_path),
            fps=args.fps,
        )

        # 暂不保存 AMF 搜索历史，只输出视频。
        # history_path.write_text(
        #     json.dumps(
        #         history,
        #         indent=2,
        #         ensure_ascii=False,
        #     )
        #     + "\n",
        #     encoding="utf-8",
        # )

        logging.info("Saved video: %s", output_path)
        # logging.info("Saved history: %s", history_path)


if __name__ == "__main__":
    main()
