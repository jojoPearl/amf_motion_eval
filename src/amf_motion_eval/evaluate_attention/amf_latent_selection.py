from __future__ import annotations

import argparse
import copy
import inspect
import gc
import json
import logging
import math
import random
import sys
import time
import types
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

try:
    from wan.modules.model import rope_apply
    from wan.textimage2video import WanTI2V, _sync_and_elapsed
    from wan.utils.fm_solvers import (
        FlowDPMSolverMultistepScheduler,
        get_sampling_sigmas,
        retrieve_timesteps,
    )
    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
    from wan.utils.utils import masks_like
    import torch.distributed as dist
    from tqdm import tqdm
except Exception:
    rope_apply = None
    WanTI2V = object
    _sync_and_elapsed = None
    FlowDPMSolverMultistepScheduler = None
    FlowUniPCMultistepScheduler = None
    get_sampling_sigmas = None
    retrieve_timesteps = None
    masks_like = None
    dist = None
    tqdm = None


Tensor = torch.Tensor

# ---------------------------------------------------------------------
#   - step: five early/early-mid selection points: 4, 6, 8, 11, 15 for 50 steps
#   - beam size 2, three candidates per parent beam
#   - branch noise scale 0.4 with temporal smoothing kernel 5
#   - three-step lookahead before scoring each branch
#   - reward = -AMF-TV + 0.5 * latent_motion
#   - Q/K head reduction = mean_qk
#   - legacy shared seed_g consumption order is preserved deliberately


@dataclass
class AMFSearchConfig:
    # B in beam search. Start with 1.
    beam_size: int = 2

    # K candidates per beam. Start with 2.
    candidates_per_beam: int = 3

    # L lookahead steps used only for reward estimation.
    lookahead_steps: int = 3

    # Select only around these normalized denoising positions.
    # step_i / (num_steps - 1) is matched to these ratios.
    select_step_ratios: Tuple[float, ...] = (0.08, 0.12, 0.16, 0.22, 0.30)

    # Max distance between current ratio and a target ratio to activate selection.
    ratio_tolerance: float = 0.025

    # Branching perturbation scale. Candidate 0 is the original latent.
    # Candidate k>0 uses z_t + branch_noise_scale * std(z_t) * eps.
    branch_noise_scale: float = 0.40

    # Used only by the optional "linear" and "group_norm" reward modes.
    # The default route uses "motion_constraint" instead.
    lambda_motion: float = 0.50

    # Keep False in this cleaned AMF-QK route. Final rerank needs its own
    # Q/K-based implementation and is intentionally disabled below.
    final_rerank: bool = False

    # Candidate-selection rule. Default:
    #   "motion_constraint": keep candidates whose motion is close to candidate0,
    #   then minimize real Q/K-based AMF-TV.
    reward_mode: str = "linear"

    # For motion_constraint:
    # select among candidates with motion >= min_motion_ratio * motion(candidate0).
    min_motion_ratio: float = 0.90

    # If True, perturb candidates using temporally smoothed noise instead of frame-wise Gaussian noise.
    temporal_smooth_noise: bool = True

    # Odd kernel size for smoothing along the frame/time dimension.
    temporal_smooth_kernel: int = 5

    # Debug mode: ignore score and randomly choose from the expanded candidates.
    # Use this only to test whether branching can change the trajectory.
    random_select_debug: bool = False

    # Add a small preference for non-zero candidates so near-ties do not collapse
    # back to the baseline trajectory.
    novelty_bonus: float = 0.0

    # If True, prints candidate scores and kept candidate IDs.
    verbose: bool = True


@dataclass
class AMFSelectionOptions:
    enabled: bool = True
    beam_size: int = 2
    candidates_per_beam: int = 3
    lookahead_steps: int = 3
    step_ratios: Tuple[float, ...] = (0.08, 0.12, 0.16, 0.22, 0.30)
    ratio_tolerance: float = 0.025
    branch_noise_scale: float = 0.40
    lambda_motion: float = 0.50
    final_rerank: bool = False
    reward_mode: str = "linear"
    min_motion_ratio: float = 0.90
    temporal_smooth_noise: bool = True
    temporal_smooth_kernel: int = 5
    random_select_debug: bool = False
    novelty_bonus: float = 0.0
    verbose: bool = True
    history_json: Optional[str] = None

    selection_amf_block_id: int = 15
    selection_max_amf_triplets: int = 4
    selection_motion_temp: float = 2.0
    selection_head_reduce: str = "mean_qk"


@dataclass
class StatefulSelectionBeam:
    latents: Tensor
    scheduler: Any
    cumulative_reward: float = 0.0
    history: Optional[List[Dict[str, float]]] = None

    def with_update(
        self,
        latents: Tensor,
        scheduler: Any,
        reward: float,
        info: Dict[str, float],
    ) -> "StatefulSelectionBeam":
        history = [] if self.history is None else list(self.history)
        history.append(info)
        return StatefulSelectionBeam(
            latents=latents.detach(),
            scheduler=scheduler,
            cumulative_reward=self.cumulative_reward + float(reward),
            history=history,
        )


def _require_wan_runtime() -> None:
    if WanTI2V is object or FlowUniPCMultistepScheduler is None or masks_like is None:
        raise ImportError(
            "Wan runtime is unavailable. Run with Wan2.2 on PYTHONPATH, e.g. "
            "export PYTHONPATH=$AMF_ROOT/src:$PROJECT_ROOT:$WAN_ROOT:$PYTHONPATH"
        )


def _parse_ratios(value: Any) -> Tuple[float, ...]:
    if isinstance(value, str):
        return tuple(float(x.strip()) for x in value.split(",") if x.strip())
    if isinstance(value, (list, tuple)):
        return tuple(float(x) for x in value)
    return (float(value),)


def _latent_motion_only(preview_latents: Tensor) -> float:
    """Motion magnitude used only as an anti-static constraint.

    This is not the selection quality score. The quality score should be
    real Q/K-based AMF-TV.
    """
    if preview_latents.ndim < 2 or preview_latents.shape[1] < 2:
        return 0.0
    velocity = preview_latents[:, 1:] - preview_latents[:, :-1]
    motion = velocity.detach().float().square().mean().sqrt()
    return float(motion.cpu())


def _amf_qk_score_from_record(q: Tensor, k: Tensor, grid_size: Any, opts: Any) -> Dict[str, Any]:
    """Compute real attention-derived AMF temporal consistency from captured Q/K.

    amf_tv is computed from AMF flow consistency:
        F_{t->t+1}(p) = sum_q A_t(p,q) * coord(q) - coord(p)
        AMF-TV = mean_p |F_{t+1->t+2}(p) - F_{t->t+1}(p)|_1

    Lower amf_tv means smoother attention-derived temporal correspondence.
    """
    loss, debug = _amf_loss(q, k, grid_size, opts)
    if loss is None:
        raise RuntimeError(f"AMF-QK score failed: {debug}")
    if not torch.isfinite(loss):
        raise RuntimeError(f"AMF-QK score is non-finite: {loss}, debug={debug}")
    return {
        "amf_tv": float(loss.detach().float().cpu()),
        "debug": debug,
    }

def _temporal_smooth_noise(x: Tensor, kernel_size: int) -> Tensor:
    if kernel_size <= 1:
        return x
    if kernel_size % 2 == 0:
        kernel_size += 1

    frame_dim = 2 if x.ndim == 5 else 1
    if x.ndim < 2 or x.shape[frame_dim] < 2:
        return x

    pad = kernel_size // 2
    y = x.transpose(frame_dim, -1)
    orig_shape = y.shape
    y = y.reshape(-1, 1, orig_shape[-1])
    weight = torch.ones(1, 1, kernel_size, device=x.device, dtype=x.dtype) / float(kernel_size)
    y = torch.nn.functional.pad(y, (pad, pad), mode="replicate")
    y = torch.nn.functional.conv1d(y, weight)
    return y.reshape(orig_shape).transpose(frame_dim, -1)


def _make_candidate_inputs(latents: Tensor, cfg: AMFSearchConfig, generator: Optional[torch.Generator]) -> List[Tensor]:
    candidates = [latents.detach()]
    if cfg.candidates_per_beam == 1:
        return candidates

    std = latents.detach().float().std().clamp(min=1e-6)
    scale = float(cfg.branch_noise_scale) * std
    for _ in range(cfg.candidates_per_beam - 1):
        noise = torch.randn(
            latents.shape,
            device=latents.device,
            dtype=latents.dtype,
            generator=generator,
        )
        if cfg.temporal_smooth_noise:
            noise = _temporal_smooth_noise(noise, int(cfg.temporal_smooth_kernel))
        noise_std = noise.detach().float().std().clamp(min=1e-6)
        noise = noise / noise_std.to(noise.dtype)
        candidates.append((latents + scale.to(latents.dtype) * noise).detach())
    return candidates


def _assign_candidate_rewards(items: List[Dict[str, Any]], cfg: AMFSearchConfig) -> None:
    if not items:
        return
    if cfg.reward_mode == "linear":
        for item in items:
            novelty = float(cfg.novelty_bonus) if int(item["candidate_id"]) > 0 else 0.0
            item["reward"] = -float(item["amf_tv"]) + float(cfg.lambda_motion) * float(item["motion"]) + novelty
        return

    tvs = torch.tensor([float(x["amf_tv"]) for x in items], dtype=torch.float32)
    motions = torch.tensor([float(x["motion"]) for x in items], dtype=torch.float32)
    eps = 1e-8

    if cfg.reward_mode == "group_norm":
        tv_norm = (tvs - tvs.min()) / (tvs.max() - tvs.min() + eps)
        motion_norm = (motions - motions.min()) / (motions.max() - motions.min() + eps)
        rewards = -tv_norm + float(cfg.lambda_motion) * motion_norm
        for item, reward in zip(items, rewards):
            novelty = float(cfg.novelty_bonus) if int(item["candidate_id"]) > 0 else 0.0
            item["reward"] = float(reward.item()) + novelty
        return

    motion0 = float(items[0]["motion"])
    min_motion = float(cfg.min_motion_ratio) * motion0
    valid = [i for i, item in enumerate(items) if float(item["motion"]) >= min_motion]
    if not valid:
        valid = list(range(len(items)))
    best_tv = min(float(items[i]["amf_tv"]) for i in valid)

    for i, item in enumerate(items):
        tv = float(item["amf_tv"])
        motion = float(item["motion"])
        novelty = float(cfg.novelty_bonus) if int(item["candidate_id"]) > 0 else 0.0
        if i in valid:
            item["reward"] = -(tv - best_tv) + novelty
        else:
            item["reward"] = -(tv - best_tv) - 1.0 - abs(min_motion - motion) + novelty


def _beams_to_jsonable(beams: Sequence[StatefulSelectionBeam]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for beam_id, beam in enumerate(beams):
        out.append(
            {
                "beam_id": int(beam_id),
                "cumulative_reward": float(beam.cumulative_reward),
                "history": beam.history or [],
            }
        )
    return out


class AMFSelectionWanTI2V(WanTI2V):
    def set_selection_options(self, args) -> None:
        self.selection_options = AMFSelectionOptions(
            enabled=bool(getattr(args, "add_selection", True)),
            beam_size=int(getattr(args, "selection_beam_size", 2)),
            candidates_per_beam=int(getattr(args, "selection_candidates_per_beam", 3)),
            lookahead_steps=int(getattr(args, "selection_lookahead_steps", 3)),
            step_ratios=_parse_ratios(getattr(args, "selection_step_ratios", "0.08,0.12,0.16,0.22,0.30")),
            ratio_tolerance=float(getattr(args, "selection_ratio_tolerance", 0.025)),
            branch_noise_scale=float(getattr(args, "selection_branch_noise_scale", 0.40)),
            lambda_motion=float(getattr(args, "selection_lambda_motion", 0.50)),
            final_rerank=bool(getattr(args, "selection_final_rerank", False)),
            reward_mode=str(getattr(args, "selection_reward_mode", "linear")),
            min_motion_ratio=float(getattr(args, "selection_min_motion_ratio", 0.90)),
            temporal_smooth_noise=bool(getattr(args, "selection_temporal_smooth_noise", True)),
            temporal_smooth_kernel=int(getattr(args, "selection_temporal_smooth_kernel", 5)),
            random_select_debug=bool(getattr(args, "selection_random_select_debug", False)),
            novelty_bonus=float(getattr(args, "selection_novelty_bonus", 0.0)),
            verbose=bool(getattr(args, "selection_verbose", True)),
            history_json=getattr(args, "selection_history_json", None),
            selection_amf_block_id=int(getattr(args, "selection_amf_block_id", getattr(args, "block_id", 15))),
            selection_max_amf_triplets=int(getattr(args, "selection_max_amf_triplets", 4)),
            selection_motion_temp=float(getattr(args, "selection_motion_temp", 2.0)),
            selection_head_reduce=str(getattr(args, "selection_head_reduce", "mean_qk")),
        )
        self.selection_history: List[Dict[str, Any]] = []

    def _selection_config(self) -> AMFSearchConfig:
        opts = self.selection_options
        return AMFSearchConfig(
            beam_size=opts.beam_size,
            candidates_per_beam=opts.candidates_per_beam,
            lookahead_steps=opts.lookahead_steps,
            select_step_ratios=opts.step_ratios,
            ratio_tolerance=opts.ratio_tolerance,
            branch_noise_scale=opts.branch_noise_scale,
            lambda_motion=opts.lambda_motion,
            final_rerank=opts.final_rerank,
            reward_mode=opts.reward_mode,
            min_motion_ratio=opts.min_motion_ratio,
            temporal_smooth_noise=opts.temporal_smooth_noise,
            temporal_smooth_kernel=opts.temporal_smooth_kernel,
            random_select_debug=opts.random_select_debug,
            novelty_bonus=opts.novelty_bonus,
            verbose=opts.verbose,
        )

    def _save_selection_history(self, beams: Sequence[StatefulSelectionBeam], selection_step_ids: Sequence[int]) -> None:
        self.selection_history = _beams_to_jsonable(beams)
        path = getattr(self.selection_options, "history_json", None)
        if not path or self.rank != 0:
            return
        payload = {
            "selection_step_ids": list(selection_step_ids),
            "options": self.selection_options.__dict__,
            "rng_profile": "legacy_shared_seed_g",
            "beams": self.selection_history,
        }
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        logging.info("Saved AMF selection history to %s", out_path)

    def generate(self, input_prompt, img=None, size=(1280, 704), max_area=704 * 1280,
                 frame_num=81, shift=5.0, sample_solver='unipc', sampling_steps=50,
                 guide_scale=5.0, n_prompt="", seed=-1, offload_model=True):
        if img is not None and getattr(self.selection_options, "enabled", True):
            raise NotImplementedError("AMF latent selection currently supports TI2V text-to-video only.")
        return super().generate(
            input_prompt=input_prompt,
            img=img,
            size=size,
            max_area=max_area,
            frame_num=frame_num,
            shift=shift,
            sample_solver=sample_solver,
            sampling_steps=sampling_steps,
            guide_scale=guide_scale,
            n_prompt=n_prompt,
            seed=seed,
            offload_model=offload_model,
        )

    def t2v(self, input_prompt, size=(1280, 704), frame_num=121, shift=5.0,
            sample_solver='unipc', sampling_steps=50, guide_scale=5.0,
            n_prompt="", seed=-1, offload_model=True):
        _require_wan_runtime()
        if not hasattr(self, "selection_options"):
            self.set_selection_options(type("Args", (), {})())
        if not self.selection_options.enabled:
            return super().t2v(input_prompt, size, frame_num, shift, sample_solver,
                               sampling_steps, guide_scale, n_prompt, seed, offload_model)

        total_start = time.perf_counter()
        target_shape = (self.vae.model.z_dim, (frame_num - 1) // self.vae_stride[0] + 1,
                        size[1] // self.vae_stride[1], size[0] // self.vae_stride[2])
        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size
        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        stage_start = time.perf_counter()
        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]
        logging.info("[timing] ti2v_selection.prompt_encode_seconds=%.3f",
                     _sync_and_elapsed(stage_start, self.device))

        stage_start = time.perf_counter()
        noise = [torch.randn(target_shape[0], target_shape[1], target_shape[2], target_shape[3],
                             dtype=torch.float32, device=self.device, generator=seed_g)]
        logging.info("[timing] ti2v_selection.noise_init_seconds=%.3f",
                     _sync_and_elapsed(stage_start, self.device))

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)
        with (torch.amp.autocast('cuda', dtype=self.param_dtype), torch.no_grad(), no_sync()):
            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(sample_scheduler, device=self.device, sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            latents = noise
            _, mask2 = masks_like(noise, zero=False)
            arg_c = {'context': context, 'seq_len': seq_len}
            arg_null = {'context': context_null, 'seq_len': seq_len}

            stage_start = time.perf_counter()
            if offload_model or self.init_on_cpu:
                self.model.to(self.device)
                torch.cuda.empty_cache()
            logging.info("[timing] ti2v_selection.model_to_gpu_seconds=%.3f",
                         _sync_and_elapsed(stage_start, self.device))

            def make_timestep(t):
                timestep = torch.stack([t])
                temp_ts = (mask2[0][0][:, ::2, ::2] * timestep).flatten()
                temp_ts = torch.cat([temp_ts, temp_ts.new_ones(seq_len - temp_ts.size(0)) * timestep])
                return temp_ts.unsqueeze(0)

            def scheduler_step(candidate_latents: Tensor, scheduler: Any, step_i: int) -> Tensor:
                t = timesteps[step_i]
                timestep = make_timestep(t)
                latent_model_input = [candidate_latents]
                noise_pred_cond = self.model(latent_model_input, t=timestep, **arg_c)[0]
                noise_pred_uncond = self.model(latent_model_input, t=timestep, **arg_null)[0]
                noise_pred = noise_pred_uncond + guide_scale * (noise_pred_cond - noise_pred_uncond)
                temp_x0 = scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    candidate_latents.unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                return temp_x0.squeeze(0).detach()

            def capture_qk_record(candidate_latents: Tensor, step_i: int) -> Optional[Dict[str, Any]]:
                """Extra forward used only for scoring.

                It captures self-attention Q/K at one Wan block and converts them
                into a real AMF-QK temporal consistency score. It does not update
                the latent and it is only used during lookahead scoring.
                """
                if step_i >= len(timesteps):
                    return None

                attn = original_forward = capture = None
                t = timesteps[step_i]
                timestep = make_timestep(t)
                score_opts = types.SimpleNamespace(
                    max_amf_triplets=int(self.selection_options.selection_max_amf_triplets),
                    motion_temp=float(self.selection_options.selection_motion_temp),
                    head_reduce=str(self.selection_options.selection_head_reduce),
                )

                try:
                    attn, original_forward, capture = _patch_attention_for_capture(
                        self.model,
                        int(self.selection_options.selection_amf_block_id),
                    )
                    try:
                        _ = self.model([candidate_latents], t=timestep, **arg_c)
                    except StopAfterCapture:
                        pass

                    if capture.query is None or capture.key is None or capture.grid_size is None:
                        raise RuntimeError("AMF-QK capture failed: no q/k/grid captured")

                    score = _amf_qk_score_from_record(
                        capture.query,
                        capture.key,
                        capture.grid_size,
                        score_opts,
                    )
                    return score
                finally:
                    if attn is not None:
                        _restore_attention(attn, original_forward)
                    if capture is not None:
                        capture.query = None
                        capture.key = None
                        capture.grid_size = None

            def score_with_lookahead(candidate_latents: Tensor, scheduler: Any, start_step_i: int) -> Dict[str, float]:
                """Score one candidate using real Q/K-based AMF-TV.

                AMF-TV is captured during lookahead from Wan self-attention Q/K.
                Latent motion is kept only as an anti-static constraint signal.
                There is no latent-proxy AMF fallback in this selection route.
                """
                preview_scheduler = copy.deepcopy(scheduler)
                z = candidate_latents.detach()
                max_j = min(cfg.lookahead_steps, max(0, len(timesteps) - start_step_i))

                amf_scores: List[float] = []

                for j in range(max_j):
                    idx = start_step_i + j

                    rec = capture_qk_record(z, idx)
                    if rec is not None:
                        amf_scores.append(float(rec["amf_tv"]))

                    # Normal lookahead denoising. This recomputes model output from z.
                    z = scheduler_step(z, preview_scheduler, idx)

                if not amf_scores:
                    raise RuntimeError("No AMF-QK records were captured during lookahead scoring")

                return {
                    "amf_tv": float(sum(amf_scores) / len(amf_scores)),
                    "motion": float(_latent_motion_only(z)),
                    "amf_qk_records": float(len(amf_scores)),
                }

            stage_start = time.perf_counter()
            cfg = self._selection_config()
            if not selection_step_ids:
                raise ValueError("AMF latent selection is enabled, but no selection step matched.")
            logging.info("[amf-selection] active selection_step_ids=%s", list(selection_step_ids))
            logging.info(
                "[amf-selection] reproduction_profile beam=%s candidates=%s lookahead=%s "
                "noise=%.3f reward=%s lambda_motion=%.3f head_reduce=%s rng=legacy_shared_seed_g",
                cfg.beam_size,
                cfg.candidates_per_beam,
                cfg.lookahead_steps,
                cfg.branch_noise_scale,
                cfg.reward_mode,
                cfg.lambda_motion,
                self.selection_options.selection_head_reduce,
            )

            beams: List[StatefulSelectionBeam] = [
                StatefulSelectionBeam(latents=latents[0].detach(), scheduler=copy.deepcopy(sample_scheduler), history=[])
            ]

            for step_i in tqdm(range(len(timesteps))):
                if step_i not in selection_step_ids:
                    beams = [
                        StatefulSelectionBeam(
                            latents=scheduler_step(beam.latents, beam.scheduler, step_i),
                            scheduler=beam.scheduler,
                            cumulative_reward=beam.cumulative_reward,
                            history=beam.history,
                        )
                        for beam in beams
                    ]
                    continue

                expanded: List[StatefulSelectionBeam] = []
                for beam_id, beam in enumerate(beams):
                    candidate_inputs = _make_candidate_inputs(beam.latents, cfg, seed_g)
                    cand0 = candidate_inputs[0].detach()
                    group_items: List[Dict[str, Any]] = []
                    for candidate_id, candidate_input in enumerate(candidate_inputs):
                        candidate_scheduler = copy.deepcopy(beam.scheduler)
                        candidate_next = scheduler_step(candidate_input, candidate_scheduler, step_i)
                        score = score_with_lookahead(candidate_next, candidate_scheduler, step_i + 1)
                        latent_dist = 0.0 if candidate_id == 0 else float(
                            (candidate_input.detach().float() - cand0.float()).square().mean().sqrt().item())
                        group_items.append({
                            "beam": beam,
                            "beam_id": beam_id,
                            "candidate_id": candidate_id,
                            "candidate_next": candidate_next,
                            "candidate_scheduler": candidate_scheduler,
                            "amf_tv": float(score["amf_tv"]),
                            "motion": float(score.get("motion", 0.0)),
                            "latent_dist": latent_dist,
                            "amf_qk_records": float(score.get("amf_qk_records", 0.0)),
                        })

                    _assign_candidate_rewards(group_items, cfg)
                    if cfg.random_select_debug:
                        perm = torch.randperm(
                            len(group_items),
                            generator=seed_g,
                            device=beam.latents.device,
                        ).tolist()
                        group_items = [group_items[i] for i in perm]
                    else:
                        group_items.sort(key=lambda x: x["beam"].cumulative_reward + x["reward"], reverse=True)

                    if cfg.verbose:
                        logging.info("[amf-selection] step=%03d beam=%s mode=%s", step_i, beam_id, cfg.reward_mode)
                        for item in group_items:
                            logging.info(
                                "[amf-selection]   cand=%s amf_qk_tv=%.6f motion=%.6f latent_dist=%.6f qk_records=%s reward=%.6f",
                                item["candidate_id"],
                                item["amf_tv"],
                                item["motion"],
                                item["latent_dist"],
                                int(item.get("amf_qk_records", 0)),
                                item["reward"],
                            )

                    for item in group_items:
                        info = {
                            "step_i": float(step_i),
                            "beam_id": float(item["beam_id"]),
                            "candidate_id": float(item["candidate_id"]),
                            "amf_tv": float(item["amf_tv"]),
                            "motion": float(item["motion"]),
                            "latent_dist": float(item["latent_dist"]),
                            "amf_qk_records": float(item.get("amf_qk_records", 0.0)),
                            "reward": float(item["reward"]),
                        }
                        expanded.append(item["beam"].with_update(
                            item["candidate_next"], item["candidate_scheduler"], item["reward"], info))

                expanded.sort(key=lambda item: item.cumulative_reward, reverse=True)
                beams = expanded[: cfg.beam_size]
                if cfg.verbose:
                    logging.info(
                        "[amf-selection] keep top-%s ids=%s cumulative_rewards=%s",
                        cfg.beam_size,
                        [(int((b.history or [])[-1]["beam_id"]), int((b.history or [])[-1]["candidate_id"])) for b in beams],
                        [round(b.cumulative_reward, 6) for b in beams],
                    )

            if cfg.final_rerank:
                raise NotImplementedError(
                    "selection_final_rerank is disabled in the cleaned AMF-QK route. "
                    "Use selection_final_rerank=False, or implement a final Q/K-based rerank explicitly."
                )

            best_beam = max(beams, key=lambda item: item.cumulative_reward)
            latents = [best_beam.latents]
            self._save_selection_history(beams, selection_step_ids)
            logging.info("[timing] ti2v_selection.sampling_seconds=%.3f steps=%s",
                         _sync_and_elapsed(stage_start, self.device), sampling_steps)

            x0 = latents
            stage_start = time.perf_counter()
            if offload_model:
                self.model.cpu()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            logging.info("[timing] ti2v_selection.model_to_cpu_seconds=%.3f",
                         _sync_and_elapsed(stage_start, self.device))

            if self.rank == 0:
                stage_start = time.perf_counter()
                videos = self.vae.decode(x0)
                logging.info("[timing] ti2v_selection.vae_decode_seconds=%.3f",
                             _sync_and_elapsed(stage_start, self.device))

        del noise, latents, sample_scheduler
        if offload_model:
            stage_start = time.perf_counter()
            gc.collect()
            torch.cuda.synchronize()
            logging.info("[timing] ti2v_selection.gc_seconds=%.3f", time.perf_counter() - stage_start)
        if dist is not None and dist.is_initialized():
            dist.barrier()
        logging.info("[timing] ti2v_selection.total_seconds=%.3f",
                     _sync_and_elapsed(total_start, self.device))
        return videos[0] if self.rank == 0 else None


class StopAfterCapture(RuntimeError):
    pass


class AMFAttentionCapture:
    def __init__(self):
        self.query = None
        self.key = None
        self.grid_size = None

    def save(self, q, k, grid_sizes):
        self.query = q
        self.key = k
        self.grid_size = grid_sizes[0]


def _patch_attention_for_capture(model, block_id: int) -> Tuple[Any, Any, AMFAttentionCapture]:
    if rope_apply is None:
        raise ImportError("Wan rope_apply is unavailable; cannot attach AMF guidance.")
    if not hasattr(model, "blocks"):
        raise AttributeError("Wan model has no blocks; cannot attach AMF guidance.")
    if block_id < 0 or block_id >= len(model.blocks):
        raise ValueError(f"block_id={block_id} out of range for {len(model.blocks)} blocks")

    attn = model.blocks[block_id].self_attn
    original_forward = attn.forward
    capture = AMFAttentionCapture()

    def forward(self, x, seq_lens, grid_sizes, freqs):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        q = rope_apply(q, grid_sizes, freqs)
        k = rope_apply(k, grid_sizes, freqs)
        capture.save(q, k, grid_sizes)
        raise StopAfterCapture()

    attn.forward = types.MethodType(forward, attn)
    return attn, original_forward, capture


def _restore_attention(attn, original_forward):
    attn.forward = original_forward


def _reshape_qk(q: Tensor, k: Tensor, grid_size):
    frames, grid_h, grid_w = [int(x) for x in grid_size]
    hw = grid_h * grid_w
    seq_len = frames * hw
    if q.ndim != 4 or k.ndim != 4:
        raise ValueError(f"expected q/k [B,S,H,D], got q={tuple(q.shape)} k={tuple(k.shape)}")
    if q.shape[1] < seq_len or k.shape[1] < seq_len:
        raise ValueError(f"q/k seq too short for grid={grid_size}: q={q.shape} k={k.shape}")
    q = q[0, :seq_len].permute(1, 0, 2).reshape(q.shape[2], frames, hw, q.shape[-1])
    k = k[0, :seq_len].permute(1, 0, 2).reshape(k.shape[2], frames, hw, k.shape[-1])
    return q, k, frames, grid_h, grid_w


def _pair_attention(q_i: Tensor, k_j: Tensor, temp: float, head_reduce: str):
    if head_reduce == "single":
        head_id = q_i.shape[0] // 2
        q = q_i[head_id]
        k = k_j[head_id]
        logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
        return torch.softmax(logits * temp, dim=-1)
    if head_reduce == "mean_qk":
        q = q_i.mean(dim=0)
        k = k_j.mean(dim=0)
        logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
        return torch.softmax(logits * temp, dim=-1)
    if head_reduce == "full":
        logits = torch.matmul(q_i, k_j.transpose(-1, -2)) / math.sqrt(q_i.shape[-1])
        return torch.softmax(logits.mean(dim=0) * temp, dim=-1)
    raise ValueError(f"unsupported head_reduce={head_reduce}")


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


def _amf_loss(q, k, grid_size, opts: Any, rng: Optional[torch.Generator] = None):
    q, k, frames, grid_h, grid_w = _reshape_qk(q, k, grid_size)
    if frames < 3:
        return None, {"reason": "need at least 3 latent frames"}
    triplet_ids = list(range(frames - 2))
    if opts.max_amf_triplets > 0 and len(triplet_ids) > opts.max_amf_triplets:
        if rng is not None:
            perm = torch.randperm(len(triplet_ids), generator=rng, device=q.device)[:opts.max_amf_triplets].tolist()
            triplet_ids = sorted([triplet_ids[i] for i in perm])
        else:
            keep = torch.linspace(0, len(triplet_ids) - 1, steps=opts.max_amf_triplets).round().long().tolist()
            triplet_ids = [triplet_ids[i] for i in keep]

    losses = []
    per_triplet = []
    for t in triplet_ids:
        attn_01 = _pair_attention(q[:, t], k[:, t + 1], opts.motion_temp, opts.head_reduce)
        attn_12 = _pair_attention(q[:, t + 1], k[:, t + 2], opts.motion_temp, opts.head_reduce)
        flow_01 = _attention_to_flow(attn_01, grid_h, grid_w)
        flow_12 = _attention_to_flow(attn_12, grid_h, grid_w)
        loss_t = (flow_12 - flow_01).abs().sum(dim=-1).mean()
        losses.append(loss_t)
        per_triplet.append(float(loss_t.detach().item()))
    return torch.stack(losses).mean(), {"triplet_ids": triplet_ids, "per_triplet_loss": per_triplet}



# ---------------------------------------------------------------------
# CLI entry point for the cleaned selection route
# ---------------------------------------------------------------------

def _parse_size(value: str) -> Tuple[int, int]:
    text = str(value).lower().replace("*", "x").replace(",", "x")
    parts = [p.strip() for p in text.split("x") if p.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("size must look like 832x480")
    return int(parts[0]), int(parts[1])


def _add_selection_cli_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--add_selection", action="store_true", default=True)
    parser.add_argument("--no_selection", dest="add_selection", action="store_false")
    parser.add_argument("--selection_beam_size", type=int, default=2)
    parser.add_argument("--selection_candidates_per_beam", type=int, default=3)
    parser.add_argument("--selection_lookahead_steps", type=int, default=3)
    parser.add_argument("--selection_step_ratios", type=str, default="0.08,0.12,0.16,0.22,0.30")
    parser.add_argument("--selection_ratio_tolerance", type=float, default=0.025)
    parser.add_argument("--selection_branch_noise_scale", type=float, default=0.40)
    parser.add_argument("--selection_lambda_motion", type=float, default=0.50)
    parser.add_argument("--selection_reward_mode", type=str, default="linear",
                        choices=["linear", "group_norm", "motion_constraint"])
    parser.add_argument("--selection_min_motion_ratio", type=float, default=0.90)
    parser.add_argument("--selection_temporal_smooth_noise", action="store_true", default=True)
    parser.add_argument("--no_selection_temporal_smooth_noise", dest="selection_temporal_smooth_noise", action="store_false")
    parser.add_argument("--selection_temporal_smooth_kernel", type=int, default=5)
    parser.add_argument("--selection_random_select_debug", action="store_true", default=False)
    parser.add_argument("--selection_novelty_bonus", type=float, default=0.0)
    parser.add_argument("--selection_verbose", action="store_true", default=True)
    parser.add_argument("--selection_quiet", dest="selection_verbose", action="store_false")
    parser.add_argument("--selection_history_json", type=str, default=None)
    parser.add_argument("--selection_final_rerank", action="store_true", default=False,
                        help="Kept for compatibility, but this cleaned route raises if enabled.")
    parser.add_argument("--selection_amf_block_id", type=int, default=15)
    parser.add_argument("--selection_max_amf_triplets", type=int, default=4)
    parser.add_argument("--selection_motion_temp", type=float, default=2.0)
    parser.add_argument("--selection_head_reduce", type=str, default="mean_qk",
                        choices=["single", "mean_qk", "full"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wan T2V generation with AMF-QK candidate selection only."
    )
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--task", type=str, default="ti2v-5B",
                        help="Key used in wan.configs.WAN_CONFIGS when available.")
    parser.add_argument("--size", type=_parse_size, default=(1280, 704))
    parser.add_argument("--frame_num", type=int, default=121)
    parser.add_argument("--shift", type=float, default=5.0)
    parser.add_argument("--sample_solver", type=str, default="unipc", choices=["unipc", "dpm++"])
    parser.add_argument("--sampling_steps", type=int, default=50)
    parser.add_argument("--guide_scale", type=float, default=5.0)
    parser.add_argument("--n_prompt", type=str, default="")
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--offload_model", action="store_true", default=True)
    parser.add_argument("--no_offload_model", dest="offload_model", action="store_false")
    parser.add_argument("--t5_cpu", action="store_true", default=False)
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--output_pt", type=str, default=None,
                        help="Optional path to torch.save the returned video tensor.")
    parser.add_argument("--output_video", type=str, default=None,
                        help="Optional path for the decoded MP4 video.")
    parser.add_argument("--output_fps", type=float, default=24.0,
                        help="Frame rate used by --output_video.")
    parser.add_argument("--log_level", type=str, default="INFO")
    parser.add_argument(
        "--allow_overwrite", action="store_true", default=False,
        help="Allow output_pt/history_json to overwrite existing files.",
    )
    parser.add_argument(
        "--expected_candidate_path",
        type=str,
        default=None,
        help=(
            "Optional comma-separated candidate IDs for reproduction validation, "
            "for example 2,1,2,2,2 for the known seed205 result."
        ),
    )
    _add_selection_cli_args(parser)
    return parser.parse_args()


def _load_wan_config(task: str) -> Any:
    try:
        from wan.configs import WAN_CONFIGS  # type: ignore
    except Exception as exc:
        raise ImportError(
            "Cannot import wan.configs.WAN_CONFIGS. Make sure Wan2.2 is on PYTHONPATH."
        ) from exc
    if task not in WAN_CONFIGS:
        available = ", ".join(sorted(str(k) for k in WAN_CONFIGS.keys()))
        raise KeyError(f"Unknown task={task!r}. Available WAN_CONFIGS keys: {available}")
    return WAN_CONFIGS[task]


def _instantiate_selection_model(args: argparse.Namespace) -> AMFSelectionWanTI2V:
    _require_wan_runtime()
    config = _load_wan_config(args.task)

    candidate_kwargs: Dict[str, Any] = {
        "config": config,
        "checkpoint_dir": args.ckpt_dir,
        "ckpt_dir": args.ckpt_dir,
        "device_id": args.device_id,
        "rank": args.rank,
        "t5_cpu": args.t5_cpu,
    }

    signature = inspect.signature(AMFSelectionWanTI2V.__init__)
    accepted = {
        name for name, param in signature.parameters.items()
        if name != "self" and param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY)
    }
    filtered_kwargs = {k: v for k, v in candidate_kwargs.items() if k in accepted}

    try:
        model = AMFSelectionWanTI2V(**filtered_kwargs)
    except TypeError:
        # Fallback for common Wan constructors that expect config as the first positional argument.
        filtered_kwargs.pop("config", None)
        model = AMFSelectionWanTI2V(config, **filtered_kwargs)

    model.set_selection_options(args)
    return model


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if not args.allow_overwrite:
        for candidate_path in (args.output_pt, args.output_video, args.selection_history_json):
            if candidate_path and Path(candidate_path).exists():
                raise FileExistsError(
                    f"Refusing to overwrite existing output: {candidate_path}. "
                    "Choose a new reproduction directory or pass --allow_overwrite."
                )

    model = _instantiate_selection_model(args)
    video = model.t2v(
        input_prompt=args.prompt,
        size=args.size,
        frame_num=args.frame_num,
        shift=args.shift,
        sample_solver=args.sample_solver,
        sampling_steps=args.sampling_steps,
        guide_scale=args.guide_scale,
        n_prompt=args.n_prompt,
        seed=args.seed,
        offload_model=args.offload_model,
    )

    if args.expected_candidate_path and args.rank == 0:
        expected = [
            int(x.strip())
            for x in args.expected_candidate_path.split(",")
            if x.strip()
        ]
        if not model.selection_history:
            raise RuntimeError("No selection history is available for path validation.")
        actual = [
            int(item["candidate_id"])
            for item in model.selection_history[0].get("history", [])
        ]
        if actual != expected:
            raise RuntimeError(
                "Reproduction path mismatch: "
                f"expected={expected}, actual={actual}. "
                "Check prompt, checkpoint, Wan commit, PyTorch/CUDA versions, and RNG order."
            )
        logging.info("Validated candidate path: %s", actual)

    if args.output_pt and args.rank == 0:
        out_path = Path(args.output_pt)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(video, out_path)
        logging.info("Saved returned video tensor to %s", out_path)

    if args.output_video and args.rank == 0:
        from wan.utils.utils import save_video

        out_path = Path(args.output_video)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        logging.info("Saving generated video to %s", out_path)
        save_video(
            tensor=video[None],
            save_file=str(out_path),
            fps=float(args.output_fps),
            nrow=1,
            normalize=True,
            value_range=(-1, 1),
        )
        if not out_path.is_file() or out_path.stat().st_size == 0:
            raise RuntimeError(f"Video encoder did not create a non-empty file: {out_path}")
        logging.info("Saved generated video to %s", out_path)


if __name__ == "__main__":
    main()
