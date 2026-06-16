#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.io import read_video


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WAN_REPO = PROJECT_ROOT / "Wan2.2"

if str(DEFAULT_WAN_REPO) not in sys.path:
    sys.path.insert(0, str(DEFAULT_WAN_REPO))

if str(PROJECT_ROOT / "amf_motion_eval") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "amf_motion_eval"))

from guidance_utils import compute_pair_attention_from_heads  # noqa: E402


BBox = Tuple[int, int, int, int]


@dataclass
class CaseConfig:
    case_id: str
    video_path: str
    frame_t: int
    block_id: int
    size: str
    frame_num: int
    reference_prompt: str
    bboxes: Dict[str, BBox]

    @property
    def frame_t1(self) -> int:
        return self.frame_t + 1


def load_cases(path: Path) -> List[CaseConfig]:
    data = json.loads(path.read_text(encoding="utf-8"))
    cases = []
    for entry in data["cases"]:
        cases.append(
            CaseConfig(
                case_id=entry["case_id"],
                video_path=entry["video_path"],
                frame_t=int(entry["frame_t"]),
                block_id=int(entry.get("block_id", data.get("default_block_id", 15))),
                size=entry.get("size", data.get("default_size", "1280*704")),
                frame_num=int(entry["frame_num"]),
                reference_prompt=entry.get("reference_prompt", ""),
                bboxes={
                    key: tuple(int(v) for v in value)
                    for key, value in entry["bboxes"].items()
                },
            )
        )
    return cases


def make_extractor(case: CaseConfig, output_dir: Path) -> WanAMFExtractor:
    from wan_hooks.wan_amf import WanAMFExtractor, parse_args as parse_wan_args

    argv = [
        "--mode",
        "extract",
        "--video_path",
        case.video_path,
        "--output_path",
        str(output_dir),
        "--size",
        case.size,
        "--frame_num",
        str(case.frame_num),
        "--guidance_blocks",
        str(case.block_id),
        "--reference_prompt",
        case.reference_prompt,
    ]
    prev_argv = sys.argv
    try:
        sys.argv = ["wan_amf.py", *argv]
        config = parse_wan_args()
    finally:
        sys.argv = prev_argv
    return WanAMFExtractor(config)


def capture_qk(extractor: WanAMFExtractor, block_id: int):
    from wan.utils.utils import masks_like

    extractor._reset_all_processors()
    extractor._set_processor_capture_mode(with_grad=False)

    context = extractor.encode_prompt()
    extractor.model.to(extractor.device)

    _, mask2 = masks_like([extractor.motion_latent], zero=False)
    ref_t = extractor.resolve_reference_timestep()
    timestep = extractor.make_timestep(ref_t, mask2)

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=extractor.dtype):
        extractor.model(
            [extractor.motion_latent],
            t=timestep,
            context=context,
            seq_len=extractor.seq_len,
        )

    processor = extractor.attn_processors[block_id]
    if processor.query is None or processor.key is None or processor.grid_size is None:
        raise RuntimeError(f"Failed to capture Q/K for block {block_id}")

    grid_size = tuple(int(v) for v in processor.grid_size.tolist())
    query = processor.query.detach().cpu()
    key = processor.key.detach().cpu()

    extractor._reset_all_processors()
    return query, key, grid_size


def reshape_qk(q: torch.Tensor, k: torch.Tensor, grid_size: Tuple[int, int, int]):
    frames, grid_h, grid_w = [int(v) for v in grid_size]
    hw = grid_h * grid_w
    seq_len = frames * hw

    q = q[0, :seq_len].permute(1, 0, 2).reshape(q.shape[2], frames, hw, q.shape[-1]).float()
    k = k[0, :seq_len].permute(1, 0, 2).reshape(k.shape[2], frames, hw, k.shape[-1]).float()
    return q, k


def pixel_bbox_to_token_bbox(bbox: BBox, image_w: int, image_h: int, grid_w: int, grid_h: int):
    x0, y0, x1, y1 = bbox
    tx0 = max(0, min(grid_w - 1, math.floor(x0 * grid_w / image_w)))
    ty0 = max(0, min(grid_h - 1, math.floor(y0 * grid_h / image_h)))
    tx1 = max(tx0 + 1, min(grid_w, math.ceil(x1 * grid_w / image_w)))
    ty1 = max(ty0 + 1, min(grid_h, math.ceil(y1 * grid_h / image_h)))
    return tx0, ty0, tx1, ty1


def bbox_to_token_indices(bbox: Tuple[int, int, int, int], grid_w: int):
    x0, y0, x1, y1 = bbox
    indices = []
    for yy in range(y0, y1):
        row_start = yy * grid_w
        for xx in range(x0, x1):
            indices.append(row_start + xx)
    return torch.tensor(indices, dtype=torch.long)


def compute_match(attn: torch.Tensor, source_idx: torch.Tensor, target_idx: torch.Tensor) -> float:
    source_attn = attn.index_select(0, source_idx)
    numerator = source_attn.index_select(1, target_idx).sum().item()
    denominator = source_attn.sum().item()
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def aggregate_heatmap(attn: torch.Tensor, source_idx: torch.Tensor, grid_h: int, grid_w: int):
    source_attn = attn.index_select(0, source_idx)
    heat = source_attn.mean(dim=0).reshape(grid_h, grid_w)
    heat = heat / heat.max().clamp_min(1e-8)
    return heat.numpy()


def read_frame_pair(video_path: str, frame_t: int):
    video, _, info = read_video(video_path, pts_unit="sec")
    if frame_t + 1 >= video.shape[0]:
        raise ValueError(f"Requested frame {frame_t + 1}, but only {video.shape[0]} frames exist")
    frame0 = video[frame_t].numpy()
    frame1 = video[frame_t + 1].numpy()
    return frame0, frame1, info


def draw_bbox(draw: ImageDraw.ImageDraw, bbox: BBox, color: Tuple[int, int, int], label: str):
    draw.rectangle(bbox, outline=color, width=5)
    label_box = (bbox[0], max(0, bbox[1] - 28), bbox[0] + 140, bbox[1])
    draw.rectangle(label_box, fill=color)
    draw.text((label_box[0] + 6, label_box[1] + 4), label, fill=(255, 255, 255))


def make_heat_overlay(frame: np.ndarray, heat: np.ndarray) -> np.ndarray:
    resized = cv2.resize(heat, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_CUBIC)
    heat_u8 = np.clip(resized * 255.0, 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(frame[:, :, ::-1], 0.55, colored, 0.45, 0.0)
    return overlay[:, :, ::-1]


def save_visual(case: CaseConfig, out_dir: Path, frame0: np.ndarray, frame1: np.ndarray, heat: np.ndarray, metrics: Dict[str, float]):
    panel_w = frame0.shape[1]
    panel_h = frame0.shape[0]

    source_img = Image.fromarray(frame0.copy())
    target_img = Image.fromarray(frame1.copy())
    heat_img = Image.fromarray(make_heat_overlay(frame1.copy(), heat))

    draw_bbox(ImageDraw.Draw(source_img), case.bboxes["normal_source"], (46, 204, 113), "source")

    target_draw = ImageDraw.Draw(target_img)
    draw_bbox(target_draw, case.bboxes["normal_target"], (52, 152, 219), "normal")
    draw_bbox(target_draw, case.bboxes["phantom_target"], (231, 76, 60), "phantom")

    heat_draw = ImageDraw.Draw(heat_img)
    draw_bbox(heat_draw, case.bboxes["normal_target"], (52, 152, 219), "normal")
    draw_bbox(heat_draw, case.bboxes["phantom_target"], (231, 76, 60), "phantom")
    heat_draw.rectangle((30, 24, 520, 110), fill=(0, 0, 0))
    heat_draw.text((42, 34), f"normal={metrics['normal_match']:.4f}", fill=(255, 255, 255))
    heat_draw.text((42, 58), f"phantom={metrics['phantom_match']:.4f}", fill=(255, 255, 255))
    heat_draw.text((42, 82), f"ratio={metrics['ratio']:.4f}", fill=(255, 255, 255))

    canvas = Image.new("RGB", (panel_w * 3, panel_h), (255, 255, 255))
    canvas.paste(source_img, (0, 0))
    canvas.paste(target_img, (panel_w, 0))
    canvas.paste(heat_img, (panel_w * 2, 0))
    canvas.save(out_dir / f"{case.case_id}_viz.png")


def write_csv(rows: List[Dict[str, object]], path: Path):
    header = [
        "case_id",
        "video_path",
        "frame_t",
        "frame_t1",
        "block_id",
        "grid_frames",
        "grid_h",
        "grid_w",
        "normal_match",
        "phantom_match",
        "ratio",
    ]
    lines = [",".join(header)]
    for row in rows:
        lines.append(",".join(str(row[key]) for key in header))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(rows: List[Dict[str, object]], path: Path):
    avg_normal = sum(float(row["normal_match"]) for row in rows) / max(len(rows), 1)
    avg_phantom = sum(float(row["phantom_match"]) for row in rows) / max(len(rows), 1)
    avg_ratio = sum(float(row["ratio"]) for row in rows) / max(len(rows), 1)
    all_support = all(float(row["ratio"]) <= 0.5 for row in rows)

    lines = [
        "# Phantom Limb AMF Match Summary",
        "",
        "| case_id | t | t+1 | normal_match | phantom_match | ratio |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['case_id']} | {row['frame_t']} | {row['frame_t1']} | "
            f"{float(row['normal_match']):.4f} | {float(row['phantom_match']):.4f} | {float(row['ratio']):.4f} |"
        )

    lines.extend(
        [
            "",
            f"- Average normal_match: {avg_normal:.4f}",
            f"- Average phantom_match: {avg_phantom:.4f}",
            f"- Average ratio: {avg_ratio:.4f}",
            f"- MVP verdict: {'supported' if all_support else 'not yet supported'} "
            "(criterion: every case ratio <= 0.5).",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def extract_attention(case: CaseConfig, case_dir: Path):
    extractor = make_extractor(case, case_dir / "wan_tmp")
    query, key, grid_size = capture_qk(extractor, case.block_id)
    q, k = reshape_qk(query, key, grid_size)

    if case.frame_t1 >= grid_size[0]:
        raise ValueError(
            f"Case {case.case_id} requests frame {case.frame_t1}, "
            f"but captured grid has only {grid_size[0]} frames"
        )

    attn = compute_pair_attention_from_heads(
        q[:, case.frame_t],
        k[:, case.frame_t1],
        temp=extractor.config.motion_temp,
        head_reduce=extractor.config.head_reduce,
    ).cpu()

    save_dict = {
        "attention": attn,
        "case_id": case.case_id,
        "video_path": case.video_path,
        "frame_t": case.frame_t,
        "frame_t1": case.frame_t1,
        "block_id": case.block_id,
        "grid_size": grid_size,
        "size": case.size,
        "bboxes": case.bboxes,
    }
    attn_path = case_dir / f"{case.case_id}_A_t_to_t1.pt"
    torch.save(save_dict, attn_path)
    return attn_path, attn, grid_size


def score_case(case: CaseConfig, attn: torch.Tensor, grid_size: Tuple[int, int, int], case_dir: Path):
    frame0, frame1, _ = read_frame_pair(case.video_path, case.frame_t)
    image_h, image_w = frame0.shape[0], frame0.shape[1]
    _, grid_h, grid_w = grid_size

    token_bboxes = {
        key: pixel_bbox_to_token_bbox(value, image_w, image_h, grid_w, grid_h)
        for key, value in case.bboxes.items()
    }

    source_idx = bbox_to_token_indices(token_bboxes["normal_source"], grid_w)
    normal_idx = bbox_to_token_indices(token_bboxes["normal_target"], grid_w)
    phantom_idx = bbox_to_token_indices(token_bboxes["phantom_target"], grid_w)

    normal_match = compute_match(attn, source_idx, normal_idx)
    phantom_match = compute_match(attn, source_idx, phantom_idx)
    ratio = phantom_match / max(normal_match, 1e-8)

    metrics = {
        "normal_match": normal_match,
        "phantom_match": phantom_match,
        "ratio": ratio,
    }
    heat = aggregate_heatmap(attn, source_idx, grid_h, grid_w)
    save_visual(case, case_dir, frame0, frame1, heat, metrics)

    json_path = case_dir / f"{case.case_id}_metrics.json"
    json_path.write_text(
        json.dumps(
            {
                "case_id": case.case_id,
                "video_path": case.video_path,
                "frame_t": case.frame_t,
                "frame_t1": case.frame_t1,
                "block_id": case.block_id,
                "grid_size": list(grid_size),
                "pixel_bboxes": case.bboxes,
                "token_bboxes": {k: list(v) for k, v in token_bboxes.items()},
                **metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "case_id": case.case_id,
        "video_path": case.video_path,
        "frame_t": case.frame_t,
        "frame_t1": case.frame_t1,
        "block_id": case.block_id,
        "grid_frames": grid_size[0],
        "grid_h": grid_size[1],
        "grid_w": grid_size[2],
        **metrics,
    }


def run_annotation(case: CaseConfig, out_dir: Path):
    frame0, frame1, _ = read_frame_pair(case.video_path, case.frame_t)
    if "DISPLAY" not in os.environ:
        raise RuntimeError("Interactive annotation requires a GUI session with DISPLAY set")
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts = [
        ("normal_source", frame0),
        ("normal_target", frame1),
        ("phantom_target", frame1),
    ]
    bboxes = {}
    for key, frame in prompts:
        roi = cv2.selectROI(f"{case.case_id}:{key}", frame[:, :, ::-1], fromCenter=False, showCrosshair=True)
        cv2.destroyWindow(f"{case.case_id}:{key}")
        x, y, w, h = [int(v) for v in roi]
        bboxes[key] = [x, y, x + w, y + h]
    (out_dir / f"{case.case_id}_annotation.json").write_text(json.dumps(bboxes, indent=2), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser("Phantom limb AMF evaluation")
    parser.add_argument(
        "--cases",
        default=str(PROJECT_ROOT / "amf_motion_eval" / "configs" / "phantom_limb_cases.json"),
    )
    parser.add_argument(
        "--output_dir",
        default=str(PROJECT_ROOT / "amf_motion_eval" / "outputs" / "phantom_limb_eval"),
    )
    parser.add_argument(
        "--mode",
        choices=["run", "annotate"],
        default="run",
    )
    parser.add_argument("--case_id", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cases = load_cases(Path(args.cases))
    if args.case_id:
        cases = [case for case in cases if case.case_id == args.case_id]
        if not cases:
            raise ValueError(f"No case found for case_id={args.case_id}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "annotate":
        for case in cases:
            run_annotation(case, output_dir / case.case_id)
        return

    rows = []
    for case in cases:
        case_dir = output_dir / case.case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        _, attn, grid_size = extract_attention(case, case_dir)
        rows.append(score_case(case, attn, grid_size, case_dir))

    write_csv(rows, output_dir / "results.csv")
    write_summary(rows, output_dir / "summary.md")


if __name__ == "__main__":
    main()
