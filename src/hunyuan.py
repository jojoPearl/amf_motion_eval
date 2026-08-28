import argparse
import json
import os
import re
from pathlib import Path

import torch
from diffusers import HunyuanVideoPipeline
from diffusers.utils import export_to_video


# 避免 tokenizers fork warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_ID = "hunyuanvideo-community/HunyuanVideo"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Hunyuan videos for judge cases."
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

    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=512)

    # 新增：只检查 transformer / attention 结构，不生成视频
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Print Hunyuan transformer attention structure and exit.",
    )

    return parser.parse_args()


def case_name_and_seed(case, index):
    baseline_stem = Path(case.get("baseline_video", "")).stem

    match = re.search(r"_seed(\d+)$", baseline_stem)

    seed = int(match.group(1)) if match else 42 + index

    name = baseline_stem or f"case_{index + 1:02d}_seed{seed}"

    return name, seed


def inspect_transformer(pipe):
    print("\n" + "=" * 80)
    print("HUNYUAN TRANSFORMER CONFIG")
    print("=" * 80)

    print(pipe.transformer.config)

    print("\n" + "=" * 80)
    print("HUNYUAN ATTENTION MODULES")
    print("=" * 80)

    for name, module in pipe.transformer.named_modules():
        lower_name = name.lower()

        if (
            "attn" in lower_name
            or "attention" in lower_name
            or "to_q" in lower_name
            or "to_k" in lower_name
        ):
            print(f"{name:100s} {type(module)}")

    print("\n" + "=" * 80)
    print("END")
    print("=" * 80)


def main():
    args = parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading HunyuanVideo...")

    pipe = HunyuanVideoPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
    )

    # ---------------------------------------------------------
    # Inspect mode
    # ---------------------------------------------------------
    if args.inspect:
        inspect_transformer(pipe)
        return

    # ---------------------------------------------------------
    # Normal baseline generation
    # ---------------------------------------------------------

    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()

    with args.config.open(encoding="utf-8") as file:
        cases = json.load(file)

    if not isinstance(cases, list) or not cases:
        raise ValueError(
            f"Expected a non-empty case list in {args.config}"
        )

    for index, case in enumerate(cases):
        prompt = case.get("prompt")

        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(
                f"Case {index + 1} has no valid prompt"
            )

        name, seed = case_name_and_seed(case, index)

        output_path = (
            args.output_dir / f"{name}_hunyuan.mp4"
        )

        print(
            f"\n[{index + 1}/{len(cases)}] "
            f"seed={seed}"
        )

        print(f"Prompt: {prompt}")

        generator = (
            torch.Generator(device="cpu")
            .manual_seed(seed)
        )

        frames = pipe(
            prompt=prompt,
            height=args.height,
            width=args.width,
            num_frames=61,
            num_inference_steps=30,
            generator=generator,
        ).frames[0]

        export_to_video(
            frames,
            str(output_path),
            fps=15,
        )

        print(
            f"Saved video to: {output_path}"
        )


if __name__ == "__main__":
    main()
