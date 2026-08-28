#!/bin/bash

#SBATCH --job-name=hunyuan_baseline_batch
#SBATCH --output=/home/ids/bjia-25/project/amf_motion_eval/scripts/%x_%j.out
#SBATCH --error=/home/ids/bjia-25/project/amf_motion_eval/scripts/%x_%j.err
#SBATCH --partition=H100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00

set -euo pipefail

PROJECT_ROOT=/home/ids/bjia-25/project
AMF_ROOT=$PROJECT_ROOT/amf_motion_eval

PROMPT_FILE=$AMF_ROOT/configs/prompt_intense.jsonl
OUTPUT_DIR=$AMF_ROOT/outputs/hunyuan_score_results1

source /home/ids/bjia-25/miniconda3/etc/profile.d/conda.sh
conda activate hunyuan15

cd "$AMF_ROOT"
mkdir -p "$OUTPUT_DIR"

export PYTHONPATH="$AMF_ROOT/src:$PROJECT_ROOT:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# =========================
# 只改这里
# =========================
# SEEDS=(205 1024 2026 4096 8901)
SEEDS=(205 1024)
# Hunyuan requires both dimensions to be divisible by 16.
# 704x1280 is the nearest valid portrait resolution to 720x1280.
HEIGHT=1280
WIDTH=704
# =========================

CONFIG=$(mktemp --tmpdir hunyuan_baseline_batch.XXXXXX.json)

python - "$PROMPT_FILE" "$CONFIG" "${SEEDS[@]}" <<'PY'
import json
import sys
from pathlib import Path

prompt_file = Path(sys.argv[1])
config_path = Path(sys.argv[2])
seeds = [int(x) for x in sys.argv[3:]]

items = []

with open(prompt_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()

        if not line:
            continue

        item = json.loads(line)

        if "name" not in item or "prompt" not in item:
            raise ValueError(
                f"Each JSONL line must contain 'name' and 'prompt': {line}"
            )

        items.append(item)

if not items:
    raise RuntimeError(f"No prompts found in {prompt_file}")

config = []

for item in items:
    name = item["name"]
    prompt = item["prompt"]

    for seed in seeds:
        config.append({
            "prompt": prompt,
            "seed": seed,
            "baseline_video": f"{name}_seed{seed}.mp4"
        })

with open(config_path, "w", encoding="utf-8") as f:
    json.dump(
        config,
        f,
        indent=2,
        ensure_ascii=False
    )

print("=" * 60)
print(f"Prompt file : {prompt_file}")
print(f"Prompts     : {len(items)}")
print(f"Seeds       : {seeds}")
print(f"Total videos: {len(config)}")
print("=" * 60)

for item in items:
    print(f"{item['name']}: {item['prompt']}")

PY

echo ""
echo "Starting HunyuanVideo baseline generation..."
echo ""

python "$AMF_ROOT/src/hunyuan.py" \
    --config "$CONFIG" \
    --output-dir "$OUTPUT_DIR" \
    --height "$HEIGHT" \
    --width "$WIDTH"

rm -f "$CONFIG"

echo ""
echo "Done."
echo "Videos saved to:"
echo "$OUTPUT_DIR"
