#!/bin/bash
#SBATCH --job-name=amf_selection_batch
#SBATCH --output=/home/ids/bjia-25/project/amf_motion_eval/scripts/%x_%j.out
#SBATCH --error=/home/ids/bjia-25/project/amf_motion_eval/scripts/%x_%j.err
#SBATCH --partition=A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00

set -euo pipefail

module purge
source ~/miniconda3/etc/profile.d/conda.sh
conda activate wan22

PROJECT_ROOT="${PROJECT_ROOT:-/home/ids/bjia-25/project}"
WAN_ROOT="${WAN_ROOT:-$PROJECT_ROOT/Wan2.2}"
CKPT_DIR="${CKPT_DIR:-$WAN_ROOT/Wan2.2-TI2V-5B}"
OUTPUT_DIR="${OUTPUT_DIR:-$WAN_ROOT/outputs/amf_selection_prompt_seed_batch}"

usage() {
  cat <<'EOF'
Usage:
  amf_selection.sh [--output-dir DIR] [--name NAME] \
    --prompt "PROMPT" --seed SEED

Legacy batch form:
  amf_selection.sh [--output-dir DIR] [--name NAME] \
    "PROMPT|SEED" ["PROMPT|SEED" ...]

Each quoted PROMPT|SEED item produces one AMF-selection MP4. All outputs are
written into the same output directory.

With --name hurdling, outputs use baseline-compatible names such as:
  hurdling_seed42_amf_selection.mp4

Without --name, the prompt-derived filename format is retained.
EOF
}

OUTPUT_NAME=""
PROMPT_ARG=""
SEED_ARG=""
PROMPT_SEED_SPECS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir)
      [[ $# -ge 2 ]] || { echo "Error: --output-dir requires a value" >&2; exit 2; }
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --name)
      [[ $# -ge 2 ]] || { echo "Error: --name requires a value" >&2; exit 2; }
      OUTPUT_NAME="$2"
      shift 2
      ;;
    --prompt)
      [[ $# -ge 2 ]] || { echo "Error: --prompt requires a value" >&2; exit 2; }
      PROMPT_ARG="$2"
      shift 2
      ;;
    --seed)
      [[ $# -ge 2 ]] || { echo "Error: --seed requires a value" >&2; exit 2; }
      SEED_ARG="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      PROMPT_SEED_SPECS+=("$@")
      break
      ;;
    -*)
      echo "Error: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      PROMPT_SEED_SPECS+=("$1")
      shift
      ;;
  esac
done

if [[ -n "$PROMPT_ARG" || -n "$SEED_ARG" ]]; then
  if [[ -z "$PROMPT_ARG" || ! "$SEED_ARG" =~ ^[0-9]+$ ]]; then
    echo "Error: --prompt and a non-negative integer --seed must be provided together" >&2
    exit 2
  fi
  if [[ ${#PROMPT_SEED_SPECS[@]} -ne 0 ]]; then
    echo "Error: do not mix --prompt/--seed with PROMPT|SEED arguments" >&2
    exit 2
  fi
  PROMPT_SEED_SPECS+=("$PROMPT_ARG|$SEED_ARG")
fi

if [[ ${#PROMPT_SEED_SPECS[@]} -eq 0 ]]; then
  echo "Error: provide --prompt/--seed or at least one quoted PROMPT|SEED item" >&2
  usage >&2
  exit 2
fi

if [[ "$OUTPUT_DIR" != /* ]]; then
  OUTPUT_DIR="$PROJECT_ROOT/$OUTPUT_DIR"
fi

if [[ -n "$OUTPUT_NAME" ]]; then
  OUTPUT_NAME="$(printf '%s' "$OUTPUT_NAME" | sed -E 's/[^a-zA-Z0-9_.-]+/_/g; s/^_+//; s/_+$//')"
  [[ -n "$OUTPUT_NAME" ]] || { echo "Error: --name contains no usable filename characters" >&2; exit 2; }
fi

cd "$WAN_ROOT"
mkdir -p "$OUTPUT_DIR"

export PYTHONPATH="$PROJECT_ROOT/amf_motion_eval/src:$PROJECT_ROOT:$WAN_ROOT:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

slugify_prompt() {
  printf '%s' "$1" \
    | tr '[:upper:]' '[:lower:]' \
    | sed -E 's/[^a-z0-9]+/_/g; s/^_+//; s/_+$//; s/_+/_/g' \
    | cut -c1-72
}

for spec in "${PROMPT_SEED_SPECS[@]}"; do
  if [[ "$spec" != *"|"* ]]; then
    echo "Error: expected PROMPT|SEED, got: $spec" >&2
    exit 2
  fi

  PROMPT="${spec%|*}"
  SEED="${spec##*|}"
  if [[ -z "$PROMPT" || ! "$SEED" =~ ^[0-9]+$ ]]; then
    echo "Error: invalid PROMPT|SEED item: $spec" >&2
    exit 2
  fi

  if [[ -n "$OUTPUT_NAME" ]]; then
    OUTPUT_STEM="${OUTPUT_NAME}_seed${SEED}_amf_selection"
  else
    PROMPT_SLUG="$(slugify_prompt "$PROMPT")"
    [[ -n "$PROMPT_SLUG" ]] || PROMPT_SLUG="prompt"
    OUTPUT_STEM="${PROMPT_SLUG}_seed${SEED}_amf_selection"
  fi
  OUTPUT_VIDEO="$OUTPUT_DIR/${OUTPUT_STEM}.mp4"

  echo "[selection] seed=$SEED"
  echo "[selection] prompt=$PROMPT"
  echo "[selection] video=$OUTPUT_VIDEO"

  python "$PROJECT_ROOT/amf_motion_eval/src/amf_motion_eval/evaluate_attention/amf_latent_selection.py" \
    --prompt "$PROMPT" \
    --ckpt_dir "$CKPT_DIR" \
    --task ti2v-5B \
    --size 1280x704 \
    --frame_num 121 \
    --sampling_steps 50 \
    --sample_solver unipc \
    --shift 5.0 \
    --guide_scale 5.0 \
    --seed "$SEED" \
    --selection_beam_size 2 \
    --selection_candidates_per_beam 3 \
    --selection_lookahead_steps 3 \
    --selection_step_ratios 0.08,0.12,0.16,0.22,0.30 \
    --selection_ratio_tolerance 0.025 \
    --selection_branch_noise_scale 0.4 \
    --selection_reward_mode linear \
    --selection_lambda_motion 0.5 \
    --selection_temporal_smooth_noise \
    --selection_temporal_smooth_kernel 5 \
    --selection_novelty_bonus 0.0 \
    --selection_amf_block_id 15 \
    --selection_max_amf_triplets 4 \
    --selection_motion_temp 2.0 \
    --selection_head_reduce mean_qk \
    --output_video "$OUTPUT_VIDEO" \
    --output_fps 24
done

echo "AMF selection batch finished at $(date)"
echo "output_dir=$OUTPUT_DIR"
