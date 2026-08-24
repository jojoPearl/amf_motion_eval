#!/bin/bash

#SBATCH --job-name=vbench_all_compare
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=H100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00

set -euo pipefail

# ============================================================
# Paths
# ============================================================

PROJECT_ROOT="/home/ids/bjia-25/project"
AMF_ROOT="$PROJECT_ROOT/amf_motion_eval"

VBENCH1_ROOT="$PROJECT_ROOT/VBench"
VBENCH2_ROOT="$PROJECT_ROOT/VBench/VBench-2.0"
YOLO_WORLD_ROOT="$VBENCH2_ROOT/vbench2/third_party/YOLO-World"

PY_SCRIPT="$AMF_ROOT/src/vbench_rank.py"

INPUT_DIR="$AMF_ROOT/outputs/hunyuan_results"

RESULTS_DIR="$AMF_ROOT/outputs/hunyuan_vbench_all_compare"

VBENCH1_CSV="$RESULTS_DIR/vbench1_tmp.csv"

VBENCH2_BASE="$RESULTS_DIR/vbench2_baseline"
VBENCH2_AMF="$RESULTS_DIR/vbench2_amf"

FINAL_CSV="$RESULTS_DIR/baseline_vs_amf_all.csv"

mkdir -p "$RESULTS_DIR"

# ============================================================
# Temporary paired directory for VBench1
# ============================================================

WORK_DIR=$(mktemp -d "$RESULTS_DIR/paired_XXXXXX")
PAIR_DIR="$WORK_DIR/vbench1"
BASELINE_DIR="$WORK_DIR/baseline"
AMF_DIR="$WORK_DIR/amf"

mkdir -p "$PAIR_DIR" "$BASELINE_DIR" "$AMF_DIR"

cleanup() {
    rm -rf "$WORK_DIR"
}

trap cleanup EXIT

# ============================================================
# Input checks
# ============================================================

if [ ! -d "$INPUT_DIR" ]; then
    echo "ERROR: input directory not found:"
    echo "$INPUT_DIR"
    exit 1
fi

if [ ! -f "$PY_SCRIPT" ]; then
    echo "ERROR: vbench_rank.py not found:"
    echo "$PY_SCRIPT"
    exit 1
fi

if [ ! -f "$VBENCH2_ROOT/evaluate.py" ]; then
    echo "ERROR: VBench2 evaluate.py not found:"
    echo "$VBENCH2_ROOT/evaluate.py"
    exit 1
fi

# ============================================================
# Select complete baseline/AMF pairs and prepare input directories
# ============================================================

echo
echo "============================================================"
echo "Preparing paired videos"
echo "============================================================"

pair_count=0

find "$INPUT_DIR" \
    -maxdepth 1 \
    -type f \
    -name "*.mp4" \
    ! -name "*_amf.mp4" \
    -print0 |
while IFS= read -r -d '' video; do

    filename="$(basename "$video")"
    stem="${filename%.mp4}"
    amf_video="$INPUT_DIR/${stem}_amf.mp4"

    # Generation may still be in progress. Evaluate only complete pairs.
    if [ ! -f "$amf_video" ]; then
        continue
    fi

    ln -s "$video" "$BASELINE_DIR/${stem}.mp4"
    ln -s "$amf_video" "$AMF_DIR/${stem}_amf.mp4"
    ln -s "$video" "$PAIR_DIR/${stem}_baseline.mp4"
    ln -s "$amf_video" "$PAIR_DIR/${stem}_amf.mp4"
done

pair_count=$(find "$BASELINE_DIR" -maxdepth 1 -type l | wc -l)

if [ "$pair_count" -eq 0 ]; then
    echo "ERROR: no complete baseline/AMF pairs found in:"
    echo "$INPUT_DIR"
    exit 1
fi

echo "Complete pairs: $pair_count"
echo "Total videos submitted to VBench1: $((pair_count * 2))"


# ============================================================
# 1/3 VBench1
# ============================================================

echo
echo "============================================================"
echo "[1/3] VBench1"
echo "============================================================"

module purge || true
module load cuda/11.8

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vbench

export PYTHONPATH="$AMF_ROOT/src:$VBENCH1_ROOT:$PROJECT_ROOT"

python "$PY_SCRIPT" \
    --video_dir "$PAIR_DIR" \
    --output_csv "$VBENCH1_CSV" \
    --vbench_root "$VBENCH1_ROOT" \
    --device cuda \
    --skip_vbench2


# ============================================================
# 2/3 VBench2 Human Anatomy
# ============================================================

echo
echo "============================================================"
echo "[2/3] VBench2 Human Anatomy"
echo "============================================================"

conda deactivate || true

module purge || true
module load cuda/11.8
module load gcc/11.5.0

conda activate vbench2

export CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"
export CC="$(which gcc)"
export CXX="$(which g++)"

export PYTHONPATH="$YOLO_WORLD_ROOT:$VBENCH2_ROOT:$PROJECT_ROOT"

rm -rf "$VBENCH2_BASE" "$VBENCH2_AMF"

mkdir -p "$VBENCH2_BASE"
mkdir -p "$VBENCH2_AMF"

cd "$VBENCH2_ROOT"

# ------------------------------------------------------------
# Baseline
# ------------------------------------------------------------

echo
echo "Running Human Anatomy: baseline"

python evaluate.py \
    --videos_path "$BASELINE_DIR" \
    --dimension Human_Anatomy \
    --output_path "$VBENCH2_BASE" \
    --mode custom_input

# ------------------------------------------------------------
# AMF
# ------------------------------------------------------------

echo
echo "Running Human Anatomy: AMF"

python evaluate.py \
    --videos_path "$AMF_DIR" \
    --dimension Human_Anatomy \
    --output_path "$VBENCH2_AMF" \
    --mode custom_input


# ============================================================
# 3/3 Merge
# ============================================================

echo
echo "============================================================"
echo "[3/3] Merge all metrics"
echo "============================================================"

python - \
    "$VBENCH1_CSV" \
    "$VBENCH2_BASE" \
    "$VBENCH2_AMF" \
    "$FINAL_CSV" <<'PY'

import csv
import json
import re
import sys
from pathlib import Path


vbench1_csv = Path(sys.argv[1])
vbench2_base_dir = Path(sys.argv[2])
vbench2_amf_dir = Path(sys.argv[3])
output_csv = Path(sys.argv[4])


# ============================================================
# Helpers
# ============================================================

def parse_key(name):

    stem = Path(name).stem

    # Remove variant suffix
    stem = re.sub(
        r"_(baseline|amf|selection)$",
        "",
        stem,
        flags=re.IGNORECASE,
    )

    # Remove hunyuan suffix
    stem = re.sub(
        r"_hunyuan$",
        "",
        stem,
        flags=re.IGNORECASE,
    )

    m = re.match(
        r"^(?P<action>.+?)_seed(?P<seed>\d+)$",
        stem,
        flags=re.IGNORECASE,
    )

    if not m:
        return None

    return (
        m.group("action"),
        int(m.group("seed")),
    )


def to_float(value):

    if value in (
        None,
        "",
        "None",
        "N/A",
    ):
        return None

    try:
        return float(value)

    except (TypeError, ValueError):
        return None


# ============================================================
# Load VBench1
# ============================================================

with vbench1_csv.open(
    "r",
    encoding="utf-8",
    newline="",
) as f:

    vbench1_rows = list(
        csv.DictReader(f)
    )


pairs = {}


for src in vbench1_rows:

    action = src.get("action")
    seed = src.get("seed")

    try:
        key = (action, int(seed)) if action and seed else None
    except ValueError:
        key = None

    if key is None:
        print("WARNING: cannot parse VBench1 row:", src)
        continue

    pairs[key] = {}

    for variant, prefix in (("baseline", "baseline"), ("amf", "selection")):
        pairs[key][variant] = {
            "variant": variant,
            "action": key[0],
            "seed": key[1],
            "human_anatomy": None,
            "motion_smoothness": to_float(
                src.get(f"{prefix}_motion_smoothness")
            ),
            "overall_consistency": to_float(
                src.get(f"{prefix}_overall_consistency")
            ),
            "imaging_quality": to_float(
                src.get(f"{prefix}_imaging_quality")
            ),
        }


# ============================================================
# VBench2 result loader
# ============================================================

def find_eval_json(folder):

    files = sorted(
        Path(folder).glob(
            "*_eval_results.json"
        ),
        key=lambda p:
            p.stat().st_mtime,
        reverse=True,
    )

    if not files:

        raise FileNotFoundError(
            f"No *_eval_results.json found in {folder}"
        )

    return files[0]


def load_anatomy(folder):

    json_path = find_eval_json(
        folder
    )

    print(
        "Loading:",
        json_path,
    )

    with json_path.open(
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(f)


    if "Human_Anatomy" not in data:

        raise KeyError(
            f"Human_Anatomy missing from {json_path}"
        )


    _, video_results = (
        data["Human_Anatomy"]
    )


    scores = {}


    for item in video_results:

        video_path = item.get(
            "video_path"
        )

        score = item.get(
            "video_results"
        )


        if not isinstance(
            video_path,
            str,
        ):
            continue


        key = parse_key(
            video_path
        )


        if key is None:

            print(
                "WARNING: cannot parse anatomy:",
                video_path,
            )

            continue


        score = float(score)

        # Keep VBench-2.0 Human Anatomy on the same 0-100 scale
        # used by vbench_rank.py for VBench-1.0 metrics.
        if 0.0 <= score <= 1.000001:
            score *= 100.0

        scores[key] = score


    return scores


baseline_anatomy = load_anatomy(
    vbench2_base_dir
)

amf_anatomy = load_anatomy(
    vbench2_amf_dir
)


# ============================================================
# Add Anatomy
# ============================================================

all_keys = (
    set(pairs.keys())
    | set(baseline_anatomy.keys())
    | set(amf_anatomy.keys())
)


def empty_row(
    variant,
    action,
    seed,
):

    return {

        "variant":
            variant,

        "action":
            action,

        "seed":
            seed,

        "human_anatomy":
            None,

        "motion_smoothness":
            None,

        "overall_consistency":
            None,

        "imaging_quality":
            None,
    }


for key in all_keys:

    action, seed = key

    pairs.setdefault(
        key,
        {}
    )


    if "baseline" not in pairs[key]:

        pairs[key]["baseline"] = (
            empty_row(
                "baseline",
                action,
                seed,
            )
        )


    if "amf" not in pairs[key]:

        pairs[key]["amf"] = (
            empty_row(
                "amf",
                action,
                seed,
            )
        )


    pairs[key]["baseline"][
        "human_anatomy"
    ] = baseline_anatomy.get(
        key
    )


    pairs[key]["amf"][
        "human_anatomy"
    ] = amf_anatomy.get(
        key
    )


# ============================================================
# Build one wide row per action + seed
#
# delta = AMF - baseline
# ============================================================

metrics = (
    "human_anatomy",
    "motion_smoothness",
    "overall_consistency",
    "imaging_quality",
)

rows = []

for key in sorted(
    pairs.keys(),
    key=lambda x: (x[0], x[1]),
):

    action, seed = key
    baseline = pairs[key]["baseline"]
    amf = pairs[key]["amf"]

    row = {
        "action": action,
        "seed": seed,
    }

    for metric in metrics:
        baseline_value = baseline.get(metric)
        selection_value = amf.get(metric)

        row[f"baseline_{metric}"] = baseline_value
        row[f"selection_{metric}"] = selection_value

        if (
            isinstance(baseline_value, (int, float))
            and isinstance(selection_value, (int, float))
        ):
            row[f"delta_{metric}"] = (
                float(selection_value)
                - float(baseline_value)
            )
        else:
            row[f"delta_{metric}"] = None

    rows.append(row)


# ============================================================
# CSV
# ============================================================

fieldnames = [
    "action",
    "seed",

    "baseline_human_anatomy",
    "selection_human_anatomy",
    "delta_human_anatomy",

    "baseline_motion_smoothness",
    "selection_motion_smoothness",
    "delta_motion_smoothness",

    "baseline_overall_consistency",
    "selection_overall_consistency",
    "delta_overall_consistency",

    "baseline_imaging_quality",
    "selection_imaging_quality",
    "delta_imaging_quality",
]

with output_csv.open(
    "w",
    encoding="utf-8",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )

    writer.writeheader()
    writer.writerows(rows)


# ============================================================
# Summary
# ============================================================

print()
print("========================================")
print("FINAL RESULT")
print("========================================")

print(
    "Pairs:",
    len(pairs),
)

print(
    "Wide rows (one per action+seed):",
    len(rows),
)

print(
    "Saved:",
    output_csv,
)

print()
print(
    "delta = AMF - baseline"
)

print()
print(
    "No overall score is computed."
)

print(
    "All reported metric scores are on a 0-100 scale when source scores are in [0, 1]."
)

PY


# ============================================================
# Done
# ============================================================

echo
echo "============================================================"
echo "DONE"
echo "============================================================"

echo
echo "Final CSV:"
echo "$FINAL_CSV"
