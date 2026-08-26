#!/bin/bash

#SBATCH --job-name=vbench_compare
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

# ------------------------------------------------------------
# Two folders to compare
# ------------------------------------------------------------
# Baseline videos, e.g. cartwheel_seed205.mp4 / cartwheel_seed205_hunyuan.mp4
BASELINE_DIR="$AMF_ROOT/outputs/hunyuan_score_results1"

# AMF videos, e.g. cartwheel_seed205_amf.mp4 / cartwheel_seed205_hunyuan_amf.mp4
AMF_DIR="$AMF_ROOT/outputs/hunyuan_score_results"

RESULTS_DIR="$AMF_ROOT/outputs/vbench_final"

VBENCH1_CSV="$RESULTS_DIR/vbench1_compare.csv"
VBENCH2_DIR="$RESULTS_DIR/vbench2_human_anatomy"
FINAL_CSV="$RESULTS_DIR/baseline_vs_amf_scores.csv"

mkdir -p "$RESULTS_DIR"

# ============================================================
# Temporary workspace
#
# vbench_rank.py expects paired filenames:
#   <key>_baseline.mp4
#   <key>_selection.mp4
# We map the AMF video to "selection" only inside WORK_DIR.
# The final CSV calls it AMF, not selection.
# ============================================================

WORK_DIR=$(mktemp -d "$RESULTS_DIR/work_XXXXXX")
VBENCH1_INPUT="$WORK_DIR/vbench1_input"
VBENCH2_INPUT="$WORK_DIR/vbench2_input"
PAIR_MAP="$WORK_DIR/pair_map.csv"

mkdir -p "$VBENCH1_INPUT" "$VBENCH2_INPUT"

cleanup() {
    rm -rf "$WORK_DIR"
}
trap cleanup EXIT

# ============================================================
# Input checks
# ============================================================

for d in "$BASELINE_DIR" "$AMF_DIR"; do
    if [ ! -d "$d" ]; then
        echo "ERROR: directory not found:"
        echo "$d"
        exit 1
    fi
done

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
# Prepare paired videos
# ============================================================

echo
echo "============================================================"
echo "Preparing baseline / AMF pairs"
echo "============================================================"
echo "Baseline: $BASELINE_DIR"
echo "AMF:      $AMF_DIR"

python - "$BASELINE_DIR" "$AMF_DIR" "$VBENCH1_INPUT" "$VBENCH2_INPUT" "$PAIR_MAP" <<'PY'
import csv
import re
import sys
from pathlib import Path

baseline_dir = Path(sys.argv[1])
amf_dir = Path(sys.argv[2])
vbench1_input = Path(sys.argv[3])
vbench2_input = Path(sys.argv[4])
pair_map_csv = Path(sys.argv[5])


def canonical_key(path: Path) -> str:
    """Normalize common baseline/AMF/Hunyuan suffixes to one pairing key."""
    stem = path.stem

    # Remove suffixes repeatedly because filenames may look like
    # xxx_hunyuan_amf, xxx_amf_hunyuan, xxx_baseline, etc.
    suffixes = (
        "_baseline",
        "_selection",
        "_amf",
        "_hunyuan",
    )

    changed = True
    while changed:
        changed = False
        lower = stem.lower()
        for suffix in suffixes:
            if lower.endswith(suffix):
                stem = stem[: -len(suffix)]
                changed = True
                break

    return stem


def collect(folder: Path):
    result = {}
    for path in sorted(folder.glob("*.mp4")):
        key = canonical_key(path)
        if key in result:
            raise RuntimeError(
                f"Duplicate canonical key '{key}' in {folder}:\n"
                f"  {result[key].name}\n"
                f"  {path.name}\n"
                "Rename the files so each action/seed has only one video per folder."
            )
        result[key] = path.resolve()
    return result


baseline = collect(baseline_dir)
amf = collect(amf_dir)

if not baseline:
    raise RuntimeError(f"No .mp4 files found in baseline folder: {baseline_dir}")
if not amf:
    raise RuntimeError(f"No .mp4 files found in AMF folder: {amf_dir}")

baseline_only = sorted(set(baseline) - set(amf))
amf_only = sorted(set(amf) - set(baseline))
paired = sorted(set(baseline) & set(amf))

if baseline_only:
    print("WARNING: baseline videos with no AMF match:")
    for key in baseline_only:
        print("  ", baseline[key].name)

if amf_only:
    print("WARNING: AMF videos with no baseline match:")
    for key in amf_only:
        print("  ", amf[key].name)

if not paired:
    raise RuntimeError(
        "No baseline/AMF pairs found. Pairing is based on filename after removing "
        "_baseline, _selection, _amf and _hunyuan suffixes."
    )

with pair_map_csv.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["key", "baseline_name", "amf_name"],
    )
    writer.writeheader()

    for key in paired:
        baseline_path = baseline[key]
        amf_path = amf[key]

        # VBench1: exploit existing baseline/selection comparison code.
        (vbench1_input / f"{key}_baseline.mp4").symlink_to(baseline_path)
        (vbench1_input / f"{key}_selection.mp4").symlink_to(amf_path)

        # VBench2: keep explicit variant suffixes so scores can be separated.
        (vbench2_input / f"{key}_baseline.mp4").symlink_to(baseline_path)
        (vbench2_input / f"{key}_amf.mp4").symlink_to(amf_path)

        writer.writerow({
            "key": key,
            "baseline_name": baseline_path.name,
            "amf_name": amf_path.name,
        })

print(f"Baseline videos: {len(baseline)}")
print(f"AMF videos:      {len(amf)}")
print(f"Matched pairs:   {len(paired)}")
PY

# ============================================================
# 1/3 VBench1 comparison
# ============================================================

echo
echo "============================================================"
echo "[1/3] VBench1 baseline vs AMF"
echo "============================================================"

module purge || true
module load cuda/11.8

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vbench

export PYTHONPATH="$AMF_ROOT/src:$VBENCH1_ROOT:$PROJECT_ROOT"

python "$PY_SCRIPT" \
    --video_dir "$VBENCH1_INPUT" \
    --output_csv "$VBENCH1_CSV" \
    --vbench_root "$VBENCH1_ROOT" \
    --device cuda \
    --skip_vbench2

# ============================================================
# 2/3 VBench2 Human Anatomy
# ============================================================

echo
echo "============================================================"
echo "[2/3] VBench2 Human Anatomy baseline vs AMF"
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

rm -rf "$VBENCH2_DIR"
mkdir -p "$VBENCH2_DIR"

cd "$VBENCH2_ROOT"

python evaluate.py \
    --videos_path "$VBENCH2_INPUT" \
    --dimension Human_Anatomy \
    --output_path "$VBENCH2_DIR" \
    --mode custom_input

# ============================================================
# 3/3 Merge VBench1 + Human Anatomy into comparison CSV
# ============================================================

echo
echo "============================================================"
echo "[3/3] Merge baseline / AMF metrics and deltas"
echo "============================================================"

python - \
    "$VBENCH1_CSV" \
    "$VBENCH2_DIR" \
    "$PAIR_MAP" \
    "$FINAL_CSV" <<'PY'

import csv
import json
import re
import sys
from pathlib import Path

vbench1_csv = Path(sys.argv[1])
vbench2_dir = Path(sys.argv[2])
pair_map_csv = Path(sys.argv[3])
output_csv = Path(sys.argv[4])


def to_float(value):
    if value in (None, "", "None", "N/A"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def delta(amf, baseline):
    if amf is None or baseline is None:
        return None
    return amf - baseline


def parse_action_seed(key):
    m = re.match(
        r"^(?P<action>.+?)_seed(?P<seed>\d+)$",
        key,
        flags=re.IGNORECASE,
    )
    if not m:
        return key, None
    return m.group("action"), int(m.group("seed"))


# ------------------------------------------------------------
# Pair map
# ------------------------------------------------------------

with pair_map_csv.open("r", encoding="utf-8", newline="") as f:
    pairs = {row["key"]: row for row in csv.DictReader(f)}

rows = {}
for key, pair in pairs.items():
    action, seed = parse_action_seed(key)
    rows[key] = {
        "action": action,
        "seed": seed,
        "baseline_video": pair["baseline_name"],
        "amf_video": pair["amf_name"],
    }


# ------------------------------------------------------------
# VBench1
# vbench_rank.py already compares *_baseline and *_selection.
# Rename selection -> AMF in the final output.
# ------------------------------------------------------------

with vbench1_csv.open("r", encoding="utf-8", newline="") as f:
    vbench1_rows = list(csv.DictReader(f))

metrics = [
    "motion_smoothness",
    "overall_consistency",
    "imaging_quality",
]

for src in vbench1_rows:
    action = src.get("action")
    seed = src.get("seed")

    if not action or seed in (None, ""):
        print("WARNING: cannot identify VBench1 row:", src)
        continue

    try:
        seed_int = int(seed)
    except ValueError:
        print("WARNING: invalid seed in VBench1 row:", src)
        continue

    key = f"{action}_seed{seed_int}"
    if key not in rows:
        print("WARNING: VBench1 key not present in pair map:", key)
        continue

    for metric in metrics:
        baseline_score = to_float(src.get(f"baseline_{metric}"))
        amf_score = to_float(src.get(f"selection_{metric}"))

        rows[key][f"baseline_{metric}"] = baseline_score
        rows[key][f"amf_{metric}"] = amf_score
        rows[key][f"delta_{metric}"] = delta(amf_score, baseline_score)


# ------------------------------------------------------------
# VBench2 Human Anatomy
# ------------------------------------------------------------

eval_files = sorted(
    vbench2_dir.glob("*_eval_results.json"),
    key=lambda p: p.stat().st_mtime,
    reverse=True,
)

if not eval_files:
    raise FileNotFoundError(f"No *_eval_results.json found in {vbench2_dir}")

json_path = eval_files[0]
print("Loading:", json_path)

with json_path.open("r", encoding="utf-8") as f:
    data = json.load(f)

if "Human_Anatomy" not in data:
    raise KeyError(f"Human_Anatomy missing from {json_path}")

_, video_results = data["Human_Anatomy"]

for item in video_results:
    video_path = item.get("video_path")
    score = item.get("video_results")

    if not isinstance(video_path, str) or score is None:
        continue

    stem = Path(video_path).stem

    if stem.endswith("_baseline"):
        key = stem[: -len("_baseline")]
        variant = "baseline"
    elif stem.endswith("_amf"):
        key = stem[: -len("_amf")]
        variant = "amf"
    else:
        print("WARNING: unknown VBench2 variant:", stem)
        continue

    if key not in rows:
        print("WARNING: VBench2 key not present in pair map:", key)
        continue

    score = float(score)
    if 0.0 <= score <= 1.000001:
        score *= 100.0

    rows[key][f"{variant}_human_anatomy"] = score

for row in rows.values():
    b = row.get("baseline_human_anatomy")
    a = row.get("amf_human_anatomy")
    row["delta_human_anatomy"] = delta(a, b)


# ------------------------------------------------------------
# Final CSV
# delta = AMF - baseline
# Positive delta means AMF scored higher.
# ------------------------------------------------------------

fieldnames = [
    "action",
    "seed",
    "baseline_video",
    "amf_video",
    "baseline_human_anatomy",
    "amf_human_anatomy",
    "delta_human_anatomy",
    "baseline_motion_smoothness",
    "amf_motion_smoothness",
    "delta_motion_smoothness",
    "baseline_overall_consistency",
    "amf_overall_consistency",
    "delta_overall_consistency",
    "baseline_imaging_quality",
    "amf_imaging_quality",
    "delta_imaging_quality",
]

final_rows = sorted(
    rows.values(),
    key=lambda r: (
        str(r.get("action") or ""),
        -1 if r.get("seed") is None else int(r["seed"]),
    ),
)

with output_csv.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in final_rows:
        writer.writerow({k: row.get(k) for k in fieldnames})

print()
print("========================================")
print("FINAL RESULT")
print("========================================")
print("Matched pairs:", len(final_rows))
print("Saved:", output_csv)
print("delta = AMF - baseline")
print("Positive delta => AMF is higher on that metric")

PY

# ============================================================
# Done
# ============================================================

echo
echo "============================================================"
echo "DONE"
echo "============================================================"
echo
echo "Final comparison CSV:"
echo "$FINAL_CSV"
