#!/bin/bash
#SBATCH --job-name=wan22_baseline
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=A40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=4:00:00

module purge

source ~/miniconda3/etc/profile.d/conda.sh
conda activate wan22

echo "Job $SLURM_JOB_ID started on $(hostname) at $(date)"

nvidia-smi

cd /home/ids/bjia-25/project/Wan2.2

echo "Python:"
which python
python -V

echo "Torch check:"
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("torch cuda:", torch.version.cuda)
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
PY

echo "Run Wan2.2 TI2V-5B baseline"

python generate.py \
  --task ti2v-5B \
  --size 1280*704 \
  --ckpt_dir ./Wan2.2-TI2V-5B \
  --offload_model True \
  --convert_model_dtype \
  --t5_cpu \
  --prompt "Leopard running up a snowy hill in a forest"

echo "Job finished at $(date)"
