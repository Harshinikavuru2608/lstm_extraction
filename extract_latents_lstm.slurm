#!/bin/bash
#SBATCH --job-name=LSTM_EXTRACT_LATENTS_288_24
#SBATCH --account=PAS0272
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=100G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kavuru.7@buckeyemail.osu.edu
#SBATCH --output=/fs/ess/PAS0272/harshinikavuru2608/logs/MAMBA_DDP/%x.out.%j
#SBATCH --error=/fs/ess/PAS0272/harshinikavuru2608/logs/MAMBA_DDP/%x.err.%j

set -euo pipefail
mkdir -p /fs/ess/PAS0272/harshinikavuru2608/logs/MAMBA_DDP

echo "========================================"
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "========================================"

cd /fs/ess/PAS0272/harshinikavuru2608/AI_READI/MAMBA_CGM_DDP

/apps/python/3.12/bin/python extract_latents_lstm.py \
    --model_pt logs/lstm_L288_H24/final_model.pt \
    --processed_parquet "/users/PAS0536/harshinikavuru26/mydata/AI_READI/mamba_cgm/data/processed_data_3.0.parquet" \
    --out_path logs/lstm_L288_H24/latents_lstm_288_24.parquet \
    --batch_size 256 \
    --device cuda \
    --mode 2

echo "========================================"
echo "Job finished at: $(date)"
echo "========================================"
