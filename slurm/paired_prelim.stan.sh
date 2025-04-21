#!/bin/bash
#SBATCH --partition=gpu-h100,gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=80:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --tasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --output=slurm/logs/%x_j.out
#SBATCH --array=1-16

module load cuDNN/8.7.0.84-CUDA-11.8.0
source .venv/bin/activate

name="paired_prelim"

WANDB_TAGS=("hpopt" $name)
orion hunt -n $name -c hparams/orion.yaml --n-workers 4 python train.py hparams/train.yaml --precision=fp16 --eval_precision=fp16 --data_folder=$DATA --hpopt hparams/hpopt.yaml --hpopt_mode orion \
  --xp_name=$name \
  --cfm_sigma~"choices([0.01, 0.05, 0.1, 0.2, 0.3])" \
  --batch_size~"choices([16, 32, 64])" \
  --learning_rate~"choices([0.001, 0.005, 0.01, 0.0005])" \
  --dropout~"choices([0.0, 0.05, 0.1, 0.15])" \
  --num_layers~"choices([1, 2, 4, 6])"
