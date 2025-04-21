#!/bin/bash
#SBATCH --partition=dcs-gpu
#SBATCH --account=dcs-res
#SBATCH --gres=gpu:1
#SBATCH --time=160:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=32G
#SBATCH --output=logs/slurm/%x-%a.out
#SBATCH --array=1-16

module load CUDA/12.4.0 
source .venv/bin/activate

name="paired_prelim"

WANDB_TAGS=("hpopt" $name)

orion hunt -n $name -c hparams/orion/orion.yaml python train.py hparams/train.yaml --precision=fp16 --eval_precision=fp16 --data_folder=$DATA --hpopt hparams/xps/${name}.yaml --hpopt_mode orion --xp_name=${name}

