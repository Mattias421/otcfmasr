#!/bin/bash
WANDB_TAGS=("hpopt" "prelim_loss_hpopt")
orion hunt -n prelim_loss_hpoptv2 -c hparams/orion.yaml python train.py hparams/train.yaml --precision=fp16 --eval_precision=fp16 --data_folder=$DATA --hpopt hparams/hpopt.yaml --hpopt_mode orion \
  --cfm_sigma~"choices([0.01, 0.05, 0.1, 0.2, 0.3])" \
  --batch_size~"choices([16, 32, 64])" \
  --learning_rate~"choices([0.001, 0.005, 0.01, 0.0005])" \
  --attention_type~"choices(['RoPEMHA', 'regularMHA', 'hypermixing'])" \
  --dropout~"choices([0.0, 0.05, 0.1, 0.15])" \
  --num_layers~"choices([1, 2, 4, 6])"

