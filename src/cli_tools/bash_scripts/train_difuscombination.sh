#!/bin/bash

# Train
dataset="er_50_100" # also "er_300_400", "er_700_800"
batch_size=32
num_epochs=50

hatch run cli difuscombination run-difuscombination \
  --task "mis" \
  --wandb_logger_name "mis_difuscombination_${dataset}_train_test_new" \
  --diffusion_type "gaussian" \
  --do_train \
  --do_test \
  --learning_rate 0.0002 \
  --weight_decay 0.0001 \
  --lr_scheduler "cosine-decay" \
  --data_path "data" \
  --models_path "models" \
  --logs_path "logs" \
  --training_graphs_dir "mis/${dataset}/train" \
  --training_samples_file "difuscombination/mis/${dataset}/train" \
  --training_labels_dir "difuscombination/mis/${dataset}/train_labels" \
  --validation_graphs_dir "mis/${dataset}/test" \
  --validation_samples_file "difuscombination/mis/${dataset}/test" \
  --validation_labels_dir "difuscombination/mis/${dataset}/test_labels" \
  --test_graphs_dir "mis/${dataset}/test" \
  --test_samples_file "difuscombination/mis/${dataset}/test" \
  --test_labels_dir "difuscombination/mis/${dataset}/test_labels" \
  --batch_size ${batch_size} \
  --num_epochs ${num_epochs} \
  --validation_examples 8 \
  --inference_schedule "cosine" \
  --inference_diffusion_steps 50 \
  --use_activation_checkpoint
