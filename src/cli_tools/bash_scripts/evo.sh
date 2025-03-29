#!/bin/bash

# echo which machine
echo "Running on $(hostname)"

dataset=er_700_800
initialization=difusco_sampling # random_feasible or difusco_sampling
recombination=classic # classic or optimal or difuscombination
pop_size=10
n_generations=10
wandb_logger_name=mis_gaussian_er_700_800_test

n_samples=3 # set to none to run on the full test set

echo "running with parameters:
dataset=$dataset
initialization=$initialization
recombination=$recombination
pop_size=$pop_size
n_generations=$n_generations
wandb_logger_name=$wandb_logger_name"
echo "-------"


hatch run cli ea run-ea \
  --task "mis" \
  --config_name "mis_inference" \
  --wandb_logger_name $wandb_logger_name \
  --data_path "/home/e12223411/repos/difusco/data" \
  --models_path "/home/e12223411/repos/difusco/models" \
  --results_path "/home/e12223411/repos/difusco/results" \
  --logs_path "/home/e12223411/repos/difusco/logs" \
  --training_split "mis/${dataset}/test" \
  --training_split_label_dir "mis/${dataset}/test_labels" \
  --validation_split "mis/${dataset}/test" \
  --validation_split_label_dir "mis/${dataset}/test_labels" \
  --test_split "mis/${dataset}/test" \
  --test_split_label_dir "mis/${dataset}/test_labels" \
  --test_samples_file "difuscombination/mis/${dataset}/test" \
  --test_labels_dir "difuscombination/mis/${dataset}/test_labels" \
  --initialization $initialization \
  --recombination $recombination \
  --pop_size $pop_size \
  --n_generations $n_generations \
  --device "cuda" \
  --diffusion_type "gaussian" \
  --ckpt_path "mis/mis_${dataset}_gaussian.ckpt" \
  --ckpt_path_difuscombination "difuscombination/mis_${dataset}_gaussian_new.ckpt" \
  --cache_dir "/home/e12223411/repos/difusco/cache/mis/${dataset}/test" \
  --parallel_sampling $pop_size \
  --wandb_logger_name $wandb_logger_name \
  --validate_samples $n_samples \
  --opt_recomb_time_limit 30 \
  --mutation_prob 0.25 \
  --preserve_optimal_recombination False \
#   --save_results True
