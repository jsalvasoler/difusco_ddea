#!/bin/sh

echo "Running Evolutionary Algorithm on $(hostname)"

# first argument is the dataset
dataset=$1
echo "dataset: $dataset"
# second arguments is the initialization
initialization=$2
echo "initialization: $initialization"
# third argument is the recombination
recombination=$3
echo "recombination: $recombination"
# fourth argument is n_generations
n_generations=$4
echo "n_generations: $n_generations"
# fifth argument is process_idx
process_idx=$5
echo "process_idx: $process_idx"
# sixth argument is n_processes
n_processes=$6
echo "n_processes: $n_processes"
# seventh argument, if given, is the pop_size
if [ -n "$7" ]; then
    pop_size=$7
else
    pop_size=24
fi
echo "pop_size: $pop_size"
# wandb logger name
wandb_logger_name="mis_ea_${dataset}_${initialization}_${pop_size}_tournament"
train_dataset=$dataset
hatch run cli ea run-ea \
    --task "mis" \
    --algo "ga" \
    --data_path "/home/e12223411/repos/difusco/data" \
    --logs_path "/home/e12223411/repos/difusco/logs" \
    --models_path "/home/e12223411/repos/difusco/models" \
    --results_path "/home/e12223411/repos/difusco/results" \
    --test_split "mis/${dataset}/test" \
    --test_split_label_dir "mis/${dataset}/test_labels" \
    --test_samples_file "difuscombination/mis/${train_dataset}/test" \
    --test_labels_dir "difuscombination/mis/${train_dataset}/test_labels" \
    --project_name "difusco" \
    --wandb_logger_name $wandb_logger_name \
    --device "cuda" \
    --pop_size $pop_size \
    --n_generations $n_generations \
    --initialization $initialization \
    --recombination $recombination \
    --config_name "mis_inference" \
    --parallel_sampling $pop_size \
    --sequential_sampling 1 \
    --diffusion_type "gaussian" \
    --ckpt_path "mis/mis_${train_dataset}_gaussian.ckpt" \
    --ckpt_path_difuscombination "difuscombination/mis_${train_dataset}_gaussian_new.ckpt" \
    --inference_diffusion_steps 50 \
    --process_idx $process_idx \
    --num_processes $n_processes \
    --save_results 1 \
    --cache_dir "/home/e12223411/repos/difusco/cache/mis/${dataset}/test" \
    --time_limit_inf_s 100000 \
    --selection_method "tournament_unique" \
