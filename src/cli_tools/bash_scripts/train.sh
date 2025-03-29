dataset_name=er_700_800
diffusion_type=gaussian

# select do_train for training, do_test for testing
# if do_train is specified, a given checkpoint will be used as the starting point for training

hatch run cli difusco run-difusco \
  --task "mis" \
  --wandb_logger_name "mis_${diffusion_type}_${dataset_name}_test" \
  --diffusion_type "${diffusion_type}" \
  --do_train \
  --do_test \
  --learning_rate 0.0002 \
  --weight_decay 0.0001 \
  --lr_scheduler "cosine-decay" \
  --data_path "/home/e12223411/repos/difusco/data" \
  --training_split "mis/${dataset_name}/train" \
  --training_split_label_dir "mis/${dataset_name}/train_labels/" \
  --validation_split "mis/${dataset_name}/test" \
  --test_split "mis/${dataset_name}/test" \
  --logs_path "/home/e12223411/repos/difusco/logs" \
  --models_path "/home/e12223411/repos/difusco/models" \
  --batch_size 4 \
  --num_epochs 50 \
  --validation_examples 8 \
  --inference_schedule "cosine" \
  --inference_diffusion_steps 50 \
  --parallel_sampling 8 \
  --use_activation_checkpoint \
  --ckpt_path "mis/mis_${dataset_name}_${diffusion_type}.ckpt" \
  --resume_weight_only