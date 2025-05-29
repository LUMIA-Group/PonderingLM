#!/bin/bash

#gpu 8
cd PonderingLM
FORCE_TORCHRUN=1  llamafactory-cli train \
--model_name_or_path EleutherAI/pythia-1.4b \
--stage pt \
--do_train \
--finetuning_type full \
--dataset pile \
--template default \
--cutoff_len 2048 \
--output_dir saves/pile/full/pt/1.4b_ponderinglm_pile \
--logging_steps 1 \
--plot_loss \
--overwrite_output_dir \
--per_device_train_batch_size 8 \
--gradient_accumulation_steps 16 \
--learning_rate 2.0e-4 \
--num_train_epochs 1.0 \
--lr_scheduler_type cosine_with_min_lr \
--warmup_ratio 0.01 \
--report_to wandb \
--adam_beta1 0.9 \
--adam_beta2 0.95 \
--weight_decay 0.01 \
--ddp_timeout 180000000 \
--eval_dataset testpile \
--per_device_eval_batch_size 16 \
--deepspeed examples/deepspeed/ds_z0_config.json \
--eval_steps 20000 \
--flash_attn fa2 \
--save_total_limit 10 \
--bf16 \
--ponderinglm true \
--pondering_steps 3 \
--scale_embeds true \
--mutiply_pondering_steps true \
--dataloader_num_workers 4 \
--train_from_scratch true \
--lr_scheduler_kwargs '{"min_lr_rate":0.1}' \
--do_eval