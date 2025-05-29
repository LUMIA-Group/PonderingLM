#!/bin/bash

#gpu 8
cd PonderingLM
FORCE_TORCHRUN=1  llamafactory-cli train \
--model_name_or_path /data0/sxsong/0427/checkpoint-155000 \
--stage sft \
--do_train \
--finetuning_type full \
--dataset alpaca_en \
--template alpaca \
--output_dir saves/alpaca/full/pt/1.4b_ponderinglm_alpaca \
--logging_steps 1 \
--save_strategy epoch \
--plot_loss \
--overwrite_output_dir \
--per_device_train_batch_size 4 \
--gradient_accumulation_steps 4 \
--learning_rate 2.5e-5 \
--num_train_epochs 5.0 \
--lr_scheduler_type cosine \
--warmup_ratio 0.02 \
--report_to wandb \
--adam_beta1 0.9 \
--adam_beta2 0.95 \
--ddp_timeout 180000000 \
--deepspeed examples/deepspeed/ds_z0_config.json \
--flash_attn fa2 \
--cutoff_len 512 \
--save_total_limit 3 \
--latent true \
--more_iterations 3 \
--disable_gradient_checkpointing true \
--dataloader_num_workers 4 \
--bf16 \
--ponderinglm true \
--pondering_steps 3 \
--scale_embeds true \
--mutiply_pondering_steps true \