#!/bin/bash
#SBATCH --job-name=addtoken
#SBATCH --partition=RTX4090
#SBATCH --ntasks-per-node=1      # 每个节点上运行1个主任务 (torchrun将管理GPU进程)
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=8
#SBATCH --account=byzeng
#SBATCH --mem=400000
#SBATCH --time=14-00:00:00
#SBATCH --output=%j.out
#SBATCH --error=%j.err

unset http_proxy
unset https_proxy
unset all_proxy
export HF_ENDPOINT=https://hf-mirror.com
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
export LD_LIBRARY_PATH=/home/byzeng/anaconda3/envs/ponderinglm/lib/:$LD_LIBRARY_PATH

cd /home/byzeng/PonderingLM
FORCE_TORCHRUN=1  llamafactory-cli train \
--model_name_or_path EleutherAI/pythia-70m \
--stage pt \
--do_train \
--finetuning_type full \
--dataset minipile \
--template default \
--cutoff_len 2048 \
--output_dir saves/minipile/full/pt/70m_vanila_test \
--logging_steps 1 \
--num_train_epochs 1.0 \
--plot_loss \
--overwrite_output_dir \
--per_device_train_batch_size 8 \
--gradient_accumulation_steps 2 \
--save_steps 2000 \
--learning_rate 1.0e-3 \
--lr_scheduler_type cosine_with_min_lr \
--warmup_ratio 0.01 \
--report_to wandb \
--adam_beta1 0.9 \
--preprocessing_num_workers 16 \
--adam_beta2 0.95 \
--weight_decay 0.01 \
--ddp_timeout 180000000 \
--deepspeed examples/deepspeed/ds_z0_config.json \
--flash_attn fa2 \
--save_total_limit 10 \
--bf16 \
--disable_gradient_checkpointing true \
--dataloader_num_workers 4 \
--train_from_scratch true \
--lr_scheduler_kwargs '{"min_lr_rate":0.1}' 
