#!/bin/bash
# Script to launch MAE pre-training with Cow ID dataset

# Activate your Python environment if needed
# source ~/mae/env/bin/activate

# Pass one of the following URLs to the --resume argument to load a pretrained model
# https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base_full.pth
# https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large_full.pth
# https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_huge_full.pth
# --model must match pretrained model: mae_vit_base_patch16, mae_vit_large_patch16, or mae_vit_huge_patch14

# Please note a batch size of 4096 was used in the paper for pre-training on ImageNet
# However, other authors have reported good results with smaller batch sizes (e.g., 512 or 1024)

# --data_path should point to the training data directory
# The directory needs at least one subdirectory (class directory) to play nice with the dataloader

# Generate random UIDs for output and log directories
UID=$(openssl rand -hex 6)
echo "UID: $UID"

python main_pretrain.py \
    --data_path /home/luke/Downloads/id-cows/ \
    --batch_size 4 \
    --accum_iter 1 \
    --epochs 800 \
    --model mae_vit_base_patch16 \
    --resume "https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base_full.pth" \
    --input_size 96 224 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --weight_decay 0.05 \
    --blr 1e-4 \
    --min_lr 0.0 \
    --warmup_epochs 5 \
    --output_dir output/pretraining/$UID \
    --log_dir output/pretraining/$UID/logs \
    --seed 0 \
    --num_workers 10 \
    --pin_mem 