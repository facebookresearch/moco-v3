#!/bin/bash

CHK_DIR=./output
PRETRAIN="moco_pretrained_tf2pt.pth"

IMAGENET_DIR=""
# For DEiT just use the original script

CIFAR10_DIR=""
python -u -m torch.distributed.launch --nproc_per_node=8 --use_env main.py \
    --batch-size 128 --output_dir ${CHK_DIR} --epochs 100 --lr 3e-4 --weight-decay 0.1 --eval-freq 10 \
    --no-pin-mem  --warmup-epochs 3 --data-set cifar10 --data-path ${CIFAR10_DIR}  --no-repeated-aug \
    --reprob 0.0 --drop-path 0.1 --mixup 0.8 --cutmix 1


CIFAR100_DIR=""
python -u -m torch.distributed.launch --nproc_per_node=8 --use_env main.py \
    --batch-size 128 --output_dir ${CHK_DIR} --epochs 100 --lr 3e-4 --weight-decay 0.1 --eval-freq 10 \
    --no-pin-mem  --warmup-epochs 3 --data-set cifar100 --data-path ${CIFAR100_DIR}  --no-repeated-aug \
    --reprob 0.0 --drop-path 0.1 --mixup 0.5 --cutmix 1


FLOWERS_DIR=""
python -u -m torch.distributed.launch --nproc_per_node=8 --use_env main.py \
    --batch-size 128 --output_dir ${CHK_DIR} --epochs 100 --lr 3e-4 --weight-decay 0.3 --eval-freq 10 \
    --no-pin-mem  --warmup-epochs 3 --data-set flowers --data-path ${FLOWERS_DIR}  --no-repeated-aug \
    --reprob 0.25 --drop-path 0.1 --mixup 0 --cutmix 0


PETS_DIR=""
python -u -m torch.distributed.launch --nproc_per_node=8 --use_env main.py \
    --batch-size 128 --output_dir ${CHK_DIR} --epochs 100 --lr 3e-4 --weight-decay 0.1 --eval-freq 10 \
    --no-pin-mem  --warmup-epochs 3 --data-set pets --data-path ${PETS_DIR}  --no-repeated-aug \
    --reprob 0 --drop-path 0 --mixup 0.8 --cutmix 0
