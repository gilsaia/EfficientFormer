#!/usr/bin/env bash

python test.py configs/mask_rcnn_efficientformerv2_s2_fpn_1x_coco.py \
../weights/mask_rcnn_efficientformerv2_s2_fpn_1x_coco.pth \
--eval bbox segm proposal

# CONFIG=$1
# CHECKPOINT=$2
# GPUS=$3
# NNODES=${NNODES:-1}
# NODE_RANK=${NODE_RANK:-0}
# PORT=${PORT:-29500}
# MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# python -m torch.distributed.launch \
#     --nnodes=$NNODES \
#     --node_rank=$NODE_RANK \
#     --master_addr=$MASTER_ADDR \
#     --nproc_per_node=$GPUS \
#     --master_port=$PORT \
#     $(dirname "$0")/test.py \
#     $CONFIG \
#     $CHECKPOINT \
#     --launcher pytorch \
#     ${@:4}
