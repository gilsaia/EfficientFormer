#!/usr/bin/env bash

# CUDA_VISIBLE_DEVICES=1 python test.py configs/mask_rcnn_efficientformerv2_s2_fpn_1x_coco.py \
# ../weights/mask_rcnn_efficientformerv2_s2_fpn_1x_coco.pth \
# --eval bbox segm proposal

CUDA_VISIBLE_DEVICES=1 python test.py configs/mask_trt_rcnn_efficientformerv2_s2_fpn_1x_coco.py \
../weights/mask_rcnn_efficientformerv2_s2_fpn_1x_coco.pth \
--eval bbox segm proposal

# python pytorch2onnx.py configs/mask_rcnn_efficientformerv2_s2_fpn_1x_coco.py \
# ../weights/mask_rcnn_efficientformerv2_s2_fpn_1x_coco.pth \
# --dynamic-export

# python deploy.py deploy_configs/mmdet/detection/detection_onnxruntime_static.py \
# configs/mask_rcnn_efficientformerv2_s2_fpn_1x_coco.py \
# ../weights/mask_rcnn_efficientformerv2_s2_fpn_1x_coco.pth \
# ./000000001242.jpg \
# --device cpu \
# --show \
# --work-dir ./deploy_res

# python pytorch2onnxdet.py configs/mask_rcnn/mask_rcnn_1x_coco.py ../weights/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth --dynamic-export --show

# python pytorch2onnx.py configs/mask_rcnn_efficientformer_l1_fpn_1x_coco.py \
# ../weights/efficientformer_l1_coco.pth \
# --dynamic-export

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
