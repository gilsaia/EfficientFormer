#!/usr/bin/env bash

# CUDA_VISIBLE_DEVICES=1 python test.py configs/mask_rcnn_efficientformerv2_s2_fpn_1x_coco.py \
# ../weights/mask_rcnn_efficientformerv2_s2_fpn_1x_coco.pth \
# --eval bbox

# CUDA_VISIBLE_DEVICES=1 python test.py configs/mask_trt_rcnn_efficientformerv2_s2_fpn_1x_coco.py \
# ../weights/mask_rcnn_efficientformerv2_s2_fpn_1x_coco.pth \
# --eval bbox

# CUDA_VISIBLE_DEVICES=1 python test_trt.py configs/mask_trt_rcnn_efficientformerv2_s2_fpn_1x_coco.py \
# ../weights/mask_rcnn_efficientformerv2_s2_fpn_1x_coco.pth \
# --eval bbox

# python pytorch2onnx.py configs/sparse_rcnn_efficientformer_l1_fpn_3x_coco.py \
# ../weights/sparse_rcnn_efficientformer_l1_fpn_3x_coco.pth --output-file end2end.onnx \
# # --dynamic-export

# python deploy.py deploy_configs/mmdet/_base_/base_tensorrt_dynamic-320x320-1344x1344.py \
# configs/mask_rcnn_efficientformerv2_s2_fpn_1x_coco.py \
# ../weights/mask_rcnn_efficientformerv2_s2_fpn_1x_coco.pth \
# ./000000001242.jpg \
# --device cuda:1 \
# --work-dir ./deploy_res

# python deploy.py deploy_configs/mmdet/detection/detection_onnxruntime_static.py \
# configs/mask_rcnn_efficientformerv2_s2_fpn_1x_coco.py \
# ../weights/mask_rcnn_efficientformerv2_s2_fpn_1x_coco.pth \
# ./000000001242.jpg \
# --device cuda:1 \
# --work-dir ./deploy_res