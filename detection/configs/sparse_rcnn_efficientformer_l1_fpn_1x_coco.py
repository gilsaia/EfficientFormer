_base_ = './sparse_rcnn_r50_fpn_1x_coco.py'

model = dict(
    backbone=dict(
        type='efficientformer_l1_feat',
        style='pytorch',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='../weights/efficientformer_l1_300d.pth',
        ),
    ),
    neck=dict(in_channels=[48, 96, 224, 448]))

# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.000025, weight_decay=0.0001)
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=1, norm_type=2))

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1)
# optimizer = dict(_delete_=True, type='AdamW', lr=0.0002, weight_decay=0.0001)
# optimizer_config = dict(grad_clip=None)
