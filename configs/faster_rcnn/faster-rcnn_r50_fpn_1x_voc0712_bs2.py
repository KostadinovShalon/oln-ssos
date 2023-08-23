_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_1x.py'
]

model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=20,))
    )
scale = 8
data_root = 'data/voc0712/'
data = dict(
    samples_per_gpu=16 // scale,
    workers_per_gpu=2,
    train=dict(
        ann_file=data_root + 'voc0712_train_all.json',
        img_prefix=data_root + 'JPEGImages/',),
    val=dict(
        ann_file=data_root + 'val_coco_format.json',
        img_prefix=data_root + 'JPEGImages/',),
    test=dict(
        ann_file=data_root + 'val_coco_format.json',
        img_prefix=data_root + 'JPEGImages/'))
# optimizer
optimizer = dict(type='SGD', lr=0.02 / scale, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500 * scale,
    warmup_ratio=0.001,
    step=[8 * scale, 11 * scale])
total_epochs = 12 * scale

checkpoint_config = dict(interval=scale)
# yapf:disable
log_config = dict(
    interval=50 * scale,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])