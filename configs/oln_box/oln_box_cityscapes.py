_base_ = './oln_box.py'

# Dataset
dataset_type = 'CocoSplitDataset'
data_root = 'data/cityscapes/'

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=8,
        dataset=dict(
            type=dataset_type,
            is_class_agnostic=True,
            ann_file=data_root +
                     'annotations/instancesonly_filtered_gtFine_train.json',
            img_prefix=data_root)),
    val=dict(
        type=dataset_type,
        is_class_agnostic=True,
        ann_file=data_root +
                 'annotations/instancesonly_filtered_gtFine_val.json',
        img_prefix=data_root),
    test=dict(
        type=dataset_type,
        is_class_agnostic=True,
        ann_file=data_root +
                 'annotations/instancesonly_filtered_gtFine_val.json',
        img_prefix=data_root))

optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[7])
total_epochs = 8

checkpoint_config = dict(interval=2)
# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
resume_from = None
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'  # noqa
workflow = [('train', 1)]

work_dir = './work_dirs/oln_box_cityscapes/'
