_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/default_runtime.py'
]

model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=20,))
    )

data_root = 'data/voc0712/'
data = dict(
    samples_per_gpu=16,
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
