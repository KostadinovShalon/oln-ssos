_base_ = '../oln_box/oln_box_cityscapes.py'

custom_imports = dict(
    imports=[
        'vos.models.detectors.epoch_faster_rcnn',
        'vos.models.roi_heads.oln_vos_roi_head',
        'vos.models.roi_heads.bbox_heads.oln_vos_bbox_head',
        'vos.datasets.vos_coco'
    ],
    allow_failed_imports=False)

model = dict(
    type='EpochFasterRCNN',
    roi_head=dict(
        type='OLNKMeansVOSRoIHead',
        start_epoch=4,
        logistic_regression_hidden_dim=512,
        negative_sampling_size=10000,
        bottomk_epsilon_dist=1,
        ood_loss_weight=0.1,
        k=10,
        repeat_ood_sampling=2,
        bbox_head=dict(
            type='VOSShared2FCBBoxScoreHead'))
    )

dataset_type = "VOSCocoSplitDataset"
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type),
    val=dict(
        type=dataset_type),
    test=dict(
        type=dataset_type)
)

custom_hooks = [dict(type='SetEpochInfoHook')]
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
work_dir = './work_dirs/oln_vos_box_cityscapes_kmeans_10/'
