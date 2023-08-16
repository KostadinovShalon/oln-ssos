_base_ = [
    '../faster_rcnn/faster-rcnn_r50_fpn_1x_voc0712.py'
]

custom_imports = dict(
    imports=[
        'vos.models.detectors.epoch_faster_rcnn',
        'vos.models.roi_heads.vos_roi_head',
        'vos.models.roi_heads.bbox_heads.vos_convfc_bbox_head',
        'vos.datasets.vos_coco'
    ],
    allow_failed_imports=False)

model = dict(
    type='EpochFasterRCNN',
    roi_head=dict(
        type='VOSRoIHead',
        vos_samples_per_class=1000,
        start_epoch=0,
        logistic_regression_hidden_dim=512,
        negative_sampling_size=10000,
        bottomk_epsilon_dist=1,
        ood_loss_weight=0.1,
        bbox_head=dict(
            type='VOSConvFCBBoxHead'))
    )

dataset_type = "VOSCocoDataset"
data = dict(
    train=dict(
        type=dataset_type),
    val=dict(
        type=dataset_type),
    test=dict(
        type=dataset_type))
custom_hooks = [dict(type='SetEpochInfoHook')]
