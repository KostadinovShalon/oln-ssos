_base_ = ['../../oln_mask/oln_mask_model.py',
          '../../_base_/datasets/db6_split_instance.py',
          '../../_base_/schedules/schedule_1x.py',
          '../../_base_/default_runtime.py'
          ]

custom_imports = dict(
    imports=[
        'vos.models.roi_heads.oln_mask_vos_roi_head',
        'vos.models.detectors.epoch_faster_rcnn',
        'vos.models.roi_heads.bbox_heads.oln_vos_bbox_head',
        'vos.datasets.vos_coco'
    ],
    allow_failed_imports=False)

model = dict(
    type='EpochMaskRCNN',
    roi_head=dict(
        type='OLNMaskKMeansVOSRoIHead',
        start_epoch=0,
        logistic_regression_hidden_dim=512,
        negative_sampling_size=10000,
        bottomk_epsilon_dist=1,
        ood_loss_weight=0.1,
        pseudo_label_loss_weight=1.,
        k=5,
        repeat_ood_sampling=4,
        use_all_proposals_ood=False,
        bbox_head=dict(
            type='VOSShared2FCBBoxScoreHead'))
    )

checkpoint_config = dict(interval=1)
dataset_type = "VOSDB6SplitDataset"

optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=0.001,
    step=[5])
total_epochs = 7

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

load_from = './work_dirs/oln_mask/epoch_8.pth'
custom_hooks = [dict(type='SetEpochInfoHook')]