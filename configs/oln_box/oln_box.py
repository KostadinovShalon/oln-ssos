_base_ = [
    './oln_box_model.py',
    '../_base_/datasets/coco_split_detection_ann_id.py',
    '../_base_/schedules/schedule_1x.py', 
    '../_base_/default_runtime.py'
]

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[6, 7])
total_epochs = 8

checkpoint_config = dict(interval=2)
# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
