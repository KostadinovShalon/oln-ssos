_base_ = "./oln_vos_mask_db6_kmeans_5.py"

model = dict(
    roi_head=dict(
        start_epoch=12,
    )
)
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=0.001,
    step=[11, 16])
total_epochs = 18
#
# Runner type
runner = dict(type='PseudoLabelEpochBasedRunner')
