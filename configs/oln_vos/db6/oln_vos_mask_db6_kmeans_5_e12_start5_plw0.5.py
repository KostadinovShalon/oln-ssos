_base_ = "./oln_vos_mask_db6_kmeans_5.py"

model = dict(
    roi_head=dict(
        start_epoch=5,
        pseudo_label_loss_weight=0.5,
    )
)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=0.001,
    step=[8, 11])
total_epochs = 12

load_from = './work_dirs/oln_mask/epoch_8.pth'
# Runner type
runner = dict(type='PseudoLabelEpochBasedRunner')