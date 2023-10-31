_base_ = '../oln_vos_mask_cityscapes_kmeans_5.py'

model = dict(
    roi_head=dict(
        start_epoch=12)
    )
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[12, 16])
total_epochs = 18

# Pretrained on COCO
load_from = './work_dirs/oln_mask/epoch_8.pth'
work_dir = './work_dirs/oln_vos_mask_cityscapes_kmeans_5_e18/'
