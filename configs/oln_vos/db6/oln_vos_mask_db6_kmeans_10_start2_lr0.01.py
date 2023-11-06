_base_ = "./oln_vos_mask_db6_kmeans_5.py"

model = dict(
    roi_head=dict(
        start_epoch=2,
        k=10,
        repeat_ood_sampling=2,
    )
)
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)

load_from = './work_dirs/oln_mask/epoch_8.pth'
# Runner type
runner = dict(type='PseudoLabelEpochBasedRunner')