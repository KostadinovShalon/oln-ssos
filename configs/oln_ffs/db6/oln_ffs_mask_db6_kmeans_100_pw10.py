_base_ = "./oln_ffs_mask_db6_kmeans_100.py"

model = dict(
    roi_head=dict(
        start_epoch=4,
        pseudo_label_loss_weight=10,
    )
)
