_base_ = "./oln_ffs_mask_db6_kmeans_20.py"

model = dict(
    roi_head=dict(
        start_epoch=4,
        nll_loss_weight=0.01,
    )
)
