_base_ = "./oln_ffs_mask_db6_kmeans_5.py"

model = dict(
    roi_head=dict(
        start_epoch=2,
        k=100,
        repeat_ood_sampling=1,
    )
)