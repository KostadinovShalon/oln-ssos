_base_ = "./oln_ffs_mask_db6_kmeans_20.py"

model = dict(
    roi_head=dict(
        negative_sampling_size=1000,
    )
)