_base_ = './oln_ffs_box_ltdimaging_kmeans_5.py'

model = dict(
    roi_head=dict(
        k=20,
        repeat_ood_sampling=1)
)
