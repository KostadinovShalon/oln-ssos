_base_ = './oln_ffs_box_sixray10_kmeans_20.py'

model = dict(
    roi_head=dict(
        negative_sampling_size=50,)
)
