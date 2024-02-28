_base_ = './oln_ffs_box_sixray10_kmeans_10.py'

model = dict(
    roi_head=dict(
        start_epoch=6,)
)
