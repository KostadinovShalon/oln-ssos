_base_ = './oln_vos_box_ltdimaging_kmeans_10.py'

model = dict(
    roi_head=dict(
        negative_sampling_size=1000)
)
