_base_ = './oln_ffs_box_voc0712_kmeans_100.py'

model = dict(roi_head=dict(negative_sampling_size=200))