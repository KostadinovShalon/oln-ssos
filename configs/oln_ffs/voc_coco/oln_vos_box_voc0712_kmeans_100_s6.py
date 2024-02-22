_base_ = './oln_vos_box_voc0712_kmeans_100.py'

model = dict(roi_head=dict(start_epoch=6))