_base_ = './oln_vos_box_voc0712_kmeans_200.py'

model = dict(roi_head=dict(start_epoch=2))