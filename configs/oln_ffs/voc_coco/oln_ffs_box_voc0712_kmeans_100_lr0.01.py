_base_ = './oln_ffs_box_voc0712_kmeans_100.py'

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)