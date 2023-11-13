_base_ = './oln_vos_box_voc0712_kmeans_100_start12.py'
model = dict(
    roi_head=dict(
        start_epoch=10,))
work_dir='./work_dirs/oln_vos_box_voc0712_kmeans_100_start10/'