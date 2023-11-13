_base_ = './oln_vos_box_bdd100_kmeans_40_start12.py'
model = dict(
    roi_head=dict(
        start_epoch=10,))
work_dir='./work_dirs/oln_vos_box_bdd100_kmeans_40_start10/'