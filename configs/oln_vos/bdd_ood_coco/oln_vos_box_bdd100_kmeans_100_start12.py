_base_ = './oln_vos_box_bdd100_kmeans_5_start12.py'
model = dict(
    roi_head=dict(
        k=100,
        repeat_ood_sampling=1))
work_dir='./work_dirs/oln_vos_box_bdd100_kmeans_100_start12/'