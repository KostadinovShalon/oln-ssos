_base_ = './oln_vos_box_bdd100_kmeans_5_start12.py'
model = dict(
    roi_head=dict(
        k=20,
        repeat_ood_sampling=1))
work_dir='./work_dirs/oln_vos_box_bdd100_kmeans_20_start12/'