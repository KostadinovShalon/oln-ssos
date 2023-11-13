_base_ = './oln_vos_box_voc0712_kmeans_5_start12.py'
model = dict(
    roi_head=dict(
        k=10,
        repeat_ood_sampling=2))
work_dir='./work_dirs/oln_vos_box_voc0712_kmeans_10_start12/'