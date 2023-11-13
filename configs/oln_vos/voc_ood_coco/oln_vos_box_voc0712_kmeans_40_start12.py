_base_ = './oln_vos_box_voc0712_kmeans_5_start12.py'
model = dict(
    roi_head=dict(
        k=40,
        repeat_ood_sampling=1))
work_dir='./work_dirs/oln_vos_box_voc0712_kmeans_40_start12/'