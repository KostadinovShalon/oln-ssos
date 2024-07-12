_base_ = './oln_ffs_box_voc0712_kmeans_5.py'
model = dict(
    roi_head=dict(
        k=100,
        repeat_ood_sampling=1,))