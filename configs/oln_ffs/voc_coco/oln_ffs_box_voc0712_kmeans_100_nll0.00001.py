_base_ = './oln_ffs_box_voc0712_kmeans_100.py'
model = dict(
    roi_head=dict(
        nll_loss_weight=0.00001))