_base_ = './oln_vos_box_cityscapes_kmeans_5.py'
model = dict(
    roi_head=dict(
        k=10,
        repeat_ood_sampling=2,)
    )
# Pretrained on COCO
load_from = './work_dirs/oln_box/epoch_8.pth'
work_dir = './work_dirs/oln_vos_box_cityscapes_kmeans_10/'
