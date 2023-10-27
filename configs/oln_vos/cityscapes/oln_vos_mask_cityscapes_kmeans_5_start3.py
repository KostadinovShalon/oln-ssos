_base_ = '../oln_vos_mask_cityscapes_kmeans_5.py'

model = dict(
    roi_head=dict(
        start_epoch=3)
    )

# Pretrained on COCO
load_from = './work_dirs/oln_mask/epoch_8.pth'
work_dir = './work_dirs/oln_vos_mask_cityscapes_kmeans_5_start3/'
