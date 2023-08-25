_base_ = [
    './vos_faster-rcnn_r50_fpn_1x_voc0712_detectron2.py'
]
ood_data_root = "data/coco/"
data = dict(
    test=dict(
        ann_file=ood_data_root + 'annotations/instances_val2017_ood_rm_overlap.json',
        img_prefix=ood_data_root + 'val2017/'),
    train=dict(
        ann_file=ood_data_root + 'annotations/instances_val2017_ood_rm_overlap.json',
        img_prefix=ood_data_root + 'val2017/')
)
