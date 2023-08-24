_base_ = [
    './vos_faster-rcnn_r50_fpn_1x_voc0712_detectron2.py'
]

model = dict(
    roi_head=dict(
        use_queue=False)
    )

