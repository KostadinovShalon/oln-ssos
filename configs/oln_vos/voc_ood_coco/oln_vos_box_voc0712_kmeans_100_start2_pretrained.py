_base_ = './oln_vos_box_voc0712_kmeans_5_start12.py'
model = dict(
    calculate_pseudo_labels_from_epoch=0,
    roi_head=dict(
        k=100,
        start_epoch=2,
        repeat_ood_sampling=1))
work_dir = './work_dirs/oln_vos_box_voc0712_kmeans_100_start2_pretrained/'
load_from = './work_dirs/oln_box_voc/epoch_8.pth'

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=0.001,
    step=[7])
total_epochs = 10

optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)