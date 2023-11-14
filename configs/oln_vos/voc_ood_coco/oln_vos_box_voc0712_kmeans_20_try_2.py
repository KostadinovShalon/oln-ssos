_base_ = './oln_vos_box_voc0712_kmeans_5.py'
model = dict(
    roi_head=dict(
        k=20,
        repeat_ood_sampling=1,
        start_epoch=2,))
work_dir = './work_dirs/oln_vos_box_voc0712_kmeans_20_try_2/'

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[5])
total_epochs = 7

optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)

load_from = './work_dirs/oln_box/epoch_8.pth'