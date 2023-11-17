_base_ = './oln_vos_mask_cityscapes_kmeans_5.py'

data_root = 'data/roadanomaly21/'
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=2,
    test=dict(
        ann_file=data_root +
                 'instances.json',
        img_prefix=data_root + 'images/')
)
