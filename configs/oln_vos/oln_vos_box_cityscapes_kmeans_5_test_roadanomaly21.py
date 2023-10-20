_base_ = './oln_vos_box_cityscapes_kmeans_5.py'

data_root = 'data/roadanomaly21/'
data = dict(
    test=dict(
        ann_file=data_root +
                 'instances.json',
        img_prefix=data_root + 'images/')
)
