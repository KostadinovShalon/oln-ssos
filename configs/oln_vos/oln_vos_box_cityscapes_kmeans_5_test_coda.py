_base_ = './oln_vos_box_cityscapes_kmeans_5.py'

data_root = '/home/brian/Documents/datasets/coda/coda_sample/CODA/sample/'
data = dict(
    test=dict(
        ann_file=data_root +
                 'corner_case.json',
        img_prefix=data_root + 'images/')
)

work_dir = './work_dirs/oln_vos_box_cityscapes_kmeans_5/'
