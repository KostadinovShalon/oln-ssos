_base_ = './oln_vos_box_cityscapes_kmeans_100.py'

dataset_type = "VOSCocoSplitDataset"
data_root = '/home/brian/Documents/datasets/coda/coda_sample/CODA/sample/'
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type),
    val=dict(
        type=dataset_type),
    test=dict(
        type=dataset_type,
        ann_file=data_root +
                 'corner_case.json',
        img_prefix=data_root + 'images/')
)