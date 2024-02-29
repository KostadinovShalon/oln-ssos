from . import CocoSplitDataset
from .builder import DATASETS


@DATASETS.register_module()
class BDDSplitDataset(CocoSplitDataset):
    CLASSES = ('pedestrian', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle',
               'traffic light', 'traffic sign')
    class_names_dict = {
        'all': CLASSES
    }
