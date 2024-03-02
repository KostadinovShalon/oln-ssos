from . import CocoSplitDataset
from .builder import DATASETS


@DATASETS.register_module()
class BDDSplitDataset(CocoSplitDataset):
    CLASSES = ('pedestrian', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle',
               'traffic light', 'traffic sign')
    class_names_dict = {
        'all': CLASSES
    }


@DATASETS.register_module()
class CocoBDDSplitDataset(CocoSplitDataset):
    CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')
    BDD_CLASSES = (
        'person', 'bicycle', 'car', 'motorcycle', 'truck', 'bus', 'train', 'traffic light', 'stop sign')
    NONBDD_CLASSES = (
        'fire hydrant', 'parking meter', 'bench',
        'elephant', 'bear', 'zebra', 'giraffe',
        'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
        'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
        'hot dog', 'pizza', 'donut', 'cake',
        'bed', 'toilet', 'laptop',
        'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
        'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush',
        'airplane', 'bird', 'boat', 'bottle',
        'cat', 'chair', 'cow', 'dining table', 'dog', 'horse',
        'potted plant', 'sheep', 'couch',
        'tv'
    )
    class_names_dict = {
        'all': CLASSES,
        'bdd': BDD_CLASSES,
        'nonbdd': NONBDD_CLASSES
    }
