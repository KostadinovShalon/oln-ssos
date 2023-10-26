from . import CocoSplitDataset
from .builder import DATASETS


@DATASETS.register_module()
class DB6SplitDataset(CocoSplitDataset):
    CLASSES = ('firearm', 'firearmpart', 'knife', 'camera', 'ceramic_knife', 'laptop')
    ID_CLASSES = ('knife', 'camera', 'ceramic_knife', 'laptop')
    OOD_CLASSES = ('firearm', 'firearmpart')
    class_names_dict = {
        'all': CLASSES,
        'id': ID_CLASSES,
        'ood': OOD_CLASSES
    }
