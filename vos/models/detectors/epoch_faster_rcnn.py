from mmdet.models import FasterRCNN, DETECTORS


@DETECTORS.register_module()
class EpochFasterRCNN(FasterRCNN):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_, epoch-aware"""

    def __init__(self, *args, **kwargs):
        super(EpochFasterRCNN, self).__init__(*args, **kwargs)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch
        if self.roi_head is not None and hasattr(self.roi_head, 'epoch'):
            self.roi_head.epoch = epoch
