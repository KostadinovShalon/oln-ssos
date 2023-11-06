import torch
import tqdm
from mmcv.runner import EpochBasedRunner
from mmcv.runner.builder import RUNNERS

from mmdet.core import bbox2roi


@RUNNERS.register_module()
class PseudoLabelEpochBasedRunner(EpochBasedRunner):

    def train(self, data_loader, **kwargs):
        # self.model.train()
        # self.mode = 'train'
        self.data_loader = data_loader
        # self._max_iters = self._max_epochs * len(self.data_loader)
        # self.call_hook('before_train_epoch')
        # time.sleep(2)  # Prevent possible deadlock during epoch transition
        # for i, data_batch in enumerate(self.data_loader):
        #     self._inner_iter = i
        #     self.call_hook('before_train_iter')
        #     self.run_iter(data_batch, train_mode=True)
        #     self.call_hook('after_train_iter')
        #     self._iter += 1
        #
        # self.call_hook('after_train_epoch')
        # self._epoch += 1
        super(PseudoLabelEpochBasedRunner, self).train(data_loader, **kwargs)
        self.run_pseudo_label_epoch()

    def run_pseudo_label_epoch(self):
        with torch.no_grad():
            for i, data_batch in enumerate(tqdm.tqdm(self.data_loader)):
                inputs, kwargs = self.model.scatter(data_batch, {}, self.model.device_ids)
                fts = self.model.module.extract_feat(inputs[0]['img'])
                rois = bbox2roi([b for b in inputs[0]['gt_bboxes']])
                self.model.module.roi_head.accumulate_pseudo_labels(fts, rois)
            self.model.module.roi_head.calculate_pseudo_labels()