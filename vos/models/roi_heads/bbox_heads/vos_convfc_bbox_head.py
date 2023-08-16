from mmcv.runner import force_fp32

from mmdet.models import HEADS, Shared2FCBBoxHead
from mmdet.core import multiclass_nms
import torch.nn.functional as F


@HEADS.register_module()
class VOSConvFCBBoxHead(Shared2FCBBoxHead):

    def forward(self, x):
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        # x_cls = x
        # x_reg = x

        cls_score = self.forward_cls_branch(x)
        bbox_pred = self.forward_bbox_branch(x)
        return cls_score, bbox_pred, x

    def forward_cls_branch(self, x_cls):
        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))
        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        return cls_score

    def forward_bbox_branch(self, x_reg):
        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        return bbox_pred

    @force_fp32(apply_to=('cls_score', 'bbox_pred', 'ood_scores'))
    def get_bboxes(self,
                   rois,
                   cls_score,
                   bbox_pred,
                   ood_scores,
                   img_shape,
                   scale_factor,
                   rescale=False,
                   cfg=None):
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        scores = F.softmax(cls_score, dim=1) if cls_score is not None else None

        if bbox_pred is not None:
            bboxes = self.bbox_coder.decode(
                rois[:, 1:], bbox_pred, max_shape=img_shape)
        else:
            bboxes = rois[:, 1:].clone()
            if img_shape is not None:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1])
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0])

        if rescale and bboxes.size(0) > 0:
            if isinstance(scale_factor, float):
                bboxes /= scale_factor
            else:
                scale_factor = bboxes.new_tensor(scale_factor)
                bboxes = (bboxes.view(bboxes.size(0), -1, 4) /
                          scale_factor).view(bboxes.size()[0], -1)

        if cfg is None:
            return bboxes, scores, ood_scores
        else:
            det_bboxes, det_labels, inds = multiclass_nms(bboxes, scores,
                                                          cfg.score_thr, cfg.nms,
                                                          cfg.max_per_img, return_inds=True)
            det_ood_scores = ood_scores[inds]

            return det_bboxes, det_labels, det_ood_scores
