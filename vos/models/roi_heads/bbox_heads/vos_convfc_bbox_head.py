import numpy as np
import torch
from mmcv.ops import batched_nms
from mmcv.runner import force_fp32

from mmdet.models import HEADS, Shared2FCBBoxHead
import torch.nn.functional as F


def multiclass_nms_with_ood(multi_bboxes,
                            multi_scores,
                            multi_ood_scores,
                            inter_feats,
                            score_thr,
                            nms_cfg,
                            max_num=-1,
                            score_factors=None,
                            return_inds=False,
                            ood_score_threshold=0.):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class), where the last column
            contains scores of the background class, but this will be ignored.
        multi_ood_scores (Tensor): shape (n,) OOD scores
        inter_feats (Tensor): shape (n, #class + 1) Class intermediate features
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int, optional): if there are more than max_num bboxes after
            NMS, only top max_num will be kept. Default to -1.
        score_factors (Tensor, optional): The factors multiplied to scores
            before applying NMS. Default to None.
        return_inds (bool, optional): Whether return the indices of kept
            bboxes. Default to False.

    Returns:
        tuple: (bboxes, labels, indices (optional)), tensors of shape (k, 5),
            (k), and (k). Labels are 0-based.
    """
    num_classes = multi_scores.size(1) - 1
    # exclude background category
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
    else:
        bboxes = multi_bboxes[:, None].expand(
            multi_scores.size(0), num_classes, 4)
    if inter_feats is not None:
        inter_feats = inter_feats[:, None].expand(
            multi_scores.size(0), num_classes, inter_feats.size(-1))

    scores = multi_scores[:, :-1]
    if score_factors is not None:
        scores = scores * score_factors[:, None]

    labels = torch.arange(num_classes, dtype=torch.long)
    labels = labels.view(1, -1).expand_as(scores)

    bboxes = bboxes.reshape(-1, 4)
    scores = scores.reshape(-1)
    labels = labels.reshape(-1)
    multi_ood_scores = multi_ood_scores.repeat_interleave(num_classes)
    if inter_feats is not None:
        inter_feats = inter_feats.reshape(-1, inter_feats.size(-1))

    # remove low scoring boxes
    valid_mask = scores > score_thr
    inds = valid_mask.nonzero(as_tuple=False).squeeze(1)
    bboxes, scores, labels, multi_ood_scores = bboxes[inds], scores[inds], labels[inds], multi_ood_scores[inds]
    if inter_feats is not None:
        inter_feats = inter_feats[inds]
    if inds.numel() == 0:
        if torch.onnx.is_in_onnx_export():
            raise RuntimeError('[ONNX Error] Can not record NMS '
                               'as it has not been executed this time')
        if return_inds:
            return bboxes, labels, multi_ood_scores, inter_feats, inds
        else:
            return bboxes, labels, multi_ood_scores, inter_feats

    # TODO: add size check before feed into batched_nms
    id_indices = multi_ood_scores >= ood_score_threshold
    ood_indices = multi_ood_scores < ood_score_threshold
    _labels = labels * id_indices.to(torch.int8).cpu() + \
              (labels.max() + 1) * (1 - id_indices.to(torch.int8).cpu())
    dets, keep = batched_nms(bboxes,
                             scores,
                             labels,
                             nms_cfg)
    # if len(bboxes[id_indices]) > 0:
    #     id_dets, id_keep = batched_nms(bboxes[id_indices],
    #                                    scores[id_indices],
    #                                    labels[id_indices],
    #                                    nms_cfg)
    # else:
    #     id_dets, id_keep = torch.empty((0, 5)).to(id_indices.device), torch.empty(0, dtype=torch.int16).to(id_indices.device)
    # if len(bboxes[ood_indices]) > 0:
    #     ood_dets, ood_keep = batched_nms(bboxes[ood_indices],
    #                                      scores[ood_indices],
    #                                      labels[ood_indices],
    #                                      nms_cfg)
    # else:
    #     ood_dets, ood_keep = torch.empty((0, 5)).to(id_indices.device), torch.empty(0, dtype=torch.int16).to(id_indices.device)
    #
    # dets = torch.cat((ood_dets, id_dets), dim=0)
    # keep = torch.cat((ood_keep, id_keep), dim=0)

    _keep = torch.cat((keep[multi_ood_scores[keep] < ood_score_threshold],
                       keep[multi_ood_scores[keep] >= ood_score_threshold]), dim=0)
    _dets = torch.cat((dets[multi_ood_scores[keep] < ood_score_threshold],
                       dets[multi_ood_scores[keep] >= ood_score_threshold]), dim=0)

    if max_num > 0:
        dets = _dets[:max_num]
        keep = _keep[:max_num]

    if inter_feats is not None:
        inter_feats = inter_feats[keep]

    if return_inds:
        return dets, labels[keep], multi_ood_scores[keep], inter_feats, keep
    else:
        return dets, labels[keep], multi_ood_scores[keep], inter_feats


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
        x_cls = x
        x_reg = x

        cls_score = self.forward_cls_branch(x_cls)
        bbox_pred = self.forward_bbox_branch(x_reg)
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

    @force_fp32(apply_to=('cls_score', 'bbox_pred', 'ood_scores', 'inter_feats'))
    def get_bboxes(self,
                   rois,
                   cls_score,
                   bbox_pred,
                   ood_scores,
                   inter_feats,
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
            return bboxes, scores, ood_scores, inter_feats
        else:
            det_bboxes, det_labels, det_ood_scores, inter_feats = multiclass_nms_with_ood(bboxes, scores, ood_scores,
                                                                                          inter_feats,
                                                                                          cfg.score_thr, cfg.nms,
                                                                                          cfg.max_per_img)

            return det_bboxes, det_labels, det_ood_scores, inter_feats
