"""This file contains code to build OLN-Box head.

Reference:
    "Learning Open-World Object Proposals without Learning to Classify",
        Aug 2021. https://arxiv.org/abs/2108.06753
        Dahun Kim, Tsung-Yi Lin, Anelia Angelova, In So Kweon and Weicheng Kuo
"""
import numpy as np
import torch

from mmdet.core import bbox2roi
from ..builder import HEADS, build_head
from .standard_roi_head import StandardRoIHead


@HEADS.register_module()
class OlnRoIHead(StandardRoIHead):
    """OLN Box head.
    
    We take the top-scoring (e.g., well-centered) proposals from OLN-RPN and
    perform RoIAlign to extract the region features from each feature pyramid
    level. Then we linearize each region features and feed it through two fc
    layers, followed by two separate fc layers, one for bbox regression and the
    other for localization quality prediction. It is recommended to use IoU as
    the localization quality target in this stage. 
    """

    def _bbox_forward(self, x, rois):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score, bbox_pred, bbox_score = self.bbox_head(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats,
            bbox_score=bbox_score)
        return bbox_results

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
                            img_metas):
        """Run forward function and calculate loss for box head in training."""
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)

        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)

        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'], 
                                        bbox_results['bbox_score'],
                                        rois,
                                        *bbox_targets)

        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

    def simple_test_bboxes(self,
                           x,
                           img_metas,
                           proposals,
                           rcnn_test_cfg,
                           rescale=False):
        """Test only det bboxes without augmentation."""
        # RPN score
        rpn_score = torch.cat([p[:, -1:] for p in proposals], 0)
        rois = bbox2roi(proposals)
        bbox_results = self._bbox_forward(x, rois)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # split batch bbox prediction back to each image
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']
        bbox_score = bbox_results['bbox_score']

        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)
        bbox_score = bbox_score.split(num_proposals_per_img, 0)          
        rpn_score = rpn_score.split(num_proposals_per_img, 0)

        # some detector with_reg is False, bbox_pred will be None
        if bbox_pred is not None:
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.bbox_head.bbox_pred_split(
                    bbox_pred, num_proposals_per_img)
        else:
            bbox_pred = (None, ) * len(proposals)

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        for i in range(len(proposals)):
            det_bbox, det_label = self.bbox_head.get_bboxes(
                rois[i],
                cls_score[i],
                bbox_pred[i],
                bbox_score[i],
                rpn_score[i],
                img_shapes[i],
                scale_factors[i],
                rescale=rescale,
                cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
        return det_bboxes, det_labels


@HEADS.register_module()
class MaskScoringOlnRoIHead(OlnRoIHead):
    """Mask Scoring RoIHead for Mask Scoring RCNN.

        https://arxiv.org/abs/1903.00241
        """

    def __init__(self, mask_iou_head, **kwargs):
        assert mask_iou_head is not None
        super(MaskScoringOlnRoIHead, self).__init__(**kwargs)
        self.mask_iou_head = build_head(mask_iou_head)


    def simple_test(self,
                    x,
                    proposal_list,
                    img_metas,
                    proposals=None,
                    rescale=False):
        results = super().simple_test(x, proposal_list, img_metas, proposals, rescale)
        for bbox_results, segm_results in results:
            masks, mask_score = segm_results
            for b, ms in zip(bbox_results, mask_score):
                if len(b) == 0:
                    continue
                bbox_score = b[:, 4]
                geometric_mask_score = np.cbrt((bbox_score ** 2) * ms)
                b[:, 4] = geometric_mask_score
        return results

    def _mask_forward(self, x, rois=None, pos_inds=None, bbox_feats=None):
        """Mask head forward function used in both training and testing."""
        assert ((rois is not None) ^
                (pos_inds is not None and bbox_feats is not None))
        if rois is not None:
            mask_feats = self.mask_roi_extractor(
                x[:self.mask_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
        else:
            assert bbox_feats is not None
            mask_feats = bbox_feats[pos_inds]

        mask_pred, mask_fcn_out = self.mask_head(mask_feats)
        mask_results = dict(mask_pred=mask_pred, mask_feats=mask_feats, mask_fcn_out=mask_fcn_out)
        return mask_results

    def _mask_forward_train(self, x, sampling_results, bbox_feats, gt_masks,
                            img_metas):
        """Run forward function and calculate loss for Mask head in
        training."""
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results]) * 0
        mask_results = super(MaskScoringOlnRoIHead,
                             self)._mask_forward_train(x, sampling_results,
                                                       bbox_feats, gt_masks,
                                                       img_metas)
        if mask_results['loss_mask'] is None:
            return mask_results

        # mask iou head forward and loss
        pos_mask_pred = mask_results['mask_pred'][
            range(mask_results['mask_pred'].size(0)), pos_labels]
        mask_iou_pred = self.mask_iou_head(mask_results['mask_fcn_out'],
                                           pos_mask_pred)
        pos_mask_iou_pred = mask_iou_pred[range(mask_iou_pred.size(0)),
                                          pos_labels]

        mask_iou_targets = self.mask_iou_head.get_targets(
            sampling_results, gt_masks, pos_mask_pred,
            mask_results['mask_targets'], self.train_cfg)
        loss_mask_iou = self.mask_iou_head.loss(pos_mask_iou_pred,
                                                mask_iou_targets)
        mask_results['loss_mask'].update(loss_mask_iou)
        return mask_results

    def simple_test_mask(self,
                         x,
                         img_metas,
                         det_bboxes,
                         det_labels,
                         rescale=False):
        """Obtain mask prediction without augmentation."""
        # image shapes of images in the batch
        ori_shapes = tuple(meta['ori_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        num_imgs = len(det_bboxes)
        if all(det_bbox.shape[0] == 0 for det_bbox in det_bboxes):
            num_classes = self.mask_head.num_classes
            segm_results = [[[] for _ in range(num_classes)]
                            for _ in range(num_imgs)]
            mask_scores = [[[] for _ in range(num_classes)]
                           for _ in range(num_imgs)]
        else:
            # if det_bboxes is rescaled to the original image size, we need to
            # rescale it back to the testing scale to obtain RoIs.
            if rescale and not isinstance(scale_factors[0], float):
                scale_factors = [
                    torch.from_numpy(scale_factor).to(det_bboxes[0].device)
                    for scale_factor in scale_factors
                ]
            _bboxes = [
                det_bboxes[i][:, :4] *
                scale_factors[i] if rescale else det_bboxes[i]
                for i in range(num_imgs)
            ]
            mask_rois = bbox2roi(_bboxes)
            mask_results = self._mask_forward(x, mask_rois)
            concat_det_labels = torch.cat(det_labels)
            # get mask scores with mask iou head
            mask_feats = mask_results['mask_fcn_out']
            mask_pred = mask_results['mask_pred']
            mask_iou_pred = self.mask_iou_head(
                mask_feats, mask_pred[range(concat_det_labels.size(0)),
                                      concat_det_labels])
            # split batch mask prediction back to each image
            num_bboxes_per_img = tuple(len(_bbox) for _bbox in _bboxes)
            mask_preds = mask_pred.split(num_bboxes_per_img, 0)
            mask_iou_preds = mask_iou_pred.split(num_bboxes_per_img, 0)

            # apply mask post-processing to each image individually
            segm_results = []
            mask_scores = []
            for i in range(num_imgs):
                if det_bboxes[i].shape[0] == 0:
                    segm_results.append(
                        [[] for _ in range(self.mask_head.num_classes)])
                    mask_scores.append(
                        [[] for _ in range(self.mask_head.num_classes)])
                else:
                    segm_result = self.mask_head.get_seg_masks(
                        mask_preds[i], _bboxes[i], det_labels[i],
                        self.test_cfg, ori_shapes[i], scale_factors[i],
                        rescale)
                    # get mask scores with mask iou head
                    mask_score = self.mask_iou_head.get_mask_scores(
                        mask_iou_preds[i], det_bboxes[i], det_labels[i])
                    segm_results.append(segm_result)
                    mask_scores.append(mask_score)
        return list(zip(segm_results, mask_scores))
