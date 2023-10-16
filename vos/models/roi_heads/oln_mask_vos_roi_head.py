import numpy as np
import torch

from mmdet.models.roi_heads.oln_roi_head import MaskScoringOlnRoIHead

from vos.models.roi_heads.oln_vos_roi_head import OLNKMeansVOSRoIHead
from vos.models.roi_heads.vos_roi_head import bbox2result_ood
from mmdet.models import HEADS


@HEADS.register_module()
class OLNMaskKMeansVOSRoIHead(OLNKMeansVOSRoIHead, MaskScoringOlnRoIHead):

    def simple_test(self,
                    x,
                    proposal_list,
                    img_metas,
                    proposals=None,
                    rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels, det_ood_scores = self.simple_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
        # det_ood_scores = self.simple_test_ood(x, proposal_list)
        if torch.onnx.is_in_onnx_export():
            if self.with_mask:
                segm_results = self.simple_test_mask(
                    x, img_metas, det_bboxes, det_labels, rescale=rescale)
                return det_bboxes, det_labels, det_ood_scores, segm_results,
            else:
                return det_bboxes, det_labels, det_ood_scores

        results = [
            bbox2result_ood(det_bboxes[i], det_labels[i], det_ood_scores[i],
                            self.bbox_head.num_classes)
            for i in range(len(det_bboxes))
        ]

        if not self.with_mask:
            return results
        else:
            segm_results = self.simple_test_mask(
                x, img_metas, det_bboxes, det_labels, rescale=rescale)
            results = list(zip(results, segm_results))
            for bbox_results, segm_results in results:
                masks, mask_score = segm_results
                for b, ms in zip(bbox_results, mask_score):
                    if len(b) == 0:
                        continue
                    bbox_score = b[:, 4]
                    geometric_mask_score = np.cbrt((bbox_score ** 2) * ms)
                    b[:, 4] = geometric_mask_score
            return results