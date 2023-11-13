import numpy as np

from mmdet.datasets import DATASETS, CocoDataset, CocoSplitDataset


@DATASETS.register_module()
class VOSCocoDataset(CocoDataset):

    def _det2json(self, results):
        """Convert detection results to COCO json style."""
        json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            result = results[idx]
            for label in range(len(result)):
                bboxes = result[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['ood_score'] = float(bboxes[i][5])
                    data['inter_feats'] = bboxes[i][6:27].tolist()
                    data['category_id'] = self.cat_ids[label]
                    json_results.append(data)
        return json_results


@DATASETS.register_module()
class VOSCocoSplitDataset(CocoSplitDataset):

    def __init__(self, **kwargs):
        super(VOSCocoSplitDataset, self).__init__(**kwargs)
        for ann in self.coco.anns.values():
            ann['pseudo_class'] = 0

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,\
                labels, masks, seg_map. "masks" are raw annotations and not \
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        gt_ann_ids = []
        gt_pseudo_class = []
        gt_weak_bboxes = []
        gt_weak_bboxes_labels = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            # A weak bounding box is a box detected by OLN but notin the gt
            is_weak = ann['weak'] if 'weak' in ann.keys() else False
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                if not is_weak:
                    gt_bboxes.append(bbox)
                    gt_labels.append(self.cat2label[ann['category_id']])
                    gt_masks_ann.append(ann.get('segmentation', None))
                    gt_ann_ids.append(ann['id'])
                    gt_pseudo_class.append(ann['pseudo_class'])
                else:
                    gt_weak_bboxes.append(bbox)
                    gt_weak_bboxes_labels.append(ann['pseudo_class'])

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
            gt_ann_ids = np.array(gt_ann_ids, dtype=np.int64)
            gt_pseudo_class = np.array(gt_pseudo_class, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)
            gt_ann_ids = np.array([], dtype=np.int64)
            gt_pseudo_class = np.array([], dtype=np.int64)

        if gt_weak_bboxes:
            gt_weak_bboxes = np.array(gt_weak_bboxes, dtype=np.float32)
            gt_weak_bboxes_labels = np.array(gt_weak_bboxes_labels, dtype=np.int64)
        else:
            gt_weak_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_weak_bboxes_labels = np.array([], dtype=np.int64)


        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            ann_ids=gt_ann_ids,
            pseudo_labels=gt_pseudo_class,
            weak_bboxes=gt_weak_bboxes,
            weak_bboxes_labels=gt_weak_bboxes_labels,
            seg_map=seg_map)

        return ann

    def _det2json(self, results):
        """Convert detection results to COCO json style."""
        json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            result = results[idx]
            for label in range(len(result)):
                bboxes = result[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['ood_score'] = float(bboxes[i][5])
                    data['category_id'] = self.cat_ids[label]
                    json_results.append(data)
        return json_results

    def _segm2json(self, results):
        """Convert instance segmentation results to COCO json style."""
        bbox_json_results = []
        segm_json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            det, seg = results[idx]
            for label in range(len(det)):
                # bbox results
                bboxes = det[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['ood_score'] = float(bboxes[i][5])
                    data['category_id'] = self.cat_ids[label]
                    bbox_json_results.append(data)

                # segm results
                # some detectors use different scores for bbox and mask
                if isinstance(seg, tuple):
                    segms = seg[0][label]
                    mask_score = seg[1][label]
                else:
                    if label < len(seg):
                        segms = seg[label]
                        mask_score = [bbox[4] for bbox in bboxes]
                    else:
                        segms = []
                        mask_score = []
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(mask_score[i])
                    data['ood_score'] = float(bboxes[i][5])
                    data['category_id'] = self.cat_ids[label]
                    if isinstance(segms[i]['counts'], bytes):
                        segms[i]['counts'] = segms[i]['counts'].decode()
                    data['segmentation'] = segms[i]
                    segm_json_results.append(data)
        return bbox_json_results, segm_json_results


@DATASETS.register_module()
class VOSDB6SplitDataset(VOSCocoSplitDataset):
    CLASSES = ('firearm', 'firearmpart', 'knife', 'camera', 'ceramic_knife', 'laptop')
    ID_CLASSES = ('knife', 'camera', 'ceramic_knife', 'laptop')
    OOD_CLASSES = ('firearm', 'firearmpart')
    class_names_dict = {
        'all': CLASSES,
        'id': ID_CLASSES,
        'ood': OOD_CLASSES
    }
