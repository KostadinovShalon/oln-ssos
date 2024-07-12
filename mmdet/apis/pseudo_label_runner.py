import torch
import tqdm
from mmcv.runner import EpochBasedRunner
from mmcv.runner.builder import RUNNERS
from pycocotools.coco import COCO
from torchvision.ops import batched_nms

from mmdet.core import bbox2roi
from mmdet.datasets import RepeatDataset


@RUNNERS.register_module()
class PseudoLabelEpochBasedRunner(EpochBasedRunner):

    def train(self, data_loader, **kwargs):
        self.data_loader = data_loader
        if self.model.module.epoch >= self.model.module.calculate_pseudo_labels_from_epoch:
            self.run_pseudo_label_epoch()
        super(PseudoLabelEpochBasedRunner, self).train(data_loader, **kwargs)

    def run_pseudo_label_epoch(self):
        weak_conf_thr = self.model.module.roi_head.weak_bbox_test_confidence
        use_weak_bboxes = self.model.module.use_weak_bboxes
        with torch.no_grad():
            ann_ids = []
            weak_img_ids = []
            weak_bboxes = []
            for i, data_batch in enumerate(tqdm.tqdm(self.data_loader)):
                inputs, kwargs = self.model.scatter(data_batch, {}, self.model.device_ids)
                fts = self.model.module.extract_feat(inputs[0]['img'])
                rois = bbox2roi(inputs[0]['gt_bboxes'])
                gt_ann_ids = inputs[0]['gt_ann_ids']
                ann_ids.extend(gt_ann_ids)
                self.model.module.roi_head.accumulate_pseudo_labels(fts, rois)

                if use_weak_bboxes:
                    # Handling weak bboxes
                    gt_ann_ids_list = [g[0].cpu().item() for g in gt_ann_ids]
                    gt_anns = self.data_loader.dataset.coco.loadAnns(gt_ann_ids_list)
                    img_ids = [a['image_id'] for a in gt_anns]
                    proposal_list = self.model.module.rpn_head.simple_test_rpn(fts, inputs[0]['img_metas'])
                    res = self.model.module.roi_head.simple_test(fts, proposal_list, inputs[0]['img_metas'],
                                                                 rescale=False, with_ood=False)
                    bboxes = [res_img[0][0] for res_img in res]
                    bboxes = [bbox[batched_nms(torch.tensor(bbox[:, :4]), torch.tensor(bbox[:, 4]),
                                               torch.ones(bbox.shape[0]), 0.5)] for bbox in bboxes]
                    bboxes_filtered = [torch.tensor(b[b[:, 4] > weak_conf_thr, :4]).to(fts[0].device) for b in bboxes]
                    weak_rois = bbox2roi(bboxes_filtered)

                    weak_img_ids.extend([im_id for im_id, b in zip(img_ids, bboxes_filtered) for _ in range(b.shape[0])])
                    weak_bboxes.extend(bboxes_filtered)

                    self.model.module.roi_head.accumulate_weak_pseudo_labels(fts, weak_rois)

            labels = self.model.module.roi_head.calculate_pseudo_labels()
            ann_ids = torch.cat(ann_ids).cpu().numpy()

            pseudo_classes = {ann_id: label for ann_id, label in zip(ann_ids, labels[:ann_ids.shape[0]])}

            dataset = self.data_loader.dataset.coco.dataset
            dataset['annotations'] = [
                a for a in dataset['annotations']
                if 'weak' not in a.keys() or not a['weak']
            ]
            for ann_id, label in zip(ann_ids, labels):
                if type(self.data_loader.dataset) == RepeatDataset:
                    self.data_loader.dataset.dataset.coco.anns[ann_id]['pseudo_class'] = label
                else:
                    self.data_loader.dataset.coco.anns[ann_id]['pseudo_class'] = label
            #
            #
            # for ann_id, label in zip(ann_ids, labels[:ann_ids.shape[0]]):
            #     self.data_loader.dataset.coco.anns[ann_id]['pseudo_class'] = label
            # # Cleaning previous weak annotations
            # self.data_loader.dataset.coco.anns = {
            #     k: v for k, v in self.data_loader.dataset.coco.anns.items()
            #     if 'weak' not in v.keys() or not v['weak']
            # }
            # self.data_loader.dataset.coco.dataset['annotations'] = [
            #     a for a in self.data_loader.dataset.coco.dataset['annotations']
            #     if 'weak' not in a.keys() or not a['weak']
            # ]
            # self.data_loader.dataset.coco.imgToAnns =
            if use_weak_bboxes:
                weak_bboxes = torch.cat(weak_bboxes, dim=0)
                next_ann_id = max(self.data_loader.dataset.coco.anns.keys()) + 1
                # Adding weak annotations
                for label, weak_box, weak_img_id in zip(labels[ann_ids.shape[0]:], weak_bboxes, weak_img_ids):
                    box = weak_bboxes[0].cpu().tolist()
                    ann = {
                        'segmentation': [],
                        'iscrowd': 0,
                        'bbox': [int(box[0]), int(box[1]), int(box[2] - box[0]), int(box[3] - box[1])],
                        'area': int(box[2] - box[0]) * int(box[3] - box[1]),
                        'image_id': weak_img_id,
                        'category_id': dataset['categories'][0]['id'],
                        'pseudo_class': label,
                        'id': next_ann_id,
                        'weak': True
                    }
                    dataset['annotations'].append(ann)
                    next_ann_id += 1
                new_coco = COCO()
                new_coco.dataset = dataset
                new_coco.createIndex()
                new_coco.img_ann_map = new_coco.imgToAnns
                new_coco.cat_img_map = new_coco.catToImgs
                self.data_loader.dataset.coco = new_coco
