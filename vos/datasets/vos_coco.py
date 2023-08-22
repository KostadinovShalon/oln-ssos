from mmdet.datasets import DATASETS, CocoDataset


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
