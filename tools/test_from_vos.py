import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import torch


id_results_path = '/home/brian/Documents/Projects/vos/detection/data/DB6-Detection/faster-rcnn/vos/random_seed_0/inference/db6_val_id/standard_nms/corruption_level_0/coco_instances_results.json'
ood_results_path = '/home/brian/Documents/Projects/vos/detection/data/DB6-Detection/faster-rcnn/vos/random_seed_0/inference/db6_val_ood/standard_nms/corruption_level_0/coco_instances_results.json'
id_test_path = '/home/brian/Documents/datasets/db6/test_db6_no_firearms.json'
ood_test_path = '/home/brian/Documents/datasets/db6/test_db6_only_firearms.json'
score_thr = 0.5

id_results_json = json.load(open(id_results_path, 'rb'))
ood_results_json = json.load(open(ood_results_path, 'rb'))

for r in id_results_json:
    r['ood_score'] = torch.logsumexp(torch.tensor(r['inter_feat'][:-1]), dim=0).sigmoid().item()

for r in ood_results_json:
    r['ood_score'] = torch.logsumexp(torch.tensor(r['inter_feat'][:-1]), dim=0).sigmoid().item()
    r['category_id'] = 1

id_gt_coco = COCO(id_test_path)
ood_gt_coco = COCO(ood_test_path)
id_res_coco = id_gt_coco.loadRes(id_results_json)

id_coco_eval = COCOeval(id_gt_coco, id_res_coco, iouType='bbox')
id_coco_eval.params.useCats = 0
id_coco_eval.evaluate()
id_coco_eval.accumulate()
id_coco_eval.summarize()

id_anomaly_scores = [r['ood_score'] for r in id_results_json]
dt_ids_with_match = [int(dt_id) for ev_im in id_coco_eval.evalImgs for dt_id in ev_im['gtMatches'][0] if dt_id > 0]
dt_ids_with_match = list(set(dt_ids_with_match))
valid_detections = id_coco_eval.cocoDt.loadAnns(dt_ids_with_match)
optimal_detections = [v for v in valid_detections if v['score'] > score_thr]
ood_scores = [o['ood_score'] for o in optimal_detections]
ood_scores.sort()
anomaly_score_threshold = ood_scores[int(len(ood_scores) * 0.05)]
print('Anomaly threshold:', anomaly_score_threshold)

ood_results_json = [r for r in ood_results_json if r['ood_score'] < anomaly_score_threshold]

ood_res_coco = ood_gt_coco.loadRes(ood_results_json)
ood_coco_eval = COCOeval(ood_gt_coco, ood_res_coco, iouType='bbox')
ood_coco_eval.params.useCats = 0
ood_coco_eval.evaluate()
ood_coco_eval.accumulate()
ood_coco_eval.summarize()