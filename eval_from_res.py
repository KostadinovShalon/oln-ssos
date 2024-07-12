import numpy as np

# gt_path = 'data/coco/annotations/instances_val2017_ood_rm_overlap.json'
# pred_path = 'work_dirs/oln_ffs_box_voc0712_kmeans_100_ns300/epoch_8.ood.bbox.json'
# anomaly_score_threshold = 0.99515
#
# gt_path = 'data/db6/test_db6_only_firearms.json'
# pred_path = 'work_dirs/oln_ffs_box_db6_kmeans_20_ns200/epoch_8.ood.bbox.json'
# anomaly_score_threshold = 0.99401
#
# gt_path = 'data/SIXRay10/annotation/SIXRay10_test_only_firearms.json'
# pred_path = 'work_dirs/oln_ffs_box_sixray10_kmeans_20/epoch_8.ood.bbox.json'
# anomaly_score_threshold = 0.99733
#
# gt_path = 'data/ltdimaging/test_week_only_vehicles.json'
# pred_path = 'work_dirs/oln_vos_box_ltdimaging_kmeans_10/epoch_8.ood.bbox.json'
# anomaly_score_threshold = 0.64829


########### PROB
# gt_path = 'data/coco/annotations/instances_val2017_ood_rm_overlap.json'
# pred_path = '/home/brian/Projects/PROB/exps/MOWODB/PROB/eval/owod_rm_overlap_test_detections.json'
#
# gt_path = 'data/db6/test_db6_only_firearms.json'
# pred_path = '/home/brian/Projects/PROB/exps/MOWODB-DBF6/PROB/eval/test_db6_only_firearms_detections.json'

# gt_path = '/home/brian/datasets/CHALearn_LTDImaging/test_week_only_vehicles.json'
# pred_path = '/home/brian/Projects/PROB/exps/MOWODB-LTD/PROB/eval/test_week_only_vehicles_detections.json'

gt_path = 'data/SIXRay10/annotation/SIXRay10_test_only_firearms.json'
pred_path = '/home/brian/Projects/PROB/exps/MOWODB-SIXRAY10/PROB/eval/SIXRay10_test_only_firearms_detections.json'


from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json


gt_coco = COCO(gt_path)
res = json.load(open(pred_path, 'r'))
# res = [r for r in res if r['ood_score'] < anomaly_score_threshold and r['score'] > 0.05]
res = [r for r in res if r['category_id'] == 5 and r['score'] > 0.05]
for r in res:
    r['category_id'] = 1
pred_coco = gt_coco.loadRes(res)

coco_eval = COCOeval(gt_coco, pred_coco, 'bbox')
coco_eval.params.useCats = 0
coco_eval.params.iouThrs = np.array([0.5])
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()
