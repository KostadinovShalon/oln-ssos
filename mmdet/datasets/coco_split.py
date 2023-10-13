"""This file contains code to build dataloader of COCO-split dataset.

Reference:
    "Learning Open-World Object Proposals without Learning to Classify",
        Aug 2021. https://arxiv.org/abs/2108.06753
        Dahun Kim, Tsung-Yi Lin, Anelia Angelova, In So Kweon and Weicheng Kuo
"""

import itertools
import logging
import os.path as osp
import tempfile
from collections import OrderedDict

import mmcv
import numpy as np
from mmcv.utils import print_log
from pycocotools.coco import COCO
# Added for cross-category evaluation
from .cocoeval_wrappers import COCOEvalWrapper, COCOEvalXclassWrapper

from terminaltables import AsciiTable

from mmdet.core import eval_recalls
from .builder import DATASETS
from .coco import CocoDataset

try:
    import pycocotools
    if not hasattr(pycocotools, '__sphinx_mock__'):  # for doc generation
        assert pycocotools.__version__ >= '12.0.2'
except AssertionError:
    raise AssertionError('Incompatible version of pycocotools is installed. '
                         'Run pip uninstall pycocotools first. Then run pip '
                         'install mmpycocotools to install open-mmlab forked '
                         'pycocotools.')


@DATASETS.register_module()
class CocoSplitDataset(CocoDataset):

    def __init__(self, 
                 is_class_agnostic=False, 
                 train_class='all',
                 eval_class='all',
                 **kwargs):
        # We convert all category IDs into 1 for the class-agnostic training and
        # evaluation. We train on train_class and evaluate on eval_class split.
        self.is_class_agnostic = is_class_agnostic
        self.train_class = train_class
        self.eval_class = eval_class
        super(CocoSplitDataset, self).__init__(**kwargs)
    
    CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')
    VOC_CLASSES = (
               'airplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'dining table', 'dog', 'horse', 
               'motorcycle', 'person', 'potted plant', 'sheep', 'couch',
               'train', 'tv')
    NONVOC_CLASSES = (
               'truck', 'traffic light', 'fire hydrant',
               'stop sign', 'parking meter', 'bench',
               'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot dog', 'pizza', 'donut', 'cake',
               'bed', 'toilet', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')
    class_names_dict = {
        'all': CLASSES,
        'voc': VOC_CLASSES,
        'nonvoc': NONVOC_CLASSES
    }

    def load_annotations(self, ann_file):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """

        self.coco = COCO(ann_file)

        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)
        self.train_cat_ids = self.coco.get_cat_ids(
            cat_names=self.class_names_dict[self.train_class]
            )
        self.eval_cat_ids = self.coco.get_cat_ids(
            cat_names=self.class_names_dict[self.eval_class]
            )
        if self.is_class_agnostic:
            self.cat2label = {cat_id: 0 for cat_id in self.cat_ids}
        else:
            self.cat2label = {
                cat_id: i for i, cat_id in enumerate(self.cat_ids)}

        self.img_ids = self.coco.get_img_ids()
        data_infos = []
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            info['filename'] = info['file_name']
            data_infos.append(info)
        return data_infos

    # Refer to custom.py -- filter_img is not used in test_mode.
    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        # obtain images that contain annotation
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        # obtain images that contain annotations of the required categories
        ids_in_cat = set()
        
        for i, class_id in enumerate(self.train_cat_ids):
            ids_in_cat |= set(self.coco.cat_img_map[class_id])
        # merge the image id sets of the two conditions and use the merged set
        # to filter out images if self.filter_empty_gt=True
        ids_in_cat &= ids_with_ann

        valid_img_ids = []
        for i, img_info in enumerate(self.data_infos):
            img_id = self.img_ids[i]
            if self.filter_empty_gt and img_id not in ids_in_cat:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
                valid_img_ids.append(img_id)
        self.img_ids = valid_img_ids
        return valid_inds

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
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.train_cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)                
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann.get('segmentation', None))

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

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
            seg_map=seg_map)

        return ann

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 proposal_nums=(10, 20, 30, 50, 100, 300, 500, 1000, 1500),
                 iou_thrs=None,
                 metric_items=None):
        """Evaluation in COCO-Split protocol.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'bbox', 'segm', 'proposal', 'proposal_fast'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float], optional): IoU threshold used for
                evaluating recalls/mAPs. If set to a list, the average of all
                IoUs will also be computed. If not specified, [0.50, 0.55,
                0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95] will be used.
                Default: None.
            metric_items (list[str] | str, optional): Metric items that will
                be returned. If not specified, ``['AR@100', 'AR@300',
                'AR@1000', 'AR_s@1000', 'AR_m@1000', 'AR_l@1000' ]`` will be
                used when ``metric=='proposal'``, ``['mAP', 'mAP_50', 'mAP_75',
                'mAP_s', 'mAP_m', 'mAP_l']`` will be used when
                ``metric=='bbox' or metric=='segm'``.

        Returns:
            dict[str, float]: COCO style evaluation metric.
        """

        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')
        if iou_thrs is None:
            iou_thrs = np.linspace(
                .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        if metric_items is not None:
            if not isinstance(metric_items, list):
                metric_items = [metric_items]

        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)

        eval_results = OrderedDict()
        cocoGt = self.coco
        for metric in metrics:
            msg = f'Evaluating {metric}...'
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)

            if metric == 'proposal_fast':
                ar = self.fast_eval_recall(
                    results, proposal_nums, iou_thrs, logger='silent')
                log_msg = []
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
                    log_msg.append(f'\nAR@{num}\t{ar[i]:.4f}')
                log_msg = ''.join(log_msg)
                print_log(log_msg, logger=logger)
                continue

            if metric not in result_files:
                raise KeyError(f'{metric} is not in results')
            try:
                cocoDt = cocoGt.loadRes(result_files[metric])
            except IndexError:
                print_log(
                    'The testing results of the whole dataset is empty.',
                    logger=logger,
                    level=logging.ERROR)
                break

            iou_type = 'bbox' if metric == 'proposal' else metric

            # Class manipulation.
            for idx, ann in enumerate(cocoGt.dataset['annotations']):
                if ann['category_id'] in self.eval_cat_ids:
                    cocoGt.dataset['annotations'][idx]['ignored_split'] = 0
                else:
                    cocoGt.dataset['annotations'][idx]['ignored_split'] = 1

            # Cross-category evaluation wrapper.
            cocoEval = COCOEvalXclassWrapper(cocoGt, cocoDt, iou_type)

            cocoEval.params.catIds = self.cat_ids
            cocoEval.params.imgIds = self.img_ids
            cocoEval.params.maxDets = list(proposal_nums)
            cocoEval.params.iouThrs = iou_thrs
            # mapping of cocoEval.stats
            coco_metric_names = {
                'mAP': 0,
                'mAP_50': 1,
                'mAP_75': 2,
                'mAP_s': 3,
                'mAP_m': 4,
                'mAP_l': 5,
                'AR@10': 6,
                'AR@20': 7,
                'AR@50': 8,
                'AR@100': 9,
                'AR@300': 10,
                'AR@500': 11,
                'AR@1000': 12,
                'AR@1500': 13,
            }
            if metric_items is not None:
                for metric_item in metric_items:
                    if metric_item not in coco_metric_names:
                        raise KeyError(
                            f'metric item {metric_item} is not supported')

            cocoEval.params.useCats = 0  # treat all FG classes as single class.
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            if classwise:  # Compute per-category AP
                # Compute per-category AP
                # from https://github.com/facebookresearch/detectron2/
                precisions = cocoEval.eval['precision']
                # precision: (iou, recall, cls, area range, max dets)
                assert len(self.cat_ids) == precisions.shape[2]

                results_per_category = []
                for idx, catId in enumerate(self.cat_ids):
                    # area range index 0: all area ranges
                    # max dets index -1: typically 100 per image
                    nm = self.coco.loadCats(catId)[0]
                    precision = precisions[:, :, idx, 0, -1]
                    precision = precision[precision > -1]
                    if precision.size:
                        ap = np.mean(precision)
                    else:
                        ap = float('nan')
                    results_per_category.append(
                        (f'{nm["name"]}', f'{float(ap):0.3f}'))

                num_columns = min(6, len(results_per_category) * 2)
                results_flatten = list(
                    itertools.chain(*results_per_category))
                headers = ['category', 'AP'] * (num_columns // 2)
                results_2d = itertools.zip_longest(*[
                    results_flatten[i::num_columns]
                    for i in range(num_columns)
                ])
                table_data = [headers]
                table_data += [result for result in results_2d]
                table = AsciiTable(table_data)
                print_log('\n' + table.table, logger=logger)

            if metric_items is None:
                metric_items = [
                    'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'
                ]

            for metric_item in metric_items:
                key = f'{metric}_{metric_item}'
                val = float(
                    f'{cocoEval.stats[coco_metric_names[metric_item]]:.3f}'
                )
                eval_results[key] = val
            ap = cocoEval.stats[:6]
            eval_results[f'{metric}_mAP_copypaste'] = (
                f'{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                f'{ap[4]:.3f} {ap[5]:.3f}')
        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results


@DATASETS.register_module()
class OI7CocoSplitDataset(CocoSplitDataset):

    CLASSES = ('Accordion', 'Adhesive tape', 'Aircraft', 'Alarm clock', 'Alpaca', 'Ambulance', 'Animal', 'Ant',
               'Antelope', 'Apple', 'Armadillo', 'Artichoke', 'Auto part', 'Axe', 'Backpack', 'Bagel', 'Baked goods',
               'Balance beam', 'Ball (Object)', 'Balloon', 'Banana', 'Band-aid', 'Banjo', 'Barge', 'Barrel',
               'Baseball bat', 'Baseball glove', 'Bat (Animal)', 'Bathroom accessory', 'Bathroom cabinet', 'Bathtub',
               'Beaker', 'Bear', 'Beard', 'Bed', 'Bee', 'Beehive', 'Beer', 'Beetle', 'Bell pepper', 'Belt', 'Bench',
               'Bicycle', 'Bicycle helmet', 'Bicycle wheel', 'Bidet', 'Billboard', 'Billiard table', 'Binoculars',
               'Bird', 'Blender', 'Blue jay', 'Boat', 'Bomb', 'Book', 'Bookcase', 'Boot', 'Bottle', 'Bottle opener',
               'Bow and arrow', 'Bowl', 'Bowling equipment', 'Box', 'Boy', 'Brassiere', 'Bread', 'Briefcase',
               'Broccoli', 'Bronze sculpture', 'Brown bear', 'Building', 'Bull', 'Burrito', 'Bus', 'Bust', 'Butterfly',
               'Cabbage', 'Cabinetry', 'Cake', 'Cake stand', 'Calculator', 'Camel', 'Camera', 'Can opener', 'Canary',
               'Candle', 'Candy', 'Cannon', 'Canoe', 'Cantaloupe', 'Car', 'Carnivore', 'Carrot', 'Cart',
               'Cassette deck', 'Castle', 'Cat', 'Cat furniture', 'Caterpillar', 'Cattle', 'Ceiling fan', 'Cello',
               'Centipede', 'Chainsaw', 'Chair', 'Cheese', 'Cheetah', 'Chest of drawers', 'Chicken', 'Chime', 'Chisel',
               'Chopsticks', 'Christmas tree', 'Clock', 'Closet', 'Clothing', 'Coat', 'Cocktail', 'Cocktail shaker',
               'Coconut', 'Coffee (drink)', 'Coffee cup', 'Coffee table', 'Coffeemaker', 'Coin', 'Common fig',
               'Common sunflower', 'Computer keyboard', 'Computer monitor', 'Computer mouse', 'Container',
               'Convenience store', 'Cookie', 'Cooking spray', 'Corded phone', 'Cosmetics', 'Couch', 'Countertop',
               'Cowboy hat', 'Crab', 'Cream', 'Cricket ball', 'Crocodile', 'Croissant', 'Crown', 'Crutch', 'Cucumber',
               'Cupboard', 'Curtain', 'Cutting board', 'Dagger', 'Dairy Product', 'Deer', 'Desk', 'Dessert', 'Diaper',
               'Dice', 'Digital clock', 'Dinosaur', 'Dishwasher', 'Dog', 'Dog bed', 'Doll', 'Dolphin', 'Door',
               'Door handle', 'Doughnut', 'Dragonfly', 'Drawer', 'Dress', 'Drill (Tool)', 'Drink', 'Drinking straw',
               'Drum', 'Duck', 'Dumbbell', 'Eagle', 'Earring', 'Egg', 'Elephant', 'Envelope', 'Eraser', 'Face powder',
               'Facial tissue holder', 'Falcon', 'Fashion accessory', 'Fast food', 'Fax', 'Fedora', 'Filing cabinet',
               'Fire hydrant', 'Fireplace', 'Fish', 'Fixed-wing aircraft', 'Flag', 'Flashlight', 'Flower', 'Flowerpot',
               'Flute', 'Flying disc', 'Food', 'Food processor', 'Football', 'Football helmet', 'Footwear', 'Fork',
               'Fountain', 'Fox', 'French fries', 'French horn', 'Frog', 'Fruit', 'Frying pan', 'Furniture',
               'Garden Asparagus', 'Gas stove', 'Giraffe', 'Girl', 'Glasses', 'Glove', 'Goat', 'Goggles', 'Goldfish',
               'Golf ball', 'Golf cart', 'Gondola', 'Goose', 'Grape', 'Grapefruit', 'Grinder', 'Guacamole', 'Guitar',
               'Hair dryer', 'Hair spray', 'Hamburger', 'Hammer', 'Hamster', 'Hand dryer', 'Handbag', 'Handgun',
               'Harbor seal', 'Harmonica', 'Harp', 'Harpsichord', 'Hat', 'Headphones', 'Heater', 'Hedgehog',
               'Helicopter', 'Helmet', 'High heels', 'Hiking equipment', 'Hippopotamus', 'Home appliance', 'Honeycomb',
               'Horizontal bar', 'Horse', 'Hot dog', 'House', 'Houseplant', 'Human arm', 'Human body', 'Human ear',
               'Human eye', 'Human face', 'Human foot', 'Human hair', 'Human hand', 'Human head', 'Human leg',
               'Human mouth', 'Human nose', 'Humidifier', 'Ice cream', 'Indoor rower', 'Infant bed', 'Insect',
               'Invertebrate', 'Ipod', 'Isopod', 'Jacket', 'Jacuzzi', 'Jaguar (Animal)', 'Jeans', 'Jellyfish',
               'Jet ski', 'Jug', 'Juice', 'Kangaroo', 'Kettle', 'Kitchen & dining room table', 'Kitchen appliance',
               'Kitchen knife', 'Kitchen utensil', 'Kitchenware', 'Kite', 'Knife', 'Koala', 'Ladder', 'Ladle',
               'Ladybug', 'Lamp', 'Land vehicle', 'Lantern', 'Laptop', 'Lavender (Plant)', 'Lemon (plant)', 'Leopard',
               'Light bulb', 'Light switch', 'Lighthouse', 'Lily', 'Limousine', 'Lion', 'Lipstick', 'Lizard', 'Lobster',
               'Loveseat', 'Luggage and bags', 'Lynx', 'Magpie', 'Mammal', 'Man', 'Mango', 'Maple', 'Maraca',
               'Marine invertebrates', 'Marine mammal', 'Measuring cup', 'Mechanical fan', 'Medical equipment',
               'Microphone', 'Microwave oven', 'Milk', 'Miniskirt', 'Mirror', 'Missile', 'Mixer', 'Mixing bowl',
               'Mobile phone', 'Monkey', 'Moths and butterflies', 'Motorcycle', 'Mouse', 'Muffin', 'Mug', 'Mule',
               'Mushroom', 'Musical instrument', 'Musical keyboard', 'Nail (Construction)', 'Necklace', 'Nightstand',
               'Oboe', 'Office building', 'Office supplies', 'Orange (fruit)', 'Organ (Musical Instrument)', 'Ostrich',
               'Otter', 'Oven', 'Owl', 'Oyster', 'Paddle', 'Palm tree', 'Pancake', 'Panda', 'Paper cutter',
               'Paper towel', 'Parachute', 'Parking meter', 'Parrot', 'Pasta', 'Pastry', 'Peach', 'Pear', 'Pen',
               'Pencil case', 'Pencil sharpener', 'Penguin', 'Perfume', 'Person', 'Personal care',
               'Personal flotation device', 'Piano', 'Picnic basket', 'Picture frame', 'Pig', 'Pillow', 'Pineapple',
               'Pitcher (Container)', 'Pizza', 'Pizza cutter', 'Plant', 'Plastic bag', 'Plate', 'Platter',
               'Plumbing fixture', 'Polar bear', 'Pomegranate', 'Popcorn', 'Porch', 'Porcupine', 'Poster', 'Potato',
               'Power plugs and sockets', 'Pressure cooker', 'Pretzel', 'Printer', 'Pumpkin', 'Punching bag', 'Rabbit',
               'Raccoon', 'Racket', 'Radish', 'Ratchet (Device)', 'Raven', 'Rays and skates', 'Red panda',
               'Refrigerator', 'Remote control', 'Reptile', 'Rhinoceros', 'Rifle', 'Ring binder', 'Rocket',
               'Roller skates', 'Rose', 'Rugby ball', 'Ruler', 'Salad', 'Salt and pepper shakers', 'Sandal', 'Sandwich',
               'Saucer', 'Saxophone', 'Scale', 'Scarf', 'Scissors', 'Scoreboard', 'Scorpion', 'Screwdriver',
               'Sculpture', 'Sea lion', 'Sea turtle', 'Seafood', 'Seahorse', 'Seat belt', 'Segway', 'Serving tray',
               'Sewing machine', 'Shark', 'Sheep', 'Shelf', 'Shellfish', 'Shirt', 'Shorts', 'Shotgun', 'Shower',
               'Shrimp', 'Sink', 'Skateboard', 'Ski', 'Skirt', 'Skull', 'Skunk', 'Skyscraper', 'Slow cooker', 'Snack',
               'Snail', 'Snake', 'Snowboard', 'Snowman', 'Snowmobile', 'Snowplow', 'Soap dispenser', 'Sock', 'Sofa bed',
               'Sombrero', 'Sparrow', 'Spatula', 'Spice rack', 'Spider', 'Spoon', 'Sports equipment', 'Sports uniform',
               'Squash (Plant)', 'Squid', 'Squirrel', 'Stairs', 'Stapler', 'Starfish', 'Stationary bicycle',
               'Stethoscope', 'Stool', 'Stop sign', 'Strawberry', 'Street light', 'Stretcher', 'Studio couch',
               'Submarine', 'Submarine sandwich', 'Suit', 'Suitcase', 'Sun hat', 'Sunglasses', 'Surfboard', 'Sushi',
               'Swan', 'Swim cap', 'Swimming pool', 'Swimwear', 'Sword', 'Syringe', 'Table', 'Table tennis racket',
               'Tablet computer', 'Tableware', 'Taco', 'Tank', 'Tap', 'Tart', 'Taxi', 'Tea', 'Teapot', 'Teddy bear',
               'Telephone', 'Television', 'Tennis ball', 'Tennis racket', 'Tent', 'Tiara', 'Tick', 'Tie', 'Tiger',
               'Tin can', 'Tire', 'Toaster', 'Toilet', 'Toilet paper', 'Tomato', 'Tool', 'Toothbrush', 'Torch',
               'Tortoise', 'Towel', 'Tower', 'Toy', 'Traffic light', 'Traffic sign', 'Train', 'Training bench',
               'Treadmill', 'Tree', 'Tree house', 'Tripod', 'Trombone', 'Trousers', 'Truck', 'Trumpet', 'Turkey',
               'Turtle', 'Umbrella', 'Unicycle', 'Van', 'Vase', 'Vegetable', 'Vehicle', 'Vehicle registration plate',
               'Violin', 'Volleyball (Ball)', 'Waffle', 'Waffle iron', 'Wall clock', 'Wardrobe', 'Washing machine',
               'Waste container', 'Watch', 'Watercraft', 'Watermelon', 'Weapon', 'Whale', 'Wheel', 'Wheelchair',
               'Whisk', 'Whiteboard', 'Willow', 'Window', 'Window blind', 'Wine', 'Wine glass', 'Wine rack',
               'Winter melon', 'Wok', 'Woman', 'Wood-burning stove', 'Woodpecker', 'Worm', 'Wrench', 'Zebra',
               'Zucchini')
    class_names_dict = {
        'all': CLASSES,
    }
