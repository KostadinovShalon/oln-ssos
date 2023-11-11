from mmdet.core import BitmapMasks, PolygonMasks
from mmdet.datasets import PIPELINES
from mmdet.datasets.pipelines import LoadAnnotations


@PIPELINES.register_module()
class LoadAnnotationsWithAnnID(LoadAnnotations):
    """Load mutiple types of annotations.

    Args:
        with_bbox (bool): Whether to parse and load the bbox annotation.
             Default: True.
        with_label (bool): Whether to parse and load the label annotation.
            Default: True.
        with_mask (bool): Whether to parse and load the mask annotation.
             Default: False.
        with_seg (bool): Whether to parse and load the semantic segmentation
            annotation. Default: False.
        poly2mask (bool): Whether to convert the instance masks from polygons
            to bitmaps. Default: True.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self, with_ann_id=True, with_pseudo_labels=True,
                 with_weak_bbox=True, **kwargs):
        super().__init__(**kwargs)
        self.with_ann_id = with_ann_id
        self.with_pseudo_labels = with_pseudo_labels
        self.with_weak_bbox = with_weak_bbox

    def _load_ann_ids(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded label annotations.
        """

        results['gt_ann_ids'] = results['ann_info']['ann_ids'].copy()
        return results

    def _load_pseudo_labels(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded label annotations.
        """

        results['gt_pseudo_labels'] = results['ann_info']['pseudo_labels'].copy()
        return results

    def _load_weak_bboxes(self, results):

        results['gt_weak_bboxes'] = results['ann_info']['weak_bboxes'].copy()
        results['gt_weak_bboxes_labels'] = results['ann_info']['weak_bboxes_labels'].copy()
        return results

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded bounding box, label, mask and
                semantic segmentation annotations.
        """

        if self.with_bbox:
            results = self._load_bboxes(results)
            if results is None:
                return None
        if self.with_label:
            results = self._load_labels(results)
        if self.with_mask:
            results = self._load_masks(results)
        if self.with_seg:
            results = self._load_semantic_seg(results)
        if self.with_ann_id:
            results = self._load_ann_ids(results)
        if self.with_pseudo_labels:
            results = self._load_pseudo_labels(results)
        if self.with_weak_bbox:
            results = self._load_weak_bboxes(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(with_bbox={self.with_bbox}, '
        repr_str += f'with_label={self.with_label}, '
        repr_str += f'with_ann_ids={self.with_ann_id}, '
        repr_str += f'with_pseudo_labels={self.with_pseudo_labels}, '
        repr_str += f'with_mask={self.with_mask}, '
        repr_str += f'with_seg={self.with_seg}, '
        repr_str += f'poly2mask={self.poly2mask}, '
        repr_str += f'poly2mask={self.file_client_args})'
        return repr_str