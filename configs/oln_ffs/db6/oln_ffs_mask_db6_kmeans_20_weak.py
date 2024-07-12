_base_ = "./oln_ffs_mask_db6_kmeans_5.py"

model = dict(
    use_weak_bboxes=True,
    roi_head=dict(
        start_epoch=4,
        k=20,
        repeat_ood_sampling=1,
        weak_bbox_test_confidence=0.5
    )
)
# Runner type
runner = dict(type='PseudoLabelEpochBasedRunner')
