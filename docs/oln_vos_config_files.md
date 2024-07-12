# OLN-VOS Config Files

Since this repo is based on a combination of the original [OLN](https://github.com/mcahny/object_localization_network) repo, developed in the MMDet 
framework, and the [VOS](https://github.com/deeplearning-wisc/vos) repo, developed in the Detectron-2 framework, new config files with several
new parameters are being introduced. This file reviews the changes in the config files.

## New Config Files

The following config files are introduced:


|                     Path                     |                            Description                             |
|:--------------------------------------------:|:------------------------------------------------------------------:|
| configs/_base_/db6_split_instance_ann_id.py  |                Base config file for the DB6 dataset                |
| configs/_base_/ltdimaging_split_detection.py |                Base config file for the Ltd dataset                |
|       configs/oln_box/oln_box_model.py       |               Base config file for the OLN-Box model               |
|        configs/oln_box/oln_box_XXX.py        | Base config file for the OLN-Box model applied to the XXX dataset  |
|      configs/oln_mask/oln_mask_model.py      |              Base config file for the OLN-Mask model               |
|       configs/oln_mask/oln_mask_XXX.py       | Base config file for the OLN-Mask model applied to the XXX dataset |
|          configs/oln_vos/XXX/YYY.py          |   Config file for a YYY OLN-VOS model applied to the XXX dataset   |
|          configs/oln_ffs/XXX/YYY.py          | Config file for a YYY OLN-VOS-FFS model applied to the XXX dataset |

## OLN-VOS Config Parameters

The OLN-VOS architecture introduces new RoI Heads and other components with new config parameters, based on the VOS 
architecture. These are described as follows:

```

model = dict(
    type='EpochFasterRCNN|EpochMaskRCNN',                           # Wrapper that sends the current epoch to the model
    calculate_pseudo_labels_from_epoch=0,                           # Starts the pseudo-label training from this epoch
    use_weak_bboxes=False,                                          # (Experimental) Uses OLN detected boxes as extra inputs for pseudo-label and Ood training
    roi_head=dict(
        type='OLNKMeansVOSRoIHead|OLNMaskKMeansVOSRoIHead',         # RoI Head for OLN-VOS (with and without mask)
        start_epoch=0,                                              # Starting epoch for training the anomaly discriminator
        logistic_regression_hidden_dim=512,                         # Internal dimension of the anomaly MLP
        negative_sampling_size=10000,                               # Number of pseudo-class distribution samplings during VOS
        bottomk_epsilon_dist=1,                                     # Bottom index from the sampled virtual outliers to consider as outlier
        ood_loss_weight=0.1,                                        # Loss weight for training the discriminator
        pseudo_label_loss_weight=1.,                                # Pseudo-label classification head loss weight
        k=5,                                                        # Number of pseudo labels
        repeat_ood_sampling=4,                                      # Number of samplings per pseudo-class
        bbox_head=dict(
            type='VOSShared2FCBBoxScoreHead',                       # OLN-VOS box head
            reg_class_agnostic=True))
    )
```

To use EpochFasterRCNN|EpochMaskRCNN, a new befor-train-hook is introduced. It is added as follows:

```
custom_hooks = [dict(type='SetEpochInfoHook')]
```