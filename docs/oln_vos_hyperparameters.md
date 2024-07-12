# OLN-VOS Hyperparameters

OLN-VOS code includes new detectors, RoI heads, bboxes heads and dataset classes, with new hyperparameters that
are described in this doc. These can also be seen in the source code.

## Models

### Detectors
- **EpochFasterRCNN** (Parent class: _FasterRCNN_). Epoch-aware Faster RCNN models for the OLN-VOS architecture. 
It contains the following extra hyperparameters:
  - **calculate_pseudo_labels_from_epoch** (_float_, default=0): Indicates the starting epoch where bbox features are 
  clustered to generate the pseudo-labels. This process is carried out before the specified epoch starts.
  - **use_weak_bboxes** (_bool_, default=False, <span style="color:red">experimental</span>): if true, all the generated
  bboxes are used to train the OLN-VOS OoD discriminator. Otherwise, only ground-truth bboxes are used instead (default behaviour).
- **EpochMaskRCNN** (Parent class: _MaskRCNN_). Same as EpochFasterRCNN but for MaskRCNN. Same hyperparameters as EpochFasterRCNN.

### RoI Heads
- **OLNKMeansVOSRoIHead** (Parent class: _OlnRoIHead_): OLN-VOS RoI head that uses k-means for unsupervised clustering of
bbox features. Hyperparameters:
  - **start_epoch** (_int_, default=0): starting epoch where VOS is done. While the default value is 0, experimental results
  show that it should start around the middle of training (Check VOS paper).
  - **logistic_regression_hidden_dim** (_int_, default=512): number of hidden dimensions for the linear module of the OoD layer in VOS. 
  - **vos_samples_per_class** (_int_, default=1000): number of samples for each (pseudo)class in VOS when calculating the means and variances
  of the Gaussians in VOS.
  - **negative_sampling_size** (_int_, default=10000): number of generated virtual samples from the Gaussian distributions for sampling 
  the lowest confidence elements as outliers. 
  - **bottomk_epsilon_dist** (_int_, default=1): number of the lowest confidence elements to use as anomalies from the virtually generated samples.
  - **ood_loss_weight** (_float_, default=0.1): OoD loss weight.
  - **pseudo_label_loss_weight** (_float_, default=1.0): pseudo label classification loss weight.
  - **k** (_int_, default=5): number of pseudoclasses.
  - **recalculate_pseudolabels_every_epoch** (_int_, default=1): number of epochs to wait until the re-calculation of the pseudo-labels.
  - **k_means_minibatch** (_bool_, default=True): whether to use the original k-means implementation of the mini-batch version.
  - **repeat_ood_sampling** (_int_, default=4): number of repetitions of the sampling/selecting the lowest confidence elements process. This
  is useful for low number of pseudo-classes. 
  - **use_all_proposals_ood** (_bool_, default=False): if True, uses all predicted bounding boxes during training as ground truth for pseudo-label generation. This is still experimental.
  - **pseudo_bbox_roi_extractor** (_dict_). Config dict for the RoIAlign layer used for generating pseudo labels. The default config is: 
  ```python
  dict(type='SingleRoIExtractor',
       roi_layer=dict(type='RoIAlign', output_size=3, sampling_ratio=0),
       out_channels=256,
       featmap_strides=[4, 8, 16, 32]),```
- **OLNMaskKMeansVOSRoIHead** (Parent classL _OLNKMeansVOSRoIHead_, _MaskScoringOlnRoIHead_): similar to OLNKMeansVOSRoIHead, but for the
mask version of OLN-VOS. Its hyperparameters are the same as OLNKMeansVOSRoIHead and MaskScoringOlnRoIHead.
- **OLNKMeansFFSRoIHead** (Parent class: _OLNKMeansVOSRoIHead_): Similar to OLNKMeansVOSRoIHead but it uses Feature Flow Synthesis.
Same hyperparameters with the following extra parameter:
  - **nll_loss_weight** (_float_, default=0.1): loss weight for the FFS generation.
- **OLNMaskKMeansFFSRoIHead** (Parent classL _OLNKMeansVOSRoIHead_, _OLNMaskKMeansVOSRoIHead_). Same as OLNKMeansFFSRoIHead but for the
mask version. It has the same hyperparameters as its parent classes.

### Bounding Box Heads

- **VOSShared2FCBBoxScoreHead** (Parent class: Shared2FCBBoxHead). Same bounding box as Shared2FCBBoxHead but it handles OoD training data.
No extra hyperparameters are defined. It should be uses for any OLN-VOS model.

## Datasets

- **VOSCocoDataset** (Parent class: _CocoDataset_). Used for VOS models. No new hyperparameters.

- **VOSCocoSplitDataset** (Parent class: _CocoSplitDataset_). Used for OLN-VOS models. It handles the use of 
pseudoclasses within OLN-VOS. No new hyperparameters.

- **VOSDB6SplitDataset** (Parent class: _VOSCocoSplitDataset_). Used for OLN-VOS models on the db6 dataset.

## Data Pipelines

- **PseudoLabelFormatBundle** (Parent Class: _DefaultFormatBundle_): pseudo label data formatter. Used when
pseudo-labelled data is needed. No extra parameters are used.
- **LoadAnnotationsWithAnnID** (Parent class: _LoadAnnotations_): annotation loader that includes the annotation id. Used
for pseudo-label generation. It defines the following extra parameters:
  - with_ann_id (*bool*, default=True): indicates if the annotation id is included.
  - with_pseudo_labels (*bool*, default=True): indicates if the pseudo-label is included.
  - with_weak_bbox (*bool*, default=True, <span style="color:red">experimental</span>): indicates if associated weak bounding boxes are also included.