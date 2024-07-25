
# OLN-SSOS: Towards Open-World Object-based Anomaly Detection via Self-Supervised Outlier Synthesis

[Brian K.S. Isaac-Medina](https://kostadinovshalon.github.io/), [Yona F. A. Gaus](https://yonafalinie.github.io/), [Neelanjan Bhowmik](https://scholar.google.co.uk/citations?user=5-8aIIoAAAAJ&hl=en),and [Toby Breckon](https://breckon.org/toby/).

## Introduction

While modern techniques for anomaly detection are focused on Out-of-Distribution classification or uncertainty estimation in semantic segmentation, little
attention has been oriented towards object-based anomaly detection. In this regard, a major difficulty is to both detect unseen objects and classify them as anomalies. 
For this reason, we combine the power of a class-agnostic object detector, namely [OLN](https://arxiv.org/abs/2108.06753), with the [VOS](https://arxiv.org/abs/2202.01197) OoD framework to
perform anomaly detection on an object level. We dub this method **OLN-SSOS**.

## Installation

This repo is based on the OLN repo. The installation follows the same instructions. Additionally, scitkit-learn==1.0.2 is needed.

## Config files
A description of the new config file options is detailed [here](docs/oln_vos_config_files.md). Additionally, a
more detailed documentation about the new classes and hyperparameters is described [here](docs/oln_vos_hyperparameters.md).

## Training
Training is done similar to MMDet 2.x, but with the train_pseudo_label.py file:
```
python tools/train_pseudo_label.py <CONFIG FILE PATH> [OPTIONS...]

```

## Testing
Testing is carried out in two steps. First, test on the normal dataset to get the optimal score threshold (bbox confidence that maximizes the F1 score) and
the anomaly threshold, which enforces that 95% of the normal data is classified as normal:
```
python tools/test_ood_det.py <CONFIG FILE PATH> <CHECKPOINT FILE PATH> --mode id [OPTIONS...]
```
Then, test the OoD data with the resulting anomaly threshold. The optimal score threshold is only used for visualization. The same config file
can be used but the config parameters pointing to the normal test dataset should be modified via the `---cfg-optioons` option:

```
python tools/test_ood_det.py <CONFIG FILE PATH> <CHECKPOINT FILE PATH> --mode ood --cfg-options "data.test.ann_file='<PATH>'" "data.test.img_prefix='<PREFIX>'" --anomaly-threshold 0.XXX [OPTIONS...]
```

Some [example scripts](scripts/) are available. The script will show the COCO metrics of the detected anomalies.

## Datasets

Three datasets have been tested: DB6, VOC(ID)/COCO(OoD) and the LTD Imaging dataset. These datasets should be under the ``data`` directory.

#### DB6
The db6 contains six classes: firearm, firearm part, knife, camera, ceramic knife and laptop. For this study, the 
firearm and firearm parts classes are considered anomalies while the other four classes are normal. The following 
annotation files are provided:

|          Filename           | Dataset | Partition |                       Comments                        |                      Location                       |
|:---------------------------:|:-------:|:---------:|:-----------------------------------------------------:|:---------------------------------------------------:|
| train_db6_no_firearms.json  |   db6   |   train   | Train partition without images that contain firearms  | [link](annotations/db6/train_db6_no_firearms.json)  |
|  test_db6_no_firearms.json  |   db6   |   test    |  Test partition without images that contain firearms  |  [link](annotations/db6/test_db6_no_firearms.json)  |
| test_db6_only_firearms.json |   db6   |   test    | Test partition with images that only contain firearms | [link](annotations/db6/test_db6_only_firearms.json) |

Data Structure:
```
data
├── db6
│   ├── annotations
│   │   ├── test_db6_no_firearms.json
│   │   ├── test_db6_only_firearms.json
│   │   ├── train_db6_no_firearms.json
│   ├── images
│   │   ├── ...

```

#### VOC(ID)/COCO(OoD)
We follow the same methodology in VOS for training the detector in the PASCAL VOC dataset and test in a COCO dataset partition
that does not contain any instances of the PASCAL VOC dataset. The train/test annotation files are the same as in VOC, while
the COCO annotation file is [the same as in the VOS repository](https://drive.google.com/file/d/1Wsg9yBcrTt2UlgBcf7lMKCw19fPXpESF/view). We
include a copy also [here](annotations/coco/instances_val2017_ood_rm_overlap.json).

Data Structure:
```
data
├── coco
│   ├── annotations
│   │   ├── instances_val2017_ood_rm_overlap.json
│   ├── val2017
│   │   ├── ...
├── voc2017
│   ├── JPEGImages
│   │   ├── ...
│   ├── val_coco_format.json
│   ├── voc0712_train_all.json

```

#### LTD Imaging
The LTD Imaging thermal dataset contains four classes: human, bicycle, motorcycle and vehicle in several temporal partitions.
In this work we use the Week partition for training and consider the vehicle class as anomaly. For testing, we use the 
Day partition. 

|          Filename           |     Dataset     | Partition |                                     Comments                                      |                      Location                       |
|:---------------------------:|:---------------:|:---------:|:---------------------------------------------------------------------------------:|:---------------------------------------------------:|
| train_week_no_vehicles.json |   ltdimaging    |   Week    |       Week partition without images that contain vehicles, used for trainig       | [link](annotations/ltdimaging/train_week_no_vehicles.json)  |
|  test_day_no_vehicles.json  |   ltdimaging    |    Day    |       Day partition without images that contain vehicles, used for testing        |  [link](annotations/ltdimaging/test_day_no_vehicles.json)  |
| test_day_only_vehicles.json |   ltdimaging    |    Day    |         Day partition with images that contain vehicles, used for testing         | [link](annotations/ltdimaging/test_day_only_vehicles.json) |

Data Structure:
```
data
├── ltdimaging
│   ├── Day
│   │   ├── images
│   │   │   ├── ...
│   ├── Day
│   │   ├── images
│   │   │   ├── ...
│   ├── test_day_no_vehicles.json
│   ├── test_day_only_vehicles.json
│   ├── train_week_no_vehicles.json
```

## Citation
You can cite this work as follows:

```
@inproceedings{isaac-medina2024oln-ssos,
 author = {Isaac-Medina, B.K.S. and Gaus, Y.F.A. and Bhowmik, N. and Breckon, T.P.},
 title = {Towards Open-World Object-based Anomaly Detection via Self-Supervised Outlier Synthesis},
 booktitle = {Proc. European Conference on Computer Vision },
 year = {2024},
 month = {September},
 publisher = {Springer},
 keywords = {},
 note = {to appear},
 category = {anomaly baggage automotive},
}
```