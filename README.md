# YOLOv4_tensorflow2.x
[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)

A minimal TensorFlow2.x implementation of YOLOv4.
- Paper Yolo v4: https://arxiv.org/abs/2004.10934
- Source code:https://github.com/AlexeyAB/darknet
- More details: http://pjreddie.com/darknet/yolo/
## YOLOv4 Overview
+ Backbone：CSPDarkNet53
+ Neck：SPP，PAN
+ Head：YOLOv3
+ Tricks (Backbone): CutMix、Mosaic、DropBlock、Label Smoothing
+ Modified(Backbone) : Mish、CSP、MiWRC
+ Tricks (Detection) : CIoU、CmBN、SAT、Eliminate grid sensitivity
+ Modified(Detection): Mish、SPP、SAM、PAN、DIoU-NMS

## Requirements
- python == 3.6
- tensorflow == 2.1.1
- tensoflow-addons == 0.9.1
- opencv-python == 4.2.0
- easydict == 1.9

## Usage
### Train on PASCAL VOC 2012
```
|——data
    |——dataset 
        |——VOCdevkit
            |——VOC2012
                |——Annotations
                |——ImageSets
                |——JPEGImages
                |——SegmentationClass
                |——SegmentationObject
```
1. Download the [PASCAL VOC 2012 dataset](http://host.robots.ox.ac.uk/pascal/VOC/).
2. Unzip the file and place it in the 'dataset' folder, make sure the directory is like this : 
3. Run ./data/write_voc_to_txt.py to generate voc2012.txt, which operation is essential. 
4. Run train.py

## Project Schedule
### Data augmentation
- [ ] Mosaic
- [ ] Cutmix
- [ ] Self-adversarial-training (SAT)
### Model
- [x] Cross-stage partial Net (CSP-DarkNet53)
- [x] Mish-activation
- [x] DropBlock regularization
- [x] SPP-block
- [ ] SAM-block
- [x] PAN block
- [ ] Cross mini-Batch Normalization (CmBN)
### Otimization
- [ ] Multi-input weighted residual connections (MiWRC)
- [ ] Eliminate grid sensitivity
- [x] Cosine annealing scheduler
- [ ] kmeans
- [x] DIoU-NMS
### Loss
- [x] Class label smoothing
- [x] CIoU loss
- [x] Focal loss
## Reference
+ tensorflow-yolov3 https://github.com/YunYang1994/tensorflow-yolov3
