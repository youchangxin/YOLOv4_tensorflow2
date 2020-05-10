# YOLOv4_tensorflow2

Details can see [paper](https://arxiv.org/pdf/2004.10934.pdf)
## YOLOv4 Overview

+ Backbone：CSPDarkNet53
+ Neck：SPP，PAN
+ Head：YOLOv3
+ Tricks (Backbone): CutMix、Mosaic、DropBlock、Label Smoothing
+ Modified(Backbone) : Mish、CSP、MiWRC
+ Tricks (Detection) : CIoU、CmBN、SAT、Eliminate grid sensitivity
+ Modified(Detection): Mish、SPP、SAM、PAN、DIoU-NMS

## Project Schedule
### Data augmentation
- [ ] Mosaic
- [ ] Cutmix
- [ ] Self-adversarial-training (SAT)
### Model
- [x] Cross-stage partial Net (CSP-DarkNet53)
- [x] Mish-activation
- [ ] DropBlock regularization
- [x] SPP-block
- [ ] SAM-block
- [ ] PAN block
- [ ] Cross mini-Batch Normalization (CmBN)
### Otimization
- [ ] Multi-input weighted residual connections (MiWRC)
- [ ] Eliminate grid sensitivity
- [ ] Cosine annealing scheduler
- [ ] kmeans
- [ ] DIoU-NMS
### Loss
- [ ] Class label smoothing
- [ ] CIoU loss

## Reference
+ 