# -*- coding: utf-8 -*-

from easydict import EasyDict as edict


__C                           = edict()
# Consumers can get config by: from config import cfg

cfg                           = __C

# YOLO options
__C.YOLO                      = edict()

__C.YOLO.STRIDES              = [8, 16, 32]
__C.YOLO.XYSCALE              = [1.2, 1.1, 1.05]
__C.YOLO.ANCHOR_PER_SCALE     = 3
__C.YOLO.IOU_LOSS_THRESH      = 0.5
__C.YOLO.SAVE_MODEL_DIR       = '/saved_weights/'
__C.YOLO.ANCHORS              = "./data/anchors/basline_anchors.txt"
__C.YOLO.CLASSES              = {"person": 1, "bird": 2, "cat": 3, "cow": 4, "dog": 5,
                                 "horse": 6, "sheep": 7, "aeroplane": 8, "bicycle": 9,
                                 "boat": 10, "bus": 11, "car": 12, "motorbike": 13,
                                 "train": 14, "bottle": 15, "chair": 16, "diningtable": 17,
                                 "pottedplant": 18, "sofa": 19, "tvmonitor": 20}





# Train options
__C.TRAIN                     = edict()

__C.TRAIN.ANNOT_PATH          = "./data/dataset/voc2012.txt"
__C.TRAIN.BATCH_SIZE          = 4
__C.TRAIN.INPUT_SIZE          = 416
__C.TRAIN.DATA_AUG            = True
__C.TRAIN.LR_INIT             = 1e-3
__C.TRAIN.LR_END              = 1e-6
__C.TRAIN.WARMUP_EPOCHS       = 2
__C.TRAIN.EPOCHS              = 30
__C.TRAIN.SAVE_FREQ           = 2



# TEST options
__C.TEST                      = edict()

__C.TEST.ANNOT_PATH           = "./data/dataset/val2017.txt"
__C.TEST.BATCH_SIZE           = 2
__C.TEST.INPUT_SIZE           = 416
__C.TEST.DATA_AUG             = False
__C.TEST.DECTECTED_IMAGE_PATH = "./data/detection/"
__C.TEST.SCORE_THRESHOLD      = 0.25
__C.TEST.IOU_THRESHOLD        = 0.5


'''
COCO_CLASSES
               {"person": 1, "bicycle": 2, "car": 3, "motorcycle": 4, "airplane": 5,
                "bus": 6, "train": 7, "truck": 8, "boat": 9, "traffic light": 10,
                "fire hydrant": 11, "stop sign": 12, "parking meter": 13, "bench": 14,
                "bird": 15, "cat": 16, "dog": 17, "horse": 18, "sheep": 19, "cow": 20,
                "elephant": 21, "bear": 22, "zebra": 23, "giraffe": 24, "backpack": 25,
                "umbrella": 26, "handbag": 27, "tie": 28, "suitcase": 29, "frisbee": 30,
                "skis": 31, "snowboard": 32, "sports ball": 33, "kite": 34, "baseball bat": 35,
                "baseball glove": 36, "skateboard": 37, "surfboard": 38, "tennis racket": 39,
                "bottle": 40, "wine glass": 41, "cup": 42, "fork": 43, "knife": 44, "spoon": 45,
                "bowl": 46, "banana": 47, "apple": 48, "sandwich": 49, "orange": 50, "broccoli": 51,
                "carrot": 52, "hot dog": 53, "pizza": 54, "donut": 55, "cake": 56, "chair": 57,
                "couch": 58, "potted plant": 59, "bed": 60, "dining table": 61, "toilet": 62,
                "tv": 63, "laptop": 64, "mouse": 65, "remote": 66, "keyboard": 67, "cell phone": 68,
                "microwave": 69, "oven": 70, "toaster": 71, "sink": 72, "refrigerator": 73,
                "book": 74, "clock": 75, "vase": 76, "scissors": 77, "teddy bear": 78,
                "hair drier": 79, "toothbrush": 80}'''