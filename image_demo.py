# -*- coding: utf-8 -*-
import cv2
import numpy as np
import core.loss as loss
import tensorflow as tf

from config import cfg
from utils.preprocess import image_preporcess
from utils.postprocess import postprocess_boxes, draw_bbox, nms
from core.loss import decode
from core.model.yolov4 import YOLOV4
from PIL import Image


input_size   = 416
image_path   = "./data/kite.jpg"

original_image      = cv2.imread(image_path)
original_image      = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
original_image_size = original_image.shape[:2]

image_data = image_preporcess(np.copy(original_image), [input_size, input_size])
image_data = image_data[np.newaxis, ...].astype(np.float32)

model = YOLOV4()
model.build(input_shape=(None, 416, 416, 3))
model.load_weights(filepath='./yolov4')

feature_maps = model.predict(image_data)
decoded_tensor = decode(feature_maps)

pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in decoded_tensor]
pred_bbox = tf.concat(pred_bbox, axis=0)

bboxes = postprocess_boxes(pred_bbox, original_image_size, input_size, 0.4)
bboxes = nms(bboxes, 0.45, method='nms')

image = draw_bbox(original_image, bboxes, show_label=True)
image = Image.fromarray(image)
image.show()


