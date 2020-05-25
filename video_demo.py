# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
import tensorflow as tf
import time

from config import cfg
from utils.preprocess import image_preporcess
from utils.postprocess import postprocess_boxes, draw_bbox, nms
from core.loss import decode
from core.model.yolov4 import YOLOV4


video_path      =  './data/road.mp4' #"E:\Movies\蒙太奇.mkv"#
num_classes     = 20
input_size      = 416

model = YOLOV4()
model.build(input_shape=(None, 416, 416, 3))
model.load_weights(filepath='./yolov4')


vid = cv2.VideoCapture(video_path)
while True:
    return_value, frame = vid.read()
    if return_value:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError("No image!")
    frame_size = frame.shape[:2]
    image_data = image_preporcess(np.copy(frame), [input_size, input_size])
    image_data = image_data[np.newaxis, ...].astype(np.float32)

    prev_time = time.time()

    feature_maps = model.predict(image_data)
    decoded_tensor = decode(feature_maps)

    curr_time = time.time()
    exec_time = curr_time - prev_time

    pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in decoded_tensor]
    pred_bbox = tf.concat(pred_bbox, axis=0)
    bboxes = postprocess_boxes(pred_bbox, frame_size, input_size, 0.7)
    bboxes = nms(bboxes, 0.213, method='nms')
    image = draw_bbox(frame, bboxes, show_label=True)
    result = np.asarray(image)
    info = "time: %.2f ms" % (1000*exec_time)
    cv2.putText(result, text=info, org=(50, 70), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1, color=(255, 0, 0), thickness=2)
    cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
    result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow("result", result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
