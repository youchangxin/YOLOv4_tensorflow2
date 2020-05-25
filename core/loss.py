# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

from config import cfg
from utils.preprocess import get_anchors
from utils.iou import bbox_ciou, bbox_iou, bbox_giou


NUM_CLASS = len(cfg.YOLO.CLASSES)
STRIDES = np.array(cfg.YOLO.STRIDES)
ANCHORS = get_anchors(cfg.YOLO.ANCHORS)
IOU_LOSS_THRESH = cfg.YOLO.IOU_LOSS_THRESH
XYSCALE = cfg.YOLO.XYSCALE


def decode(conv_outputs):
    decoded_fm = []
    for i, con in enumerate(conv_outputs):
        conv_shape = tf.shape(con)
        batch_size = conv_shape[0]
        output_size = conv_shape[1]

        conv_output = tf.reshape(con, (batch_size, output_size, output_size, 3, 5 + NUM_CLASS))
        conv_raw_dxdy, conv_raw_dwdh, conv_raw_conf, conv_raw_prob = tf.split(conv_output, (2, 2, 1, NUM_CLASS), axis=-1)

        x = tf.tile(tf.expand_dims(tf.range(output_size, dtype=tf.int32), axis=0), [output_size, 1])
        y = tf.tile(tf.expand_dims(tf.range(output_size, dtype=tf.int32), axis=1), [1, output_size])
        xy_grid = tf.expand_dims(tf.stack([x, y], axis=-1), axis=2)  # [gx, gy, 1, 2]

        xy_grid = tf.tile(tf.expand_dims(xy_grid, axis=0), [batch_size, 1, 1, 3, 1])
        xy_grid = tf.cast(xy_grid, tf.float32)

        pred_xy = ((tf.sigmoid(conv_raw_dxdy) * XYSCALE[i]) - 0.5 * (XYSCALE[i] - 1) + xy_grid) * STRIDES[i]
        pred_wh = tf.exp(conv_raw_dwdh) * ANCHORS[i]
        max_scale = tf.cast(output_size * STRIDES[i], dtype=tf.float32)

        pred_wh = tf.clip_by_value(pred_wh, clip_value_min=0.0, clip_value_max=max_scale)
        pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)
        pred_conf = tf.sigmoid(conv_raw_conf)
        pred_prob = tf.sigmoid(conv_raw_prob)

        decoded_fm.append(tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1))
    return decoded_fm


def yolov4_loss(pred, label, bboxes, i=0, eps=1e-15):
    conv_shape = tf.shape(pred)
    output_size = conv_shape[1]
    input_size = output_size * STRIDES[i]

    pred_xywh  = pred[:, :, :, :, 0:4]
    pred_conf  = pred[:, :, :, :, 4:5]
    pred_prob  = pred[:, :, :, :, 5: ]

    label_xywh   = label[:, :, :, :, 0:4]
    respond_bbox = label[:, :, :, :, 4:5]
    label_prob   = label[:, :, :, :, 5: ]

    ciou = tf.expand_dims(bbox_ciou(pred_xywh, label_xywh), axis=-1)
    input_size = tf.cast(input_size, tf.float32)
    # 边界框的尺寸越小，bbox_loss_scale 的值就越大，可以弱化边界框尺寸对损失值的影响
    bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
    # 两个边界框之间的 GIoU 值越大，giou 的损失值就会越小
    ciou_loss = respond_bbox * bbox_loss_scale * (1 - ciou)

    iou = bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :], bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
    # 找出与真实框 iou 值最大的预测框
    max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)
    # 如果最大的 iou 小于阈值，那么认为该预测框不包含物体,则为背景框
    respond_bgd = (1.0 - respond_bbox) * tf.cast(max_iou < IOU_LOSS_THRESH, tf.float32)

    conf_focal = tf.pow(respond_bbox - pred_conf, 2)
    # Focal Loss, 通过修改标准的交叉熵损失函数，降低对能够很好分类样本的权重
    conf_loss = conf_focal * (
            respond_bbox * -(respond_bbox * tf.math.log(tf.clip_by_value(pred_conf, eps, 1.0)))
            +
            respond_bgd * -(respond_bgd * tf.math.log(tf.clip_by_value((1- pred_conf), eps, 1.0)))
            )
    prob_loss = respond_bbox * -(label_prob * tf.math.log(tf.clip_by_value(pred_prob, eps, 1.0))
                                 +
                                 (1 - label_prob) * tf.math.log(tf.clip_by_value((1 - pred_prob), eps, 1.0)))

    # 将各部分损失值的和，除以均值，累加，作为最终的图片损失值
    ciou_loss = tf.reduce_mean(tf.reduce_sum(ciou_loss, axis=[1, 2, 3, 4]))
    conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1, 2, 3, 4]))
    prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1, 2, 3, 4]))

    return ciou_loss, conf_loss, prob_loss
