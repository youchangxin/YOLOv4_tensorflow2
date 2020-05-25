# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf


def bbox_iou(boxes1, boxes2):
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = boxes1[..., 2] * boxes1[..., 3]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]

    boxes1 = np.concatenate([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                             boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = np.concatenate([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                             boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    iou = tf.math.divide_no_nan(inter_area, union_area)
    return iou


def bbox_giou(boxes1, boxes2, eps=1e-7):

    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    boxes1 = tf.concat([tf.minimum(boxes1[..., :2], boxes1[..., 2:]),
                        tf.maximum(boxes1[..., :2], boxes1[..., 2:])], axis=-1)
    boxes2 = tf.concat([tf.minimum(boxes2[..., :2], boxes2[..., 2:]),
                        tf.maximum(boxes2[..., :2], boxes2[..., 2:])], axis=-1)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    iou = inter_area / (union_area + eps)

    enclose_left_up = tf.minimum(boxes1[..., :2], boxes2[..., :2])
    enclose_right_down = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])
    enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)
    enclose_area = enclose[..., 0] * enclose[..., 1]
    giou = iou - 1.0 * (enclose_area - union_area) / enclose_area

    return giou


def bbox_diou(boxes1, boxes2, eps=1e-7):
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    inter_left_up    = np.maximum(boxes1[..., :2], boxes2[..., :2])
    inter_right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(inter_right_down - inter_left_up, 0.0)
    inter_area    = inter_section[..., 0] * inter_section[..., 1]
    union_area    = boxes1_area + boxes2_area - inter_area

    iou =  inter_area / union_area

    left_up     = np.minimum(boxes1[..., :2], boxes2[..., :2])
    right_down  = np.maximum(boxes1[..., 2:], boxes2[..., 2:])
    c = np.maximum(right_down - left_up, 0.0)
    c = np.power(c[..., 0], 2) + np.power(c[..., 1], 2)

    boxes1_coor = (boxes1[..., 2:] - boxes1[..., 0:2]) * 0.5 + boxes1[..., 0:2]
    boxes2_coor = (boxes2[..., 2:] - boxes2[..., 0:2]) * 0.5 + boxes2[..., 0:2]

    u = np.power((boxes1_coor[..., 0] - boxes2_coor[..., 0]), 2) + \
        np.power((boxes1_coor[..., 1] - boxes2_coor[..., 1]), 2)
    d = u / c

    diou = iou - d
    return diou


def bbox_ciou(boxes1, boxes2, eps=1e-7):
    boxes1_coor = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                             boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2_coor = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                             boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    boxes1_area = boxes1[..., 2] * boxes1[..., 3]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]

    enclose_left_up    = tf.maximum(boxes1_coor[..., :2], boxes2_coor[..., :2])
    enclose_right_down = tf.minimum(boxes1_coor[..., 2:], boxes2_coor[..., 2:])

    inter_section = tf.maximum(enclose_right_down - enclose_left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    iou = tf.math.divide_no_nan(inter_area, union_area)

    left_up     = tf.minimum(boxes1_coor[..., :2], boxes2_coor[..., :2])
    right_down  = tf.maximum(boxes1_coor[..., 2:], boxes2_coor[..., 2:])
    c = tf.maximum(right_down - left_up, 0.0)
    c = tf.pow(c[..., 0], 2) + tf.pow(c[..., 1], 2)

    u = (boxes1[..., 0] - boxes2[..., 0]) * (boxes1[..., 0] - boxes2[..., 0]) + \
        (boxes1[..., 1] - boxes2[..., 1]) * (boxes1[..., 1] - boxes2[..., 1])
    d = tf.math.divide_no_nan(u, c)

    ar_gt = tf.math.divide_no_nan(boxes2[..., 2] , boxes2[..., 3])
    ar_pred = tf.math.divide_no_nan(boxes1[..., 2], boxes1[..., 3])

    pi = tf.convert_to_tensor(np.pi)
    ar_loss = tf.math.divide_no_nan(4.0, pi * pi ) * tf.pow((tf.atan(ar_gt) - tf.atan(ar_pred)), 2)
    alpha = tf.math.divide_no_nan(ar_loss ,(1 - iou + ar_loss))
    ciou_term = d + alpha * ar_loss
    ciou = iou - ciou_term   #   1
    ciou = tf.clip_by_value(ciou, clip_value_min=-1.0, clip_value_max=0.99)
    return ciou