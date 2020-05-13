# -*- coding: utf-8 -*-
import os
import shutil
import numpy as np
import tensorflow as tf
import utils.preprocess as preprocess

from core.model.yolov4 import YOLOV4
from core.loss import decode, yolov4_loss
from dataset import Dataset
from config import cfg


os.environ['CUDA_VISIBLE_DEVICES'] = '0'


logdir = "./log"
save_frequency = cfg.TRAIN.SAVE_FREQ

# GPU Setting
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

trainset = Dataset('train')
steps_per_epoch = len(trainset)
EPOCHS = cfg.TRAIN.EPOCHS
warmup_steps = cfg.TRAIN.WARMUP_EPOCHS * steps_per_epoch
total_steps = EPOCHS * steps_per_epoch
NUM_CLASS = len(cfg.YOLO.CLASSES)
STRIDES = np.array(cfg.YOLO.STRIDES)
IOU_LOSS_THRESH = cfg.YOLO.IOU_LOSS_THRESH
ANCHORS = preprocess.get_anchors(cfg.YOLO.ANCHORS)
global_steps = tf.Variable(initial_value=1, trainable=False, dtype=tf.int64)


# TensorBoard
if os.path.exists(logdir): shutil.rmtree(logdir)
writer = tf.summary.create_file_writer(logdir)

# model
model = YOLOV4()
model.build(input_shape=(None, cfg.TRAIN.INPUT_SIZE, cfg.TRAIN.INPUT_SIZE, 3))
model.summary()

if os.listdir('./saved_weights'):
    latest_weight = tf.train.latest_checkpoint('./saved_weights')
    model.load_weights(latest_weight)

optimizer = tf.keras.optimizers.Adam()


def train_step(img, target, epoch):
    with tf.GradientTape() as tape:
        pred_result = model(img, training=True)
        decoded_result = decode(pred_result)
        ciou_loss = conf_loss = prob_loss = 0

        # optimizing process
        for i in range(3):
            loss_items = yolov4_loss(decoded_result[i], target[i][0], target[i][1], i=i)
            ciou_loss += loss_items[0]
            conf_loss += loss_items[1]
            prob_loss += loss_items[2]

        total_loss = ciou_loss + conf_loss + prob_loss

        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        tf.print("=>EPOCH %3d  STEP %4d   lr: %.6f   ciou_loss: %4.2f   conf_loss: %4.2f   "
                 "prob_loss: %4.2f   total_loss: %4.2f" % (epoch, global_steps, optimizer.lr.numpy(),
                                                           ciou_loss, conf_loss,
                                                           prob_loss, total_loss))
        # update learning rate
        global_steps.assign_add(1)
        if global_steps < warmup_steps:
            lr = global_steps / warmup_steps * cfg.TRAIN.LR_INIT
        else:
            lr = cfg.TRAIN.LR_END + 0.5 * (cfg.TRAIN.LR_INIT - cfg.TRAIN.LR_END) * (
                (1 + tf.cos((global_steps - warmup_steps) / (total_steps - warmup_steps) * np.pi))
            )
        optimizer.lr.assign(lr.numpy())

        # writing summary data
        with writer.as_default():
            tf.summary.scalar("lr", optimizer.lr, step=global_steps)
            tf.summary.scalar("loss/total_loss", total_loss, step=global_steps)
            tf.summary.scalar("loss/ciou_loss", ciou_loss, step=global_steps)
            tf.summary.scalar("loss/conf_loss", conf_loss, step=global_steps)
            tf.summary.scalar("loss/prob_loss", prob_loss, step=global_steps)
        writer.flush()


if __name__ == '__main__':
    for epoch in range(EPOCHS):
        for image_data, target in trainset:
            train_step(image_data, target, epoch)
            # save model weights
        if (epoch+1) % save_frequency == 0:
            model.save_weights(filepath=cfg.YOLO.SAVE_MODEL_DIR + "YOLOv4_epoch-{}".format(epoch), save_format="tf")

        model.save_weights(filepath="/saved_model", save_format="h5")
