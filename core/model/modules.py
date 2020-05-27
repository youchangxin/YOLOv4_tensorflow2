# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorflow_addons as tfa
from core.model.dropblock import DropBlock2D


class BatchNormalization(tf.keras.layers.BatchNormalization):
    """
    "Frozen state" and "inference mode" are two separate concepts.
    `layer.trainable = False` is to freeze the layer, so the layer will use
    stored moving `var` and `mean` in the "inference mode", and both `gama`
    and `beta` will not be updated !
    """
    def call(self, x, training=False):
        if not training:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)


def Mish(x):
    return tfa.activations.mish(x)


class Conv2d(tf.keras.layers.Layer):
    def __init__(self, filters, kernel, stride, padding='same', activation='Mish', use_bias=True):
        super(Conv2d, self).__init__()
        self.activation = activation
        self.conv = tf.keras.layers.Conv2D(filters=filters,
                                           kernel_size=kernel,
                                           strides=stride,
                                           kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                                           kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                           padding=padding,
                                           use_bias=use_bias
                                           )
        self.bn = BatchNormalization()

    def call(self, inputs, training=False, **kwargs):
        x = self.conv(inputs)
        x = self.bn(x)
        if self.activation == 'Mish':
            x = Mish(x)
        elif self.activation == 'LeakRelu':
            x = tf.nn.leaky_relu(x)
        elif self.activation == 'Linear':
            x = tf.keras.activations.linear(x)
        return x


def build_Res_block(filters, repeat_num):
    block = tf.keras.Sequential()
    for _ in range(repeat_num):
        block.add(CSP_Res(filters=filters))
    return block


class CSP_Res(tf.keras.layers.Layer):
    def __init__(self, filters):
        super(CSP_Res, self).__init__()
        self.conv1 = Conv2d(filters=filters, kernel=(1, 1), stride=1)
        self.conv2 = Conv2d(filters=filters, kernel=(3, 3), stride=1)
        self.dropblock = DropBlock2D(keep_prob=0.9, block_size=3)

    def call(self, inputs, training=None, **kwargs):
        x = self.conv1(inputs, training=training)
        x = self.conv2(x, training=training)
        x = self.dropblock(x, training=training)
        output = tf.keras.layers.Add()([inputs, x])

        return output


class CSP_Block(tf.keras.layers.Layer):
    def __init__(self, num_filters, num_iter, allow_narrow=True):
        '''
        :param num_filters: number of filter
        :param num_iter: the times that iter CSP_Block
        :param allow_narrow: Boolean value; True: halves num_filter  False: spilt_filter=num_filter
        '''
        super(CSP_Block, self).__init__()
        split_filters = num_filters // 2 if allow_narrow else num_filters
        self.zeropadding = tf.keras.layers.ZeroPadding2D(padding=((1, 0), (1, 0)))
        self.preconv = Conv2d(filters=num_filters, kernel=(3, 3), stride=2)
        self.res_block = build_Res_block(filters=split_filters, repeat_num=num_iter)
        self.shortconv = Conv2d(filters=split_filters, kernel=(1, 1), stride=1)
        self.mainconv = Conv2d(filters=split_filters, kernel=(1, 1), stride=1)

        self.postconv = Conv2d(filters=split_filters, kernel=(3, 3), stride=1)

        self.transition = Conv2d(filters=num_filters, kernel=(1, 1), stride=1)

    def call(self, inputs, training=False, **kwargs):
        # x = self.zeropadding(inputs)
        x = self.preconv(inputs, training=training)
        shortcut = self.shortconv(x, training=training)
        mainsteam = self.mainconv(x, training=training)
        res = self.res_block(mainsteam)
        mainsteam = self.postconv(res, training=training)
        outputs = tf.keras.layers.Concatenate()([mainsteam, shortcut])
        outputs = self.transition(outputs, training=training)

        return outputs


class SPP(tf.keras.layers.Layer):
    def __init__(self):
        super(SPP, self).__init__()
        self.conv1 = Conv2d(filters=512, kernel=(1, 1), stride=1, activation='LeakRelu')
        self.conv2 = Conv2d(filters=1024, kernel=(3, 3), stride=1, activation='LeakRelu')
        # MaxPool section
        self.maxpool_5 = tf.keras.layers.MaxPool2D(pool_size=(5, 5), strides=1, padding='same')
        self.maxpool_9 = tf.keras.layers.MaxPool2D(pool_size=(9, 9), strides=1, padding='same')
        self.maxpool_13 = tf.keras.layers.MaxPool2D(pool_size=(13, 13), strides=1, padding='same')

        self.conv3 = Conv2d(filters=512, kernel=(1, 1), stride=1, activation='LeakRelu')
        self.conv4 = Conv2d(filters=512, kernel=(1, 1), stride=1, activation='LeakRelu')
        self.conv5 = Conv2d(filters=1024, kernel=(3, 3), stride=1, activation='LeakRelu')
        self.conv6 = Conv2d(filters=512, kernel=(1, 1), stride=1, activation='LeakRelu')

    def call(self, inputs, training=False, **kwargs):
        x = self.conv1(inputs, training=training)
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)

        x1 = self.maxpool_5(x)
        x2 = self.maxpool_9(x)
        x3 = self.maxpool_13(x)
        x = tf.keras.layers.concatenate([x1, x2, x3, x], axis=-1)

        x = self.conv4(x, training=training)
        x = self.conv5(x, training=training)
        x = self.conv6(x, training=training)

        return x


class Upper_Concate(tf.keras.layers.Layer):
    def __init__(self, num_filter1, num_filter2):
        super(Upper_Concate, self).__init__()
        self.conv1 = Conv2d(filters=num_filter1, kernel=(1, 1), stride=1, activation='LeakRelu')
        self.conv2 = Conv2d(filters=num_filter1, kernel=(1, 1), stride=1, activation='LeakRelu')
        self.upsamp = tf.keras.layers.UpSampling2D()
        self.conv3 = Conv2d(filters=num_filter1, kernel=(1, 1), stride=1, activation='LeakRelu')
        self.conv4 = Conv2d(filters=num_filter2, kernel=(3, 3), stride=1, activation='LeakRelu')
        self.conv5 = Conv2d(filters=num_filter1, kernel=(1, 1), stride=1, activation='LeakRelu')
        self.conv6 = Conv2d(filters=num_filter2, kernel=(3, 3), stride=1, activation='LeakRelu')
        self.conv7 = Conv2d(filters=num_filter1, kernel=(1, 1), stride=1, activation='LeakRelu')

    def call(self, input1, input2, training=False, **kwargs):
        x1 = self.conv1(input1, training=training)
        x2 = self.conv2(input2, training=training)
        x2 = self.upsamp(x2)
        x = tf.keras.layers.Concatenate()([x1, x2])
        x = self.conv3(x, training=training)
        x = self.conv4(x, training=training)
        x = self.conv5(x, training=training)
        x = self.conv6(x, training=training)
        x = self.conv7(x, training=training)

        return x


class Merge(tf.keras.layers.Layer):
    def __init__(self, num_filter1, num_filter2):
        super(Merge, self).__init__()
        self.conv = Conv2d(filters=num_filter1, kernel=(3, 3), stride=2, activation='LeakRelu')
        self.conv1 = Conv2d(filters=num_filter1, kernel=(1, 1), stride=1, activation='LeakRelu')
        self.conv2 = Conv2d(filters=num_filter2, kernel=(3, 3), stride=1, activation='LeakRelu')
        self.conv3 = Conv2d(filters=num_filter1, kernel=(1, 1), stride=1, activation='LeakRelu')
        self.conv4 = Conv2d(filters=num_filter2, kernel=(3, 3), stride=1, activation='LeakRelu')
        self.conv5 = Conv2d(filters=num_filter1, kernel=(1, 1), stride=1, activation='LeakRelu')

    def call(self, input1, input2, training=False, **kwargs):
        x1 = self.conv(input1, training=training)
        x = tf.keras.layers.Concatenate()([x1, input2])
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)
        x = self.conv4(x, training=training)
        x = self.conv5(x, training=training)

        return x


class Linear(tf.keras.layers.Layer):
    def __init__(self, num_filter1, num_filter2):
        super(Linear, self).__init__()
        self.conv1 = Conv2d(filters=num_filter1, kernel=(3, 3), stride=1, activation='LeakRelu')
        self.conv2 = Conv2d(filters=num_filter2, kernel=(1, 1), stride=1, activation='Linear')

    def call(self, inputs, training=False, **kwargs):
        x = self.conv1(inputs, training=training)
        x = self.conv2(x, training=training)

        return x

