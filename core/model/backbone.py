# -*- coding: utf-8 -*-
import tensorflow as tf
from core.model.modules import Conv2d, CSP_Block


class CSPDarkNet53(tf.keras.layers.Layer):
    def __init__(self):
        super(CSPDarkNet53, self).__init__()
        self.conv = Conv2d(filters=32, kernel=(3, 3), stride=1)
        self.csp_block1 = CSP_Block(64, 1, allow_narrow=False)
        self.csp_block2 = CSP_Block(128, 2)
        self.csp_block3 = CSP_Block(256, 8)
        self.csp_block4 = CSP_Block(512, 8)
        self.csp_block5 = CSP_Block(1024, 4)

    def call(self, inputs, training=False, **kwargs):
        x = self.conv(inputs, training=training)
        x = self.csp_block1(x, training=training)
        x = self.csp_block2(x, training=training)
        route_small = self.csp_block3(x, training=training)
        route_medial = self.csp_block4(route_small, training=training)
        route_large = self.csp_block5(route_medial, training=training)

        return route_small, route_medial, route_large
