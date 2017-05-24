#! -*- coding:utf-8 -*-

import os
import sys
import numpy as np
import tensorflow as tf

from tf_util import conv, linear, Layers, batch_norm, flatten, lrelu

class Encoder(Layers):
    def __init__(self, name_scopes, out_dim, layer_channels):
        assert(len(name_scopes) == 2)
        super().__init__(name_scopes)
        self.layer_channels = layer_channels
        self.out_dim = out_dim
        
    def set_model(self, inputs, batch_size, is_training = True, reuse = False):
        assert(self.layer_channels[0] == inputs.get_shape().as_list()[-1])
        
        h = inputs

        # convolution
        with tf.variable_scope(self.name_scopes[0], reuse = reuse):
            for i, out_chan in enumerate(self.layer_channels[1:]):
                # convolution
                conved = conv(i, h, out_chan, 5, 5, 2)

                # batch normalization
                bn_conved = batch_norm(i, conved, 0.99, is_training)

                # activation
                h = lrelu(bn_conved)
                
        # fully connect
        with tf.variable_scope(self.name_scopes[1], reuse = reuse):
            encoded = linear('fc', flatten(h), self.out_dim)
            
        return encoded
    
if __name__ == u'__main__':
    e = Encoder([u'convolution', u'fc'], 100, [3, 64, 256, 512])
    z = tf.placeholder(tf.float32, [None, 256, 256, 3])
    h = e.set_model(z, 10)
    h = e.set_model(z, 10, True, True)    
    print(h)
