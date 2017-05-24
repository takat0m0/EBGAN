#! -*- coding:utf-8 -*-

import os
import sys
import numpy as np
import tensorflow as tf

from tf_util import deconv, linear, Layers, batch_norm
from encoder import Encoder
from decoder import Decoder

class Discriminator(object):
    def __init__(self, dis_hidden_dim, enc_layers, dec_in_dim, dec_layers):
        self.enc = Encoder([u'enc_conv', u'enc_fc'], dis_hidden_dim,
                           enc_layers)
        self.dec = Decoder([u'dec_reshape', u'dec_deconv'], dec_in_dim, dec_layers)
        
    def get_variables(self):
        ret = self.enc.get_variables()
        ret.extend(self.dec.get_variables())
        return ret
    
    def set_model(self, input_img, batch_size, is_training = True, reuse = False):
        encoded = self.enc.set_model(input_img, batch_size, is_training, reuse)
        auto_encoded_img = self.dec.set_model(encoded, batch_size, is_training, reuse)
        disc_loss = tf.reduce_mean(
            tf.reduce_sum(tf.abs(input_img - auto_encoded_img), (1, 2, 3))
        )
        return encoded, disc_loss
        
    
if __name__ == u'__main__':
    dis = Discriminator(100, [3, 64, 256, 512],
                        4, [1024, 512, 512, 256, 256, 128, 3])
    imgs = tf.placeholder(tf.float32, [None, 256, 256, 3])
    h = dis.set_model(imgs, 10)
    h = dis.set_model(imgs, 10, True, True)    
    print(h)
