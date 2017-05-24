#! -*- coding:utf-8 -*-

import os
import sys

import numpy as np
import tensorflow as tf

from generator import Generator
from discriminator import Discriminator

def get_pt(embeddings):

    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    similarity = tf.matmul(
        normalized_embeddings, normalized_embeddings, transpose_b=True)
    batch_size = tf.cast(tf.shape(embeddings)[0], tf.float32)
    pt_loss = (tf.reduce_sum(similarity) - batch_size) / (batch_size * (batch_size - 1))
    return pt_loss
    
class Model(object):
    def __init__(self, m, pt_coeff, z_dim, ae_hidden_dim, batch_size):

        self.input_size = 256
        self.z_dim = z_dim
        self.batch_size = batch_size
        self.m = m
        self.pt_coeff = pt_coeff
        
        # generator config
        gen_layer = [1024, 512, 512, 256, 256, 128, 3]
        gen_in_dim = int(self.input_size/2**(len(gen_layer) - 1))

        #discriminato config
        disc_enc_layer = [3, 64, 256, 512]
        disc_dec_layer = [1024, 512, 512, 256, 256, 128, 3]
        disc_dec_in_dim = int(self.input_size/2**(len(disc_dec_layer) - 1))

        # -- generator -----
        self.gen = Generator([u'gen_reshape', u'gen_deconv'],
                             gen_in_dim, gen_layer)

        # -- discriminator --
        self.disc = Discriminator(ae_hidden_dim, disc_enc_layer,
                                  disc_dec_in_dim, disc_dec_layer)
        self.lr = 0.0002

        
    def set_model(self):
        # -- z -> gen_fig -> disc ---

        self.z = tf.placeholder(tf.float32, [self.batch_size, self.z_dim])

        gen_figs = self.gen.set_model(self.z, self.batch_size, True, False)
        g_encoded, g_loss = self.disc.set_model(gen_figs, self.batch_size, True, False)
        self.g_obj = g_loss + self.pt_coeff * get_pt(g_encoded)
        
        self.train_gen  = tf.train.AdamOptimizer(self.lr, beta1 = 0.5).minimize(self.g_obj, var_list = self.gen.get_variables())
        
        # -- true_fig -> disc --------
        self.figs= tf.placeholder(tf.float32, [self.batch_size, self.input_size, self.input_size, 3])        

        _, d_loss = self.disc.set_model(self.figs, self.batch_size, True, True)

        d_obj_true = d_loss
        d_obj_fake = tf.maximum(0.0, self.m - g_loss)
    
        self.d_obj = d_obj_true + d_obj_fake

        self.train_disc = tf.train.AdamOptimizer(self.lr, beta1 = 0.5).minimize(self.d_obj, var_list = self.disc.get_variables())

        # -- for figure generation -------
        self.gen_figs = self.gen.set_model(self.z, self.batch_size, False, True)
        
    def training_gen(self, sess, z_list):
        _, g_obj = sess.run([self.train_gen, self.g_obj],
                            feed_dict = {self.z: z_list})
        return g_obj
        
    def training_disc(self, sess, z_list, figs):
        _, d_obj = sess.run([self.train_disc, self.d_obj],
                            feed_dict = {self.z: z_list,
                              self.figs:figs})
        return d_obj
    
    def gen_fig(self, sess, z):
        ret = sess.run(self.gen_figs,
                       feed_dict = {self.z: z})
        return ret

if __name__ == u'__main__':
    model = Model(m = 30, pt_coeff = 50, z_dim = 30, ae_hidden_dim = 100, batch_size = 10)
    model.set_model()
    
