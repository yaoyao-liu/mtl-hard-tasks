##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Yaoyao Liu
## NUS School of Computing
## Email: yaoyao.liu@mail.m2i.ac.cn
## Copyright (c) 2019
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import numpy as np
import sys
import tensorflow as tf
from tensorflow.python.platform import flags
from utils.misc import mse, softmaxloss, xent, resnet_conv_block, resnet_nob_conv_block, normalize

FLAGS = flags.FLAGS

class Models:
    def __init__(self):
        self.dim_input = FLAGS.img_size * FLAGS.img_size * 3
        self.dim_output = FLAGS.way_num
        self.update_lr = FLAGS.base_lr
        self.pretrain_class_num = FLAGS.pretrain_class_num
        self.meta_lr = tf.placeholder_with_default(FLAGS.meta_lr, ())
        self.pretrain_lr = tf.placeholder_with_default(FLAGS.pre_lr, ())

        self.loss_func = xent
        self.pretrain_loss_func = softmaxloss

        self.channels = 3
        self.img_size = FLAGS.img_size

    def process_ss_weights(self, weights, ss_weights, label):     
        [dim0, dim1] = weights[label].get_shape().as_list()[0:2]
        this_ss_weights = tf.tile(ss_weights[label], multiples=[dim0, dim1, 1, 1])
        return tf.multiply(weights[label], this_ss_weights)

    def forward_pretrain_resnet(self, inp, weights, reuse=False, scope=''):
        inp = tf.reshape(inp, [-1, self.img_size, self.img_size, self.channels])
        inp = tf.image.resize_images(inp, size=[224,224], method=tf.image.ResizeMethod.BILINEAR)
        net = self.pretrain_first_block_forward(inp, weights, 'block0_1', reuse, scope)

        net = self.pretrain_block_forward(net, weights, 'block1_1', reuse, scope)
        net = self.pretrain_block_forward(net, weights, 'block1_2', reuse, scope, block_last_layer=True)

        net = self.pretrain_block_forward(net, weights, 'block2_1', reuse, scope)
        net = self.pretrain_block_forward(net, weights, 'block2_2', reuse, scope, block_last_layer=True)

        net = self.pretrain_block_forward(net, weights, 'block3_1', reuse, scope)
        net = self.pretrain_block_forward(net, weights, 'block3_2', reuse, scope, block_last_layer=True)

        net = self.pretrain_block_forward(net, weights, 'block4_1', reuse, scope)
        net = self.pretrain_block_forward(net, weights, 'block4_2', reuse, scope, block_last_layer=True)

        net = tf.nn.avg_pool(net, [1,7,7,1], [1,7,7,1], 'SAME')
        net = tf.reshape(net, [-1, np.prod([int(dim) for dim in net.get_shape()[1:]])])
        return net

    def forward_resnet(self, inp, weights, ss_weights, reuse=False, scope=''):
        inp = tf.reshape(inp, [-1, self.img_size, self.img_size, self.channels])
        inp = tf.image.resize_images(inp, size=[224,224], method=tf.image.ResizeMethod.BILINEAR)
        net = self.first_block_forward(inp, weights, ss_weights, 'block0_1', reuse, scope)

        net = self.block_forward(net, weights, ss_weights, 'block1_1', reuse, scope)
        net = self.block_forward(net, weights, ss_weights, 'block1_2', reuse, scope, block_last_layer=True)

        net = self.block_forward(net, weights, ss_weights, 'block2_1', reuse, scope)
        net = self.block_forward(net, weights, ss_weights, 'block2_2', reuse, scope, block_last_layer=True)

        net = self.block_forward(net, weights, ss_weights, 'block3_1', reuse, scope)
        net = self.block_forward(net, weights, ss_weights, 'block3_2', reuse, scope, block_last_layer=True)

        net = self.block_forward(net, weights, ss_weights, 'block4_1', reuse, scope)
        net = self.block_forward(net, weights, ss_weights, 'block4_2', reuse, scope, block_last_layer=True)

        net = tf.nn.avg_pool(net, [1,7,7,1], [1,7,7,1], 'SAME')
        net = tf.reshape(net, [-1, np.prod([int(dim) for dim in net.get_shape()[1:]])])
        return net

    def forward_fc(self, inp, fc_weights):
        net = tf.matmul(inp, fc_weights['w5']) + fc_weights['b5']
        return net

    def pretrain_block_forward(self, inp, weights, block, reuse, scope, block_last_layer=False):
        net = resnet_conv_block(inp, weights[block + '_conv1'], weights[block + '_bias1'], reuse, scope+block+'0')
        net = resnet_conv_block(net, weights[block + '_conv2'], weights[block + '_bias2'], reuse, scope+block+'1')
        res = resnet_nob_conv_block(inp, weights[block + '_conv_res'], reuse, scope+block+'res')
        net = net + res
        if block_last_layer:
            net = tf.nn.max_pool(net, [1,2,2,1], [1,2,2,1], 'SAME')
        net = tf.nn.dropout(net, keep_prob=FLAGS.pretrain_dropout_keep)
        return net

    def pretrain_first_block_forward(self, inp, weights, block, reuse, scope):
        net = resnet_conv_block(inp, weights[block + '_conv1'], weights[block + '_bias1'], reuse, scope+block+'0')
        net = tf.nn.max_pool(net, [1,3,3,1], [1,2,2,1], 'SAME')
        net = tf.nn.dropout(net, keep_prob=FLAGS.pretrain_dropout_keep)
        return net

    def block_forward(self, inp, weights, ss_weights, block, reuse, scope, block_last_layer=False):
        net = resnet_conv_block(inp, self.process_ss_weights(weights, ss_weights, block + '_conv1'), ss_weights[block + '_bias1'], reuse, scope+block+'0')
        net = resnet_conv_block(net, self.process_ss_weights(weights, ss_weights, block + '_conv2'), ss_weights[block + '_bias2'], reuse, scope+block+'1')
        res = resnet_nob_conv_block(inp, weights[block + '_conv_res'], reuse, scope+block+'res')
        net = net + res
        if block_last_layer:
            net = tf.nn.max_pool(net, [1,2,2,1], [1,2,2,1], 'SAME')
        net = tf.nn.dropout(net, keep_prob=1)
        return net

    def first_block_forward(self, inp, weights, ss_weights, block, reuse, scope, block_last_layer=False):
        net = resnet_conv_block(inp, self.process_ss_weights(weights, ss_weights, block + '_conv1'), ss_weights[block + '_bias1'], reuse, scope+block+'0')
        net = tf.nn.max_pool(net, [1,3,3,1], [1,2,2,1], 'SAME')
        net = tf.nn.dropout(net, keep_prob=1)
        return net

    def construct_fc_weights(self):
        dtype = tf.float32        
        fc_weights = {}
        fc_initializer =  tf.contrib.layers.xavier_initializer(dtype=dtype)
        if FLAGS.phase=='pre':
            fc_weights['w5'] = tf.get_variable('fc_w5', [512, FLAGS.pretrain_class_num], initializer=fc_initializer)
            fc_weights['b5'] = tf.Variable(tf.zeros([FLAGS.pretrain_class_num]), name='fc_b5')
        else:
            fc_weights['w5'] = tf.get_variable('fc_w5', [512, self.dim_output], initializer=fc_initializer)
            fc_weights['b5'] = tf.Variable(tf.zeros([self.dim_output]), name='fc_b5')
        return fc_weights

    def construct_resnet_weights(self):
        weights = {}
        dtype = tf.float32
        conv_initializer =  tf.contrib.layers.xavier_initializer_conv2d(dtype=dtype)
        fc_initializer =  tf.contrib.layers.xavier_initializer(dtype=dtype)
        weights = self.construct_first_block_weights(weights, 7, 3, 64, conv_initializer, dtype, 'block0_1')

        weights = self.construct_residual_block_weights(weights, 3, 64, 64, conv_initializer, dtype, 'block1_1')
        weights = self.construct_residual_block_weights(weights, 3, 64, 64, conv_initializer, dtype, 'block1_2')

        weights = self.construct_residual_block_weights(weights, 3, 64, 128, conv_initializer, dtype, 'block2_1')
        weights = self.construct_residual_block_weights(weights, 3, 128, 128, conv_initializer, dtype, 'block2_2')

        weights = self.construct_residual_block_weights(weights, 3, 128, 256, conv_initializer, dtype, 'block3_1')
        weights = self.construct_residual_block_weights(weights, 3, 256, 256, conv_initializer, dtype, 'block3_2')

        weights = self.construct_residual_block_weights(weights, 3, 256, 512, conv_initializer, dtype, 'block4_1')
        weights = self.construct_residual_block_weights(weights, 3, 512, 512, conv_initializer, dtype, 'block4_2')

        weights['w5'] = tf.get_variable('w5', [512, FLAGS.pretrain_class_num], initializer=fc_initializer)
        weights['b5'] = tf.Variable(tf.zeros([FLAGS.pretrain_class_num]), name='b5')
        return weights

    def construct_residual_block_weights(self, weights, k, last_dim_hidden, dim_hidden, conv_initializer, dtype, scope='block0'):
        weights[scope + '_conv1'] = tf.get_variable(scope + '_conv1', [k, k, last_dim_hidden, dim_hidden], initializer=conv_initializer, dtype=dtype)
        weights[scope + '_bias1'] = tf.Variable(tf.zeros([dim_hidden]), name=scope + '_bias1')
        weights[scope + '_conv2'] = tf.get_variable(scope + '_conv2', [k, k, dim_hidden, dim_hidden], initializer=conv_initializer, dtype=dtype)
        weights[scope + '_bias2'] = tf.Variable(tf.zeros([dim_hidden]), name=scope + '_bias2')
        weights[scope + '_conv_res'] = tf.get_variable(scope + '_conv_res', [1, 1, last_dim_hidden, dim_hidden], initializer=conv_initializer, dtype=dtype)
        return weights

    def construct_first_block_weights(self, weights, k, last_dim_hidden, dim_hidden, conv_initializer, dtype, scope='block0'):
        weights[scope + '_conv1'] = tf.get_variable(scope + '_conv1', [k, k, last_dim_hidden, dim_hidden], initializer=conv_initializer, dtype=dtype)
        weights[scope + '_bias1'] = tf.Variable(tf.zeros([dim_hidden]), name=scope + '_bias1')
        return weights

    def construct_first_block_ss_weights(self, ss_weights, last_dim_hidden, dim_hidden, scope='block0'):
        ss_weights[scope + '_conv1'] = tf.Variable(tf.ones([1, 1, last_dim_hidden, dim_hidden]), name=scope + '_conv1')
        ss_weights[scope + '_bias1'] = tf.Variable(tf.zeros([dim_hidden]), name=scope + '_bias1')
        return ss_weights

    def construct_resnet_ss_weights(self):
        ss_weights = {}
        ss_weights = self.construct_first_block_ss_weights(ss_weights, 3, 64, 'block0_1')

        ss_weights = self.construct_residual_block_ss_weights(ss_weights, 64, 64, 'block1_1')
        ss_weights = self.construct_residual_block_ss_weights(ss_weights, 64, 64, 'block1_2')

        ss_weights = self.construct_residual_block_ss_weights(ss_weights, 64, 128, 'block2_1')
        ss_weights = self.construct_residual_block_ss_weights(ss_weights, 128, 128, 'block2_2')

        ss_weights = self.construct_residual_block_ss_weights(ss_weights, 128, 256, 'block3_1')
        ss_weights = self.construct_residual_block_ss_weights(ss_weights, 256, 256, 'block3_2')

        ss_weights = self.construct_residual_block_ss_weights(ss_weights, 256, 512, 'block4_1')
        ss_weights = self.construct_residual_block_ss_weights(ss_weights, 512, 512, 'block4_2')

        return ss_weights

    def construct_residual_block_ss_weights(self, ss_weights, last_dim_hidden, dim_hidden, scope='block0'):
        ss_weights[scope + '_conv1'] = tf.Variable(tf.ones([1, 1, last_dim_hidden, dim_hidden]), name=scope + '_conv1')
        ss_weights[scope + '_bias1'] = tf.Variable(tf.zeros([dim_hidden]), name=scope + '_bias1')
        ss_weights[scope + '_conv2'] = tf.Variable(tf.ones([1, 1, dim_hidden, dim_hidden]), name=scope + '_conv2')
        ss_weights[scope + '_bias2'] = tf.Variable(tf.zeros([dim_hidden]), name=scope + '_bias2')
        return ss_weights




