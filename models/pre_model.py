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
#from models import Models
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS


def MakePreModel():
    if FLAGS.backbone_arch=='resnet12':
        from resnet12 import Models
    elif FLAGS.backbone_arch=='resnet18':
        from resnet18 import Models
    elif FLAGS.backbone_arch=='resnet34':
        from resnet34 import Models
    else:
        print('Please set the correct backbone')

    class PreModel(Models):
        def construct_pretrain_model(self, input_tensors=None, is_val=False):
            self.input = input_tensors['pretrain_input']
            self.label = input_tensors['pretrain_label']
            with tf.variable_scope('pretrain-model', reuse=None) as training_scope:
                if 'weights' in dir(self):
                    training_scope.reuse_variables()
                    weights = self.weights
                    fc_weights = self.fc_weights
                else:
                    self.weights = weights = self.construct_resnet_weights()
                    self.fc_weights = fc_weights = self.construct_fc_weights()

                if is_val==False:
                    self.pretrain_task_output = self.forward_fc(self.forward_pretrain_resnet(self.input, weights, reuse=False), fc_weights)
                    self.pretrain_task_loss = self.pretrain_loss_func(self.pretrain_task_output, self.label)
                    optimizer = tf.train.AdamOptimizer(self.pretrain_lr)
                    self.pretrain_op = optimizer.minimize(self.pretrain_task_loss, var_list=weights.values()+fc_weights.values())
                    self.pretrain_task_accuracy = tf.reduce_mean(tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(self.pretrain_task_output), 1), tf.argmax(self.label, 1)))
                    tf.summary.scalar('pretrain train loss', self.pretrain_task_loss)
                    tf.summary.scalar('pretrain train accuracy', self.pretrain_task_accuracy)
                else:
                    self.pretrain_task_output_val = self.forward_fc(self.forward_pretrain_resnet(self.input, weights, reuse=True), fc_weights)
                    self.pretrain_task_accuracy_val = tf.reduce_mean(tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(self.pretrain_task_output_val), 1), tf.argmax(self.label, 1)))
                    tf.summary.scalar('pretrain val accuracy', self.pretrain_task_accuracy_val)

    return PreModel()



