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
from utils.misc import mse, softmaxloss, xent, resnet_conv_block, resnet_nob_conv_block, normalize

FLAGS = flags.FLAGS


def MakeMetaModel():

    if FLAGS.backbone_arch=='resnet12':
        from resnet12 import Models
    elif FLAGS.backbone_arch=='resnet18':
        from resnet18 import Models
    elif FLAGS.backbone_arch=='resnet34':
        from resnet34 import Models
    else:
        print('Please set the correct backbone')

    class MetaModel(Models):
        '''
        def __init__(self):
            Models.__init__(self)
            if FLAGS.base_arch=='fc':
                self.construct_base_weights = self.construct_fc_weights
                self.forward_base = self.forward_fc
        '''
        def support_set_encoder(self, emb_outputa, lstm_weights):
            lstm_input = [emb_outputa]
            out_G, out_state_fw, out_state_bw = tf.contrib.rnn.static_bidirectional_rnn(lstm_weights['forward_cell'], lstm_weights['backward_cell'], inputs=lstm_input, dtype=tf.float32)
            out_G = out_G[0]
            G = emb_outputa + out_G[:,:512] + out_G[:,512:]
            G_norm = tf.norm(G, axis=1)
            [dim0, dim1] = G.get_shape().as_list()[0:2]
            G_norm = tf.tile(tf.expand_dims(G_norm, axis=1), multiples=[1, dim1])
            G_normalized = G / (G_norm+ 0.00001) 
            return G, G_normalized

        def construct_lstm_weights(self):
            with tf.variable_scope("mentornet", reuse=False) as vs:
                lstm_weights = {}
                lstm_weights['forward_cell'] = tf.contrib.rnn.BasicLSTMCell(512, forget_bias=0.0)
                lstm_weights['backward_cell'] = tf.contrib.rnn.BasicLSTMCell(512, forget_bias=0.0)
            variables = tf.contrib.framework.get_variables(vs)
            return lstm_weights, variables

        def construct_model(self):
            self.inputa = tf.placeholder(tf.float32, shape=(None, FLAGS.shot_num*FLAGS.way_num, 21168))
            self.labela = tf.placeholder(tf.float32, shape=(None, FLAGS.shot_num*FLAGS.way_num, FLAGS.way_num))
            self.inputb = tf.placeholder(tf.float32, shape=(None, None, 21168))
            self.labelb = tf.placeholder(tf.float32, shape=(None, None, FLAGS.way_num))
            #self.labelb = tf.placeholder(tf.float32, shape=(FLAGS.meta_batch_size, FLAGS.metatrain_epite_sample_num*FLAGS.way_num, FLAGS.way_num))

            with tf.variable_scope('meta-model', reuse=None) as training_scope:
                self.ss_weights = ss_weights = self.construct_resnet_ss_weights()
                self.weights = weights = self.construct_resnet_weights()
                lstm_weights, lstm_variables = self.construct_lstm_weights()
                self.lstm_weights = lstm_weights

                num_updates = FLAGS.train_base_epoch_num

                def task_metalearn(inp, reuse=True):

                    inputa, inputb, labela, labelb = inp
                    lossa_list = []
                    lossb_list = []

                    emb_outputa = self.forward_resnet(inputa, weights, ss_weights, reuse=reuse)
                    emb_outputb = self.forward_resnet(inputb, weights, ss_weights, reuse=True)

                    G, G_normalized = self.support_set_encoder(emb_outputa, lstm_weights)

                    F = emb_outputb
                    F_norm = tf.norm(F, axis=1)
                    F_norm = tf.tile(tf.expand_dims(F_norm, axis=1), multiples=[1, 512])
                    F_normalized = F / (F_norm+ 0.00001) 
                    G_T = tf.transpose(G_normalized)
                    scores = tf.nn.relu(tf.matmul(F_normalized, G_T)) *100
                    softmax_scores = tf.nn.softmax(scores)
                    logprobs =tf.log(tf.matmul(softmax_scores, labela)+1e-6)

                    ce_loss = -tf.reduce_mean(tf.reshape(tf.reduce_sum(tf.multiply(labelb, logprobs), axis=-1), [-1]))
                    accb = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(logprobs), 1), tf.argmax(labelb, 1))
                    task_output = [ce_loss, accb]

                    return task_output

                if FLAGS.norm is not 'None':
                    unused = task_metalearn((self.inputa[0], self.inputb[0], self.labela[0], self.labelb[0]), False)

                out_dtype = [tf.float32, tf.float32]

                result = tf.map_fn(task_metalearn, elems=(self.inputa, self.inputb, self.labela, self.labelb), \
                    dtype=out_dtype, parallel_iterations=FLAGS.meta_batch_size)
                lossb, accb = result

            self.total_loss = total_loss = tf.reduce_sum(lossb) / tf.to_float(FLAGS.meta_batch_size)
            self.total_accuracy = total_accuracy = tf.reduce_sum(accb) / tf.to_float(FLAGS.meta_batch_size)

            optimizer = tf.train.AdamOptimizer(self.meta_lr)
            if FLAGS.use_mtl:
                self.metatrain_op = optimizer.minimize(total_loss, var_list=ss_weights.values()+lstm_variables)
            else:
                self.metatrain_op = optimizer.minimize(total_loss, var_list=weights.values()+lstm_variables)

            self.training_summaries = []
            self.training_summaries.append(tf.summary.scalar('Meta Train Loss', (total_loss)))
            self.training_summaries.append(tf.summary.scalar('Meta Train Accuracy', total_accuracy))
            self.training_summ_op = tf.summary.merge(self.training_summaries)

            self.input_val_loss = tf.placeholder(tf.float32)
            self.input_val_acc = tf.placeholder(tf.float32)
            self.val_summaries = []
            self.val_summaries.append(tf.summary.scalar('Meta Val Loss', self.input_val_loss))
            self.val_summaries.append(tf.summary.scalar('Meta Val Accuracy', self.input_val_acc))
            self.val_summ_op = tf.summary.merge(self.val_summaries)

        '''
        def construct_test_model(self):
            self.inputa = tf.placeholder(tf.float32)
            self.inputb = tf.placeholder(tf.float32)
            self.labela = tf.placeholder(tf.float32)
            self.labelb = tf.placeholder(tf.float32)

            with tf.variable_scope('meta-test-model', reuse=None) as training_scope:             
                self.ss_weights = ss_weights = self.construct_resnet_ss_weights()
                self.weights = weights = self.construct_resnet_weights()
                self.base_weights = base_weights = self.construct_base_weights()

                num_updates = FLAGS.test_base_epoch_num

                def task_metalearn(inp, reuse=True):
                    """ Perform gradient descent for one task in the meta-batch. """
                    inputa, inputb, labela, labelb = inp
                    accb_list = []

                    emb_outputa = self.forward_resnet(inputa, weights, ss_weights, reuse=reuse)
                    emb_outputb = self.forward_resnet(inputb, weights, ss_weights, reuse=True)

                    outputa = self.forward_base(emb_outputa, base_weights)
                    lossa = self.loss_func(outputa, labela)     
                    grads = tf.gradients(lossa, list(base_weights.values()))
                    gradients = dict(zip(base_weights.keys(), grads))
                    fast_base_weights = dict(zip(base_weights.keys(), [base_weights[key] - \
                        self.update_lr*gradients[key] for key in base_weights.keys()]))
                    outputb = self.forward_base(emb_outputb, fast_base_weights)
                    accb = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(outputb), 1), tf.argmax(labelb, 1))
                    accb_list.append(accb)
              
                    for j in range(num_updates - 1):
                        lossa = self.loss_func(self.forward_base(emb_outputa, fast_base_weights), labela)
                        grads = tf.gradients(lossa, list(fast_base_weights.values()))
                        gradients = dict(zip(fast_base_weights.keys(), grads))
                        fast_base_weights = dict(zip(fast_base_weights.keys(), [fast_base_weights[key] - \
                            self.update_lr*gradients[key] for key in fast_base_weights.keys()]))
                        outputb = self.forward_base(emb_outputb, fast_base_weights)
                        accb = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(outputb), 1), tf.argmax(labelb, 1))
                        accb_list.append(accb)

                    lossb = self.loss_func(outputb, labelb)

                    task_output = [lossb, accb, accb_list]

                    return task_output

                if FLAGS.norm is not 'None':
                    unused = task_metalearn((self.inputa[0], self.inputb[0], self.labela[0], self.labelb[0]), False)

                out_dtype = [tf.float32, tf.float32, [tf.float32]*num_updates]

                result = tf.map_fn(task_metalearn, elems=(self.inputa, self.inputb, self.labela, self.labelb), \
                    dtype=out_dtype, parallel_iterations=FLAGS.meta_batch_size)
                lossesb, accsb, accsb_list = result

            self.metaval_total_loss = total_loss = tf.reduce_sum(lossesb)
            self.metaval_total_accuracy = total_accuracy = tf.reduce_sum(accsb)
            self.metaval_total_accuracies = total_accuracies =[tf.reduce_sum(accsb_list[j]) for j in range(num_updates)]
        '''

    return MetaModel()
