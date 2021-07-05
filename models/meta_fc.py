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
            self.construct_base_weights = self.construct_fc_weights
            self.forward_base = self.forward_fc
        '''
        def construct_base_weights(self):
            dtype = tf.float32        
            fc_weights = {}
            fc_initializer =  tf.contrib.layers.xavier_initializer(dtype=dtype)
            fc_weights['w5'] = tf.get_variable('fc_w5', [512, self.dim_output], initializer=fc_initializer)
            fc_weights['b5'] = tf.Variable(tf.zeros([self.dim_output]), name='fc_b5')
            return fc_weights

        def forward_base(self, inp, fc_weights):
            net = tf.matmul(inp, fc_weights['w5']) + fc_weights['b5']
            return net

        def construct_model(self):
            self.inputa = tf.placeholder(tf.float32)
            self.inputb = tf.placeholder(tf.float32)
            self.labela = tf.placeholder(tf.float32)
            self.labelb = tf.placeholder(tf.float32)

            with tf.variable_scope('meta-model', reuse=None) as training_scope:
                self.ss_weights = ss_weights = self.construct_resnet_ss_weights()
                self.weights = weights = self.construct_resnet_weights()
                self.base_weights = base_weights = self.construct_base_weights()

                num_updates = FLAGS.train_base_epoch_num

                def task_metalearn(inp, reuse=True):

                    inputa, inputb, labela, labelb = inp
                    lossa_list = []
                    lossb_list = []

                    emb_outputa = self.forward_resnet(inputa, weights, ss_weights, reuse=reuse)
                    emb_outputb = self.forward_resnet(inputb, weights, ss_weights, reuse=True)

                    outputa = self.forward_base(emb_outputa, base_weights)
                    lossa = self.loss_func(outputa, labela)
                    lossa_list.append(lossa)
                    outputb = self.forward_base(emb_outputb, base_weights)
                    lossb = self.loss_func(outputb, labelb)
                    lossb_list.append(lossb)  
                    grads = tf.gradients(lossa, list(base_weights.values()))
                    gradients = dict(zip(base_weights.keys(), grads))
                    fast_base_weights = dict(zip(base_weights.keys(), [base_weights[key] - \
                        self.update_lr*gradients[key] for key in base_weights.keys()]))
              
                    for j in range(num_updates - 1):
                        lossa = self.loss_func(self.forward_base(emb_outputa, fast_base_weights), labela)
                        lossa_list.append(lossa)
                        lossb = self.loss_func(self.forward_base(emb_outputb, fast_base_weights), labelb)
                        lossb_list.append(lossb) 
                        grads = tf.gradients(lossa, list(fast_base_weights.values()))
                        gradients = dict(zip(fast_base_weights.keys(), grads))
                        fast_base_weights = dict(zip(fast_base_weights.keys(), [fast_base_weights[key] - \
                            self.update_lr*gradients[key] for key in fast_base_weights.keys()]))

                    outputb = self.forward_base(emb_outputb, fast_base_weights)
                    final_lossb = self.loss_func(outputb, labelb)
                    accb = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(outputb), 1), tf.argmax(labelb, 1))

                    outputb_softmax = tf.nn.softmax(outputb)
                    task_output = [final_lossb, lossb_list, lossa_list, accb, outputb_softmax, labelb]

                    return task_output

                if FLAGS.norm is not 'None':
                    unused = task_metalearn((self.inputa[0], self.inputb[0], self.labela[0], self.labelb[0]), False)

                out_dtype = [tf.float32, [tf.float32]*num_updates, [tf.float32]*num_updates, tf.float32, tf.float32, tf.float32]

                result = tf.map_fn(task_metalearn, elems=(self.inputa, self.inputb, self.labela, self.labelb), \
                    dtype=out_dtype, parallel_iterations=FLAGS.meta_batch_size)
                lossb, lossesb, lossesa, accsb, outputsb, outlabelsb = result

            self.out_predb = outputsb
            self.out_labelb = outlabelsb
            self.total_loss = total_loss = tf.reduce_sum(lossb) / tf.to_float(FLAGS.meta_batch_size)
            self.total_accuracy = total_accuracy = tf.reduce_sum(accsb) / tf.to_float(FLAGS.meta_batch_size)
            self.total_lossa = total_lossa = [tf.reduce_sum(lossesa[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]
            self.total_lossb = total_lossb = [tf.reduce_sum(lossesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]

            optimizer = tf.train.AdamOptimizer(self.meta_lr)
            if FLAGS.use_mtl:
                self.metatrain_op = optimizer.minimize(total_loss, var_list=ss_weights.values() + base_weights.values())
            else:
                self.metatrain_op = optimizer.minimize(total_loss, var_list=weights.values() + base_weights.values())   

            self.training_summaries = []
            self.training_summaries.append(tf.summary.scalar('Meta Train Loss', (total_loss / tf.to_float(FLAGS.metatrain_epite_sample_num))))
            self.training_summaries.append(tf.summary.scalar('Meta Train Accuracy', total_accuracy))
            for j in range(num_updates):
                self.training_summaries.append(tf.summary.scalar('Base Train Loss Step' + str(j+1), total_lossa[j]))
            for j in range(num_updates):
                self.training_summaries.append(tf.summary.scalar('Base Val Loss Step' + str(j+1), total_lossb[j]))

            self.training_summ_op = tf.summary.merge(self.training_summaries)

            self.input_val_loss = tf.placeholder(tf.float32)
            self.input_val_acc = tf.placeholder(tf.float32)
            self.val_summaries = []
            self.val_summaries.append(tf.summary.scalar('Meta Val Loss', self.input_val_loss))
            self.val_summaries.append(tf.summary.scalar('Meta Val Accuracy', self.input_val_acc))
            self.val_summ_op = tf.summary.merge(self.val_summaries)


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

    return MetaModel()


