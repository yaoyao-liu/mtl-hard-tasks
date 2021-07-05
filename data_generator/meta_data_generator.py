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
import os
import random
import tensorflow as tf

from tqdm import trange
from tensorflow.python.platform import flags
from utils.misc import get_images, process_batch, process_batch_augmentation

FLAGS = flags.FLAGS

class MetaDataGenerator(object):
    def __init__(self):
        filename_dir = FLAGS.logdir_base + FLAGS.dataset +'_processed_data/'
        if not os.path.exists(filename_dir):
            os.mkdir(filename_dir)

        self.this_setting_filename_dir = filename_dir + 'shot(' + str(FLAGS.shot_num) + ').way(' + str(FLAGS.way_num) \
            + ').metatr_epite(' + str(FLAGS.metatrain_epite_sample_num) + ').metate_epite(' + str(FLAGS.metatest_epite_sample_num) + ')/'
        if not os.path.exists(self.this_setting_filename_dir):
            os.mkdir(self.this_setting_filename_dir)    

    def generate_data(self, data_type='train'):
        if data_type=='train':
            metatrain_folder = FLAGS.metatrain_dir
            folders = [os.path.join(metatrain_folder, label) \
                for label in os.listdir(metatrain_folder) \
                if os.path.isdir(os.path.join(metatrain_folder, label)) \
                ]
            num_total_batches = FLAGS.metatrain_iterations * FLAGS.meta_batch_size + 10
            num_samples_per_class = FLAGS.shot_num + FLAGS.metatrain_epite_sample_num

        elif data_type=='test':
            metatest_folder = FLAGS.metatest_dir
            folders = [os.path.join(metatest_folder, label) \
                for label in os.listdir(metatest_folder) \
                if os.path.isdir(os.path.join(metatest_folder, label)) \
                ]
            num_total_batches = 3600
            if FLAGS.metatest_epite_sample_num==0:
                num_samples_per_class = FLAGS.shot_num*2
            else:
                num_samples_per_class = FLAGS.shot_num + FLAGS.metatest_epite_sample_num
        elif data_type=='val':
            metaval_folder = FLAGS.metaval_dir
            folders = [os.path.join(metaval_folder, label) \
                for label in os.listdir(metaval_folder) \
                if os.path.isdir(os.path.join(metaval_folder, label)) \
                ]
            num_total_batches = 600
            if FLAGS.metatest_epite_sample_num==0:
                num_samples_per_class = FLAGS.shot_num*2
            else:
                num_samples_per_class = FLAGS.shot_num + FLAGS.metatest_epite_sample_num
        else:
            print('[Error] Please check data list type')

        task_num = FLAGS.way_num * num_samples_per_class
        epitr_sample_num = FLAGS.shot_num

        if not os.path.exists(self.this_setting_filename_dir+'/' + data_type + '_data.npy'):
            print('Generating ' + data_type + ' data')
            data_list = []
            for epi_idx in trange(num_total_batches):
                sampled_character_folders = random.sample(folders, FLAGS.way_num)
                random.shuffle(sampled_character_folders)
                labels_and_images = get_images(sampled_character_folders, \
                    range(FLAGS.way_num), nb_samples=num_samples_per_class, shuffle=False)
                labels = [li[0] for li in labels_and_images]
                filenames = [li[1] for li in labels_and_images]
                this_task_tr_filenames = []
                this_task_tr_labels = []
                this_task_te_filenames = []
                this_task_te_labels = []
                for class_idx in range(FLAGS.way_num):
                    this_class_filenames = filenames[class_idx*num_samples_per_class:(class_idx+1)*num_samples_per_class]
                    this_class_label = labels[class_idx*num_samples_per_class:(class_idx+1)*num_samples_per_class]
                    this_task_tr_filenames += this_class_filenames[0:epitr_sample_num]
                    this_task_tr_labels += this_class_label[0:epitr_sample_num]
                    this_task_te_filenames += this_class_filenames[epitr_sample_num:]
                    this_task_te_labels += this_class_label[epitr_sample_num:]

                this_batch_data = {'filenamea': this_task_tr_filenames, 'filenameb': this_task_te_filenames, 'labela': this_task_tr_labels, \
                    'labelb': this_task_te_labels}
                data_list.append(this_batch_data)

            np.save(self.this_setting_filename_dir+'/' + data_type + '_data.npy', data_list)
            print('The ' + data_type + ' data is saved')
        else:
            print('The ' + data_type + ' data has already been created')

    def load_data(self, data_type='test'):
        data_list = np.load(self.this_setting_filename_dir+'/' + data_type + '_data.npy')
        if data_type=='train':
            self.train_data = data_list
        elif data_type=='test':
            self.test_data = data_list
        elif data_type=='val':
            self.val_data = data_list
        else:
            print('[Error] Please check data list type')

    def load_episode(self, index, data_type='train'):
        if data_type=='train':
            data_list = self.train_data
            epite_sample_num = FLAGS.metatrain_epite_sample_num
        elif data_type=='test':
            data_list = self.test_data
            if FLAGS.metatest_epite_sample_num==0:
                epite_sample_num = FLAGS.shot_num
            else:
                epite_sample_num = FLAGS.metatest_episode_test_sample
        elif data_type=='val':
            data_list = self.val_data
            if FLAGS.metatest_epite_sample_num==0:
                epite_sample_num = FLAGS.shot_num
            else:
                epite_sample_num = FLAGS.metatest_episode_test_sample
        else:
            print('[Error] Please check data list type')

        dim_input = FLAGS.img_size * FLAGS.img_size * 3
        epitr_sample_num = FLAGS.shot_num

        this_episode = data_list[index]
        this_task_tr_filenames = this_episode['filenamea']
        this_task_te_filenames = this_episode['filenameb']
        this_task_tr_labels = this_episode['labela']
        this_task_te_labels = this_episode['labelb']

        #import pdb
        #pdb.set_trace()
        if data_type=='train' and FLAGS.backbone_arch=='resnet18' and FLAGS.meta_batch_size>=2:
            this_task_te_labels = this_task_te_labels[0:10] + this_task_te_labels[15:25] + this_task_te_labels[30:40] + this_task_te_labels[45:55] + this_task_te_labels[60:70]
            this_task_te_filenames = this_task_te_filenames[0:10] + this_task_te_filenames[15:25] + this_task_te_filenames[30:40] + this_task_te_filenames[45:55] + this_task_te_filenames[60:70]

            if FLAGS.base_augmentation:
                this_inputa, this_labela = process_batch_augmentation(this_task_tr_filenames, \
                    this_task_tr_labels, dim_input, epitr_sample_num)
                this_inputb, this_labelb = process_batch(this_task_te_filenames, \
                    this_task_te_labels, dim_input, 10)
            else:
                this_inputa, this_labela = process_batch(this_task_tr_filenames, \
                    this_task_tr_labels, dim_input, epitr_sample_num)
                this_inputb, this_labelb = process_batch(this_task_te_filenames, \
                    this_task_te_labels, dim_input, 10)
        else:
            if FLAGS.base_augmentation:
                this_inputa, this_labela = process_batch_augmentation(this_task_tr_filenames, \
                    this_task_tr_labels, dim_input, epitr_sample_num)
                this_inputb, this_labelb = process_batch(this_task_te_filenames, \
                    this_task_te_labels, dim_input, epite_sample_num)
            else:
                this_inputa, this_labela = process_batch(this_task_tr_filenames, \
                    this_task_tr_labels, dim_input, epitr_sample_num)
                this_inputb, this_labelb = process_batch(this_task_te_filenames, \
                    this_task_te_labels, dim_input, epite_sample_num)

        return this_inputa, this_labela, this_inputb, this_labelb 


    def load_cls_name(self, index, data_type='train'):
        if data_type=='train':
            data_list = self.train_data
            epite_sample_num = FLAGS.metatrain_epite_sample_num
        elif data_type=='test':
            data_list = self.test_data
            if FLAGS.metatest_epite_sample_num==0:
                epite_sample_num = FLAGS.shot_num
            else:
                epite_sample_num = FLAGS.metatest_episode_test_sample
        elif data_type=='val':
            data_list = self.val_data
            if FLAGS.metatest_epite_sample_num==0:
                epite_sample_num = FLAGS.shot_num
            else:
                epite_sample_num = FLAGS.metatest_episode_test_sample
        else:
            print('[Error] Please check data list type')

        dim_input = FLAGS.img_size * FLAGS.img_size * 3
        epitr_sample_num = FLAGS.shot_num

        this_episode = data_list[index]
        this_task_tr_filenames = this_episode['filenamea']
        this_task_te_filenames = this_episode['filenameb']
        this_task_tr_labels = this_episode['labela']
        this_task_te_labels = this_episode['labelb']

        this_task_cls_name = [os.path.basename(os.path.dirname(this_path)) for this_path in this_task_te_filenames]
        this_task_cls_name = self.get_norepeat_list(this_task_cls_name)

        return this_task_cls_name

    def create_new_episode(self, cls_list):
        dim_input = FLAGS.img_size * FLAGS.img_size * 3
        epitr_sample_num = FLAGS.shot_num
        epite_sample_num = FLAGS.metatrain_epite_sample_num
        metatrain_folders = [os.path.join(FLAGS.metatrain_dir, label) for label in cls_list if os.path.isdir(os.path.join(FLAGS.metatrain_dir, label))] 
        labels_and_images_1 = get_images(metatrain_folders, range(FLAGS.way_num), nb_samples=epitr_sample_num, shuffle=False)
        add_labels_a = [li[0] for li in labels_and_images_1]
        add_filenames_a = [li[1] for li in labels_and_images_1]

        labels_and_images_2 = get_images(metatrain_folders, range(FLAGS.way_num), nb_samples=epite_sample_num, shuffle=False)
        add_labels_b = [li[0] for li in labels_and_images_2]
        add_filenames_b = [li[1] for li in labels_and_images_2]

        if FLAGS.backbone_arch=='resnet18' and FLAGS.meta_batch_size>=2 and FLAGS.shot_num>1:
            add_labels_b = add_labels_b[0:10] + add_labels_b[15:25] + add_labels_b[30:40] + add_labels_b[45:55] + add_labels_b[60:70]
            add_filenames_b = add_filenames_b[0:10] + add_filenames_b[15:25] + add_filenames_b[30:40] + add_filenames_b[45:55] + add_filenames_b[60:70]

            this_add_inputa, this_add_labela = process_batch(add_filenames_a, add_labels_a, dim_input, epitr_sample_num)
            this_add_inputb, this_add_labelb = process_batch(add_filenames_b, add_labels_b, dim_input, 10)
        else:
            this_add_inputa, this_add_labela = process_batch(add_filenames_a, add_labels_a, dim_input, epitr_sample_num)
            this_add_inputb, this_add_labelb = process_batch(add_filenames_b, add_labels_b, dim_input, epite_sample_num)            
        return this_add_inputa, this_add_labela, this_add_inputb, this_add_labelb 

    def get_norepeat_list(self, inp):
        out_list=[inp[0]]
        for item in inp:
            if not item == out_list[-1]:
                out_list.append(item)
        return out_list
              


