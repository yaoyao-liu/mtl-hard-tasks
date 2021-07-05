##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Yaoyao Liu
## NUS School of Computing
## yaoyaoliu@outlook.com
## Copyright (c) 2019
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
from tensorflow.python.platform import flags
from trainer.meta import MetaTrainer
from trainer.pre import PreTrainer

FLAGS = flags.FLAGS

### Basic Options
flags.DEFINE_integer('img_size', 84, 'image size')
flags.DEFINE_integer('device_id', 0, 'GPU device ID to run the job.')
flags.DEFINE_float('gpu_rate', 0.95, 'the parameter for the full_gpu_memory_mode')
flags.DEFINE_string('phase', 'meta', 'pre or meta')
flags.DEFINE_string('exp_log_label', 'experiment_results', 'directory for summaries and checkpoints')
flags.DEFINE_string('logdir_base', './logs/', 'directory for logs')
flags.DEFINE_bool('full_gpu_memory_mode', False, 'in this mode, the code occupies GPU memory in advance')
flags.DEFINE_string('dataset', 'mini', '-')

### Pretrain Phase Options
flags.DEFINE_integer('pre_lr_dropstep', 5000, 'the step number to drop pre_lr')
#flags.DEFINE_integer('pretrain_class_num', 64, 'number of classes used in the pre-train phase')
flags.DEFINE_integer('pretrain_batch_size', 64, 'batch_size for the pre-train phase')
flags.DEFINE_integer('pretrain_iterations', 30000, 'number of pretraining iterations.')
flags.DEFINE_integer('pre_sum_step', 10, 'the step number to summary during pretraining')
flags.DEFINE_integer('pre_save_step', 1000, 'the step number to save the pretrain model')
flags.DEFINE_integer('pre_print_step', 1000, 'the step number to print the pretrain results')
flags.DEFINE_float('pre_lr', 0.001, 'the pretrain learning rate')
flags.DEFINE_float('min_pre_lr', 0.0001, 'the pretrain learning rate min')
flags.DEFINE_float('pretrain_dropout_keep', 0.9, 'the dropout keep parameter in the pre-train phase')
#flags.DEFINE_string('pretrain_folders', './minia/mini_nh/train', 'directory for pre-train data')
#flags.DEFINE_string('pretrain_val_folders', './data/pre-train/pre-train_val', 'directory for pre-train val data')
flags.DEFINE_string('pretrain_label', 'mini_nh', 'additional label for the pre-train log folder')
flags.DEFINE_bool('pre_lr_stop', False, 'whether stop decrease the pre_lr when it is low')

### Meta Phase Options
flags.DEFINE_integer('way_num', 5, 'number of classes (e.g. 5-way classification)')
flags.DEFINE_integer('shot_num', 5, 'number of examples per class (K for K-shot learning)')
flags.DEFINE_integer('metatrain_epite_sample_num', 15, 'number of meta train episode-test samples')
flags.DEFINE_integer('metatest_epite_sample_num', 0, 'number of meta test episode-test samples, 0 means metatest_epite_sample_num=shot_num')
flags.DEFINE_integer('meta_sum_step', 10, 'the step number to summary during meta-training')
flags.DEFINE_integer('meta_save_step', 500, 'the step number to save the model')
flags.DEFINE_integer('meta_intrain_val_sample', 600, '-')
flags.DEFINE_integer('meta_print_step', 100, 'the step number to print the meta-train results')
flags.DEFINE_integer('meta_val_print_step', 200, 'the step number to print the meta-val results during meta-training')
flags.DEFINE_integer('metatrain_iterations', 15000, 'number of meta-train iterations.') 
flags.DEFINE_integer('meta_batch_size', 2, 'number of tasks sampled per meta-update')
flags.DEFINE_integer('train_base_epoch_num', 20, 'number of inner gradient updates during training.')
flags.DEFINE_integer('test_base_epoch_num', 100, 'number of inner gradient updates during test.')
flags.DEFINE_integer('lr_drop_step', 5000, 'the step number to drop meta_lr')
flags.DEFINE_integer('test_iter', 1000, 'iteration to load model')
flags.DEFINE_float('meta_lr', 0.001, 'the meta learning rate of the generator')
flags.DEFINE_float('lr_drop_rate', 0.5, '-')
#flags.DEFINE_float('min_meta_lr', 0.0001, 'the min meta learning rate of the generator')
flags.DEFINE_float('base_lr', 1e-3, 'step size alpha for inner gradient update.')
#flags.DEFINE_string('metatrain_dir', './data/meta-train/train', 'directory for meta-train set')
#flags.DEFINE_string('metaval_dir', './data/meta-train/val', 'directory for meta-val set')
#flags.DEFINE_string('metatest_dir', './data/meta-train/test', 'directory for meta-test set')
flags.DEFINE_string('activation', 'leaky_relu', 'leaky_relu, relu, or None')
flags.DEFINE_string('norm', 'batch_norm', 'batch_norm, layer_norm, or None')
flags.DEFINE_bool('metatrain', True, 'is this the meta-train phase')
flags.DEFINE_bool('base_augmentation', False, 'whether do data augmentation during base learning')
flags.DEFINE_bool('redo_init', False, '-')
flags.DEFINE_string('base_arch', 'fc', '-')
flags.DEFINE_string('backbone_arch', 'resnet12', '-')
flags.DEFINE_bool('use_mtl', True, '-')
flags.DEFINE_integer('htb_normal', 10, '-')
flags.DEFINE_integer('htb_hard', 5, '-')

if FLAGS.dataset=='mini':
    FLAGS.pretrain_class_num = 64
    FLAGS.pretrain_folders = './data/mini/pre-train/mini_nh/train'
    FLAGS.pretrain_val_folders = './data/mini/pre-train/pre-train_val'
    FLAGS.metatrain_dir = './data/mini/meta-train/train'
    FLAGS.metaval_dir = './data/mini/meta-train/val'
    FLAGS.metatest_dir = './data/mini/meta-train/test'
elif FLAGS.dataset=='tiered':
    FLAGS.pretrain_class_num = 351
    FLAGS.pretrain_folders = './data/tiered/train'
    FLAGS.pretrain_val_folders = './data/tiered/train'
    FLAGS.metatrain_dir = './data/tiered/train'
    FLAGS.metaval_dir = './data/tiered/val'
    FLAGS.metatest_dir = './data/tiered/test'
elif FLAGS.dataset=='tiered2mini':
    FLAGS.pretrain_class_num = 351
    FLAGS.pretrain_folders = './data/tiered/train'
    FLAGS.pretrain_val_folders = './data/tiered/train'
    FLAGS.metatrain_dir = './data/mini/meta-train/train'
    FLAGS.metaval_dir = './data/mini/meta-train/val'
    FLAGS.metatest_dir = './data/mini/meta-train/test'
else:
    print('Please set correct dataset')

if FLAGS.base_arch=='proto' or FLAGS.base_arch=='matching':
    FLAGS.save_base = False
else:
    FLAGS.save_base = True

# Generate Experiment Key Words String
exp_string = 'data(' + FLAGS.dataset + ')'
exp_string += '.arch(' + FLAGS.backbone_arch + ')'
exp_string += '.head(' + FLAGS.base_arch + ')'
exp_string += '.cls(' + str(FLAGS.way_num) + ')'
exp_string += '.shot(' + str(FLAGS.shot_num) + ')'
exp_string += '.mbatch(' + str(FLAGS.meta_batch_size) + ')'
exp_string += '.bepoch(' + str(FLAGS.train_base_epoch_num) + ')'
exp_string += '.mlr(' + str(FLAGS.meta_lr) + ')'
exp_string += '.blr(' + str(FLAGS.base_lr) + ')'
exp_string += '.pre_iter(' + str(FLAGS.pretrain_iterations) + ')'
exp_string += '.pre_drop(' + str(FLAGS.pretrain_dropout_keep) + ')'
exp_string += '.lrdstep(' + str(FLAGS.lr_drop_step) + ')'
exp_string += '.lrdrate(' + str(FLAGS.lr_drop_rate) + ')'
exp_string += '.ht(' + str(FLAGS.htb_hard) + '_' + str(FLAGS.htb_normal) + ')'

if FLAGS.use_mtl:
    exp_string += '.mtl(True)'
else:
    exp_string += '.mtl(False)'

FLAGS.exp_string = exp_string
print('Parameters: ' + exp_string)

# Generate Pre-train Key Words String
pre_save_str = 'arch(' + FLAGS.backbone_arch + ')'
pre_save_str += '.pre_lr(' + str(FLAGS.pre_lr) + ')'
pre_save_str += '.pre_lrdrop(' + str(FLAGS.pre_lr_dropstep) + ')'
pre_save_str += '.pre_class(' + str(FLAGS.pretrain_class_num) + ')'
pre_save_str += '.pre_batch(' + str(FLAGS.pretrain_batch_size) + ')'
pre_save_str += '.pre_dropout(' + str(FLAGS.pretrain_dropout_keep) + ')'
if FLAGS.pre_lr_stop:
    pre_save_str += '.pre_lr_stop(True)'
pre_save_str += '.pre_label(' + FLAGS.pretrain_label + ')'
FLAGS.pre_string = pre_save_str

# Generate Log Folders
FLAGS.logdir = FLAGS.logdir_base + FLAGS.exp_log_label
FLAGS.pretrain_dir = FLAGS.logdir_base + FLAGS.dataset + '_pretrain_weights'

if not os.path.exists(FLAGS.logdir_base):
    os.mkdir(FLAGS.logdir_base)
if not os.path.exists(FLAGS.logdir):
    os.mkdir(FLAGS.logdir)
if not os.path.exists(FLAGS.pretrain_dir):
    os.mkdir(FLAGS.pretrain_dir)

if FLAGS.redo_init:
    os.system('rm -r ./logs/init_weights')
    print('Init weights have been deleted')

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.device_id)
    if FLAGS.phase=='pre':
        trainer = PreTrainer()
    elif FLAGS.phase=='meta':   
        trainer = MetaTrainer()
    else:
        print('Please set correct phase')                

if __name__ == "__main__":
    main()
