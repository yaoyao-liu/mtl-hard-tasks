import os

def run_exp(SHOT_NUM=1, HEAD='fc', BACKBONE='resnet12', MTL='True', DATASET='mini', PHASE='META'):
    GPU_ID = 0
    if DATASET=='tiered' or DATASET=='tiered2mini':
        if BACKBONE=='resnet12':
            PRE_ITER = 100000
        else:
            PRE_ITER = 120000
        if BACKBONE=='resnet12':
            PRE_LR_DROP = 5000   
        else:     
            PRE_LR_DROP = 20000
        if BACKBONE=='resnet12':
            PRE_BATCH_SIZE = 64    
        else:
            PRE_BATCH_SIZE = 32
        PRE_TRA_LAB = 'normal'
    else:
        PRE_ITER = 10000
        PRE_LR_DROP = 5000   
        PRE_BATCH_SIZE = 64    
        PRE_TRA_LAB = 'mini_nh'        

    if MTL=='True':
        MAX_ITER = 10000
    else:
        MAX_ITER = 5000
    META_BATCH_SIZE = 2
    UPDATE_NUM = 20
    WAY_NUM = 5
    GPU_MODE = 'True'
    LOG_DIR = 'experiment_results_1'
    PRE_TRA_DROP = 0.9
    SAVE_STEP = 1000
    LR_DROP_STEP = 1000
    META_LR = 0.001 
    PRE_LR = 0.001
    HTB_HARD = 5
    HTB_NORMAL = 10
    

    base_command = 'python main.py' \
        + ' --metatrain_iterations=' + str(MAX_ITER) \
        + ' --pretrain_batch_size=' + str(PRE_BATCH_SIZE) \
        + ' --meta_batch_size=' + str(META_BATCH_SIZE) \
        + ' --shot_num=' + str(SHOT_NUM) \
        + ' --base_lr=0.01' \
        + ' --train_base_epoch_num=' + str(UPDATE_NUM) \
        + ' --way_num=' + str(WAY_NUM) \
        + ' --exp_log_label=' + LOG_DIR \
        + ' --pretrain_dropout=' + str(PRE_TRA_DROP) \
        + ' --activation=leaky_relu' \
        + ' --pre_lr=' + str(PRE_LR) \
        + ' --pre_lr_dropstep=' + str(PRE_LR_DROP)\
        + ' --meta_save_step=' + str(SAVE_STEP) \
        + ' --lr_drop_step=' + str(LR_DROP_STEP) \
        + ' --pretrain_label=' + PRE_TRA_LAB \
        + ' --full_gpu_memory_mode=' + GPU_MODE \
        + ' --device_id=' + str(GPU_ID) \
        + ' --use_mtl=' + str(MTL) \
        + ' --base_arch=' + HEAD \
        + ' --backbone_arch=' + BACKBONE \
        + ' --meta_lr=' + str(META_LR) \
        + ' --dataset=' + DATASET \
        + ' --htb_hard=' + str(HTB_HARD) \
        + ' --htb_normal=' + str(HTB_NORMAL)

    def process_test_command(TEST_STEP, in_command):
        output_test_command = in_command \
            + ' --phase=meta' \
            + ' --pretrain_iterations=' + str(PRE_ITER) \
            + ' --metatrain=False' \
            + ' --test_iter=' + str(TEST_STEP)
        return output_test_command

    if PHASE=='PRE':
        print('****** Start Pre-training Phase ******')
        pre_command = base_command + ' --phase=pre' + ' --pretrain_iterations=20000'
        os.system(pre_command)

    if PHASE=='META':
        print('****** Start Meta-training Phase ******')
        meta_train_command = base_command + ' --phase=meta' + ' --pretrain_iterations=' + str(PRE_ITER)
        os.system(meta_train_command)

    if PHASE=='META_TE':
        print('****** Start Meta-test Phase ******')

        test_command = process_test_command(8000, base_command)
        os.system(test_command)


run_exp(SHOT_NUM=1, HEAD='fc', BACKBONE='resnet12', MTL='True', DATASET='mini', PHASE='META')
run_exp(SHOT_NUM=5, HEAD='fc', BACKBONE='resnet12', MTL='True', DATASET='mini', PHASE='META')

run_exp(SHOT_NUM=1, HEAD='cosine', BACKBONE='resnet12', MTL='True', DATASET='mini', PHASE='META')
run_exp(SHOT_NUM=5, HEAD='cosine', BACKBONE='resnet12', MTL='True', DATASET='mini', PHASE='META')

run_exp(SHOT_NUM=1, HEAD='fc', BACKBONE='resnet12', MTL='True', DATASET='tiered', PHASE='META')
run_exp(SHOT_NUM=5, HEAD='fc', BACKBONE='resnet12', MTL='True', DATASET='tiered', PHASE='META')

run_exp(SHOT_NUM=1, HEAD='cosine', BACKBONE='resnet12', MTL='True', DATASET='tiered', PHASE='META')
run_exp(SHOT_NUM=5, HEAD='cosine', BACKBONE='resnet12', MTL='True', DATASET='tiered', PHASE='META')
