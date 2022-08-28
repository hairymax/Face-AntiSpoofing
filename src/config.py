# Original code https://github.com/minivision-ai/Silent-Face-Anti-Spoofing
# Author : @zhuyingSeu , Company : Minivision
# Modified by @hairymax
"""
default config for training
"""

import os
import torch
from datetime import datetime
from easydict import EasyDict
from src.utility import get_kernel

class CelebAattr(object):
    # indexes 0 - 39
    FACE_ATTR = [
        '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald',
        'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
        'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
        'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
        'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
        'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks',
        'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings',
        'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young'
    ]
    # index 40
    SPOOF_TYPE = ['Live',                               # 0     - live
        'Photo', 'Poster', 'A4',                        # 1,2,3 - PRINT 
        'Face Mask', 'Upper Body Mask', 'Region Mask',  # 4,5,6 - PAPER CUT 
        'PC', 'Pad', 'Phone',                           # 7,8,9 - REPLAY 
        '3D Mask'                                       # 10    - 3D MASK 
    ]
    # index 41
    ILLUMINATION = ['Live', 'Normal', 'Strong', 'Back', 'Dark']
    # index 41
    ENVIRONMENT = ['Live', 'Indoor', 'Outdoor']


SNAPSHOT_PATH = './logs/snapshot'
LOG_PATH = './logs/jobs'


def get_train_config(input_size=128, batch_size=256, spoof_categories='binary', 
                     class_balancing=None):
    cnf = EasyDict()
    
    # training
    cnf.lr = 1e-1
    cnf.milestones = [10, 15, 22]  # down learing rate  # [9, 13, 15]
    cnf.gamma = 0.1
    cnf.epochs = 50
    cnf.momentum = 0.9
    cnf.batch_size = batch_size
    cnf.valid_size = 0.2
    cnf.class_balancing = class_balancing
    
    # dataset
    cnf.input_size = input_size
    cnf.train_path = './CelebA_Spoof_crop/data{}/train'.format(cnf.input_size)
    cnf.labels_path = './CelebA_Spoof_crop/data{}/train/train_target.csv'.format(cnf.input_size)
    cnf.spoof_categories = spoof_categories 
    # [
    #     [0],     # 0     - live
    #     [1,2,3], # 1,2,3 - PRINT
    #     [4,5,6], # 4,5,6 - PAPER CUT
    #     [7,8,9], # 7,8,9 - REPLAY
    #     [10]     # 10    - 3D MASK
    # ]

    # model
    if spoof_categories == 'binary':
        cnf.num_classes = 2
    else:
        assert isinstance(spoof_categories, list), "spoof_categories expected to be list of spoof labels lists, got {}".format(spoof_categories)
        cnf.num_classes = len(spoof_categories)
    cnf.input_channel = 3
    cnf.embedding_size = 128
    cnf.kernel_size = get_kernel(cnf.input_size, cnf.input_size)
    # fourier image size
    cnf.ft_size = [2*s for s in cnf.kernel_size]
    
    # tensorboard
    cnf.board_loss_per_epoch = 10
    # save model/iter
    cnf.save_model_per_epoch = 5

    return cnf


def set_train_job(cnf, name):

    current_time = datetime.now().strftime('%b%d_%H-%M-%S')

    cnf.device = "cuda:{}".format(0) if torch.cuda.is_available() else "cpu"
    
    cnf.job_name = "{}_{}".format(name, cnf.input_size)
    # log path
    cnf.log_path = "{}/{}/{}".format(LOG_PATH, cnf.job_name, current_time)
    if not os.path.exists(cnf.log_path):
        os.makedirs(cnf.log_path)
    
    # save file path
    cnf.model_path = '{}/{}'.format(SNAPSHOT_PATH, cnf.job_name)
    if not os.path.exists(cnf.model_path):
        os.makedirs(cnf.model_path)

    return cnf
