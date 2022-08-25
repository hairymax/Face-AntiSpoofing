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


SNAPSHOT_PATH = './logs/snapshot'
LOG_PATH = './logs/jobs'

# TODO сделать аргументами input_size, batch_size, spoof_categories
def get_default_config():
    cnf = EasyDict()
    
    # training
    cnf.lr = 1e-1
    cnf.milestones = [10, 15, 22]  # down learing rate  # [9, 13, 15]
    cnf.gamma = 0.1
    cnf.epochs = 50
    cnf.momentum = 0.9
    cnf.batch_size = 256
    cnf.valid_size = 0.2
    # dataset
    cnf.input_size = 128
    cnf.train_path = './CelebA_Spoof_crop/data{}/train'.format(cnf.input_size)
    cnf.labels_path = './CelebA_Spoof_crop/data{}/train/train_target.csv'.format(cnf.input_size)
    cnf.spoof_categories = 'binary' 
    # [
    #     [0],     # 0     - live
    #     [1,2,3], # 1,2,3 - PRINT
    #     [4,5,6], # 4,5,6 - PAPER CUT
    #     [7,8,9], # 7,8,9 - REPLAY
    #     [10]     # 10    - 3D MASK
    # ]
    
    # model
    # TODO num_classes брать из spoof_categories?
    cnf.num_classes = 2
    cnf.input_channel = 3
    cnf.embedding_size = 128
    cnf.kernel_size = get_kernel(cnf.input_size, cnf.input_size)
    # fourier image size
    cnf.ft_size = 2*cnf.kernel_size[0]
    
    # tensorboard
    cnf.board_loss_per_epoch = 20
    # save model/iter
    cnf.save_model_per_epoch = 10

    return cnf


def set_job(cnf):

    current_time = datetime.now().strftime('%b%d_%H-%M-%S')

    cnf.device = "cuda:{}".format(0) if torch.cuda.is_available() else "cpu"
    
    cnf.job_name = "Anti_Spoofing_{}".format(cnf.input_size)
    # log path
    cnf.log_path = "{}/{}/{}".format(LOG_PATH, cnf.job_name, current_time)
    if not os.path.exists(cnf.log_path):
        os.makedirs(cnf.log_path)
    
    # save file path
    cnf.model_path = '{}/{}'.format(SNAPSHOT_PATH, cnf.job_name)
    if not os.path.exists(cnf.model_path):
        os.makedirs(cnf.model_path)

    return cnf
