# -*- coding: utf-8 -*-
# @Time : 20-6-4 上午9:12
# @Author : zhuying
# @Company : Minivision
# @File : default_config.py
# @Software : PyCharm
# --*-- coding: utf-8 --*--
"""
default config for training
"""

import torch
from datetime import datetime
from easydict import EasyDict
from src.utility import make_if_not_exist, get_width_height, get_kernel


def get_default_config():
    cnf = EasyDict()

    # training
    cnf.train = EasyDict()
    # dataset
    cnf.dataset = EasyDict()
    # model
    cnf.model = EasyDict()
    
    
    cnf.train.lr = 1e-1
    # [9, 13, 15]
    cnf.train.milestones = [10, 15, 22]  # down learing rate
    cnf.train.gamma = 0.1
    cnf.train.epochs = 25
    cnf.train.momentum = 0.9
    cnf.train.batch_size = 1024
    
    cnf.dataset.train_path = './Celeba_Spoof_crop/data128/train'
    cnf.dataset.labels_path = './Celeba_Spoof_crop/data128/train/train_target.csv'
    cnf.dataset.input_size = 128
    cnf.dataset.ft_size = 2*((cnf.dataset.input_size + 15) // 16)
    cnf.dataset.spoof_categories = [
        [0],     # 0     - live
        [1,2,3], # 1,2,3 - PRINT
        # [4,5,6], # 4,5,6 - PAPER CUT
        [7,8,9], # 7,8,9 - REPLAY
        # [10]     # 10    - 3D MASK
    ]
    
    cnf.model.num_classes = 2
    cnf.model.input_channel = 3
    cnf.model.embedding_size = 128
    cnf.model.kernel_size = get_kernel(cnf.dataset.input_size, cnf.dataset.input_size)
    
    # save file path
    cnf.snapshot_dir_path = './logs/snapshot'
    # log path
    cnf.log_path = './logs/jobs'
    # tensorboard
    cnf.board_loss_every = 10
    # save model/iter
    cnf.save_every = 30

    cnf.device = "cuda:{}".format(0) if torch.cuda.is_available() else "cpu"
    return cnf


def update_config(args, cnf):
    cnf.devices = args.devices
    w_input, h_input = get_width_height(args.patch_info)
    cnf.input_size = [h_input, w_input]
    cnf.model.kernel_size = get_kernel(h_input, w_input)

    # resize fourier image size

    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    job_name = 'Anti_Spoofing_{}'.format(args.patch_info)
    log_path = '{}/{}/{} '.format(cnf.log_path, job_name, current_time)
    snapshot_dir = '{}/{}'.format(cnf.snapshot_dir_path, job_name)

    make_if_not_exist(snapshot_dir)
    make_if_not_exist(log_path)

    cnf.model_path = snapshot_dir
    cnf.log_path = log_path
    cnf.job_name = job_name
    return cnf
