
import os
import torch
from datetime import datetime

SNAPSHOT_PATH = './logs/snapshot'
LOG_PATH = './logs/jobs'

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
    # index 42
    ENVIRONMENT = ['Live', 'Indoor', 'Outdoor']


def get_num_classes(spoof_categories):
    '''
    `0    ` : live
    `1,2,3` : PRINT
    `4,5,6` : PAPER CUT
    `7,8,9` : REPLAY
    `10   ` : 3D MASK
    '''
    
    if spoof_categories == 'binary':
        num_classes = 2
    else:
        assert isinstance(spoof_categories, list), "spoof_categories expected to be list of spoof labels lists, got {}".format(spoof_categories)
        num_classes = len(spoof_categories)
    return num_classes

def get_kernel(height, width):
    kernel_size = ((height + 15) // 16, (width + 15) // 16)
    return kernel_size

class TrainConfig(object):
    def __init__(self, img_size=128, input_size=128, batch_size=256, 
                spoof_categories='binary', class_balancing=None):
        # training
        self.lr = 1e-1
        self.milestones = [10, 15, 22]  # down learing rate  # [9, 13, 15]
        self.gamma = 0.1
        self.epochs = 50
        self.momentum = 0.9
        self.batch_size = batch_size
        self.valid_size = 0.2
        self.class_balancing = class_balancing
        
        # dataset
        self.img_size = img_size
        self.input_size = input_size
        self.train_path = './CelebA_Spoof_crop/data{}/train'.format(img_size)
        self.labels_path = './CelebA_Spoof_crop/data{}/train/train_target.csv'.format(img_size)
        self.spoof_categories = spoof_categories 

        # model
        self.num_classes = get_num_classes(spoof_categories)
        self.input_channel = 3
        self.embedding_size = 128
        self.kernel_size = get_kernel(input_size, input_size)
        # fourier image size
        self.ft_size = [2*s for s in self.kernel_size]
        
        # tensorboard
        self.board_loss_per_epoch = 10
      
    def set_job(self, name, device_id=0):
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')

        self.device = "cuda:{}".format(device_id) if torch.cuda.is_available() else "cpu"
        
        self.job_dir = "AntiSpoofing_{}".format(self.img_size)
        self.job_name = "AntiSpoofing_{}_{}".format(name, self.input_size)
        # log path
        self.log_path = "{}/{}/{}_{}".format(
            LOG_PATH, self.job_dir, name, current_time)
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        
        # save file path
        self.model_path = '{}/{}/{}_{}'.format(
            SNAPSHOT_PATH, self.job_dir, name, current_time)
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)


class PretrainedConfig(object):
    def __init__(self, model_path, device_id=0, input_size=128, num_classes=2):
        self.model_path = model_path
        self.device = "cuda:{}".format(device_id) if torch.cuda.is_available() else "cpu"
        self.input_size = input_size
        self.kernel_size = get_kernel(input_size, input_size)
        self.num_classes = num_classes


class TestConfig(PretrainedConfig):
    def __init__(self, model_path, device_id=0, img_size=128, input_size=128, 
                 batch_size=1, spoof_categories='binary'):
        super().__init__(model_path, device_id, input_size, 
                         get_num_classes(spoof_categories))
        self.test_path = './CelebA_Spoof_crop/data{}/test'.format(img_size)
        self.labels_path = './CelebA_Spoof_crop/data{}/test/test_target.csv'.format(img_size)
        self.spoof_categories = spoof_categories
        self.batch_size = batch_size        

# class TestConfig(object):
#     def __init__(self, img_size=128, input_size=128, batch_size=1,
#                  spoof_categories='binary', device_id=0, 
#                  model_path='saved_models/AntiSpoofing_bin_128.pth'):
#         # dataset
#         self.model_path = model_path
#         self.device = "cuda:{}".format(device_id) if torch.cuda.is_available() else "cpu"
#         self.input_size = input_size
#         self.kernel_size = get_kernel(input_size, input_size)
#         self.test_path = './CelebA_Spoof_crop/data{}/test'.format(img_size)
#         self.labels_path = './CelebA_Spoof_crop/data{}/test/test_target.csv'.format(img_size)
#         self.spoof_categories = spoof_categories
#         self.num_classes = get_num_classes(spoof_categories)
#         self.batch_size = batch_size
