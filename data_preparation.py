import os
from tqdm import tqdm as tqdm
import pandas as pd

from src.image_io import read_image, save_image

ORG_DIR = 'CelebA_Spoof/'
CROP_DIR = 'CelebA_Spoof_crop/' 

# Spoof атаки, которые оставляем
spoof_filter = [0, 1, 2, 3, 7, 8, 9]
size = 120

def process_images(labels, size, scaleup=False):
    for img_path in tqdm(labels):
        img = read_image(ORG_DIR+img_path, bbox_inc = 0.4)

        new_img_path = img_path.replace('Data', CROP_DIR+'data'+str(size))
        new_img_dir = os.path.dirname(new_img_path)

        if not os.path.exists(new_img_dir):
            os.makedirs(new_img_dir)
        save_image(new_img_path, img, size, scaleup=scaleup)
        

org_lbl_dir = ORG_DIR + 'metas/intra_test/'

# read labels
print('\nReading labels...')
train_label = pd.read_json(org_lbl_dir+'train_label.json', orient='index'
                ).apply(pd.to_numeric, downcast='integer')[:1000]
test_label = pd.read_json(org_lbl_dir+'test_label.json', orient='index'
                ).apply(pd.to_numeric, downcast='integer')[:100]
print('Train / Test shape {} / {}:'.format(train_label.shape, test_label.shape))

# filter dataset with specified spoof types
print('\nFiltering labels...')
train_label = train_label[train_label[40].isin(spoof_filter)]
test_label  = test_label[test_label[40].isin(spoof_filter)]
print('Train / Test shape {} / {}:'.format(
    train_label.shape, test_label.shape))

# Read, Crop, Save images
print('\nProcessing train data...') 
process_images(train_label.index, size, scaleup=False)
print('\nProcessing test data...')
process_images(test_label.index, size, scaleup=False)

# write labels
print('\nWriting labels')
train_label.index = train_label.index.str.replace('Data/', '')
test_label.index = test_label.index.str.replace('Data/', '')
pd.concat([train_label, test_label]).to_csv(CROP_DIR+'data'+str(size)+'/label.csv')

print('Finished\n')