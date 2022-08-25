# Author : @hairymax

import os
from tqdm import tqdm as tqdm
import pandas as pd
import argparse
import logging
import cv2

#from src.image_io import read_image, save_image

#ORIG_DIR = 'CelebA_Spoof/'
#CROP_DIR = 'CelebA_Spoof_crop/' 
#spoof_types = [0, 1, 2, 3, 7, 8, 9] # Spoof атаки, которые оставляем


def read_image(image_path, bbox_inc = 0.3):
    """
    Read an image from input path and crop it with bbox
    
    params:
        - `image_path` : str - the path of image.
        - `bbox_inc` : float - image bbox increasing
    return:
        - `image`: Cropped image.
    """

    #image_path = LOCAL_ROOT + image_path

    img = cv2.imread(image_path)
    # Get the shape of input image
    real_h, real_w = img.shape[:2]
    assert os.path.exists(image_path[:-4] + '_BB.txt'), 'path not exists' + ' ' + image_path
    
    with open(image_path[:-4] + '_BB.txt','r') as f:
        material = f.readline()
        try:
            x, y, w, h = material.strip().split(' ')[:-1]
        except:
            logging.info('Bounding Box of' + ' ' + image_path + ' ' + 'is wrong')   

        try:
            w = int( int(float(w))*(real_w / 224) )
            h = int( int(float(h))*(real_h / 224) )
            x = int( int(float(x))*(real_w / 224) - bbox_inc/2*w )
            y = int( int(float(y))*(real_h / 224) - bbox_inc/2*h )
            # Crop face based on its bounding box
            x1 = 0 if x < 0 else x 
            y1 = 0 if y < 0 else y
            x2 = real_w if x1 + (1+bbox_inc)*w > real_w else int(x + (1+bbox_inc)*w)
            y2 = real_h if y1 + (1+bbox_inc)*h > real_h else int(y + (1+bbox_inc)*h)
            img = img[y1:y2,x1:x2,:]

        except:
            logging.info('Cropping Bounding Box of' + ' ' + image_path + ' ' + 'goes wrong')   

    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def save_image(image_path, img, largest_size, scaleup = True):
    """
    Save an image to image_path
    
    params:
        - `img`: cv image
        - `image_path` : str - the path of image to save.
        - `largest_size` : int - the size of the largest side of the shape to save
    """

    #image_path = LOCAL_ROOT + image_path

    # Get the shape of input image
    h, w = img.shape[:2]
    
    ratio = largest_size/max(h,w)
    if not scaleup:
        ratio = min(ratio, 1.)
        
    if ratio != 1.0:
        new_shape = int(w*ratio+0.01), int(h*ratio+0.01)
        img = cv2.resize(img, new_shape)
    
    return cv2.imwrite(image_path, img)


def process_images(orig_dir, crop_dir, labels, size, scaleup=False, bbox_inc = 0.3):
    for img_path in tqdm(labels):
        img = read_image(orig_dir+img_path, bbox_inc=bbox_inc)

        new_img_path = img_path.replace('Data', crop_dir+'data'+str(size))
        new_img_dir = os.path.dirname(new_img_path)

        if not os.path.exists(new_img_dir):
            os.makedirs(new_img_dir)
        save_image(new_img_path, img, size, scaleup=scaleup)

        
def parse_args():
    """parsing arguments"""
    
    def check_zero_to_one(value):
        fvalue = float(value)
        if fvalue < 0 or fvalue > 1:
            raise argparse.ArgumentTypeError("%s is an invalid value" % value)
        return fvalue
    
    p = argparse.ArgumentParser(description="Cropping images by bbox")
    p.add_argument("--orig_dir", type=str, default="CelebA_Spoof/", 
                   help="Directory with original Celeba_Spoof dataset")
    p.add_argument("--crop_dir", type=str, default="CelebA_Spoof_crop/",
                   help="Directory to save cropped dataset")
    p.add_argument("--size", type=int, default=256,
                   help="Size of the largest side of the image, px")
    p.add_argument("--bbox_inc", type=check_zero_to_one, default=0.3,
                   help="Image bbox increasing")
    p.add_argument("--spoof_types", type=int, nargs="+", default=list(range(10)),
                   help="Spoof types to keep")
    args = p.parse_args()
    if args.orig_dir[-1] != '/': args.orig_dir += '/'
    if args.crop_dir[-1] != '/': args.crop_dir += '/'

    return args


if __name__ == "__main__":
    args = parse_args()
    print('Check arguments:')
    print('    Original dataset directory       :', args.orig_dir)
    print('    Directory to save cropped images :', args.crop_dir+'data'+str(args.size))
    print('    Spoof types to keep in dataset   :', args.spoof_types)
    print('    Crop size, bbox increasing       :', (args.size, args.bbox_inc))
    
    proceed = input('\nProceed? [y/n] : ').lower()[:1] == 'y'
    if proceed:
        org_lbl_dir = args.orig_dir + 'metas/intra_test/'

        # read labels
        print('\nReading labels...')
        train_label = pd.read_json(org_lbl_dir+'train_label.json', orient='index'
                       ).apply(pd.to_numeric, downcast='integer')#[:10]
        test_label  = pd.read_json(org_lbl_dir+'test_label.json', orient='index'
                       ).apply(pd.to_numeric, downcast='integer')#[:10]
        print('Train / Test shape : {} / {}:'.format(train_label.shape, test_label.shape))

        # filter dataset with specified spoof types
        print('\nFiltering labels...')
        train_label = train_label[train_label[40].isin(args.spoof_types)]
        test_label  = test_label[test_label[40].isin(args.spoof_types)]
        print('Train / Test shape : {} / {}:'.format(train_label.shape, test_label.shape))

        # Read, Crop, Save images
        print('\nProcessing train data...') 
        process_images(args.orig_dir, args.crop_dir, 
                       train_label.index, args.size, scaleup=False)
        print('\nProcessing test data...')
        process_images(args.orig_dir, args.crop_dir, 
                       test_label.index, args.size, scaleup=False)

        # write labels
        print('\nWriting labels...')
        data_dir = args.crop_dir+'data'+str(args.size)
        train_label.index = train_label.index.str.replace('Data/', '')
        test_label.index  = test_label.index.str.replace('Data/', '')
        pd.concat([train_label, test_label]).to_csv(data_dir+'/label.csv')
        
        pd.DataFrame({'path': train_label.index.str.replace('train/', ''),
                      'spoof_type': train_label[40].values}
            ).to_csv(data_dir+'/train/train_target.csv', index=False)
        
        pd.DataFrame({'path': test_label.index.str.replace('test/', ''),
                      'spoof_type': test_label[40].values}
            ).to_csv(data_dir+'/test/test_target.csv', index=False)
        
        print('\nFinished\n')
    
    else:
        print('\nCancelled\n')