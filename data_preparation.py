# Author : @hairymax

import os
from tqdm import tqdm as tqdm
import pandas as pd
import argparse
import cv2

CELEBA_DIR = 'CelebA_Spoof/'
CROP_DIR = 'CelebA_Spoof_crop/' 
#spoof_types = [0, 1, 2, 3, 7, 8, 9] # Spoof атаки, которые оставляем

def read_image(image_path, bbox_inc = 1.5):
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
            print('Bounding Box of ' + image_path + ' is wrong')   
        
        try:
            w = int( float(w)*(real_w / 224) )
            h = int( float(h)*(real_h / 224) )
            x = int( float(x)*(real_w / 224) )
            y = int( float(y)*(real_h / 224) )

            # Crop face based on its bounding box
            l = max(w, h)
            # cv2.rectangle(img, (x, y), (x+w, y+h), (0,0,255), int(l/50))
            xc, yc = x + w/2, y + h/2
            x, y = int(xc - l*bbox_inc/2), int(yc - l*bbox_inc/2)

            x1 = 0 if x < 0 else x 
            y1 = 0 if y < 0 else y
            x2 = real_w if x + l*bbox_inc > real_w else x + int(l*bbox_inc)
            y2 = real_h if y + l*bbox_inc > real_h else y + int(l*bbox_inc)
            img = img[y1:y2,x1:x2,:]
            img = cv2.copyMakeBorder(img, 
                                     y1-y, int(l*bbox_inc-y2+y), 
                                     x1-x, int(l*bbox_inc-x2+x), 
                                     cv2.BORDER_CONSTANT, value=[0, 0, 0])
        except:
            print('Cropping Bounding Box of ' + image_path + ' goes wrong')   

    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def save_image(image_path, img, largest_size, scaleup=True):
    """
    Save an image to image_path
    
    params:
        - `img`: cv image
        - `image_path` : str - the path of image to save.
        - `largest_size` : int - the size of the largest side of the shape to save
    """
    # Get the shape of input image
    h, w = img.shape[:2]
    
    ratio = largest_size/max(h,w)
    if not scaleup:
        ratio = min(ratio, 1.)
        
    if ratio != 1.0:
        new_shape = int(w*ratio+0.01), int(h*ratio+0.01)
        img = cv2.resize(img, new_shape)
    
    return cv2.imwrite(image_path, img)


def read_orig_labels(orig_dir, spoof_filter=None):
    # read labels
    org_lbl_dir = orig_dir + 'metas/intra_test/'

    train_label = pd.read_json(org_lbl_dir+'train_label.json', orient='index'
                    ).apply(pd.to_numeric, downcast='integer')
    test_label  = pd.read_json(org_lbl_dir+'test_label.json', orient='index'
                    ).apply(pd.to_numeric, downcast='integer')
    
    print('Train / Test shape')
    print('          original: {} / {}'.format(
        train_label.shape, test_label.shape))
    # filter dataset with specified spoof types
    if spoof_filter:
        train_label = train_label[train_label[40].isin(spoof_filter)]
        test_label  = test_label[test_label[40].isin(spoof_filter)]
        print('          filtered: {} / {}'.format(
            train_label.shape, test_label.shape))
        
    return train_label, test_label


def save_labels(train_label, test_label, dir):
    #data_dir = dir+'data'+str(size)
    train_label.index = train_label.index.str.replace('Data/', '')
    test_label.index  = test_label.index.str.replace('Data/', '')
    pd.concat([train_label, test_label]).to_csv(dir+'/label.csv')
    
    if not os.path.exists(dir+'/train'):
        os.makedirs(dir+'/train')
    pd.DataFrame({'path': train_label.index.str.replace('train/', ''),
                    'spoof_type': train_label[40].values}
        ).to_csv(dir+'/train/train_target.csv', index=False)
    
    if not os.path.exists(dir+'/test'):
        os.makedirs(dir+'/test')
    pd.DataFrame({'path': test_label.index.str.replace('test/', ''),
                    'spoof_type': test_label[40].values}
        ).to_csv(dir+'/test/test_target.csv', index=False)


def process_images(orig_dir, crop_dir, labels, size, bbox_inc=1.5, scaleup=False):
    for img_path in tqdm(labels):
        img = read_image(orig_dir+img_path, bbox_inc=bbox_inc)

        new_img_path = img_path.replace('Data', crop_dir)
        new_img_dir = os.path.dirname(new_img_path)

        if not os.path.exists(new_img_dir):
            os.makedirs(new_img_dir)
        save_image(new_img_path, img, size, scaleup=scaleup)

        
if __name__ == "__main__":
    # parsing arguments
    def check_zero(value):
        fvalue = float(value)
        if fvalue < 0:
            raise argparse.ArgumentTypeError("%s is an invalid value" % value)
        return fvalue
    
    p = argparse.ArgumentParser(description="Cropping images by bbox")
    p.add_argument("--orig_dir", type=str, default=CELEBA_DIR, 
                   help="Directory with original Celeba_Spoof dataset")
    p.add_argument("--crop_dir", type=str, default=CROP_DIR,
                   help="Directory to save cropped dataset")
    p.add_argument("--size", type=int, default=128,
                   help="Size of the largest side of the image, px")
    p.add_argument("--bbox_inc", type=check_zero, default=1.5,
                   help="Image bbox increasing, value 1 makes no effect")
    p.add_argument("--spoof_types", type=int, nargs="+", default=list(range(10)),
                   help="Spoof types to keep")
    args = p.parse_args()
    if args.orig_dir[-1] != '/': args.orig_dir += '/'
    if args.crop_dir[-1] != '/': args.crop_dir += '/'
    
    data_dir = '{}data_{}_{}'.format(args.crop_dir, args.bbox_inc, args.size)
    print('Check arguments:')
    print('    Original dataset directory       :', args.orig_dir)
    print('    Directory to save cropped images :', data_dir)
    print('    Spoof types to keep in dataset   :', args.spoof_types)
    print('    Crop size, bbox increasing       :', (args.size, args.bbox_inc))
    
    # process images
    proceed = input('\nProceed? [y/n] : ').lower()[:1] == 'y'
    if proceed:
        # Read and filter labels
        print('\nReading and filtering labels...')
        train_label, test_label = read_orig_labels(args.orig_dir, 
                                                   spoof_filter=args.spoof_types)
        
        # Read, Crop, Save images
        print('\nProcessing train data...') 
        process_images(args.orig_dir, data_dir, train_label.index, 
                       args.size, bbox_inc=args.bbox_inc)
        print('\nProcessing test data...')
        process_images(args.orig_dir, data_dir, test_label.index, 
                       args.size, bbox_inc=args.bbox_inc)

        # write labels
        print('\nWriting labels...')
        save_labels(train_label, test_label, data_dir)
        
        print('\nFinished\n')
    
    else:
        print('\nCancelled\n')
        
        

# def read_image(image_path, bbox_inc = 0.3):
#     """
#     Read an image from input path and crop it with bbox
    
#     params:
#         - `image_path` : str - the path of image.
#         - `bbox_inc` : float - image bbox increasing
#     return:
#         - `image`: Cropped image.
#     """

#     #image_path = LOCAL_ROOT + image_path

#     img = cv2.imread(image_path)
#     # Get the shape of input image
#     real_h, real_w = img.shape[:2]
#     assert os.path.exists(image_path[:-4] + '_BB.txt'), 'path not exists' + ' ' + image_path
    
#     with open(image_path[:-4] + '_BB.txt','r') as f:
#         material = f.readline()
#         try:
#             x, y, w, h = material.strip().split(' ')[:-1]
#         except:
#             logging.info('Bounding Box of' + ' ' + image_path + ' ' + 'is wrong')   

#         try:
#             w = int( float(w)*(real_w / 224) )
#             h = int( float(h)*(real_h / 224) )
#             x = int( float(x)*(real_w / 224) - bbox_inc/2*w )
#             y = int( float(y)*(real_h / 224) - bbox_inc/2*h )
#             # Crop face based on its bounding box
#             x1 = 0 if x < 0 else x 
#             y1 = 0 if y < 0 else y
#             x2 = real_w if x1 + (1+bbox_inc)*w > real_w else int(x + (1+bbox_inc)*w)
#             y2 = real_h if y1 + (1+bbox_inc)*h > real_h else int(y + (1+bbox_inc)*h)
#             img = img[y1:y2,x1:x2,:]

#         except:
#             logging.info('Cropping Bounding Box of' + ' ' + image_path + ' ' + 'goes wrong')   

#     #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     return img