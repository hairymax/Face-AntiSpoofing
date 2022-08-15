import cv2
import os
import logging

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
