from src.face_detector import YOLOv5
from src.FaceAntiSpoofing import AntiSpoof
import cv2
import numpy as np

COLOR_REAL = (0, 255, 0)
COLOR_FAKE = (0, 0, 255)

def increased_crop(img, bbox : tuple, bbox_inc : float = 1.5):
    # Crop face based on its bounding box
    real_h, real_w = img.shape[:2]
    
    x, y, w, h = bbox
    w, h = w - x, h - y
    l = max(w, h)
    
    xc, yc = x + w/2, y + h/2
    x, y = int(xc - l*bbox_inc/2), int(yc - l*bbox_inc/2)
    x1 = 0 if x < 0 else x 
    y1 = 0 if y < 0 else y
    x2 = real_w if x + l*bbox_inc > real_w else x + int(l*bbox_inc)
    y2 = real_h if y + l*bbox_inc > real_h else y + int(l*bbox_inc)
    
    img = img[y1:y2,x1:x2,:]
    img = cv2.copyMakeBorder(img, 
                             y1-y, int(l*bbox_inc-y2+y), 
                             x1-x, int(l*bbox_inc)-x2+x, 
                             cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return img

def make_prediction(img, face_detector, anti_spoof):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    bbox = face_detector([img])[0]
    
    if bbox.shape[0] > 0:
        bbox = bbox.flatten()[:4].astype(int)
    else:
        return None

    pred = anti_spoof([increased_crop(img, bbox, bbox_inc=1.5)])[0]
    label = np.argmax(pred)
    score = pred[0][0]
    
    return bbox, label, score

face_detector = YOLOv5('saved_models/yolov5s-face.onnx')
anti_spoof = AntiSpoof('saved_models/AntiSpoofing_bin_1.5_128.onnx')

# Create a video capture object
vid_capture = cv2.VideoCapture('test_imgs/test_video.mp4')

frame_width = int(vid_capture.get(3))
frame_height = int(vid_capture.get(4))
frame_size = (frame_width,frame_height)

if (vid_capture.isOpened() == False):
    print("Error opening a video file")
# Reading fps and frame rate
else:
    fps = vid_capture.get(5)    # Get information about frame rate
    print('Frame rate: ', fps, 'FPS')
    frame_count = vid_capture.get(7)    # Get the number of frames
    print('Frames count: ', frame_count) 

output = cv2.VideoWriter('test_imgs/test_video_pred_bin.mp4', 
                         cv2.VideoWriter_fourcc(* 'XVID'), fps, frame_size)

while vid_capture.isOpened():
    ret, frame = vid_capture.read()
    if ret == True:
        pred = make_prediction(frame, face_detector, anti_spoof)
        if pred is not None:
            (x1, y1, x2, y2), label, score = pred
            if label == 0:
                res_text = "REAL      {:.2f}".format(score)
                color = COLOR_REAL
            else:
                res_text = "FAKE      {:.2f}".format(score)
                color = COLOR_FAKE
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 8)
            cv2.putText(frame, res_text, 
                (x1, y1-20), cv2.FONT_HERSHEY_COMPLEX, (x2-x1)/250, color, 3)
            
        output.write(frame)
    else:
        print("Streaming is Off")
        break
# Release the video capture and writer objects
vid_capture.release()
output.release()