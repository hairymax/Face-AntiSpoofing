import cv2
import onnxruntime as ort
import time
import numpy as np
from .utils import non_max_suppression, scale_coords, letterbox
import os


# onnx model
class YOLOv5:
    def __init__(self,
                 weights: str = None,
                 input_res: tuple = (640, 640),
                 batch_size: int = 1):
        super().__init__()
        self.weights = weights
        self.input_res = input_res
        self.batch_size = batch_size
        self.ort_session, self.input_name = self._init_session_(self.weights)
        self.max_detection = 1000

    def _init_session_(self, path_onnx_model: str):
        ort_session = None
        input_name = None
        if os.path.isfile(path_onnx_model):
            try:
                ort_session = ort.InferenceSession(path_onnx_model, providers=['CUDAExecutionProvider'])
            except:
                ort_session = ort.InferenceSession(path_onnx_model, providers=['CPUExecutionProvider'])
            input_name = ort_session.get_inputs()[0].name
        return ort_session, input_name

    def preprocessing(self, imgs: list):
        imgs_input = []
        for img in imgs:
            img_input, ratio, (dw, dh) = letterbox(img,
                                                   self.input_res,
                                                   auto=False,
                                                   scaleFill=False,
                                                   scaleup=True,
                                                   stride=32)
            img_input = img_input.transpose(2, 0, 1)  # to CHW  to 3x416x416
            img_input = np.ascontiguousarray(img_input)
            img_input = img_input.astype(np.float32)
            img_input /= 255.0
            img_input = np.expand_dims(img_input, axis=0)
            imgs_input.append(img_input)
        return imgs_input

    def postprocessing(self, prediction_bboxes, imgs, conf_thresh=0.25, iou_thresh=0.1, max_detection=1):
        assert len(prediction_bboxes) == len(imgs), f"Size prediction {len(prediction_bboxes)} not equal size images {len(imgs)}"
        pred = non_max_suppression(prediction_bboxes,
                                   conf_thresh=conf_thresh,
                                   iou_thresh=iou_thresh,
                                   max_det=max_detection)
        for i, det in enumerate(pred):  # detections per image
            if len(det):
                det[:, :4] = scale_coords(self.input_res, det[:, :4], imgs[i].shape).round()
        return pred

    def __call__(self, imgs, conf_thresh=0.25, iou_thresh=0.45, max_detection=1):
        if not self.ort_session:
            return False

        if self.batch_size == 1:
            preds = []
            for img in imgs:
                onnx_result = self.ort_session.run([],
                                                   {self.input_name: self.preprocessing([img])[0]})
                pred = onnx_result[0]
                pred = self.postprocessing(prediction_bboxes=pred,
                                           imgs=[img],
                                           conf_thresh=conf_thresh,
                                           iou_thresh=iou_thresh,
                                           max_detection=max_detection)
                preds.append(pred[0])
            return preds

        else:
            input_imgs = self.preprocessing(imgs)
            input_imgs = np.concatenate(input_imgs, axis=0)
            onnx_result = self.ort_session.run([], {self.input_name: input_imgs})
            pred = onnx_result[0]
            pred = self.postprocessing(prediction_bboxes=pred,
                                       imgs=imgs,
                                       conf_thresh=conf_thresh,
                                       iou_thresh=iou_thresh,
                                       max_detection=max_detection)
        return pred
