import cv2
import onnxruntime as ort
import numpy as np
import os

# onnx model
class AntiSpoof:
    def __init__(self,
                 weights: str = None,
                 model_img_size: int = 128):
        super().__init__()
        self.weights = weights
        self.model_img_size = model_img_size
        self.ort_session, self.input_name = self._init_session_(self.weights)

    def _init_session_(self, onnx_model_path: str):
        ort_session = None
        input_name = None
        if os.path.isfile(onnx_model_path):
            try:
                ort_session = ort.InferenceSession(onnx_model_path, 
                                                   providers=['CUDAExecutionProvider'])
            except:
                ort_session = ort.InferenceSession(onnx_model_path, 
                                                   providers=['CPUExecutionProvider']) 
            input_name = ort_session.get_inputs()[0].name
        return ort_session, input_name

    def preprocessing(self, img): 
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        new_size = self.model_img_size
        old_size = img.shape[:2] # old_size is in (height, width) format

        ratio = float(new_size)/max(old_size)
        scaled_shape = tuple([int(x*ratio) for x in old_size])

        # new_size should be in (width, height) format
        img = cv2.resize(img, (scaled_shape[1], scaled_shape[0]))

        delta_w = new_size - scaled_shape[1]
        delta_h = new_size - scaled_shape[0]
        top, bottom = delta_h//2, delta_h-(delta_h//2)
        left, right = delta_w//2, delta_w-(delta_w//2)

        img = cv2.copyMakeBorder(img, top, bottom, left, right, 
                                 cv2.BORDER_CONSTANT, value=[0, 0, 0])
        img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        return img

    def postprocessing(self, prediction):
        softmax = lambda x: np.exp(x)/np.sum(np.exp(x))
        pred = softmax(prediction)
        return pred
        #return np.argmax(pred)

    def __call__(self, imgs : list):
        if not self.ort_session:
            return False

        preds = []
        for img in imgs:
            onnx_result = self.ort_session.run([],
                {self.input_name: self.preprocessing(img)})
            pred = onnx_result[0]
            pred = self.postprocessing(pred)
            preds.append(pred)
        return preds