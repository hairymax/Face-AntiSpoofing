# Face-AntiSpoofing
Face Anti-Spoofing project. Lanit-Tercom summer school 2022

**Spoofing attack** - an attempt to deceive the identification system by presenting it with a fake image of a face

**Facial anti-spoofing** is the task of preventing false facial verification by using a photo, video, mask or a different substitute for an authorized person’s face.

- **Print attack**: The attacker shows the picture of other person printed on a sheet of paper 
- **Replay attack**: The attacker shows the screen of another device that plays a pre-recorded photo/video of the other person.
## Demo
https://user-images.githubusercontent.com/6792913/190157993-5c484b2f-834b-4914-b230-4a92021b8ef4.mp4

To process your videofile for detecting spoofing attacks run
```sh
python video_predict.py -i path/to/video.mp4 -o path/to/out_video.mp4 -m path/to/model.onnx -t 0.75 
```
Be sure to replace the arguments with your own. Running the script without the `-i` argument will start the prediction on the video stream from the webcam. 
<details><summary> video_predict.py arguments details:</summary>
<p>

`--input` (`-i`) - Path to video for predictions, if not passed webcam stream will be processed  
`--output` (`-o`) - Path to save processed video, if not passed the processed video will not be saved   
`--model_path` (`-m`) - Path to pretrained ONNX model    
`--threshold` (`-t`) - Real face probability threshold above which the prediction is considered true, default 0.5    
</p>
</details>

Pretrained models located in the directory *.\saved_models*

## DataSet 
Training was performed on the *CelebA Spoof* dataset ([GitHub](https://github.com/ZhangYuanhan-AI/CelebA-Spoof) | [Kaggle](https://www.kaggle.com/datasets/attentionlayer241/celeba-spoof-for-face-antispoofing)).
## Model
In this project we train and test the CNN models with architecture presented in [Silent-Face-Anti-Spoofing GitHub repository](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/) to detect Spoof attacks. The model architecture consists of the main branch of classification of attack type and the auxiliary supervision branch of Fourier spectrum. The Fourier transform is used only in the training stage.
### Example images of train loader with Fourier spectrum
![image](https://user-images.githubusercontent.com/6792913/190170749-437e443d-0ce7-43e6-9238-f4b2212da348.png)
## Tasks
Training was performed for two types of classification tasks:
1. **Live Face** / **Spoof Attack** (binary classification)
2. **Live Face** / **Print Attack** / **Replay Attack**  

Examples and the results of model training you can find in Jupyter Notebook 
### [FaceAntiSpoofing.ipynb Gist](https://gist.github.com/hairymax/021a8cd550a3c0fa14c8e6ae815265c9) (or [via nbviewer](https://nbviewer.org/gist/hairymax/021a8cd550a3c0fa14c8e6ae815265c9))

Best models metrics on the test set

|Task | Accuracy | AUC-ROC | Precision | Recall | F1 score |
| ---- | ---- | ---- | ---- | ---- | ---- |
|Live / Spoof | 0.9292 | 0.986919 | 0.974257 | 0.904714 | 0.938198 |
|Live / Print / Replay | 0.9328 | 0.990246 | 0.982771 | 0.902694 | 0.941032 |

In order to evaluate metrics Print / Replay classes have been reduced to one Spoof class after predictions
## Using the code
### Step 1. Dataset Preparation
- Download [CelebA Spoof dataset](https://www.kaggle.com/datasets/attentionlayer241/celeba-spoof-for-face-antispoofing) to `./CelebA_Spoof` directory.
- Generate cropped dataset of faces.  Example:
```sh
python data_preparation.py --spoof_types 0 1 2 3 7 8 9 --bbox_inc 1.5 --size 128
```
Generates dataset of squared crops of faces with live, print and replay images of shape 3x128x128 in subdir `/data_1.5_128` of `./CelebA_Spoof_crop` directory.
<details><summary> data_preparation.py arguments details</summary>
<p>

`--spoof_types` - list of spoof types to keep, according to original labels:
   - `0`     - Live
   - `1`,`2`,`3` - Print
   - `4`,`5`,`6` - Paper Cut
   - `7`,`8`,`9` - Replay  

`--bbox_inc` - Image bbox increasing, value 1 makes no effect. Crops were made according to bbox markup, which is recorded in the files '\*_BB.txt' for each photo.     
`--size` - the size of the cropped image (height = width = `size`)   
`--orig_dir` - Directory with original Celeba_Spoof dataset (*'./CelebA_Spoof'* by default)    
`--crop_dir` - Directory to save cropped dataset (*'./CelebA_Spoof_crop'* by default)    
</p>
</details>

### Step 2. Train Model
An example of a training script call
```sh
python train.py --crop_dir data_1.5_128 --input_size 128 --batch_size 256 --num_classes 2
```
Trains the model with PyTorch, records metrics in *./logs/jobs/* with tensorboard, and stores the weights of the trained models in *./logs/snapshot/*
<details><summary> train.py arguments details</summary>
<p>

`--crop_dir` - Name of subdir with cropped images in *./CelebA_Spoof_crop* directory     
`--input_size` - Input size of images passed to model (height = width = `input_size`)   
`--batch_size` - Count of images in the batch    
`--num_classes` - **2** for binary or **3** for live-print-replay classification    
`--job_name` - Suffix for model name saved in snapshots dir    
</p>
</details>

### Step 3. Convert Model to ONNX format
```sh
python model_to_onnx.py path/to/model.pth num_classes
```

## Repository
The required libraries are listed in the file `requirements.txt`
```sh
pip install -r requirements.txt
```
### Sources (`./src`)
- `train_main.py` : *TrainMain* class for model training with checking on a validation sample.
- `dataset_loader.py` : *CelebADatasetFT*, *CelebADataset* classes for train (with Fourier spectrum) and test loaders, functions to get loaders, conversion of spoof attacks from microclasses to macroclasses 
- `NN.py` : CNN implementation
- `antispoof_pretrained.py` : classes for loading pretrained .pth models: *AntiSpoofPretrained* only for predictions, *AntiSpoofPretrainedFT* with branch of Fourier spectrum for the possibility of additional training 
- `config.py` : configuration classes for Train, Loading pretrained models and for Tests
- `FaceAntiSpoofing.py` : module for using ready-made ONNX models for predictions 

