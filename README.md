# Face-AntiSpoofing
Face Anti-Spoofing project. Lanit-Tercom summer school 2022

**Spoofing attack** - an attempt to deceive the identification system by presenting it with a fake image of a face

**Facial anti-spoofing** is the task of preventing false facial verification by using a photo, video, mask or a different substitute for an authorized person’s face.

- **Print attack**: The attacker shows the picture of other person printed on a sheet of paper 
- **Replay attack**: The attacker shows the screen of another device that plays a pre-recorded photo/video of the other person.

## DataSet 
Training was performed on the *CelebA Spoof* dataset ([GitHub](https://github.com/ZhangYuanhan-AI/CelebA-Spoof) | [Kaggle](https://www.kaggle.com/datasets/attentionlayer241/celeba-spoof-for-face-antispoofing)).
## Model
In this project we train and test the CNN model with architecture presented in [Silent-Face-Anti-Spoofing GitHub repository](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/) to detect Spoof attacks. The model architecture consists of the main branch of classification of attack type and the auxiliary supervision branch of Fourier spectrum. The Fourier transform is used only in the training stage.

## Tasks
We trained models for two types of classification tasks:
1. **Live Face** / **Spoof Attack** (binary classification)
2. **Live Face** / **Print Attack** / **Replay Attack**  
   
### Step 1. Dataset Preparation
Download [CelebA Spoof dataset](https://www.kaggle.com/datasets/attentionlayer241/celeba-spoof-for-face-antispoofing) to `./CelebA_Spoof` directory.
```sh
python data_preparation.py --spoof_types 0 1 2 3 7 8 9 --bbox_inc 1.5 --size 128
```
Generates cropped dataset of faces with live, print and replay images of shape 3x128x128 in.

`spoof_types` - Spoof types to keep:
   - `0`     - Live
   - `1`,`2`,`3` - Print
   - `4`,`5`,`6` - Paper Cut
   - `7`,`8`,`9` - Replay  

`bbox_inc` - Image bbox increasing, value 1 makes no effect. Crops were made according to bbox markup, which is recorded in the files '*_BB.txt' for each photo.  
`size` - size

### Step 2. Train Model
```sh
python train.py --crop_dir data_1.5_128 --input_size 128 --batch_size 256 --num_classes 2
```