# Face-AntiSpoofing
Face Anti-Spoofing project. Lanit-Tercom summer school 2022

### DataSet 
Training was performed on the *CelebA Spoof* dataset ([GitHub](https://github.com/ZhangYuanhan-AI/CelebA-Spoof) | [Kaggle](https://www.kaggle.com/datasets/attentionlayer241/celeba-spoof-for-face-antispoofing)).    

<details><summary>Dataset Preparation</summary>
<p>

#### Cropping Faces

```sh
   python data_preparation.py --spoof_filter 0 1 2 3 7 8 9
```

</p>
</details> 

`archive.zip` - archive with original dataset

2 versions of cropped dataset are prepared from original one. 
The crops were made according to bbox markup, which is recorded in the files '*_BB.txt' for each photo
with a tolerance of 15% on each side.

data128.zip - crop scaled to 128 pixels on the largest side. 

data256.zip - crop scaled to 256 pixels on the largest side.
              If the size of the cropped image was smaller than 256 on the largest side, it was not scaled

Data preparation
```sh
   python data_preparation.py --spoof_filter 0 1 2 3 7 8 9
```