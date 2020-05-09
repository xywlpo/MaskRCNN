## Introduction
This code is for learning maskrcnn algorithm from https://github.com/matterport/Mask_RCNN. I have make very detailed code annotation for it. And I have created a ros node of maskrcnn for robot grasping.
## Current Process
- [x] make detailed code annotation
- [x] write a ros node of maskrcnn
- [x] write a data augmentation python script to generate more data
## File Funciton
* mrcnn files：core algorithm of maskrcnn
* data_aug.py：data augmentation
* train_segment.py：model train and test 
* MaskRCNNTrainGraph.mmap：flow chart of maskrcnn algorithm
* segment_node.py: ros node of maskrcnn inference
## Install
You should install following libraries:
```
numpy scipy Pillow cython matplotlib scikit-image tensorflow>=1.3.0 keras>=2.0.8 opencv-python h5py imgaug IPython
```
## Model
- Pre-trained model：[Baidu Disk](https://pan.baidu.com/s/1PU-s1ymzfms9-O6xMk9Rtg)，code：8wiu
- Trained model for workpiece：[Baidu Disk](https://pan.baidu.com/s/1PFnDiM7bPzXg9nYhMNu4SA)，code：qzl9
The workpiece is as following:


