"""
Mask R-CNN
Train on the toy Balloon dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

Modified by xywlpo, 2018.6.1

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=imagenet

    # Apply color splash to an image
    python3 balloon.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 balloon.py splash --weights=last --video=<URL or path to file>
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import argparse
import cv2
import imgaug


# Root directory of the project
ROOT_DIR = os.path.abspath("./")

# Import Mask RCNN
#sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils, visualize

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


############################################################
#  Configurations
############################################################
class SegmentConfig(Config):

    # Give the configuration a recognizable name
    NAME = "Segment"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1
    
    # Numbers of GPUs (default is 1)
    GPU_COUNT = 1

    # Number of classes (including background)
    NUM_CLASSES = 2 + 1  # Background + Foreground

    # Number of training steps per epoch
    #STEPS_PER_EPOCH = 100
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.80


############################################################
#  Dataset
############################################################
class SegmentDataset(utils.Dataset):

    def load_object(self, dataset_dir, subset):
        
        # add classes which you have
        self.add_class("object", 1, "yp1")
        self.add_class("object", 2, "yp2")

        # train or validation dataset
        assert subset in ["train", "val", "test"]
        dataset_dir = os.path.join(dataset_dir, subset)
        image_ids = list(next(os.walk(dataset_dir))[1])

        # add images
        for image_id in image_ids:          
            if os.listdir(dataset_dir + "/" + image_id + "/images/") and os.listdir(dataset_dir + "/" + image_id + "/masks/"):
                self.add_image(
                    "object",
                    image_id=image_id,
                    path=os.path.join(dataset_dir, image_id, "images/{}.png".format(image_id)))

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        mask_dir = os.path.join(os.path.dirname(os.path.dirname(info['path'])), "masks")
        #print("mask_dir = ", mask_dir)
        mask = []
        label = []
        for f in next(os.walk(mask_dir))[2]:
            if f.endswith(".png"):
                m = skimage.io.imread(os.path.join(mask_dir, f)).astype(np.bool)
                if m.ndim == 3:
                    mask.append(m[:,:,0])
                else:
                    mask.append(m)
                class_name = f.split('_')
                #print('class-name = ', class_name)
                if class_name[0] == 'yp1':
                    label.append(1)
                elif class_name[0] == 'yp2':
                    label.append(2)
                
        mask = np.stack(mask, axis=-1)
        #print("mask shape = ", mask.shape)
        #print('mask_dir = ', mask_dir)
        #print("label = ", label)
        return mask, np.array(label)
            
    def image_reference(self, image_id):
        """ return the path of the image """
        info = self.image_info[image_id]
        if info["source"] == "object":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

############################################################
#  train
############################################################
def train(model):   

    # training dataset
    dataset_train = SegmentDataset()
    dataset_train.load_object(args.dataset, "train")
    dataset_train.prepare()

    # validation dataset
    dataset_val = SegmentDataset()
    dataset_val.load_object(args.dataset, "val")
    dataset_val.prepare()

    # we use pretrained coco weights to fine the parameters.
    # no need to train all layers, just the heads should do it
    print ("Training the network heads")
    model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=30, layers='heads')
    model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=30, layers="3+")
    model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE / 10, epochs=40, layers="all")
    


############################################################
#  detect
############################################################
def detect(model, frame):
    ## object detection and segmentation    
    imageRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    t1 = cv2.getTickCount()
    r = model.detect([imageRGB], verbose=0)[0]
    t2 = cv2.getTickCount()
    time = (t2-t1)*1000.0/cv2.getTickFrequency()
    print ("It costs ", time)

    ## display the results
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    merged = cv2.merge([gray,gray,gray])
    #mask = r['masks']
    #mask = (np.sum(mask, -1, keepdims=True) >= 1)
    display = merged
    
    if r['rois'].shape[0] > 0:
        #display = np.where(mask, frame, merged).astype(np.uint8)
        a = r['rois'].shape[0]
        print (r['class_ids'])
        
        for x in range(a):
            tmp_mask = r['masks'][:,:,x]
            binary_mask = tmp_mask.astype(np.uint8)
            binary_mask[binary_mask > 0] = 255
            merged_mask = cv2.merge([binary_mask,binary_mask,binary_mask])
            
            if r['class_ids'][x] == 1:
                str_disp = 'yp1' + '.' + str(round(r['scores'][x], 2))
                img_blue = np.zeros([1544, 2064, 3], np.uint8)
                img_blue[:, :, 0] = np.zeros([1544, 2064]) + 255
                img_blue_plus_org = img_blue*0.5 + merged*0.5
                display = np.where(merged_mask, img_blue_plus_org, display).astype(np.uint8)
            elif r['class_ids'][x] == 2:
                str_disp = 'yp2' + '.' + str(round(r['scores'][x], 2))
                img_other = np.zeros([1544, 2064, 3], np.uint8)
                img_other[:, :, 1] = np.zeros([1544, 2064]) + 100
                img_other[:, :, 2] = np.zeros([1544, 2064]) + 140
                img_other_plus_org = img_other*0.5 + merged*0.5
                display = np.where(merged_mask, img_other_plus_org, display).astype(np.uint8)
    
    newdisplay = cv2.resize(display, (640, 480))
    cv2.imshow("results", newdisplay)
    cv2.waitKey(0)  
    

def compute_mAP(model, inference_config):

    # validation dataset
    dataset_test = SegmentDataset()
    dataset_test.load_object(args.dataset, "test")
    dataset_test.prepare()
    
    image_ids = np.random.choice(dataset_test.image_ids, 1000)
    APs = []
    for image_id in image_ids:
        #original_image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset_test, inference_config, image_id, use_mini_mask=False)
        #visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id, dataset_train.class_names, figsize=(8, 8))
        image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset_test, inference_config, image_id, use_mini_mask=False)
        molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
        results = model.detect([image], verbose=0)
        r = results[0]
        if r['rois'].shape[0] > 0:
            #visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'], dataset_val.class_names, r['scores'])
            AP, precisions, recalls, overlaps = utils.compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'])
            APs.append(AP)
    
    print("mAP: ", np.mean(APs))
    
    
    
############################################################
#  main
############################################################
if __name__ == '__main__':

    #os.environ['CUDA_VISIBLE_DEVICES']='3,2'
    
    # command line
    parser = argparse.ArgumentParser(description='train mask-rcnn to detect objects')
    parser.add_argument("command")
    parser.add_argument('--dataset', default='./Datasets/aug/yuanpan_aug', required=False)
    parser.add_argument('--weights', default='./Models/yuanpan_model.h5', required=False)
    parser.add_argument('--logs', default='./log_yuanpan_tf1.7_newmaskrcnn', required=False)
    parser.add_argument('--image', required=False)
    args = parser.parse_args()

    class InferenceConfig(SegmentConfig):
        GPU_COUNT = 1

    # configurations
    if args.command == "train":
        config = SegmentConfig()
    else:   
        config = InferenceConfig()
    config.display()

    # create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config, model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config, model_dir=args.logs)

    # select weights to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        if not os.path.exists(weights_path):
            print("You need coco weights file!\n")
    elif args.weights.lower() == "last":
        weights_path = model.find_last()[1]
    elif args.weights.lower() == "imagenet":
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights
    
    if args.command == "train":
            model.load_weights(weights_path, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
    else:
            model.load_weights(weights_path, by_name=True)
        

    # train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "test":
        compute_mAP(model, config)
    elif args.command == "detect-video":
        capture = cv2.VideoCapture(0)
        while True:
            ret, frame = capture.read()
            detect(model, frame)
    elif args.command == "image":
        while True:
            strg = input("Enter your input: ")
            img = cv2.imread(strg)
            detect(model, img)  




