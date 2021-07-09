import base64
import io
from posixpath import dirname
from shapely.geometry import Polygon as shapely_poly
from shapely.geometry import box
import argparse
import pickle
from pathlib import Path
from mrcnn.model import MaskRCNN
import mrcnn.utils
import mrcnn.config
import cv2
import numpy as np
#import git
import os

# if not os.path.exists("Mask_RCNN"):
#     print("Cloning M-RCNN repository...")
#     git.Git("./").clone("https://github.com/matterport/Mask_RCNN.git")


class Config(mrcnn.config.Config):
    NAME = "model_config"
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    NUM_CLASSES = 81



config = Config()
config.display()

COCO_MODEL_PATH = os.path.join("../Model/", "mask_rcnn_coco.h5")

print("Download in ",COCO_MODEL_PATH)

dirName = "../Model/"

if not os.path.exists(dirName):
    os.makedirs(dirName)
    print("Directory " , dirName ,  " Created ")
else:    
    print("Directory " , dirName ,  " already exists")   

if not os.path.exists(COCO_MODEL_PATH):
    mrcnn.utils.download_trained_weights(COCO_MODEL_PATH)