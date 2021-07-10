import argparse
import base64
import io
#import git
import os
import pickle
from pathlib import Path
from posixpath import dirname

import cv2
import mrcnn.config
import mrcnn.utils
import numpy as np
from mrcnn.model import MaskRCNN
from shapely.geometry import Polygon as shapely_poly
from shapely.geometry import box


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
