import math
import os
import random
import sys

import matplotlib
import matplotlib.pyplot as plt
import mrcnn.model as modellib
import numpy as np
import skimage.io
from mrcnn import utils, visualize

import coco
import parameters

# Directory to save logs and trained model
MODEL_DIR = os.path.join(parameters.RESULTS_PATH, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(parameters.MODEL_PATH, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(parameters.DATA_PATH, "")


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)


class_names = parameters.original_class_names#CLASS_NAMES


# Load a random image from the images folder
file_names = next(os.walk(IMAGE_DIR))[2]

print(file_names)

name_img = 'espacio-libre.jpg'#random.choice(file_names)

print('name_img : ',name_img)

image = skimage.io.imread(os.path.join(IMAGE_DIR, name_img))

# Run detection
results = model.detect([image], verbose=1)

# Visualize results
r = results[0]
print(r)

visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            class_names, r['scores'])


