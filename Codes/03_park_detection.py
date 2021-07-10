import sys

module_path = "../packages"
if module_path not in sys.path:
    sys.path.append(module_path)


import argparse
import os
import sys

import cv2
import matplotlib.pyplot as plt
import mrcnn.model as modellib
import numpy as np

from utils import *
import parameters
import coco

# Directory to save logs and trained model
MODEL_DIR = os.path.join(parameters.RESULTS_PATH, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(parameters.MODEL_PATH, "mask_rcnn_coco.h5")


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




parser = argparse.ArgumentParser()
parser.add_argument("-i",'--image_path', help="Path of image file", default='espacio-libre.jpg',type=str)
parser.add_argument("-p",'--point_path', help="Path of points file", default="regions.p",type=str)
args = parser.parse_args()



# Define parameters
points_filename = os.path.join(parameters.RESULTS_PATH,args.point_path)
image_filename = os.path.join(parameters.DATA_PATH,args.image_path)    

print("points_filename",points_filename)
print("image_filename",image_filename)

bgr_image = cv2.imread(image_filename)
rgb_image = cv2.cvtColor(bgr_image,cv2.COLOR_BGR2RGB)

park_points = read_points(points_filename)
car_points = get_car_points(rgb_image,model)
        


states_is_free = get_states_optimizado(bgr_image,car_points,park_points)
# plot points and states

park_spaces_image = plot_points(bgr_image.copy(),park_points,states_is_free,0.2)
park_spaces_image = plot_cars(park_spaces_image,car_points,alpha=0.5)

park_spaces_image = cv2.cvtColor(park_spaces_image,cv2.COLOR_BGR2RGB)

# show image
fig = plt.figure(figsize=(15,15))
plt.imshow(park_spaces_image)
plt.show()

name_img = image_filename.split('/')[-1].split('.')[0]
name_point = points_filename.split('/')[-1].split('.')[0]

park_spaces_image = cv2.cvtColor(park_spaces_image,cv2.COLOR_RGB2BGR)
cv2.imwrite(parameters.RESULTS_PATH+'Result_park_spaces_'+name_img+'_'+name_point+'_.jpg',park_spaces_image)


states_is_free = np.array(states_is_free)

n_free = int(np.sum(states_is_free*1.0))

print("Total park spaces : ",len(states_is_free))
print("Total free spaces : ",n_free)





