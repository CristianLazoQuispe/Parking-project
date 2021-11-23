import sys

module_path = "../packages"
if module_path not in sys.path:
    sys.path.append(module_path)
    
import os
import mrcnn.config
import mrcnn.utils
import parameters

class Config(mrcnn.config.Config):
    NAME = "model_config"
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    NUM_CLASSES = 81



config = Config()
config.display()

COCO_MODEL_PATH = os.path.join(parameters.MODEL_PATH,parameters.MODEL_NAME)

print("Download in ",COCO_MODEL_PATH)

dirName = parameters.MODEL_PATH

if not os.path.exists(dirName):
    os.makedirs(dirName)
    print("Directory " , dirName ,  " Created ")
else:    
    print("Directory " , dirName ,  " already exists")   

if not os.path.exists(COCO_MODEL_PATH):
    mrcnn.utils.download_trained_weights(COCO_MODEL_PATH)
