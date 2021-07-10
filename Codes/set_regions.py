import argparse
import os
import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from matplotlib.widgets import PolygonSelector
from shapely.geometry import Polygon as shapely_poly
from shapely.geometry import box

import parameters

points = []
prev_points = []
patches = []
total_points = []
breaker = False

# PARAMETERS INITIALIZATIONS

# COLOR TO PLOT
COLOR_SPACE_FREE     = parameters.COLOR_SPACE_FREE 
COLOR_SPACE_NOT_FREE = parameters.COLOR_SPACE_NOT_FREE 
COLOR_EDGES = parameters.COLOR_EDGES

#PATHS

DATA_PATH = parameters.DATA_PATH
RESULTS_PATH = parameters.RESULTS_PATH



class SelectFromCollection(object):
    def __init__(self, ax):
        self.canvas = ax.figure.canvas

        self.poly = PolygonSelector(ax, self.onselect)
        self.ind = []

    def onselect(self, verts):
        global points
        points = verts
        self.canvas.draw_idle()

    def disconnect(self):
        self.poly.disconnect_events()
        self.canvas.draw_idle()

def break_loop(event):
    global breaker
    global globSelect
    global savePath
    if event.key == 'b':
        globSelect.disconnect()
        if os.path.exists(savePath):
            os.remove(savePath)
        print('Total points :',total_points)
        print("data saved in "+ savePath + " file")    
        with open(savePath, 'wb') as f:
            pickle.dump(total_points, f, protocol=pickle.HIGHEST_PROTOCOL)
            exit()


def onkeypress(event):
    global points, prev_points, total_points
    if event.key == 'n': 
        pts = np.array(points, dtype=np.int32)   
        if points != prev_points and len(set(points)) == 4:
            print("Points : "+str(pts))
            patches.append(Polygon(pts))
            total_points.append(pts)
            prev_points = points

def get_image_from_video(video_path,n_frame = 5):

    video_capture = cv2.VideoCapture(video_path)
    cnt=0
    rgb_image = None
    while video_capture.isOpened():
        success, frame = video_capture.read()
        if not success:
            break
        if cnt == n_frame:
            rgb_image = frame
            break
        cnt += 1
        cv2.imshow('Frame',frame)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()
    cv2.imwrite('../Data/parking_space.jpg',rgb_image)

    return cv2.cvtColor(rgb_image,cv2.COLOR_BGR2RGB)

def read_points(filename):
    '''
    Input:
         - filename: str
    Output:
        - data : array of points
    '''
    print("Reading : ",filename)
    data = None
    with open(filename,'rb') as f:
        data = pickle.load(f)
        print("File ok ")
    return data


def plot_points(image,points,states_is_free,alpha=0.1):
    '''
    Input:
         - image: BGR matrix [:,:,3]
         - points: arrays of points 
            [ [[x1,y1],[x2,y2],...,[xn,yn]],
              [[x1,y1],[x2,y2],...,[xn,yn]],
              ..,
              [[x1,y1],[x2,y2],...,[xn,yn]]
            ]
        - states_is_free: arrays of boolean states
            True : is free
            False: not is free
        - alpha: int
          number of transparent overlays 
    Output:
        - image : BGR matrix with spaces

    '''
    output_image = image.copy()

    for contours,state_is_free in zip(points,states_is_free):
        if state_is_free:
            image = cv2.fillPoly(image, pts =[contours], color=COLOR_SPACE_FREE)
        else:
            image = cv2.fillPoly(image, pts =[contours], color=COLOR_SPACE_NOT_FREE)
        image = cv2.polylines(image, pts =[contours], isClosed=True, color=COLOR_EDGES, thickness=2)

    output_image = cv2.addWeighted(image, alpha, output_image, 1 - alpha,0, output_image)

    return output_image

def make_points(rgb_image):
    '''
        Input : RGB image

        Ouput: save points 
    '''
    while True:
        fig, ax = plt.subplots()
        image = rgb_image
        ax.imshow(image)
    
        p = PatchCollection(patches, alpha=0.7)
        p.set_array(10*np.ones(len(patches)))
        ax.add_collection(p)
            
        globSelect = SelectFromCollection(ax)
        bbox = plt.connect('key_press_event', onkeypress)
        break_event = plt.connect('key_press_event', break_loop)
        print(bbox)
        manager = plt.get_current_fig_manager()
        manager.window.showMaximized()

        plt.show()
        globSelect.disconnect()
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-v",'--video_path', help="Path of video file", default="stace_park3.mp4",type=str)
    parser.add_argument("-f",'--n_frame', help="Number of frame to process", default=5,type=int)
    parser.add_argument("-o",'--out_file', help="Name of the output file", default="regions.p",type=str)
    args = parser.parse_args()

    global globSelect
    global savePath

    # Define parameters
    n_frame = args.n_frame
    savePath = args.out_file if args.out_file.endswith(".p") else args.out_file+".p"
    savePath = os.path.join(RESULTS_PATH,savePath)
    video_path = os.path.join(DATA_PATH,args.video_path)    

    # Print parameters
    print("Path of video file         : ",video_path)
    print("Number of frame to process : ",n_frame)
    print("Name of the output file    : ",savePath)

    # Check if the points exits
    if os.path.exists(savePath):
        print('File already exists :',savePath)

        # get image 
        rgb_image = get_image_from_video(video_path,n_frame = n_frame)
        bgr_image = cv2.cvtColor(rgb_image,cv2.COLOR_RGB2BGR).copy()

        # get points        
        points = read_points(savePath)
        
        # simulate states
        states_is_free = [True for i in range(len(points))]

        # plot points and states
        output_image = plot_points(bgr_image,points,states_is_free,0.2)

        # show image
        cv2.imshow("Plot parking spaces ", output_image)
        cv2.waitKey()

        # save image
        filename_image = os.path.join(RESULTS_PATH,'plot_points_'+savePath.split('/')[-1].split('.')[0]+'_parking_spaces.jpg')
        cv2.imwrite(filename_image, output_image)

    else:
        print(10*'*','Make points',10*'*')
        print("\n> Select a region in the figure by enclosing them within a quadrilateral.")
        print("> Press the 'f' key to go full screen.")
        print("> Press the 'esc' key to discard current quadrilateral.")
        print("> Try holding the 'shift' key to move all of the vertices.")
        print("> Try holding the 'ctrl' key to move a single vertex.")
        print("> After marking a quadrilateral press 'n' to save current quadrilateral and then press 'q' to start marking a new quadrilateral")
        print("> When you are done press 'b' to Exit the program\n")
        

        rgb_image = get_image_from_video(video_path,n_frame = n_frame)
        
        make_points(rgb_image)
