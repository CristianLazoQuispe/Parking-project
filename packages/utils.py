import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np
from imantics import Mask, Polygons

import parameters


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
    image_draw = image.copy()

    for contours,state_is_free in zip(points,states_is_free):
        if state_is_free:
            image_draw = cv2.fillPoly(image_draw, pts =[contours], color=parameters.COLOR_SPACE_FREE)
        else:
            image_draw = cv2.fillPoly(image_draw, pts =[contours], color=parameters.COLOR_SPACE_NOT_FREE)
        image_draw = cv2.polylines(image_draw, pts =[contours], isClosed=True, color=parameters.COLOR_EDGES, thickness=2)

    output_image = cv2.addWeighted(image_draw, alpha, output_image, 1 - alpha,0, output_image)

    return output_image


def plot_cars(image,points,alpha=0.1):
    '''
        Input : image, polygs of cars
        Output: image with draws
    '''
    output_image = image.copy()

    for contours in points:
        contours = np.array(contours)
        image = cv2.fillPoly(image, pts =[contours], color=parameters.COLOR_CARS)
        image = cv2.polylines(image, pts =[contours], isClosed=True, color=parameters.COLOR_EDGES, thickness=2)
    output_image = cv2.addWeighted(image, alpha, output_image, 1 - alpha,0, output_image)

    return output_image


def get_car_points(rgb_image,model):
    '''
        Input: image, maskrcnn model

        Output: car images, points
    '''
    
    results = model.detect([rgb_image], verbose=0)
    r = results[0]
    car_points = []

    for i in range(r['rois'].shape[0]):
        if parameters.original_class_names[r['class_ids'][i]] in parameters.CLASS_NAMES:
            mask_image = np.array(r['masks'][:, :, i]*1.0)
            polygons = Mask(mask_image).polygons()
            car_points.append(polygons.points[0])
    return car_points


def DrawPolygon(ImShape,Polygon,Color):
    '''
        Draw polygon in image
    '''
    Im = np.zeros(ImShape, np.uint8)
    try:
                 cv2.fillPoly(Im, Polygon, Color) # Only using this function may cause errors, I don’t know why
    except:
        try:
            cv2.fillConvexPoly(Im, Polygon, Color)
        except:
            print('cant fill\n')
 
    return Im

def get_dice(bgr_image,points1,points2):

    ImShape = bgr_image.shape

    mask_space =DrawPolygon(ImShape,points1,122)
    mask_car =DrawPolygon(ImShape, points2, 133) 
    join_mask = mask_space+mask_car


    ret, OverlapIm = cv2.threshold(join_mask, 200, 255, cv2.THRESH_BINARY)#According to the above filling value, so the pixel value in the new image is 255 as the overlapping place
    IntersectArea=np.sum(np.greater(OverlapIm, 0))#Find the area of ​​the overlapping area of ​​two polygons

    area_space = np.sum(np.greater(mask_space, 0)) # Area of space
    area_car = np.sum(np.greater(mask_car, 0)) # Area of car
    dice = 2*IntersectArea/(area_space + area_car) # Calculate Dice metric of overlaping

    return dice



def get_states(bgr_image,car_points,park_points,threshold=0.35):
    '''
    Compute if a space has overlap with a car with a dice metric

        Input : image, points of cars, points of spaces, threshold

        Output : array of boolen if the space is free or not

    '''


    visited_car = [False for i in range(len(car_points))]

    states_is_free = [True for i in range(len(park_points))]

    for idx in range(len(park_points)):
        aux_park_point = park_points[idx]

        for idy in range(len(car_points)):
            aux_car_point = car_points[idy]
            if not visited_car[idy]:
                dice = get_dice(bgr_image,aux_park_point,aux_car_point)
                if dice>threshold:
                    visited_car[idy] = True
                    states_is_free[idx] = False
                    break
    return states_is_free


def get_states_optimizado(bgr_image,car_points,park_points,threshold=0.35):
    '''
    Compute if a space has overlap with a car with a dice metric

        Input : image, points of cars, points of spaces, threshold

        Output : array of boolen if the space is free or not

    '''

    ImShape = bgr_image.shape

    mask_cars = np.zeros(ImShape, np.uint8)

    for idy in range(len(car_points)):
        aux_car_point = car_points[idy]
        mask_space =DrawPolygon(ImShape,aux_car_point,122)
        mask_cars+=mask_space
        

    states_is_free = [True for i in range(len(park_points))]

    for idx in range(len(park_points)):
        aux_park_point = park_points[idx]


        mask_space =DrawPolygon(ImShape,aux_park_point,122)

        join_mask = mask_space+mask_cars


        ret, OverlapIm = cv2.threshold(join_mask, 200, 255, cv2.THRESH_BINARY)#According to the above filling value, so the pixel value in the new image is 255 as the overlapping place
        IntersectArea=np.sum(np.greater(OverlapIm, 0))#Find the area of ​​the overlapping area of ​​two polygons
        area_space = np.sum(np.greater(mask_space, 0)) # Area of space
        overlap = IntersectArea/(area_space) # Calculate overlap metric Area interseccion/ Area del space parking
        if overlap>threshold:
            states_is_free[idx]=False

    return states_is_free

