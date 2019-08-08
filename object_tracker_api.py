import my_yolov3_api as yolo
import cv2
import numpy as np
import time
import os



MAX_DIST = 200
tracked_vehicles = {}
id_for_new_vehicle = 0

def check_if_close_enough(cx0, cy0, cx1, cy1):
    distance = np.sqrt((cx0-cx1)**2 + (cy0-cy1)**2 )
    return distance,  distance < MAX_DIST

def track_vehicle(boxes):
    global tracked_vehicles, id_for_new_vehicle

    new_tracked = {}
    for b in boxes:
        area = (b[0] - b[2]) * (b[1] - b[3])
        cx, cy = (b[0] + b[2])/2 , (b[1] + b[3])/2
        
        # checking with previous boxes to track id
        min_dist = 9999999
        min_dist_id = None
        for id, (area1, cx1, cy1) in tracked_vehicles.items():
            # check if this box matches any of the tracked vehicles
            dist, close = check_if_close_enough(cx, cy, cx1, cy1)
            if close and dist < min_dist:
                min_dist = dist
                min_dist_id = id

        if min_dist_id is not None: # closest object is found
            my_id = min_dist_id
        else:
            my_id = id_for_new_vehicle
            id_for_new_vehicle +=1    
        
        new_tracked[my_id] = (area, cx, cy)

    tracked_vehicles = new_tracked
    return tracked_vehicles, id_for_new_vehicle

def get_bbox_without_class(bbox):
    box_list = []
    for clas, box in bbox.items():
        box = box.numpy()
        for b in box:
            box_list.append(b)
    return box_list


def check_exact_match(cx0,cy0, bbox):
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    dist, _ = check_if_close_enough(cx0,cy0,cx,cy)
    return dist < 4