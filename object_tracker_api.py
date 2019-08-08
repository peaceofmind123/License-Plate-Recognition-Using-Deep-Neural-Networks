import numpy as np
import cv2
import os
MAX_DIST = 2 # if the two objects provided are more than MAX_DIST, consider them different


def check_if_close_enough(cx0, cy0, cx1, cy1):
    distance = np.sqrt((cx0 - cx1) ** 2 + (cy0 - cy1) ** 2)
    return distance, distance < MAX_DIST

# used to find which bounding box corresponds to the returned centroid on the same frame
def check_exact_match(cx,cy, bbox):
    cx1,cx2 = (bbox[2]+bbox[0]) / 2, (bbox[3]+bbox[1]) / 2
    distance, _ = check_if_close_enough(cx,cy, cx1, cx2)
    if distance < 4: # empirical.. theoretically should be exactly 0
        return True
    else:
        return False


def track_vehicle(vbboxes, tracked_vehicles=None, id_for_new_vehicle=0):
    """
    Tracks vehicles across frames
    Recursive function that needs to be provided state info through params
    :param vbboxes: the set of vehicle bounding boxes seen in two frames
    :param tracked_vehicles: previously tracked vehicles
    :param id_for_new_vehicle: first id to be given to a new vehicle
    :return:
    """

    if tracked_vehicles is None:
        tracked_vehicles = {}
    new_tracked = {}

    for b in vbboxes:
        area = (b[0] - b[2]) * (b[1] - b[3])
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2

        # checking with previous boxes to track id
        min_dist = 9999999
        min_dist_id = None
        for id, (area1, cx1, cy1) in tracked_vehicles.items():
            # check if this box matches any of the tracked vehicles
            dist, close = check_if_close_enough(cx, cy, cx1, cy1)
            print(dist, close)
            if close and dist < min_dist:
                min_dist = dist
                min_dist_id = id

        if min_dist_id is not None:  # closest object is found
            my_id = min_dist_id
        else:
            my_id = id_for_new_vehicle
            id_for_new_vehicle += 1

        new_tracked[my_id] = (area, cx, cy)

    tracked_vehicles = new_tracked
    return tracked_vehicles, id_for_new_vehicle # use the returned id in subsequent calls


def get_bbox_without_class(bbox):
    box_list = []
    for clas, box in bbox.items():
        box = box.numpy()
        for b in box:
            box_list.append(b)
    return box_list
