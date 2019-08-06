import my_yolov3_api as yolo
import cv2
import numpy as np
import time

videofile = '/media/tsuman/98D2644AD2642EA6/vehicle_videos/DSC_0051.MOV'

cap = cv2.VideoCapture(videofile)
frames = -1
yolo.load_model()

start=0
print('Video Capture ',cap)
print('Video opened ?',cap.isOpened())


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
            print(dist, close)
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
    return tracked_vehicles

def get_bbox_without_class(bbox):
    box_list = []
    for clas, box in bbox.items():
        box = box.numpy()
        for b in box:
            box_list.append(b)
    return box_list


while cap.isOpened():
    ret, frame = cap.read()
    frames +=1
    if frames%5 != 0: #skip some frames
        continue
        
    img, factor = yolo.resize_image(frame, yolo.imgsize)

    imgs = yolo.prepare_tensor_image(np.array([img]))

    preds = yolo.predict_images(imgs)

    bboxes = yolo.post_process_predictions(preds)
    if len(bboxes) == 0: # if no prediction is made
        print('no prediction found')
        continue    

    bboxes = yolo.select_objects(bboxes, indices=yolo.vehicles)
    bboxes = yolo.refactor_bboxes(bboxes, factors=[factor])
    bboxes = yolo.select_boxes_below(bboxes, below=600)
    if len(bboxes) == 0: # if no prediction is made
        print('no vehicles found')
        continue 
    bbox = bboxes[0]
    frame = yolo.draw_bbox(frame, bbox)
    
    bbox = get_bbox_without_class(bbox)
    track_info = track_vehicle(bbox)
    print(track_info)
    ## draw track info on the image
    for id, (area, cx, cy) in track_info.items():
        cx, cy = int(cx), int(cy)
        cv2.putText(frame, f'ID-{id}', (cx, cy), cv2.FONT_HERSHEY_PLAIN, 4, [0,255,0], 4)
    

    frame = cv2.resize(frame, None,fx=0.5, fy=0.5)
    cv2.imshow("prediction", frame)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
            break

