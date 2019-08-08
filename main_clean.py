import Yolo_high as yolo
import localizer_high as localizer
import ocr_api as ocr
import sys
import cv2
import numpy
import os
import object_tracker_api as obj_tracker
from Vehicle import Vehicle
# constants
VIDEO_FILENAME = '1.MOV'

def main():
    frame_no = 0

    # initialization
    yolo.initialize_model()
    localizer.initialize_model()

    # load video
    video_file_path = os.path.join(os.getcwd(), 'video',VIDEO_FILENAME)
    cap = cv2.VideoCapture(video_file_path)

    tracked_vehicle_ids = []
    current_tracked_id = 0
    tracked_vehicles = [] # stores the actual vehicle object
    # main loop
    while cap.isOpened():
        ret, frame = cap.read()


        # perform object detection
        vehicle_imgs, vbbox, bbox = yolo.get_vehicle_imgs(frame, 750, 700)

        # perform object tracking
        tracked_vehicles_info, id_for_new_vehicle = obj_tracker.track_vehicle(obj_tracker.get_bbox_without_class(bbox))
        for key in tracked_vehicles_info.keys():
            _, centroid_x, centroid_y = tracked_vehicles_info[key]
            try:
                v_index = list(map(lambda obj: obj.id,tracked_vehicles)).index(key)
                # vehicle is present in the current list of vehicles
                vehicle = tracked_vehicles[v_index]

            except ValueError: # vehicle is not found
                vehicle = Vehicle(key)
                tracked_vehicles.append(vehicle)

            vehicle.current_bounding_box_centroid = (centroid_x, centroid_y)



        # height to width heuristic to remove partial images
        for vehicle_img in vehicle_imgs:
            if vehicle_img.shape[0] > 1.5* vehicle_img.shape[1]:

                vehicle_imgs.remove(vehicle_img)

        if len(vehicle_imgs) == 0: # if no vehicle was detected, just skip
            frame_no += 1
            cv2.imshow('prediction',frame)
            continue

        # perform license plate localization
        lp_imgs, lp_bbox, frame = localizer.predict_license_plate(vehicle_imgs, frame, vbbox)

        # perform ocr
        ocr_outputs = []
        for lp_img in lp_imgs:

            # convert image to grayscale
            lp_img = cv2.cvtColor(lp_img,cv2.COLOR_RGB2GRAY)
            ocr_outputs.append(ocr.predict_ocr(lp_img) if not None else '')

        # processing done, time for outputs
        print(frame_no, ocr_outputs)

        # create the frame to display
        frame = yolo.draw_bbox(frame, bbox)

        # resize the frame for display
        frame = cv2.resize(frame, (1080,700))
        cv2.imshow("prediction", frame)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

        frame_no += 1


if __name__ == '__main__':
    main()
