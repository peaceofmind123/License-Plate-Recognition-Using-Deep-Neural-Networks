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
    id_for_new_vehicle = 0
    # initialization
    yolo.initialize_model()
    localizer.initialize_model()
    tracked_vehicles_info = {}
    # load video
    video_file_path = os.path.join(os.getcwd(), 'video',VIDEO_FILENAME)
    cap = cv2.VideoCapture(video_file_path)

    tracked_vehicle_ids = []
    current_tracked_id = 0
    tracked_vehicles = [] # stores the actual vehicle object

    # main loop
    while cap.isOpened():
        ret, frame = cap.read()

        vehicle_ids_this_frame = []
        ocr_outputs_this_frame = []
        # perform object detection
        vehicle_imgs, vbbox, bbox = yolo.get_vehicle_imgs(frame, 750, 700)
        if len(vehicle_imgs) == 0 or len(vbbox) == 0 or len(bbox) == 0:
            continue
        # perform object tracking
        tracked_vehicles_info, id_for_new_vehicle = obj_tracker.track_vehicle(
            obj_tracker.get_bbox_without_class(bbox))
        print(id_for_new_vehicle)
        # create or update the vehicles tracked till now
        for key in tracked_vehicles_info.keys():
            _, centroid_x, centroid_y = tracked_vehicles_info[key]
            try:
                v_index = list(map(lambda obj: obj.id,tracked_vehicles)).index(key)
                # vehicle is present in the current list of vehicles
                vehicle = tracked_vehicles[v_index]

            except ValueError as e: # vehicle is not found
                vehicle = Vehicle(key)
                tracked_vehicles.append(vehicle)

            vehicle_ids_this_frame.append(vehicle.id)
            vehicle.current_bounding_box_centroid = (centroid_x, centroid_y)

            # find the vehicle's image in the current frame
            for v_img, v_bbox in zip(vehicle_imgs,vbbox):
                # vehicle_imgs and vbbox have one-to-one direct correspondence

                if obj_tracker.check_exact_match(vehicle.current_bounding_box_centroid[0],
                                                 vehicle.current_bounding_box_centroid[1],
                                                 v_bbox):
                    vehicle.img_current = v_img
                    vehicle.bbox_current = v_bbox
                    vehicle.bboxes.append(v_bbox)
                    vehicle.vehicle_imgs.append(v_img)

            # if vehicle has no corresponding image, no point to process it further
            if vehicle.img_current is None or vehicle.bbox_current is None:
                tracked_vehicles.remove(vehicle)

        if len(tracked_vehicles) == 0: # if no vehicle was detected, just skip
            frame_no += 1
            cv2.imshow('prediction',frame)
            continue

        for v in tracked_vehicles:
            # perform lp localization
            if v.id in vehicle_ids_this_frame:
                lp_imgs, lp_bboxes, frame = localizer.predict_license_plate([v.img_current],frame,[v.bbox_current])
                lp_img = lp_imgs[0]
                lp_img = cv2.cvtColor(lp_img,cv2.COLOR_RGB2GRAY)
                ocr_output = ocr.predict_ocr(lp_img)
                v.license_number_predictions.append(ocr_output if ocr_output is not None else '')
                ocr_outputs_this_frame.append(ocr_output)

        # processing done, time for outputs
        # print(frame_no, ocr_outputs_this_frame)
        l = len(tracked_vehicles)
        print(tracked_vehicles[l-1].license_number_predictions)
        # create the frame to display
        frame = yolo.draw_bbox(frame, bbox)
        for v in tracked_vehicles:
            if v.id in vehicle_ids_this_frame:
                cx = int(v.current_bounding_box_centroid[0])
                cy = int(v.current_bounding_box_centroid[1])
                frame = cv2.putText(frame,f'{v.id}',(cx,cy),cv2.FONT_HERSHEY_COMPLEX,4,(0,255,0),4)
        # resize the frame for display
        frame = cv2.resize(frame, (1080,700))
        cv2.imshow("prediction", frame)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

        frame_no += 1


if __name__ == '__main__':
    main()
