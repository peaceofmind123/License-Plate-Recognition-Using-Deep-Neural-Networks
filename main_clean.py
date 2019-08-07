import Yolo_high as yolo
import localizer_high as localizer
import ocr_api as ocr
import sys
import cv2
import numpy
import os
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


    # main loop
    while cap.isOpened():
        ret, frame = cap.read()


        # perform object detection
        vehicle_imgs, vbbox, bbox = yolo.get_vehicle_imgs(frame, 750, 700)

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
