import my_yolov3_api as yolo
import cv2
import numpy as np
import time

videofile = '/media/tsuman/98D2644AD2642EA6/vehicle_videos/DSC_0051.MOV'
# videofile = '/media/tsuman/508EA2738EA251701/Mov/47.Ronin.2013.1080p.BluRay.x264.YIFY.mp4'

cap = cv2.VideoCapture(videofile)
frames = -1
yolo.load_model()

start=0
print('Video Capture ',cap)
print('Video opened ?',cap.isOpened())

while cap.isOpened():
    ret, frame = cap.read()
    frames +=1
    if frames%5 != 0: #skip some frames
        continue
        
    # frame = cv2.resize(frame, None,fx=0.5, fy=0.5)
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
    bbox = bboxes[0]

    # print(bbox)
    start = yolo.save_bbox_image(frame, bbox, start=start)
    print(f'========== stop = {start} ============')
    
    plotted = yolo.draw_bbox(frame, bbox)
    plotted = cv2.resize(plotted, None,fx=0.5, fy=0.5)

    cv2.imshow("prediction", plotted)
    # cv2.imwrite(f'det/temp/{frames}_img.png',plotted)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
            break
    # break

