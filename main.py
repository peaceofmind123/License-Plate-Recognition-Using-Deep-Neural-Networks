import my_yolov3_api as yolo
import my_localization_api as localizer
import cv2
import numpy as np
import time
import torch
import torchvision.transforms as transforms
from OCR.predictor.model import *
from OCR.predictor.predict import *
from OCR.utilities.character_image_assembler import *
from OCR.utilities.get_character_bboxes import *
from OCR.utilities.line_parser import *
import my_segmentation_api as segment
import os

I = 1080

video_file = os.path.join(os.getcwd(), 'video', '2.MOV')

cap = cv2.VideoCapture(video_file)

# print(dir(cap))
# members = [attr for attr in dir(cap) if not callable(getattr(cap, attr)) and not attr.startswith("__")]
# print(members)
# exit()
# Define the codec and create VideoWriter object

frames = -1
# TODO: yolo
yolo.load_model()
CUDA = yolo.CUDA

# TODO: Localizer
localizer.load_model()

print('Video Capture ', cap)
print('Video opened ?', cap.isOpened())

# TODO: OCR
charNet = Net()
charNet.load_weights()
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

start = 0
counter = 1
counter1 = 0
while cap.isOpened():

    ret, frame = cap.read()


    frames += 1

    if frames % 7 != 0:
        continue

    # TODO: Yolo
    # frame = cv2.resize(frame, None,fx=0.5, fy=0.5)
    img, factor = yolo.resize_image(frame, yolo.imgsize)

    imgs = yolo.prepare_tensor_image(np.array([img]))
    preds = yolo.predict_images(imgs)

    bboxes = yolo.post_process_predictions(preds)
    if len(bboxes) == 0:  # if no prediction is made
        print('no prediction found')
        continue
    bboxes = yolo.select_objects(bboxes, indices=yolo.vehicles)
    bboxes = yolo.refactor_bboxes(bboxes, factors=[factor])
    bboxes = yolo.select_boxes_below(bboxes, below=750, below1=600)
    bbox = bboxes[0]
    # print("num_vehicles = ", len(bbox))
    if len(bbox) > 0:  # vehicle has been detected
        vehicle_imgs, vbbox = yolo.get_bbox_image(
            frame, bbox, return_bbox=True)
        # print(vehicle_imgs[0].shape)
        MAXIMUM_HEIGHT = 65 / 100 * I
        flag_vbbox_invalid = 0
        for vbbox_i in vbbox:

            height_of_vehicle = vbbox_i[3] - vbbox_i[1]
            if height_of_vehicle > MAXIMUM_HEIGHT:
                flag_vbbox_invalid = 1

        if flag_vbbox_invalid == 1:
            print('skipped')
            continue

        lp_bbox = localizer.predict_bbox(vehicle_imgs)

        # start = localizer.save_lp_box_only(vehicle_imgs, lp_bbox, start)
        lp_imgs = localizer.get_transformed_lp_bbox_only(vehicle_imgs, lp_bbox)

        cv2.imwrite(os.path.join(os.getcwd(), 'lp_detection',
                                 f'${counter1}.png'), lp_imgs[0])
        # print(len(lp_imgs), lp_imgs[0].shape)
        # print(len(lp_bbox))
        frame = localizer.draw_lp_box_onframe(frame, vbbox, lp_bbox)
        # localizer.save_lp_box_only(vehicle_imgs, lp_imgs, 0)
        counter1 += 1
        # performing ocr on the detected license plate
        char_imgs_list_rgb = segment.get_all_license_plate_chars(lp_imgs)

        for i, char_imgs in enumerate(char_imgs_list_rgb):
            if char_imgs is None:
                continue

            for char_img in char_imgs:
                fname = f'{counter}.png'
                # print(fname)
                cv2.imwrite(os.path.join(
                    os.getcwd(), 'char_imgs', fname), char_img)
                counter += 1
        char_imgs_list = segment.get_all_license_plate_chars_binary(lp_imgs)
        # print(len(char_box_list))

        # not_none_indx = []

        char_imgs_tensor = []
        for i, char_imgs in enumerate(char_imgs_list):
            if char_imgs is None:
                continue
            # not_none_indx.append(i)
            for char_img in char_imgs:
                char_img = cv2.cvtColor(char_img, cv2.COLOR_GRAY2RGB)

                char_img = cv2.resize(char_img, (50, 50))
                char_img = transform(char_img)
                char_imgs_tensor.append(char_img)
        if len(char_imgs_tensor) > 0:
            char_imgs_tensor = torch.stack(char_imgs_tensor)
            # print(char_imgs_tensor.shape)
            output = predict_characters(charNet, char_imgs_tensor)
            print(output)
            cv2.rectangle(frame, (180, 180), (820, 220), (0,0,0), -1)
            cv2.putText(frame, str(output), (200, 200), cv2.FONT_HERSHEY_COMPLEX, 1, (250, 250, 250), 1)
    # start = yolo.save_bbox_image(frame, bbox, start=start)
    print(f'========== stop = {start} ============')

    frame = yolo.draw_bbox(frame, bbox)

    # frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
    # cv2.imwrite(f'output/lp_detection/{frames}_img.png', frame)
    cv2.imshow("prediction", frame)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    # break

cap.release()
cv2.destroyAllWindows()
