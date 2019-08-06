import my_yolov3_api as yolo
import numpy as np

def get_vehicle_imgs(frame, below_left=750, below_right=600):
    """Get a list of vehicle images from a given frame
    :arg below_left: the line to cross by a vehicle on the left half to be considered
         below_right: the line to cross by a vehicle on the right half to be considered
    :returns vehicle_imgs: list of vehicle images as numpy arrays
             vbboxes: the vehicles' bounding boxes w.r.t. the frame supplied
             bbox: the actual bounding box image"""

    img, factor = yolo.resize_image(frame, yolo.imgsize)

    imgs = yolo.prepare_tensor_image(np.array([img]))
    preds = yolo.predict_images(imgs)

    bboxes = yolo.post_process_predictions(preds)
    if len(bboxes) == 0:  # if no prediction is made
        return [], [], []

    bboxes = yolo.select_objects(bboxes, indices=yolo.vehicles)
    bboxes = yolo.refactor_bboxes(bboxes, factors=[factor])
    bboxes = yolo.select_boxes_below(bboxes, below=750, below1=600)
    bbox = bboxes[0]

    if len(bbox) > 0:  # vehicle has been detected
        vehicle_imgs, vbboxes = yolo.get_bbox_image(
            frame, bbox, return_bbox=True)

        return vehicle_imgs, vbboxes, bbox

    else: return [], [], []
