import numpy as np
import my_localization_api as localizer

def initialize_model():
    localizer.load_model()

def predict_license_plate(vehicle_imgs, frame=None, vbboxes=None):
    """Predicts a license plate image given cropped vehicle images
        :arg vehicle_imgs: the list of vehicle images
        :arg frame: optional frame to draw the prediction on
        :arg vbboxes: optional list of vehicle bboxes drawn on the frame
        :returns lp_imgs: the license plate images
                 lp_bboxes: the license plate boxes
                 frame: the frame with the prediction drawn upon
    """

    lp_bboxes = localizer.predict_bbox(vehicle_imgs)
    lp_imgs = localizer.get_transformed_lp_bbox_only(vehicle_imgs, lp_bboxes)
    if frame is not None and vbboxes is not None:
        frame = localizer.draw_lp_box_onframe(frame,vbboxes,lp_bboxes)

    return lp_imgs, lp_bboxes, frame
