import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob

import random
import pandas as pd
from OCR.summer.a import *
from OCR.summer.b import *


def predict_ocr(img):
    # section 1
    h, w = img.shape
    img = img[12: h - 8, 10: w - 10]
    # img = cv2.resize(img, (int(w / h * 70), 70))
    min_height = img.shape[0] // 6
    max_width = img.shape[1] // 4
    unravelled_img = img.ravel()
    max_ = unravelled_img.max()
    img = np.clip(img, 0, max_ - 10)
    min_ = unravelled_img.min()
    img = img.astype(int)
    img = (img - min_) * 255 / (max_ - min_)
    img = img.astype(np.uint8)

    #section 2
    threshold_val, thres = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
    thres = cv2.threshold(img, threshold_val + 45, 255, cv2.THRESH_BINARY)[1]

    #section 3
    retval, labels = cv2.connectedComponents(thres)

    filtered_img = np.zeros_like(labels)
    color = 20

    bbox_list = []

    for i in range(retval):
        positions = (labels == i)
        no_points = positions.sum()

        if no_points > 30 and no_points < 1000:
            points = np.where(positions)
            bbox = cv2.boundingRect(np.array(tuple(zip(*points))))

            if bbox[2] > min_height and bbox[3] < max_width and no_points < bbox[2] * bbox[3] * 0.85:
                bbox_list.append(bbox)
                color += 2
                filtered_img[positions] = color


    filtered_img = filtered_img / color * 255

    #section 4
    if len(bbox_list) == 0:
        return None
    bbox_list_only_characters = bbox_list.copy()
    bbox_list_only_characters = np.array(bbox_list_only_characters)
    bbox_list_only_characters[:, 2] = bbox_list_only_characters[:, 0] + bbox_list_only_characters[:, 2]
    bbox_list_only_characters[:, 3] = bbox_list_only_characters[:, 1] + bbox_list_only_characters[:, 3]
    corner_points = np.concatenate([bbox_list_only_characters[:, 0:2], bbox_list_only_characters[:, 2:]])
    y1c, x1c, hc, wc = cv2.boundingRect(corner_points)
    cropped_only_characters_img = img[y1c:y1c + hc, x1c:x1c + wc]
    img = cropped_only_characters_img
    threshold_val, thres = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
    thres = cv2.threshold(img, threshold_val + 35, 255, cv2.THRESH_BINARY)[1]

    #section 5
    retval, labels = cv2.connectedComponents(thres)

    filtered_img = np.zeros_like(labels)
    count = 10

    bbox_list = []

    for i in range(retval):
        positions = (labels == i)
        no_points = positions.sum()

        if no_points > 20 and no_points < 1000:
            points = np.where(positions)
            bbox = cv2.boundingRect(np.array(tuple(zip(*points))))

            if bbox[2] > (min_height - 10) and bbox[3] < max_width and no_points < bbox[2] * bbox[3] * 0.85:
                bbox_list.append(bbox)
                count += 2
                filtered_img[positions] = count


    filtered_img = filtered_img / count * 255
    return b(bbox_list, img, net, classes, transform)