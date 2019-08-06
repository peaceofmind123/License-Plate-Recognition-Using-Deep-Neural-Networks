import numpy as np
import cv2


def get_character_bboxes(binary_img, min_points=100, max_points=1500):
    """
    It does connected component analysis on the binary image of license plate.
    The labels for each component are filtered to obtain list of suspected characters

    :param binary_img: image of license plate to be processed, must be white on black image
    :param min_points: minimum number of points for a character
    :param max_points: maximum number of points for a character
    :return: list of bbox for each character

    """
    min_height = binary_img.shape[0] // 6
    max_width = binary_img.shape[1] // 4

    labels_no, labels = cv2.connectedComponents(binary_img)

    filtered_img = np.zeros_like(labels)
    count = 0

    bbox_list = []

    for i in range(labels_no):
        positions = (labels == i)
        no_points = positions.sum()

        if min_points < no_points < max_points:
            points = np.where(positions)
            bbox = cv2.boundingRect(np.array(tuple(zip(*points))))

            if bbox[2] > min_height and bbox[3] < max_width and bbox[3] < 2 * bbox[2]:
                bbox_list.append(bbox)
                count += 1
                filtered_img[positions] = count

    return bbox_list
