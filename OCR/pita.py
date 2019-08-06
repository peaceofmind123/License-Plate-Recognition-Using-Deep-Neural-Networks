from predictor import *
from utilities import *
import os

if __name__ == '__main__':
    img = cv2.imread('data/2.png', cv2.IMREAD_GRAYSCALE)
    binary_img = cv2.threshold(img, 140, 255, cv2.THRESH_BINARY)[1]

    net = Net.build_network()
    net.load_weights(os.path.join(os.getcwd(),'weights','classifier'))

    bbox_list = get_character_bboxes(binary_img)
    lines = parse_line(bbox_list)
    print(lines)

    inputs = assemble_character_images(lines, binary_img)
    output = predict_characters(net, inputs)
    print(output)