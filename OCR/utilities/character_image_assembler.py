import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import transforms

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def assemble_character_images(lines_, binary_img):
    """
    It extracts out images of single characters from the binary image of license plate and assembles into a
    tensor to be passed to neural network.

    :param lines_: Array of bboxes assembled in accordance to the lines of characters
    :param binary_img: binary image of extracted license plate, must be white in dark
    :return: tensor of (3, 50, 50) tensors

    """

    binary_img = 255 - binary_img
    inputs = []
    image_rgb = Image.new("RGB", (50, 50))

    for line in lines_:
        for x1, y1, x2, y2 in line:
            character_img = binary_img[y1:y2, x1:x2]
            character_img = cv2.resize(character_img, (34, 41))
            final_img = np.zeros((50, 50)) + 255

            final_img[4:4 + 41, 8:34 + 8] = character_img

            image = Image.fromarray(final_img.astype('uint8'), 'L')
            image_rgb.paste(image)
            inp = transform(image_rgb)
            inputs.append(inp)

    if len(inputs) == 1:
        inputs = inputs.view((1, 3, 50, 50))
    else:
        inputs = torch.stack(inputs)

    return inputs
