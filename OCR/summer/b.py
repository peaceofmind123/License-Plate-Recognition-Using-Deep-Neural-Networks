import pandas as pd
import torch
import cv2
import numpy as np
from PIL import Image


def b(bbox_list, thres, net, classes, transform):
    df = pd.DataFrame(bbox_list, columns=['y1', 'x1', 'h', 'w'])
    df = df.sort_values(by=['x1', 'y1'])

    df['y2'] = df['y1'] + df['h']
    df['x2'] = df['x1'] + df['w']
    df2 = df.drop(columns=['h', 'w'])

    lines2 = [[tuple(df2.iloc[0, [1, 0, 3, 2]].values)]]

    for row_no in range(len(df2.values))[1:]:
        x1, y1, x2, y2 = df2.iloc[row_no, [1, 0, 3, 2]].values
        added = False

        for line_no in range(len(lines2)):
            x1_, y1_, x2_, y2_ = lines2[line_no][-1]
            y1_min, y1_max = min(y1, y1_), max(y1, y1_)
            y2_min, y2_max = min(y2, y2_), max(y2, y2_)
            IOU = (y2_min - y1_max) / (y2_max - y1_min)
            if IOU > .6:
                lines2[line_no].append((x1, y1, x2, y2))
                added = True
                break

        if not added:
            lines2.append([(x1, y1, x2, y2)])

    vertical_alignment = [(line[0][1], i) for i, line in enumerate(lines2)]
    sorted_alignment = sorted(vertical_alignment, key=lambda x: x[0])
    lines2 = [lines2[i[1]] for i in sorted_alignment]
    binary_img = 255 - thres

    characters = []
    inputs = []

    image_rgb = Image.new("RGB", (50, 50))
    for line in lines2:
        for x1, y1, x2, y2 in line:
            character_img = binary_img[y1:y2, x1:x2]
            character_img = cv2.resize(character_img, (34, 41))
            final_img = np.zeros((50, 50)) + 255
            final_img[4:4 + 41, 8:34 + 8] = character_img

            characters.append(final_img)
            image = Image.fromarray(final_img.astype('uint8'), 'L')
            image_rgb.paste(image)
            inp = transform(image_rgb)
            inputs.append(inp)

    if len(inputs) == 1:
        inputs = inputs.view((1, 3, 50, 50))
    else:
        inputs = torch.stack(inputs)

    out = net(inputs)
    outputs = out.argmax(1).tolist()
    characters = [classes[i] for i in outputs]

    return ' '.join(characters)
