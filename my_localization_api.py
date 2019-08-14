import numpy as np
# import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
# import matplotlib.pyplot as plt

# from torch.utils.data import Dataset
# from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from PIL import Image  # pip3 install pillow
import random
import cv2
# import time
import os
import copy

import copy
# ### Image Transform

class ResizeAspect(object):
    def __init__(self, h, w):
        self.hw = (h, w)
        self.rescale_factor = None
        self.shift_h = None
        self.shift_w = None

    def do_image(self, img):
        h, w = self.hw
        img_h, img_w = img.shape[0], img.shape[1]
        rescale_factor = min(w/img_w, h/img_h)
        new_w = int(img_w * rescale_factor)
        new_h = int(img_h * rescale_factor)
        resized_image = cv2.resize(
            img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        canvas = np.full((h, w, 3), 128, dtype=np.uint8)
        shift_h = (h-new_h)//2
        shift_w = (w-new_w)//2
        canvas[(h-new_h)//2:(h-new_h)//2 + new_h, (w-new_w) //
               2:(w-new_w)//2 + new_w, :] = resized_image
        img = canvas.copy()
        self.rescale_factor = rescale_factor
        self.shift_h = shift_h
        self.shift_w = shift_w
        return img

    def do_box(self, box):
        box = box.reshape(-1, 2)
        box *= self.rescale_factor
        box[:, 0] += self.shift_w
        box[:, 1] += self.shift_h
        box = box.reshape(-1)
        return box

    def undo_box(self, box):
        box = box.reshape(-1, 2)
        box[:, 0] -= self.shift_w
        box[:, 1] -= self.shift_h
        box /= self.rescale_factor
        box = box.reshape(-1)
        return box


class FinalTransform:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

    def transform_inv(self, img):
        inp = img.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        return inp


################################

# ### Defining Model
model = None


def load_model(saved_dir='model_state_v0.pth', CUDA=False):
    global model
    print("Loading LP-Localization network.....")
    model = models.resnet18(pretrained=True)
    '''
    output of our model is :
    x1, y1,
    x2, y2,
    x3, y3,
    x4, y4,
    conf -> only when no bounding box images are taken
    '''
    num_feature = model.fc.in_features
    num_output = 8  # 9
    model.fc = nn.Linear(num_feature, num_output)
    model = model.cpu()
    model.load_state_dict(torch.load(
        os.path.join(os.getcwd(),"saved_states",saved_dir), map_location='cuda' if CUDA else 'cpu'))
    model.eval()
    print("Network successfully loaded")
    return model


resizerG = ResizeAspect(h=224, w=224)
final_transform = FinalTransform()


def predict_bbox(img_list, batch_size=32):
    n = len(img_list)
    tensor_list = []
    resizer_list = []
    for index in range(len(img_list)):
        img = img_list[index]
        resizer = copy.deepcopy(resizerG)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = resizer.do_image(img)
        img = final_transform.transform(img)
        resizer_list.append(resizer)
        tensor_list.append(img)

    pred_list = []
    # combines tensors along new dimension
    tensor_imgs = torch.stack(tensor_list)
    batch_indx = 0
    while True:
        start = batch_indx*batch_size
        stop = (batch_indx+1)*batch_size
        if stop > n:
            stop = n

        inputs = tensor_imgs[start:stop]
        outputs = model(inputs)
        resizers = resizer_list[start:stop]

        outputs = outputs.data.numpy()
        for indx in range(len(outputs)):
            resizer = resizers[indx]
            out = outputs[indx]
            out = resizer.undo_box(out)

            # out = out.data.numpy()
            pred_list.append(out)

        if stop == n:
            break
    return pred_list


def draw_lp_box_onframe(img, vbbox, lplist):
    # print(len(vbbox), len(lplist))
    for vout, lout in zip(vbbox, lplist):
        # connecting the boxes (8+2 pts now)
        lout = np.append(lout, lout[:2]).reshape(-1, 2)
        # print(vout, lout)
        lout = lout + vout[[0, 1]]
        lout = lout.astype(int)
        # lout = lout.data.numpy()
        lout[0, 0] -= 5
        lout[3, 0] -= 5
        lout[1, 0] += 5
        lout[2, 0] += 5
        lout[0:2, 1] -= 5
        lout[2:4, 1] += 5
        lout[4, 0] -= 5
        lout[4, 1] -= 5
        for i in range(len(lout)-1):
            img = cv2.line(img, tuple(lout[i]), tuple(
                lout[i+1]), color=(0, 255, 100), thickness=4, lineType=cv2.LINE_AA)
    return img


def save_lp_box_onvehicle(vehicle_imgs, lp_list, start: int = 0):
    for img, lout in zip(vehicle_imgs, lp_list):
        # connecting the boxes (8+2 pts now)
        lout = np.append(lout, lout[:2]).reshape(-1, 2)
        lout = lout.astype(int)
        for i in range(len(lout)-1):
            img = cv2.line(img, tuple(lout[i]), tuple(
                lout[i+1]), color=(0, 255, 100), thickness=4, lineType=cv2.LINE_AA)
        cv2.imwrite(f"output/lp_detection/img_{start}.jpg", img)
        start += 1

    return start


def save_lp_box_only(vehicle_imgs, lp_list, start: int = 0):
    # Transformation will be applied to skewed license plate
    lp_imgs = get_transformed_lp_bbox_only(vehicle_imgs, lp_list)
    for img_out in lp_imgs:
        cv2.imwrite(f"output/lp_images/img_{start}.png", img_out)
        start += 1
    return start
    # for img, lout in zip(vehicle_imgs, lp_list):
    #     lout = lout.reshape(-1,2).astype(np.float32)
    #     widthx, heighty = lout[:,0].max()-lout[:,0].min(), lout[:,1].max()-lout[:,1].min()
    #     target = np.array([[0,0], [widthx, 0], [widthx, heighty], [0, heighty]], dtype=np.float32)
    #     pmatrix = cv2.getPerspectiveTransform(lout, target)
    #     img_out = cv2.warpPerspective(img, pmatrix, (widthx, heighty))
    #     cv2.imwrite(f"output/lp_images/img_{start}.png", img_out)
    #     start+=1
    # return start


def get_transformed_lp_bbox_only(vehicle_imgs, lp_list):
    # Transformation will be applied to skewed license plate
    lp_imlist = []
    for img, lout in zip(vehicle_imgs, lp_list):

        lout = lout.reshape(-1, 2).astype(np.float32)
        lout[0, 0] -= 20
        lout[3, 0] -= 20
        lout[1, 0] += 20
        lout[2, 0] += 20
        lout[0:2, 1] -= 10
        lout[2:4, 1] += 10
        # lout[4, 0] -= 20
        # lout[4, 1] -= 10

        widthx, heighty = lout[:, 0].max(
        )-lout[:, 0].min(), lout[:, 1].max()-lout[:, 1].min()
        target = np.array([[0, 0], [widthx, 0], [widthx, heighty], [
                          0, heighty]], dtype=np.float32)
        pmatrix = cv2.getPerspectiveTransform(lout, target)
        img_out = cv2.warpPerspective(img, pmatrix, (widthx, heighty))
        lp_imlist.append(img_out)
    return lp_imlist
