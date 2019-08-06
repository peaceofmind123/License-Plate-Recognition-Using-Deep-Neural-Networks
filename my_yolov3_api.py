from darknet import *
import torch
import torch.nn as nn
import os
import os.path as osp
import cv2
import pickle as pkl
import random

cfgFileName: str = 'data/yolov3.cfg'
weightsFileName: str = 'data/yolov3.weights'  # download from : https://pjreddie.com/media/files/yolov3.weights
imageHeight: int = 416
imgsize: tuple = (imageHeight, imageHeight)
minConfidence: float = 0.6  # 0.5
nms_conf: float = 0.6  # 0.4
colors: list = pkl.load(open("data/pallete", "rb"))
SCREEN_WIDTH = 1920

def load_classes(namesfile):
    fp = open(namesfile, "r")
    names = fp.read().split("\n")[:-1]
    return names


classes: list = load_classes("data/coco.names")
# classes:list = ['car','motorbike','bus','truck',]
# vehicle_names:list = ['car','motorbike','bus','truck',]
# vehicles:list = [2,3,5,7,]
vehicle_names: list = ['car', 'bus', 'truck', ]
vehicles: list = [2, 5, 7, ]

numclasses: int = len(classes)

model: Darknet = None
CUDA: bool = torch.cuda.is_available()
device = torch.device("cuda:0" if CUDA else "cpu")


def load_model():
    global model
    print("Loading YOLOv3 network.....")
    model = Darknet(cfgFileName)

    model.load_weights(weightsFileName)
    print("Network successfully loaded")

    model.net_info["height"] = str(imageHeight)
    # model.set_classes(vehicles)

    inp_dim: int = imageHeight
    assert inp_dim % 32 == 0  # Limit the input image size
    assert inp_dim > 32

    # If there's a GPU availible, put the model on GPU
    if CUDA:
        print("Darknet running on GPU")
        model = model.cuda()
    else:
        print("Darknet running on CPU")
        model = model.cpu()
    # Set the model in evaluation mode
    model.eval()
    return model


def predict_images(image_batch: torch.Tensor):
    with torch.no_grad():
        prediction = model(image_batch, CUDA)
    return prediction


def file_names_from_folder(folder: str):
    try:
        imlist: list = [osp.join(osp.realpath('.'), folder, img) for img in os.listdir(folder)]
    except NotADirectoryError:
        imlist: list = []
        imlist.append(osp.join(osp.realpath('.'), folder))
    except FileNotFoundError:
        print("No file or directory with the name {}".format(folder))
        exit()
    return imlist


def save_images_to_folder(folder: str):
    if not os.path.exists(folder):
        os.makedirs(folder)


def resize_image(img, input_dim):
    img_h, img_w = img.shape[0], img.shape[1]

    h, w = input_dim
    # rescale to fit the target dimension
    rescale_factor = min(w / img_w, h / img_h)
    new_w = int(img_w * rescale_factor)
    new_h = int(img_h * rescale_factor)
    resized_image = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    canvas = np.full((h, w, 3), 128)
    shift_h = (h - new_h) // 2
    shift_w = (w - new_w) // 2
    canvas[(h - new_h) // 2:(h - new_h) // 2 + new_h, (w - new_w) // 2:(w - new_w) // 2 + new_w, :] = resized_image
    # BGR -> RGB | H X W C -> C X H X W 
    img = canvas[:, :, ::-1].transpose((2, 0, 1)).copy()
    return img, (rescale_factor, shift_w, shift_h)


# def preprocess_image_(img, inp_dim):
#     '''resize image with unchanged aspect ratio using padding'''
#     # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#     img_w, img_h = img.shape[1], img.shape[0]
#     w, h = inp_dim
#     new_w = int(img_w * min(w/img_w, h/img_h))
#     new_h = int(img_h * min(w/img_w, h/img_h))
#     resized_image = cv2.resize(img, (new_w,new_h), interpolation = cv2.INTER_CUBIC)

#     canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)
#     canvas[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w,  :] = resized_image
#     # BGR -> RGB | H X W C -> C X H X W 
#     img = canvas[:,:,::-1].transpose((2,0,1)).copy()

#     return img


def resize_images(images: list):
    imgs = []
    factors = []
    for image in images:
        img, factor = resize_image(image, (imageHeight, imageHeight))
        imgs.append(img)
        factors.append(factor)

    imgs: np.ndarray = np.array(imgs)
    factors: np.ndarray = np.array(factors)

    # prepared_imgs = list(map(resize_image, images, [(imageHeight, imageHeight) for _ in range(len(images))]))
    # print('--->',prepared_imgs)

    # prepared_imgs = []
    # for img in loaded_ims:
    #     img = prep_image(img, inp_dim)
    #     prepared_imgs.append(img, inp_dim)

    # print(len(prepared_imgs))
    # prepared_imgs:np.ndarray = np.array(prepared_imgs)
    # print(prepared_imgs.shape)
    return imgs, factors


def load_images(imageNames: list):
    loaded_ims = [cv2.imread(x) for x in imageNames]
    return loaded_ims


def prepare_tensor_image(img):
    """
    Prepare image for inputting to the torch neural network. 
    
    Returns a Variable 
    """
    img = torch.from_numpy(img).float().div(255.0)
    if CUDA:
        img = img.cuda()
    return img


# def post_process_predictions(prediction):
#     conf_mask = (prediction[:,:,4] > minConfidence).float().unsqueeze(2)
#     prediction = prediction*conf_mask
#     # conf_mask = (prediction[:,:,4] > minConfidence)
#     # print(conf_mask.shape, prediction.shape)
#     # print(prediction[conf_mask].shape)
#     box_corner = prediction[:,:,:4].data.clone()
#     box_corner[:,:,0] = (prediction[:,:,0] - prediction[:,:,2]/2)
#     box_corner[:,:,1] = (prediction[:,:,1] - prediction[:,:,3]/2)
#     box_corner[:,:,2] = (prediction[:,:,0] + prediction[:,:,2]/2) 
#     box_corner[:,:,3] = (prediction[:,:,1] + prediction[:,:,3]/2)
#     prediction[:,:,:4] = box_corner
#     # print(box_corner[:,:,:4][box_corner[:,:,:4] > 0.])
#     return prediction

def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes 
    
    
    """
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1,
                                                                                     min=0)

    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou


def post_process_predictions(predictions):
    # for every input image
    toreturn = []

    for prediction in predictions:

        # threshold the confidence
        conf_mask = (prediction[:, 4] > minConfidence)
        prediction = prediction[conf_mask]

        # find the bounding box corner given x,y,h,w
        box_corner = prediction[:, :4].data.clone()
        box_corner[:, 0] = (prediction[:, 0] - prediction[:, 2] / 2)
        box_corner[:, 1] = (prediction[:, 1] - prediction[:, 3] / 2)
        box_corner[:, 2] = (prediction[:, 0] + prediction[:, 2] / 2)
        box_corner[:, 3] = (prediction[:, 1] + prediction[:, 3] / 2)
        confidence = prediction[:, 4]
        prediction = prediction[:, 5:5 + numclasses]
        # print(conf_mask.shape, prediction.shape)

        # print(prediction.shape, prediction)
        if len(prediction) == 0:
            continue

        max_conf, arg_max_conf = torch.max(prediction, dim=1)
        # print(max_conf, arg_max_conf)
        # print(box_corner[:,:,:4][box_corner[:,:,:4] > 0.])

        # all class confidence sort_index
        sorted_index = torch.argsort(max_conf, descending=True)
        sorted_max_conf = max_conf[sorted_index]
        sorted_arg_max_conf = arg_max_conf[sorted_index]
        sorted_box_corner = box_corner[sorted_index]
        # print('sorted box corner',sorted_box_corner)

        uniq_classes, uniq_index = torch.unique(sorted_arg_max_conf, return_inverse=True)
        # print(uniq_classes, uniq_index)

        img_outputs = {}

        for i, uclas in enumerate(uniq_classes):
            # if uclas not in [2, 5]:
            #     continue
            # print('class=',uclas)
            # print('items ',uniq_index == i)
            # sorted by confidence
            class_mask = uniq_index == i
            # print('classmask',class_mask)

            class_indx = sorted_index[class_mask]
            # print(class_indx)
            # print(box_corner.shape, len(class_indx))

            class_bbox = sorted_box_corner[class_mask]
            # print(uclas, class_bbox, len(class_indx))

            for i in range(len(class_indx)):
                # class_box = box_corner[]
                ious = bbox_iou(class_bbox[i:i + 1], class_bbox[i + 1:])
                # print('ious=',ious)
                # Zero out all the detections that have IoU > treshhold
                # iou_mask = (ious < nms_conf)

                if len(ious) == 0:
                    # print('merging bbox finished')
                    break

                iou_index = (ious < nms_conf).nonzero().reshape(-1)
                # print(ious)

                # ious = ious[iou_mask]
                # print('ious = ',ious)
                # print('ious indx =', iou_index)
                # print('ious shape , bboxes=',ious.shape, class_bbox[i+1:].shape)
                # print('classbbox = ', class_bbox[i+1:][iou_index])
                # class_bbox[i+1:] = class_bbox[i+1:][iou_index]
                # class_bbox_ = np.c_[class_bbox[:i+1]]
                iou_index = iou_index + i + 1
                indices = torch.cat((torch.arange(i + 1, device=device), iou_index))
                # print(torch.arange(i+1),iou_index)
                class_bbox = class_bbox[indices]
            img_outputs[int(uclas)] = class_bbox.cpu()

        #     print(len(class_bbox))
        #     print('classbbox = ', class_bbox)
        # print('---------------')
        toreturn.append(img_outputs)
    return toreturn

    # class_conf = max_conf[class_mask]
    # conf_sort_index = torch.argsort(class_conf, descending = True )
    # print('class ',i, uclas)
    # print(conf_sort_index)


def draw_boxes(images, boxes):
    assert len(images) == len(boxes)
    for i in range(len(boxes)):
        plotted = draw_bbox(images[i], boxes[i])
        images[i] = plotted
        # cv2.imwrite('det/temp/temp{}.jpg'.format(i),plotted)
    return images


def refactor_bboxes(bboxes, factors):
    assert len(bboxes) == len(factors)
    toret = []

    for bbox, factor in zip(bboxes, factors):
        bbox_vehicles = {}
        for clas, box in bbox.items():
            box[:, [0, 2]] -= factor[1]
            box[:, [1, 3]] -= factor[2]
            box /= factor[0]
            bbox_vehicles[clas] = box
        toret.append(bbox_vehicles)
    return toret


def draw_bbox(img, boxes):
    # BGR -> RGB | H X W X C -> C X H X W  inverse opration
    img_ = img.astype(np.uint8)
    # cv2.imwrite('temp1.jpg', img_)
    # img_ = img.copy()
    # print(img_.shape)
    for clas, box in boxes.items():
        box = box.numpy().astype(int)
        # print(clas, box)
        # cv2.rectangle(img, box[:,:2], box[:,2:4], color=(200,200,200))
        for b in box:
            c1, c2 = tuple(b[:2]), tuple(b[2:4])
            # print(c1, c2)
            # cv2.rectangle(img_, c1, c2, color=(200,0,200))
            # cv2.putText(img_, classes[clas],\
            #     (c1[0], c1[1] + 12 + 4),cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1)

            color = (200, 130, 140)
            label = classes[clas]
            cv2.rectangle(img_, c1, c2, color=color, thickness=3)
            # for plotting label in colored box
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 1)[0]
            c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
            cv2.rectangle(img_, c1, c2, color, -1)
            cv2.putText(img_, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [225, 255, 255], 1)

            # cv2.rectangle(img_, (100,150), (150,250), color=(200,0,200), thickness=2)

    return img_


def select_objects(bboxes, indices):
    toret = []
    for bbox in bboxes:
        # print(indices)
        bbox_vehicles = {}
        for clas, box in bbox.items():
            if clas in indices:
                bbox_vehicles[clas] = box
        toret.append(bbox_vehicles)
    return toret


def select_boxes_below(bboxes, below=800, below1=600):
    toret = []
    original_below = int(below)
    global SCREEN_WIDTH
    for bbox in bboxes:
        bbox_vehicles = {}
        for clas, box in bbox.items():
            centroid_xs = (box[:,0]+box[:,2])/2
            for centroid_x in centroid_xs:

                if float(centroid_x) > SCREEN_WIDTH/2:
                    below = below1
                else:
                    below = original_below

                mask = box[:, 3] > below
                box = box[mask]
                if len(box) == 0:
                    continue
                bbox_vehicles[clas] = box
        toret.append(bbox_vehicles)
    return toret


def save_bbox_image(image, bbox, start: int = 0):
    for clas, box in bbox.items():
        box = box.numpy().astype(int)
        for b in box:
            b[b < 0.] = 0.
            crop_img = image[b[1]:b[3], b[0]:b[2], :]
            cv2.imwrite(f'det/vehicle_images/{start}_img.png', crop_img)
            start += 1
    return start


def get_bbox_image(image, bbox, return_bbox=False):
    img_list = []
    bbox_list = []
    for clas, box in bbox.items():
        box = box.numpy().astype(int)
        for b in box:
            b[b < 0.] = 0.
            crop_img = image[b[1]:b[3], b[0]:b[2], :]
            img_list.append(crop_img)
            bbox_list.append(b)
    if return_bbox:
        return img_list, bbox_list
    else:
        return img_list
