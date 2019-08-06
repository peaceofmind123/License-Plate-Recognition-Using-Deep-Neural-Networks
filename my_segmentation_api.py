#!/usr/bin/env python

import numpy as np
import cv2
import os

##_ALL criteria for clustering
MIN_POINTS = 2
MAX_DEVIATION = 9
MAX_TRY = 5

def process_image(img, width=200):
    h,w,c = img.shape
    fx = width/w
    img = cv2.resize(img,None, fx=fx, fy=fx)
    return img

def process_images(image_list:list, width=200):
    for i in range(len(image_list)):
        img = image_list[i]
        img = process_image(img)
        image_list[i] = img
    return image_list

class KMeans1D(object):
    
    def __init__(self, data, threshold=1e-4):
        assert not len(data) < 2 
        self.threshold = threshold
        self.centers = np.array([data.min(), data.max()])
        self.data = data
        self.indices = np.array(range(len(data)), dtype=np.int64)
        self.prev_centers = None
        self.data_cluster = None
        self.distances = None

    
    def calculate_distance(self):
        dist = []
        for center in self.centers:
            d = np.abs(self.data - center)
            dist.append(d) # collecting all the sum

        self.distances = np.array(dist)
        return self.distances
    
    def assign_clusters(self):
        self.data_cluster = np.argmin(self.distances,
                                      axis = 0) # finding minimum over the the centers
        return self.data_cluster
    
    def update_centers(self):
        self.prev_centers = np.copy(self.centers)
        for index in range(len(self.centers)):
            self.centers[index] = np.mean(self.data[self.data_cluster == index])
        return self.centers
    
    def is_optimal(self):
        non_optimal = np.abs(self.prev_centers - self.centers) > self.threshold
        if non_optimal.astype(int).sum() == 0:
            return True
        return False
    

    def _processing1(self):
        # REMOVE points deviating more than MAX_DEVIATION from their centers
        deviation = np.abs(self.data - self.centers[self.data_cluster])
        mask = deviation < MAX_DEVIATION
        self.data = self.data[mask]
        self.data_cluster = self.data_cluster[mask]
        self.indices = self.indices[mask]
    
    def _processing2(self):
        # COMBINE Clusters if they are seperated small value eg. MAX_DEVIATION
        if np.abs(self.centers[0] - self.centers[1]) < MAX_DEVIATION:
            self.data_cluster = np.zeros_like(self.data_cluster)
            self.centers = self.data.mean(keepdims=True)
            # this will combine 2 centers into 1
    
    def _processing3(self):
        # REMOVE Cluster if they have less than MIN_POINTS assigned to them
        for i in range(len(self.centers)):
            mask = self.data_cluster == i
            if mask.astype(int).sum() < MIN_POINTS:
                if len(self.centers) == i:i =0
                self.centers = np.delete(self.centers, i)
                self.data_cluster = self.data_cluster[~mask]
                self.data = self.data[~mask]
                self.indices = self.indices[~mask]
    
    def do_what_is_needed(self):
        for i in range(MAX_TRY):
            self.calculate_distance()
            self.assign_clusters()
            self.update_centers()
            if self.is_optimal():
                break
        self._processing1()
        self._processing2()        
        self._processing3()     
        
        return self.indices



## All criteria here
MAX_AREA = 5000
MIN_AREA = 150
MAX_ASPECT_RATIO = 4
MIN_ASPECT_RATIO = 0.25

def get_license_char_bbox(image): ## requires preprocessed images
    #returns rectangle bounding box for each possible characters
    grayimg = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    thresh_val, thresh = cv2.threshold(grayimg,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ignore_image = False
    for i in range(2):
        cnts0 = []
        midx, midy = [], []
        rect_xywh = []
        redo_processing = False
        for cnt in cnts:
            x,y,w,h = cv2.boundingRect(cnt)
            cntArea = w*h
            aspect_ratio = w/h
            if cntArea > MAX_AREA:
                cntArea_true = cv2.contourArea(cnt)
                if cntArea_true > MAX_AREA:
                    thresh = cv2.bitwise_not(thresh)
                    cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    redo_processing = True
                    if i == 1:
                        print('The image cannot be processed furthur !!!!!............')
                        ignore_image = True
                        break
                    print('The image seems like dark-text on white-background, inverting the binary')
                    break
            if MAX_AREA > cntArea > MIN_AREA and MIN_ASPECT_RATIO < aspect_ratio < MAX_ASPECT_RATIO:
                cnts0.append(cnt)
                mx, my = x+w/2, y+h/2
                midx.append(mx)
                midy.append(my)
                rect_xywh.append((x,y,w,h))
                
        if not redo_processing: 
            if len(cnts0) < 2:
                print('No hint of text found, cannot be processed furthur !!!!!............')
                ignore_image = True        
            break
    
    if ignore_image:
        return None

    center = np.c_[np.array(midx), np.array(midy)]
    rect_xywh = np.array(rect_xywh)

    
    kmeans = KMeans1D(center[:,1])
    ind = kmeans.do_what_is_needed()
    cluster = kmeans.data_cluster

    sort_indx_ = np.argsort(center[:,0][ind]+kmeans.data_cluster*500)
    sort_indx = ind[sort_indx_]
    # center1 = center[sort_indx]
    # cnts1 = [cnts0[i] for i in sort_indx]
    rect_xywh1 = rect_xywh[sort_indx]
    # cluster1 = cluster[sort_indx_]

    # tempimg = image.copy()
    # cv2.drawContours(tempimg,cnts1,-1, (255,0,0),1)
    # for xywh, cxy, c in zip(rect_xywh1, center1, cluster1):
    #     col = [0, 0, 0]; col[c] = 255
    #     cv2.circle(tempimg, (int(cxy[0]), int(cxy[1])), 2, tuple(col), -1)
    #     x,y,w,h = xywh
    #     cv2.rectangle(tempimg,(x,y),(x+w,y+h),(0,200,240),1)
    # plt.imsave(file_names[indx].replace('lp_images', 'lp_segmentation'),tempimg)
    return rect_xywh1


# def process_all_license_plates(image_list:list):
#     char_box_list = []
#     for indx in range(len(image_list)):
#         image = image_list[indx]
#         char_box = get_license_char_bbox(image)
#         char_box_list.append(char_box)
#     return char_box_list

def get_all_license_plate_chars(image_list:list):
    char_box_list = []
    for indx in range(len(image_list)):
        image = image_list[indx]
        image = process_image(image)
        char_box = get_license_char_bbox(image)
        if char_box is None:
            char_box_list.append(None)
            continue
        
        chars = []
        for xywh in char_box:
            x,y,w,h = xywh
            char = image[y:y+h,x:x+w,:]
            chars.append(char)

        char_box_list.append(chars)
    return char_box_list



def get_license_char_binary(image):
    #returns binary characters
    image = process_image(image)
    grayimg = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    thresh_val, thresh = cv2.threshold(grayimg,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ignore_image = False
    for i in range(2):
        cnts0 = []
        midx, midy = [], []
        rect_xywh = []
        redo_processing = False
        for cnt in cnts:
            x,y,w,h = cv2.boundingRect(cnt)
            cntArea = w*h
            aspect_ratio = w/h
            if cntArea > MAX_AREA:
                cntArea_true = cv2.contourArea(cnt)
                if cntArea_true > MAX_AREA:
                    thresh = cv2.bitwise_not(thresh)
                    cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    redo_processing = True
                    if i == 1:
                        print('The image cannot be processed furthur !!!!!............')
                        ignore_image = True
                        break
                    print('The image seems like dark-text on white-background, inverting the binary')
                    break
            if MAX_AREA > cntArea > MIN_AREA and MIN_ASPECT_RATIO < aspect_ratio < MAX_ASPECT_RATIO:
                cnts0.append(cnt)
                mx, my = x+w/2, y+h/2
                midx.append(mx)
                midy.append(my)
                rect_xywh.append((x,y,w,h))
                
        if not redo_processing: 
            if len(cnts0) < 2:
                print('No hint of text found, cannot be processed furthur !!!!!............')
                ignore_image = True        
            break
    
    if ignore_image:
        return None

    center = np.c_[np.array(midx), np.array(midy)]
    rect_xywh = np.array(rect_xywh)

    
    kmeans = KMeans1D(center[:,1])
    ind = kmeans.do_what_is_needed()
    cluster = kmeans.data_cluster

    sort_indx_ = np.argsort(center[:,0][ind]+kmeans.data_cluster*500)
    sort_indx = ind[sort_indx_]
    rect_xywh1 = rect_xywh[sort_indx]
    chars = []
    print('thresh shape', thresh.shape)
    for xywh in rect_xywh1:
        x,y,w,h = xywh
        char = thresh[y:y+h,x:x+w]
        chars.append(char)
    return chars

def get_all_license_plate_chars_binary(image_list:list):
    char_box_list = []
    for indx in range(len(image_list)):
        image = image_list[indx]
        char_box = get_license_char_binary(image)
        char_box_list.append(char_box)
    return char_box_list
