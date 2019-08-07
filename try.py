import ocr_api
import os
import cv2
if __name__ == '__main__':
    img = cv2.imread(os.path.join(os.getcwd(),'lp_detection','$0.png'),cv2.IMREAD_GRAYSCALE)
    print(img.shape)
    print(ocr_api.predict_ocr(img))