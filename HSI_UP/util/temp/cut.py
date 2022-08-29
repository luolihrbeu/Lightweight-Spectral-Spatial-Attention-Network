import numpy as np
import cv2
import os

file_names = os.listdir("./Fig/")
for file_name in file_names:
    img = cv2.imread("./Fig/" + file_name, 1)
    cutimg = img[109:801, 116:808]
    # cv2.imshow('origin', img)
    # cv2.imshow('image', cutimg)
    cv2.imwrite('%s.png' % file_name[:-4], cutimg)
    k = cv2.waitKey(0)
    # if k == 27:
    #     cv2.destroyAllWindows()
