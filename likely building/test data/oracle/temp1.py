# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 19:12:07 2019

@author: Liurui
"""

import cv2






a=20
while a<50:
    a=a+1
    path = 'oracle(1).png'
    img = cv2.imread(path, flags=cv2.IMREAD_COLOR)
    bbox = cv2.selectROI(img, False)
    cut = img[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
    cv2.imwrite(str(a)+'.jpg', cut)
    cv2.destroyAllWindows()


