import os
import cv2
import numpy as np
path = "pspnet-pytorch-master/VOCdevkit/VOC2007/SegmentationClass"
file = os.listdir(path)
for filename in file:
    path1 = os.path.join(path,filename)
    pic=cv2.imread(path1)
    pic = pic[:, :, 0]
    for j in range(pic.shape[0]):
        for k in range(pic.shape[1]):
            if pic[j][k] ==128:
                pic[j][k]=1


