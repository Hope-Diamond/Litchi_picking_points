# -*- coding: utf-8 -*-
import os
import cv2
import time
import matplotlib.pyplot as plt

start=time.perf_counter()
path = "Test_pic/Lichi_pic" # Put the yolov5 detected image and the corresponding generated txt label file in the same folder
img_total = []
txt_total = []
file = os.listdir(path)
for filename in file:
    first,last = os.path.splitext(filename)
    if last == ".jpg":
        img_total.append(first)

    else:
        txt_total.append(first)
n = 1
for img_ in img_total:
    if img_ in txt_total:
        filename_img = img_+".jpg"
        path1 = os.path.join(path,filename_img)
        img = cv2.imread(path1)
        sp = img.shape
        h= sp[0]  # height(rows) of image
        w= sp[1]  # width(colums) of image
        img = cv2.resize(img,(w,h),interpolation = cv2.INTER_CUBIC)
        filename_txt = img_+".txt"

        with open(os.path.join(path,filename_txt),"r+",encoding="utf-8",errors="ignore") as f:
            for line in f:
                aa = line.split(" ")
                if aa[0]=="0":
                    x_center = w * float(aa[1])       # aa[1] is the x coordinate of the upper left point
                    y_center = h * float(aa[2])       # aa[2] is the y coordinate of the upper left point
                    width = int(w*float(aa[3]))       # aa[3] is the width of the image
                    height = int(h*float(aa[4]))      # aa[4] is the picture height
                    lefttopx = int(x_center-width/2.0)
                    lefttopy = int(y_center-height/2.0)
                    roi = img[lefttopy:lefttopy + height, lefttopx:lefttopx + width ]
                    path3 = "Test_pic/Roi_pic"+"\\"+img_+"_"+str(n)+".jpg" # Save ROI image of main stem
                    cv2.imwrite(path3,roi)
                    n = n + 1
    else:
        continue
end = time.perf_counter()
print("Total time：",end-start)