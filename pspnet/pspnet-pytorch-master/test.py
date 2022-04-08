# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import matplotlib as plt
# 1. 创建图像及txt文件夹
path = "jpg_text"         # jpg图片和对应的生成结果的txt标注文件，放在一起
path3 = "bboxcut"    # 裁剪出来的小图保存的根目录
img_total = []
txt_total = []

file = os.listdir(path)
for filename in file:
    first,last = os.path.splitext(filename)
    if last == ".jpg":                      # 图片的后缀名
        img_total.append(first)
        print(img_total)
    else:
        txt_total.append(first)

# 2. 提取yolov5预测框图像
n=1
for img_ in img_total:
    if img_ in txt_total:
        filename_img = img_+".jpg"          # 图片的后缀名
        print('filename_img:', filename_img)
        path1 = os.path.join(path,filename_img)
        img = cv2.imread(path1)

        sp = img.shape
        h = sp[0]  # height(rows) of image
        w = sp[1]  # width(colums) of image

        img = cv2.resize(img,(w,h),interpolation = cv2.INTER_CUBIC)        # resize 图像大小，否则roi区域可能会报错
        filename_txt = img_+".txt"
        print('filename_txt:', filename_txt)

        with open(os.path.join(path,filename_txt),"r+",encoding="utf-8",errors="ignore") as f:
            for line in f:
                aa = line.split(" ")
                if aa[0] == "0":#类别标签为0，即文txt文件
                    x_center = w * float(aa[1])       # aa[1]左上点的x坐标
                    y_center = h * float(aa[2])       # aa[2]左上点的y坐标
                    width = int(w*float(aa[3]))       # aa[3]图片width
                    height = int(h*float(aa[4]))      # aa[4]图片height
                    lefttopx = int(x_center-width/2.0)
                    lefttopy = int(y_center-height/2.0)
                    roi = img[lefttopy:lefttopy+height,lefttopx:lefttopx+width]   # [左上y:右下y,左上x:右下x] (y1:y2,x1:x2)需要调参，否则裁剪出来的小图可能不太好
                    print('roi:', roi)                        # 如果不resize图片统一大小，可能会得到有的roi为[]导致报错

                    filename_last = img_+"_"+str(n)+".jpg"    # 裁剪出来的小图文件名
                    # print(filename_last)
                    path2 = os.path.join(path3,"roi")           # 需要在path3路径下创建一个roi文件夹
                    print('path2:', path2)                    # 裁剪小图的保存位置
                    cv2.imwrite(os.path.join(path2,filename_last),roi)

                    # plt.imshow(roi[:,:,::-1])
                    # plt.show()

# 3. K-means聚类算法
                    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    # change img(2D) to 1D
                    img1 = gray.reshape((gray.shape[0] * gray.shape[1], 1))
                    img1 = np.float32(img1)

                    # define criteria = (type,max_iter,epsilon)
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 5, 21)

                    # set flags: hou to choose the initial center
                    # ---cv2.KMEANS_PP_CENTERS ; cv2.KMEANS_RANDOM_CENTERS
                    flags = cv2.KMEANS_RANDOM_CENTERS
                    # apply kmenas
                    compactness, labels, centers = cv2.kmeans(img1, 2, None, criteria, 10, flags)

                    img2 = labels.reshape((gray.shape[0], gray.shape[1]))
                    #plt.imshow(img2, cmap=plt.cm.gray)
                    #plt.show()
                    kmean_name ="bboxcut/K-means" + '\\'+ img_+"_k"+ str(n) + ".jpg"
                    #print(kmean_name)
                    img2=img2*255
                    cv2.imwrite(kmean_name, img2)

# 4. 形态学开运算
                    # 读取图像
                    img1 = cv2.imread(kmean_name)
                    # 创建核结构
                    kernel = np.ones((3, 3), np.uint8)
                    # 图像的开运算
                    cvOpen = cv2.morphologyEx(img1, cv2.MORPH_OPEN, kernel)  # 开运算
                    # 图像保存
                    open_name = "bboxcut/Open-operation" + '\\' +img_+ "_open" + str(n) + ".jpg"
                    cv2.imwrite(open_name , cvOpen)
                    n = n + 1

    else:
        continue
