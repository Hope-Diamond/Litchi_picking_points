<<<<<<< HEAD
import cv2,time
import os
import numpy as np


start = time.clock()
path = "yolo_txt/images"
img_total = []
txt_total = []
file = os.listdir(path)
for filename in file:
    first, last = os.path.splitext(filename)
    if last == ".jpg":
        img_total.append(first)
    # print(img_total)
    else:
        txt_total.append(first)
n = 1
for img_ in img_total:
    if img_ in txt_total:
        filename_img = img_ + ".jpg"
        path1 = os.path.join(path, filename_img)
        img = cv2.imread(path1)

        sp = img.shape
        h = sp[0]  # height(rows) of image
        w = sp[1]  # width(colums) of image

        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
        filename_txt = img_ + ".txt"

        with open(os.path.join(path, filename_txt), "r+", encoding="utf-8", errors="ignore") as f:
            for line in f:
                aa = line.split(" ")
                if aa[0] == "0":
                    x_center = w * float(aa[1])  # aa[1] is the x coordinate of the upper left point
                    y_center = h * float(aa[2])  # aa[2] is the y coordinate of the upper left point
                    width = int(w * float(aa[3]))  # aa[3] is the width of the image
                    height = int(h * float(aa[4]))  # aa[4] is the picture height
                    x1 = int(x_center - width / 2.0)
                    x2 = int(x_center + width / 2.0)
                    y1= int(y_center - height / 2.0)
                    y2=int(y_center + height / 2.0)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            path3 = "yolo_txt/bbx_results" + "\\" + img_ + "_" + str(n) + ".jpg"
            # print('path2:', path2)
            cv2.imwrite(path3, img)
            n = n + 1

    else:
        continue
end = time.clock()
print("Total time：", end - start)
=======
import cv2,time
import os
import numpy as np


start = time.clock()
path = "yolo_txt/images"
img_total = []
txt_total = []
file = os.listdir(path)
for filename in file:
    first, last = os.path.splitext(filename)
    if last == ".jpg":
        img_total.append(first)
    # print(img_total)
    else:
        txt_total.append(first)
n = 1
for img_ in img_total:
    if img_ in txt_total:
        filename_img = img_ + ".jpg"
        path1 = os.path.join(path, filename_img)
        img = cv2.imread(path1)

        sp = img.shape
        h = sp[0]  # height(rows) of image
        w = sp[1]  # width(colums) of image

        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
        filename_txt = img_ + ".txt"

        with open(os.path.join(path, filename_txt), "r+", encoding="utf-8", errors="ignore") as f:
            for line in f:
                aa = line.split(" ")
                if aa[0] == "0":
                    x_center = w * float(aa[1])  # aa[1] is the x coordinate of the upper left point
                    y_center = h * float(aa[2])  # aa[2] is the y coordinate of the upper left point
                    width = int(w * float(aa[3]))  # aa[3] is the width of the image
                    height = int(h * float(aa[4]))  # aa[4] is the picture height
                    x1 = int(x_center - width / 2.0)
                    x2 = int(x_center + width / 2.0)
                    y1= int(y_center - height / 2.0)
                    y2=int(y_center + height / 2.0)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            path3 = "yolo_txt/bbx_results" + "\\" + img_ + "_" + str(n) + ".jpg"
            # print('path2:', path2)
            cv2.imwrite(path3, img)
            n = n + 1

    else:
        continue
end = time.clock()
print("Total time：", end - start)
>>>>>>> b201b31cc22f62ca79d4167c14e19308f24f3ead
