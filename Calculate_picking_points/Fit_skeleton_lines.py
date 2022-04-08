<<<<<<< HEAD
import cv2
import os
import numpy as np
import time
import matplotlib.pyplot as plt
import core

start=time.perf_counter()
# # Clear the picking point coordinate file content
with open(r'C:\Users\Administrator\Desktop\capture_points\code\Test_save_file\Lichi_pic\Coordinate.txt','a+',encoding='utf-8') as test:
    test.truncate(0)

def image_binarization(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    retval, dst = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return dst

def Zhang_Suen_thining(img):
    # get shape
    H, W,C= img.shape

    # prepare out image
    out = np.zeros((H, W), dtype=np.int)
    out[img[..., 0] > 0] = 1

    # inverse
    out = 1 - out

    while True:
        s1 = []
        s2 = []

        # step 1 ( rasta scan )
        for y in range(1, H - 1):
            for x in range(1, W - 1):

                # condition 1
                if out[y, x] > 0:
                    continue

                # condition 2
                f1 = 0
                if (out[y - 1, x + 1] - out[y - 1, x]) == 1:
                    f1 += 1
                if (out[y, x + 1] - out[y - 1, x + 1]) == 1:
                    f1 += 1
                if (out[y + 1, x + 1] - out[y, x + 1]) == 1:
                    f1 += 1
                if (out[y + 1, x] - out[y + 1, x + 1]) == 1:
                    f1 += 1
                if (out[y + 1, x - 1] - out[y + 1, x]) == 1:
                    f1 += 1
                if (out[y, x - 1] - out[y + 1, x - 1]) == 1:
                    f1 += 1
                if (out[y - 1, x - 1] - out[y, x - 1]) == 1:
                    f1 += 1
                if (out[y - 1, x] - out[y - 1, x - 1]) == 1:
                    f1 += 1

                if f1 != 1:
                    continue

                # condition 3
                f2 = np.sum(out[y - 1:y + 2, x - 1:x + 2])
                if f2 < 2 or f2 > 6:
                    continue

                # condition 4
                # x2 x4 x6
                if (out[y - 1, x] + out[y, x + 1] + out[y + 1, x]) < 1:
                    continue

                # condition 5
                # x4 x6 x8
                if (out[y, x + 1] + out[y + 1, x] + out[y, x - 1]) < 1:
                    continue

                s1.append([y, x])

        for v in s1:
            out[v[0], v[1]] = 1

        # step 2 ( rasta scan )
        for y in range(1, H - 1):
            for x in range(1, W - 1):

                # condition 1
                if out[y, x] > 0:
                    continue

                # condition 2
                f1 = 0
                if (out[y - 1, x + 1] - out[y - 1, x]) == 1:
                    f1 += 1
                if (out[y, x + 1] - out[y - 1, x + 1]) == 1:
                    f1 += 1
                if (out[y + 1, x + 1] - out[y, x + 1]) == 1:
                    f1 += 1
                if (out[y + 1, x] - out[y + 1, x + 1]) == 1:
                    f1 += 1
                if (out[y + 1, x - 1] - out[y + 1, x]) == 1:
                    f1 += 1
                if (out[y, x - 1] - out[y + 1, x - 1]) == 1:
                    f1 += 1
                if (out[y - 1, x - 1] - out[y, x - 1]) == 1:
                    f1 += 1
                if (out[y - 1, x] - out[y - 1, x - 1]) == 1:
                    f1 += 1

                if f1 != 1:
                    continue

                # condition 3
                f2 = np.sum(out[y - 1:y + 2, x - 1:x + 2])
                if f2 < 2 or f2 > 6:
                    continue

                # condition 4
                # x2 x4 x8
                if (out[y - 1, x] + out[y, x + 1] + out[y, x - 1]) < 1:
                    continue

                # condition 5
                # x2 x6 x8
                if (out[y - 1, x] + out[y + 1, x] + out[y, x - 1]) < 1:
                    continue

                s2.append([y, x])

        for v in s2:
            out[v[0], v[1]] = 1

        # if not any pixel is changed
        if len(s1) < 1 and len(s2) < 1:
            break

    out = 1 - out
    out = out.astype(np.uint8) * 255

    return out

# path
Seg_path = r"C:\Users\Administrator\Desktop\capture_points\code\Test_pic\Seg_pic"  # Path of main stem ROI semantic segmentation image
ROI_path = r"C:\Users\Administrator\Desktop\capture_points\code\Test_pic\Roi_pic"  # Path of ROI image of main stem
litchi_path = r"C:\Users\Administrator\Desktop\capture_points\code\Test_save_pic\Lichi_pic_save"  # Path of original litchi image
lizhi_label_path = r"C:\Users\Administrator\Desktop\capture_points\code\Test_pic\Label_txt"  # Path of litchi label file

file = os.listdir(Seg_path)
n = 1
print(file)
for image_name in file:
    s1, s2 = image_name.split("_")
    lab_txt_name = s1 + ".txt"  # Litchi label file name
    first, last = os.path.splitext(image_name)
    original_name = first + ".jpg"  # Main stem ROI area image name
    lizhi_pic_name = s1 + ".jpg"  # Litchi image name

    # path
    path1 = os.path.join(Seg_path, image_name)  # Path of image after semantic segmentation
    original_pic_path1 = os.path.join(ROI_path, original_name)  # Main stem ROI area image path
    original_lizhi_path1 = os.path.join(litchi_path, lizhi_pic_name)  # Original litchi image path

    # original_lizhi_path1 = os.path.join(os.path.abspath(original_lizhi_path), lizhi_pic_name)

    # Read image
    image = cv2.imread(path1)  # Read the main stem image after semantic segmentation
    H_seg, W_seg = image.shape[:2]
    original_pic = cv2.imread(original_pic_path1, 1)  # Read the original main stem image
    original_lizhi_pic = cv2.imread(original_lizhi_path1)  # Original read litchi image
    H_img, W_img = original_lizhi_pic.shape[:2]  # Extract the height and width of the original image

    # Read litchi label file
    lab_txt = []  # The center coordinate used to store the target frame
    with open(os.path.join(lizhi_label_path, lab_txt_name), "r+", encoding="utf-8", errors="ignore") as f:
        for line in f:
            aa = line.split(" ")
            if aa[0] == "0" and H_seg == int(H_img * float(aa[4])) and W_seg == int(W_img * float(aa[3])):
                X_center = W_img * float(aa[1])  # aa[1] is the x coordinate of the center of the target box
                Y_center = H_img * float(aa[2])  # aa[2]is the y coordinate of the center of the target box
                lab_txt.append(X_center)
                lab_txt.append(Y_center)
                break

    # Binarization
    erzhihua = image_binarization(image)
    erzhi = "Test_save_pic/Binarization" + '\\' + first + "erzhi" + str(n) + ".png"
    cv2.imwrite(erzhi, erzhihua)

    # Morphological open operation
    kernel = np.ones((5, 5), np.uint8)
    close_pic = cv2.morphologyEx(erzhihua, cv2.MORPH_CLOSE, kernel)
    xingtaixue = "Test_save_pic/Morphology" + '\\' + first + "xingtai" + str(n) + ".png"
    cv2.imwrite(xingtaixue, close_pic)

    # Skeletonization
    close_pic = cv2.cvtColor(close_pic, cv2.COLOR_BGR2RGB)
    image = close_pic.astype(np.float32)
    out = Zhang_Suen_thining(image)
    gugehua = "Test_save_pic/Skeletonization" + '\\' + first + "guge" + str(n) + ".png"
    cv2.imwrite(gugehua, out)

    # Locating picking points
    H, W = out.shape[:2]
    X = []
    Y = []
    for y in range(0, H - 1):
        for x in range(0, W - 1):
            if out[y,x] == 255:
                X.append(x)
                Y.append(y)
    if X!=[] and Y!=[]:
        X_list=X[:]
        X = np.array(X)
        print("X=",X)
        print("Y=",Y)
        a, b = core.leastSquare(X, Y)

        for i in range(len(X_list)):
            if i<len(X_list)-1:
                x1=int(X_list[i])


                y1=int(a * x1 + b)
                x2=int(X_list[i+1])
                y2=int(a * x2 + b)
            else:
                break
            cv2.line(out, (x1,y1), (x2, y2), (255, 0, 0), 1)
        nihexian = "Fitting_skeleton_lines" + '\\' + first + "nihe" + str(n) + ".png"
        cv2.imwrite(nihexian, out)  # Save the marked skeleton line pick point image
        # plt.show()
    else:
        print("The main stem was not recognized")
    print("n=",n)
    n = n + 1
end = time.perf_counter()
print("Total time：",end-start)





=======
import cv2
import os
import numpy as np
import time
import matplotlib.pyplot as plt
import core

start=time.perf_counter()
# # Clear the picking point coordinate file content
with open(r'C:\Users\Administrator\Desktop\capture_points\code\Test_save_file\Lichi_pic\Coordinate.txt','a+',encoding='utf-8') as test:
    test.truncate(0)

def image_binarization(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    retval, dst = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return dst

def Zhang_Suen_thining(img):
    # get shape
    H, W,C= img.shape

    # prepare out image
    out = np.zeros((H, W), dtype=np.int)
    out[img[..., 0] > 0] = 1

    # inverse
    out = 1 - out

    while True:
        s1 = []
        s2 = []

        # step 1 ( rasta scan )
        for y in range(1, H - 1):
            for x in range(1, W - 1):

                # condition 1
                if out[y, x] > 0:
                    continue

                # condition 2
                f1 = 0
                if (out[y - 1, x + 1] - out[y - 1, x]) == 1:
                    f1 += 1
                if (out[y, x + 1] - out[y - 1, x + 1]) == 1:
                    f1 += 1
                if (out[y + 1, x + 1] - out[y, x + 1]) == 1:
                    f1 += 1
                if (out[y + 1, x] - out[y + 1, x + 1]) == 1:
                    f1 += 1
                if (out[y + 1, x - 1] - out[y + 1, x]) == 1:
                    f1 += 1
                if (out[y, x - 1] - out[y + 1, x - 1]) == 1:
                    f1 += 1
                if (out[y - 1, x - 1] - out[y, x - 1]) == 1:
                    f1 += 1
                if (out[y - 1, x] - out[y - 1, x - 1]) == 1:
                    f1 += 1

                if f1 != 1:
                    continue

                # condition 3
                f2 = np.sum(out[y - 1:y + 2, x - 1:x + 2])
                if f2 < 2 or f2 > 6:
                    continue

                # condition 4
                # x2 x4 x6
                if (out[y - 1, x] + out[y, x + 1] + out[y + 1, x]) < 1:
                    continue

                # condition 5
                # x4 x6 x8
                if (out[y, x + 1] + out[y + 1, x] + out[y, x - 1]) < 1:
                    continue

                s1.append([y, x])

        for v in s1:
            out[v[0], v[1]] = 1

        # step 2 ( rasta scan )
        for y in range(1, H - 1):
            for x in range(1, W - 1):

                # condition 1
                if out[y, x] > 0:
                    continue

                # condition 2
                f1 = 0
                if (out[y - 1, x + 1] - out[y - 1, x]) == 1:
                    f1 += 1
                if (out[y, x + 1] - out[y - 1, x + 1]) == 1:
                    f1 += 1
                if (out[y + 1, x + 1] - out[y, x + 1]) == 1:
                    f1 += 1
                if (out[y + 1, x] - out[y + 1, x + 1]) == 1:
                    f1 += 1
                if (out[y + 1, x - 1] - out[y + 1, x]) == 1:
                    f1 += 1
                if (out[y, x - 1] - out[y + 1, x - 1]) == 1:
                    f1 += 1
                if (out[y - 1, x - 1] - out[y, x - 1]) == 1:
                    f1 += 1
                if (out[y - 1, x] - out[y - 1, x - 1]) == 1:
                    f1 += 1

                if f1 != 1:
                    continue

                # condition 3
                f2 = np.sum(out[y - 1:y + 2, x - 1:x + 2])
                if f2 < 2 or f2 > 6:
                    continue

                # condition 4
                # x2 x4 x8
                if (out[y - 1, x] + out[y, x + 1] + out[y, x - 1]) < 1:
                    continue

                # condition 5
                # x2 x6 x8
                if (out[y - 1, x] + out[y + 1, x] + out[y, x - 1]) < 1:
                    continue

                s2.append([y, x])

        for v in s2:
            out[v[0], v[1]] = 1

        # if not any pixel is changed
        if len(s1) < 1 and len(s2) < 1:
            break

    out = 1 - out
    out = out.astype(np.uint8) * 255

    return out

# path
Seg_path = r"C:\Users\Administrator\Desktop\capture_points\code\Test_pic\Seg_pic"  # Path of main stem ROI semantic segmentation image
ROI_path = r"C:\Users\Administrator\Desktop\capture_points\code\Test_pic\Roi_pic"  # Path of ROI image of main stem
litchi_path = r"C:\Users\Administrator\Desktop\capture_points\code\Test_save_pic\Lichi_pic_save"  # Path of original litchi image
lizhi_label_path = r"C:\Users\Administrator\Desktop\capture_points\code\Test_pic\Label_txt"  # Path of litchi label file

file = os.listdir(Seg_path)
n = 1
print(file)
for image_name in file:
    s1, s2 = image_name.split("_")
    lab_txt_name = s1 + ".txt"  # Litchi label file name
    first, last = os.path.splitext(image_name)
    original_name = first + ".jpg"  # Main stem ROI area image name
    lizhi_pic_name = s1 + ".jpg"  # Litchi image name

    # path
    path1 = os.path.join(Seg_path, image_name)  # Path of image after semantic segmentation
    original_pic_path1 = os.path.join(ROI_path, original_name)  # Main stem ROI area image path
    original_lizhi_path1 = os.path.join(litchi_path, lizhi_pic_name)  # Original litchi image path

    # original_lizhi_path1 = os.path.join(os.path.abspath(original_lizhi_path), lizhi_pic_name)

    # Read image
    image = cv2.imread(path1)  # Read the main stem image after semantic segmentation
    H_seg, W_seg = image.shape[:2]
    original_pic = cv2.imread(original_pic_path1, 1)  # Read the original main stem image
    original_lizhi_pic = cv2.imread(original_lizhi_path1)  # Original read litchi image
    H_img, W_img = original_lizhi_pic.shape[:2]  # Extract the height and width of the original image

    # Read litchi label file
    lab_txt = []  # The center coordinate used to store the target frame
    with open(os.path.join(lizhi_label_path, lab_txt_name), "r+", encoding="utf-8", errors="ignore") as f:
        for line in f:
            aa = line.split(" ")
            if aa[0] == "0" and H_seg == int(H_img * float(aa[4])) and W_seg == int(W_img * float(aa[3])):
                X_center = W_img * float(aa[1])  # aa[1] is the x coordinate of the center of the target box
                Y_center = H_img * float(aa[2])  # aa[2]is the y coordinate of the center of the target box
                lab_txt.append(X_center)
                lab_txt.append(Y_center)
                break

    # Binarization
    erzhihua = image_binarization(image)
    erzhi = "Test_save_pic/Binarization" + '\\' + first + "erzhi" + str(n) + ".png"
    cv2.imwrite(erzhi, erzhihua)

    # Morphological open operation
    kernel = np.ones((5, 5), np.uint8)
    close_pic = cv2.morphologyEx(erzhihua, cv2.MORPH_CLOSE, kernel)
    xingtaixue = "Test_save_pic/Morphology" + '\\' + first + "xingtai" + str(n) + ".png"
    cv2.imwrite(xingtaixue, close_pic)

    # Skeletonization
    close_pic = cv2.cvtColor(close_pic, cv2.COLOR_BGR2RGB)
    image = close_pic.astype(np.float32)
    out = Zhang_Suen_thining(image)
    gugehua = "Test_save_pic/Skeletonization" + '\\' + first + "guge" + str(n) + ".png"
    cv2.imwrite(gugehua, out)

    # Locating picking points
    H, W = out.shape[:2]
    X = []
    Y = []
    for y in range(0, H - 1):
        for x in range(0, W - 1):
            if out[y,x] == 255:
                X.append(x)
                Y.append(y)
    if X!=[] and Y!=[]:
        X_list=X[:]
        X = np.array(X)
        print("X=",X)
        print("Y=",Y)
        a, b = core.leastSquare(X, Y)

        for i in range(len(X_list)):
            if i<len(X_list)-1:
                x1=int(X_list[i])


                y1=int(a * x1 + b)
                x2=int(X_list[i+1])
                y2=int(a * x2 + b)
            else:
                break
            cv2.line(out, (x1,y1), (x2, y2), (255, 0, 0), 1)
        nihexian = "Fitting_skeleton_lines" + '\\' + first + "nihe" + str(n) + ".png"
        cv2.imwrite(nihexian, out)  # Save the marked skeleton line pick point image
        # plt.show()
    else:
        print("The main stem was not recognized")
    print("n=",n)
    n = n + 1
end = time.perf_counter()
print("Total time：",end-start)





>>>>>>> b201b31cc22f62ca79d4167c14e19308f24f3ead
