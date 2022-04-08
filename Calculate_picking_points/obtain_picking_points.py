<<<<<<< HEAD
import cv2
import os
import numpy as np
import time
import matplotlib.pyplot as plt

start=time.perf_counter()

# Clear the picking point coordinate file content
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
    out = np.zeros((H, W), dtype=np.int32)
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


def Picking_points(Seg_path,ROI_path,litchi_path,lizhi_label_path):

    file=os.listdir(Seg_path)
    n=1
    print(file)
    for image_name in file:
        s1,s2=image_name.split("_")
        lab_txt_name=s1+".txt" # Litchi label file name
        first,last=os.path.splitext(image_name)
        original_name=first+".jpg" # Main stem ROI area image name
        lizhi_pic_name=s1+".jpg" # Litchi image name

        # path
        path1=os.path.join(Seg_path,image_name) # Path of image after semantic segmentation
        original_pic_path1=os.path.join(ROI_path,original_name) # Main stem ROI area image path
        original_lizhi_path1=os.path.join(litchi_path,lizhi_pic_name) # Original litchi image path

        #original_lizhi_path1 = os.path.join(os.path.abspath(original_lizhi_path), lizhi_pic_name)

        # Read image
        image = cv2.imread(path1)  # Read the main stem image after semantic segmentation
        H_seg,W_seg=image.shape[:2]
        original_pic=cv2.imread(original_pic_path1,1) # Read the original main stem image
        original_lizhi_pic=cv2.imread(original_lizhi_path1) # Original read litchi image
        H_img,W_img =original_lizhi_pic.shape[:2] # Extract the height and width of the original image

        # Read litchi label file
        lab_txt=[] # The center coordinate used to store the target frame
        with open(os.path.join(lizhi_label_path, lab_txt_name), "r+", encoding="utf-8", errors="ignore") as f:
            for line in f:
                aa = line.split(" ")
                if aa[0] == "0" and H_seg ==int(H_img * float(aa[4])) and W_seg==int(W_img * float(aa[3])):
                    X_center = W_img* float(aa[1])  # aa[1] is the x coordinate of the center of the target box
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
        close_pic=cv2.cvtColor(close_pic,cv2.COLOR_BGR2RGB)
        image = close_pic.astype(np.float32)
        out = Zhang_Suen_thining(image)
        gugehua = "Test_save_pic/Skeletonization" + '\\' + first + "guge" + str(n) + ".png"
        cv2.imwrite(gugehua, out)

        # Locating picking points
        H, W = out.shape
        spl1 = round(float(H / 4)) - 1  # 1/4 split line
        spl2 = round(float(H / 2)) - 1  # 1/2 split line
        spl3 = round(float(H * 3 / 4)) - 1  # 3/4 split line
        p1_loc = []
        p2_loc = []
        p3_loc = []

        for x in range(0, W - 1):
            # for y in range(0, H - 1):
            p1 = out[spl1, x]
            p2 = out[spl2, x]
            p3 = out[spl3, x]
            if p1 == 255:  # Extract the intersection coordinates of the main stem bone line and the image 1/4 split line
                p1p = (x, spl1)
                p1_loc.append(p1p)
            elif p2 == 255:
                # print(spl2,x)
                p2p = (x, spl2)
                p2_loc.append(p2p)
            elif p3 == 255:
                # print(spl3,x)
                p3p = (x, spl3)
                p3_loc.append(p3p)


        points_coordinate = open(os.path.join(litchi_path, 'Coordinate.txt'), 'a+') # Write the picking point coordinate value in the Coordinate.txt file

        if p2_loc != []:  # Mark the intersection of the bone line and the 1/2 split line
            p2p = p2_loc[0]

            # Calculate the picking point coordinates in the original litchi image
            x1=p2p[0] # x coordinate of picking point on bone line
            y1=p2p[1] # y coordinate of picking point on bone line
            X = lab_txt[0] - (1 / 2) * W_seg + x1   # X coordinate of picking point on the original litchi image
            Y = lab_txt[1] - (1 / 2) * H_seg + y1  # Y coordinate of picking point on the original litchi image
            X=round(X)
            Y=round(Y)
            # Mark picking point
            result = cv2.circle(out, p2p, 3, (255), -1) # Mark the picking point on the bone line
            result1 = cv2.circle(original_pic, p2p, 5, (0, 0, 255), -1) # Mark the picking point on the ROI image of the main stem
            lizhi_pic_point=cv2.circle(original_lizhi_pic, (X,Y), 5, (0, 0, 255), -1) # Mark the picking point on the original litchi image

            # Write picking point coordinate value
            coordinates = original_name + "  " + str((X, Y))+"\n"
            points_coordinate.write(coordinates)

            # Save the image after marking the picking point
            pick_point = "Test_save_pic/ske_pic_save" + '\\' + first + "ppt" + str(n) + ".png"
            cv2.imwrite(pick_point, result)  # Save the image of the picking point marked on the bone line
            pick_point1 = "Test_save_pic/Roi_pic_save" + '\\' + first + "orppt" + str(n) + ".png"
            cv2.imwrite(pick_point1, result1)  # Save the image of the picking point marked on the main stem ROI
            cv2.imwrite(original_lizhi_path1, lizhi_pic_point)  # Save the image of the picking point marked on the original litchi image

        elif p2_loc == [] and p1_loc != []: # Mark the intersection of the bone line and the 3/4 split line
            p1p = p1_loc[0]

            x1 = p1p[0]
            y1 = p1p[1]
            X = lab_txt[0] - (1 / 2) * W_seg + x1
            Y = lab_txt[1] - (1 / 2) * H_seg + y1
            X=round(X)
            Y=round(Y)
            result = cv2.circle(out, p1p, 3, (255), -1)
            result1= cv2.circle(original_pic, p1p, 5, (0, 0,255), -1)
            lizhi_pic_point = cv2.circle(original_lizhi_pic, (X, Y), 5, (0, 0,255), -1)

            # Write picking point coordinate value
            coordinates = original_name + "  " + str((X, Y)) + "\n"
            points_coordinate.write(coordinates)

            # Save the image after marking the picking point
            pick_point = "Test_save_pic/ske_pic_save" + '\\' + first + "ppt" + str(n) + ".png"
            cv2.imwrite(pick_point, result)
            pick_point1 = "Test_save_pic/Roi_pic_save" + '\\' + first + "orppt" + str(n) + ".png"
            cv2.imwrite(pick_point1, result1)
            cv2.imwrite(original_lizhi_path1, lizhi_pic_point)


        elif p2_loc == [] and p1_loc == [] and p3_loc != []: # Mark the intersection of the bone line and the 1/4 split line
            p3p = p3_loc[0]

            x1 =p3p[0]
            y1 =p3p[1]
            X = lab_txt[0] - (1 / 2) * W_seg + x1
            Y = lab_txt[1] - (1 / 2) * H_seg + y1
            X=round(X)
            Y=round(Y)
            result = cv2.circle(out, p3p, 3, (255), -1)
            result1 = cv2.circle(original_pic, p3p, 5, (0, 0,255), -1)
            lizhi_pic_point = cv2.circle(original_lizhi_pic, (X, Y), 5, (0, 0,255), -1)

            # Write picking point coordinate value
            coordinates = original_name + "  " + str((X, Y)) + "\n"
            points_coordinate.write(coordinates)

            # Save the image after marking the picking point
            pick_point = "Test_save_pic/ske_pic_save" + '\\' + first + "ppt" + str(n) + ".png"
            cv2.imwrite(pick_point, result)
            pick_point1="Test_save_pic/Roi_pic_save" + '\\' + first + "orppt" + str(n) + ".png"
            cv2.imwrite(pick_point1, result1)
            cv2.imwrite(original_lizhi_path1,lizhi_pic_point)

        points_coordinate.close()

    n = n + 1
    end = time.perf_counter()
    print("Total time：",end-start)

if __name__ == "__main__":

    Seg_path = r"C:\Users\Administrator\Desktop\capture_points\code\Test_pic\Seg_pic"  # Path of main stem ROI semantic segmentation image
    ROI_path = r"C:\Users\Administrator\Desktop\capture_points\code\Test_pic\Roi_pic"  # Path of ROI image of main stem
    litchi_path = r"C:\Users\Administrator\Desktop\capture_points\code\Test_save_pic\Lichi_pic_save"  # Path of original litchi image
    lizhi_label_path = r"C:\Users\Administrator\Desktop\capture_points\code\Test_pic\Label_txt"  # Path of litchi label file

    Picking_points(Seg_path,ROI_path,litchi_path,lizhi_label_path)




=======
import cv2
import os
import numpy as np
import time
import matplotlib.pyplot as plt

start=time.perf_counter()

# Clear the picking point coordinate file content
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
    out = np.zeros((H, W), dtype=np.int32)
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


def Picking_points(Seg_path,ROI_path,litchi_path,lizhi_label_path):

    file=os.listdir(Seg_path)
    n=1
    print(file)
    for image_name in file:
        s1,s2=image_name.split("_")
        lab_txt_name=s1+".txt" # Litchi label file name
        first,last=os.path.splitext(image_name)
        original_name=first+".jpg" # Main stem ROI area image name
        lizhi_pic_name=s1+".jpg" # Litchi image name

        # path
        path1=os.path.join(Seg_path,image_name) # Path of image after semantic segmentation
        original_pic_path1=os.path.join(ROI_path,original_name) # Main stem ROI area image path
        original_lizhi_path1=os.path.join(litchi_path,lizhi_pic_name) # Original litchi image path

        #original_lizhi_path1 = os.path.join(os.path.abspath(original_lizhi_path), lizhi_pic_name)

        # Read image
        image = cv2.imread(path1)  # Read the main stem image after semantic segmentation
        H_seg,W_seg=image.shape[:2]
        original_pic=cv2.imread(original_pic_path1,1) # Read the original main stem image
        original_lizhi_pic=cv2.imread(original_lizhi_path1) # Original read litchi image
        H_img,W_img =original_lizhi_pic.shape[:2] # Extract the height and width of the original image

        # Read litchi label file
        lab_txt=[] # The center coordinate used to store the target frame
        with open(os.path.join(lizhi_label_path, lab_txt_name), "r+", encoding="utf-8", errors="ignore") as f:
            for line in f:
                aa = line.split(" ")
                if aa[0] == "0" and H_seg ==int(H_img * float(aa[4])) and W_seg==int(W_img * float(aa[3])):
                    X_center = W_img* float(aa[1])  # aa[1] is the x coordinate of the center of the target box
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
        close_pic=cv2.cvtColor(close_pic,cv2.COLOR_BGR2RGB)
        image = close_pic.astype(np.float32)
        out = Zhang_Suen_thining(image)
        gugehua = "Test_save_pic/Skeletonization" + '\\' + first + "guge" + str(n) + ".png"
        cv2.imwrite(gugehua, out)

        # Locating picking points
        H, W = out.shape
        spl1 = round(float(H / 4)) - 1  # 1/4 split line
        spl2 = round(float(H / 2)) - 1  # 1/2 split line
        spl3 = round(float(H * 3 / 4)) - 1  # 3/4 split line
        p1_loc = []
        p2_loc = []
        p3_loc = []

        for x in range(0, W - 1):
            # for y in range(0, H - 1):
            p1 = out[spl1, x]
            p2 = out[spl2, x]
            p3 = out[spl3, x]
            if p1 == 255:  # Extract the intersection coordinates of the main stem bone line and the image 1/4 split line
                p1p = (x, spl1)
                p1_loc.append(p1p)
            elif p2 == 255:
                # print(spl2,x)
                p2p = (x, spl2)
                p2_loc.append(p2p)
            elif p3 == 255:
                # print(spl3,x)
                p3p = (x, spl3)
                p3_loc.append(p3p)


        points_coordinate = open(os.path.join(litchi_path, 'Coordinate.txt'), 'a+') # Write the picking point coordinate value in the Coordinate.txt file

        if p2_loc != []:  # Mark the intersection of the bone line and the 1/2 split line
            p2p = p2_loc[0]

            # Calculate the picking point coordinates in the original litchi image
            x1=p2p[0] # x coordinate of picking point on bone line
            y1=p2p[1] # y coordinate of picking point on bone line
            X = lab_txt[0] - (1 / 2) * W_seg + x1   # X coordinate of picking point on the original litchi image
            Y = lab_txt[1] - (1 / 2) * H_seg + y1  # Y coordinate of picking point on the original litchi image
            X=round(X)
            Y=round(Y)
            # Mark picking point
            result = cv2.circle(out, p2p, 3, (255), -1) # Mark the picking point on the bone line
            result1 = cv2.circle(original_pic, p2p, 5, (0, 0, 255), -1) # Mark the picking point on the ROI image of the main stem
            lizhi_pic_point=cv2.circle(original_lizhi_pic, (X,Y), 5, (0, 0, 255), -1) # Mark the picking point on the original litchi image

            # Write picking point coordinate value
            coordinates = original_name + "  " + str((X, Y))+"\n"
            points_coordinate.write(coordinates)

            # Save the image after marking the picking point
            pick_point = "Test_save_pic/ske_pic_save" + '\\' + first + "ppt" + str(n) + ".png"
            cv2.imwrite(pick_point, result)  # Save the image of the picking point marked on the bone line
            pick_point1 = "Test_save_pic/Roi_pic_save" + '\\' + first + "orppt" + str(n) + ".png"
            cv2.imwrite(pick_point1, result1)  # Save the image of the picking point marked on the main stem ROI
            cv2.imwrite(original_lizhi_path1, lizhi_pic_point)  # Save the image of the picking point marked on the original litchi image

        elif p2_loc == [] and p1_loc != []: # Mark the intersection of the bone line and the 3/4 split line
            p1p = p1_loc[0]

            x1 = p1p[0]
            y1 = p1p[1]
            X = lab_txt[0] - (1 / 2) * W_seg + x1
            Y = lab_txt[1] - (1 / 2) * H_seg + y1
            X=round(X)
            Y=round(Y)
            result = cv2.circle(out, p1p, 3, (255), -1)
            result1= cv2.circle(original_pic, p1p, 5, (0, 0,255), -1)
            lizhi_pic_point = cv2.circle(original_lizhi_pic, (X, Y), 5, (0, 0,255), -1)

            # Write picking point coordinate value
            coordinates = original_name + "  " + str((X, Y)) + "\n"
            points_coordinate.write(coordinates)

            # Save the image after marking the picking point
            pick_point = "Test_save_pic/ske_pic_save" + '\\' + first + "ppt" + str(n) + ".png"
            cv2.imwrite(pick_point, result)
            pick_point1 = "Test_save_pic/Roi_pic_save" + '\\' + first + "orppt" + str(n) + ".png"
            cv2.imwrite(pick_point1, result1)
            cv2.imwrite(original_lizhi_path1, lizhi_pic_point)


        elif p2_loc == [] and p1_loc == [] and p3_loc != []: # Mark the intersection of the bone line and the 1/4 split line
            p3p = p3_loc[0]

            x1 =p3p[0]
            y1 =p3p[1]
            X = lab_txt[0] - (1 / 2) * W_seg + x1
            Y = lab_txt[1] - (1 / 2) * H_seg + y1
            X=round(X)
            Y=round(Y)
            result = cv2.circle(out, p3p, 3, (255), -1)
            result1 = cv2.circle(original_pic, p3p, 5, (0, 0,255), -1)
            lizhi_pic_point = cv2.circle(original_lizhi_pic, (X, Y), 5, (0, 0,255), -1)

            # Write picking point coordinate value
            coordinates = original_name + "  " + str((X, Y)) + "\n"
            points_coordinate.write(coordinates)

            # Save the image after marking the picking point
            pick_point = "Test_save_pic/ske_pic_save" + '\\' + first + "ppt" + str(n) + ".png"
            cv2.imwrite(pick_point, result)
            pick_point1="Test_save_pic/Roi_pic_save" + '\\' + first + "orppt" + str(n) + ".png"
            cv2.imwrite(pick_point1, result1)
            cv2.imwrite(original_lizhi_path1,lizhi_pic_point)

        points_coordinate.close()

    n = n + 1
    end = time.perf_counter()
    print("Total time：",end-start)

if __name__ == "__main__":

    Seg_path = r"C:\Users\Administrator\Desktop\capture_points\code\Test_pic\Seg_pic"  # Path of main stem ROI semantic segmentation image
    ROI_path = r"C:\Users\Administrator\Desktop\capture_points\code\Test_pic\Roi_pic"  # Path of ROI image of main stem
    litchi_path = r"C:\Users\Administrator\Desktop\capture_points\code\Test_save_pic\Lichi_pic_save"  # Path of original litchi image
    lizhi_label_path = r"C:\Users\Administrator\Desktop\capture_points\code\Test_pic\Label_txt"  # Path of litchi label file

    Picking_points(Seg_path,ROI_path,litchi_path,lizhi_label_path)




>>>>>>> b201b31cc22f62ca79d4167c14e19308f24f3ead
