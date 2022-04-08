<<<<<<< HEAD
import cv2
import os
import numpy as np

path="Seg_2zhihua/01seg_dalvsuanfa"
file=os.listdir(path)
print(file)
n=1
for image_name in file:
    first,last=os.path.splitext(image_name)
    path1=os.path.join(path,image_name)
    image=cv2.imread(path1,-1)
    # Create convolution kernel
    kernel = np.ones((8, 8), np.uint8)
    # Morphological open operation
    cvOpen = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    # Save results
    open_name = "Seg_open/08" + '\\' + first + "_08op" + str(n) + ".png"
    cv2.imwrite(open_name , cvOpen)
    n=n+1

=======
import cv2
import os
import numpy as np

path="Seg_2zhihua/01seg_dalvsuanfa"
file=os.listdir(path)
print(file)
n=1
for image_name in file:
    first,last=os.path.splitext(image_name)
    path1=os.path.join(path,image_name)
    image=cv2.imread(path1,-1)
    # Create convolution kernel
    kernel = np.ones((8, 8), np.uint8)
    # Morphological open operation
    cvOpen = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    # Save results
    open_name = "Seg_open/08" + '\\' + first + "_08op" + str(n) + ".png"
    cv2.imwrite(open_name , cvOpen)
    n=n+1

>>>>>>> b201b31cc22f62ca79d4167c14e19308f24f3ead
