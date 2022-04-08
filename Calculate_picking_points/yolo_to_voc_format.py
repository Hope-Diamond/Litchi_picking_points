<<<<<<< HEAD
import os
import xml.etree.ElementTree as ET
from xml.dom.minidom import Document
import cv2

'''
import xml
xml.dom.minidom.Document().writexml()
def writexml(self,
             writer: Any,
             indent: str = "",
             addindent: str = "",
             newl: str = "",
             encoding: Any = None) -> None
'''


class YOLO2VOCConvert:
    def __init__(self, txts_path, xmls_path, imgs_path):
        self.txts_path = txts_path  # The Yolo format label file path of the label
        self.xmls_path = xmls_path  # Save path after converting to VOC format label
        self.imgs_path = imgs_path  # Read the path of film reading and the name of each picture, and store it in the XML tag file
        '''#Note: it can be changed according to your category name and category'''
        self.classes = ['0','1']  # Note: it can be changed according to your category name and category

    # Extract all categories from all txt files. The label format categories of Yolo format are numbers 0,1...
    def search_all_classes(self, writer=False):
        # Read each txt label file and take out the label information of each target
        all_names = set()
        txts = os.listdir(self.txts_path)
        txts = [txt for txt in txts if txt.split('.')[-1] == 'txt']
        print(len(txts), txts)
        for txt in txts:
            txt_file = os.path.join(self.txts_path, txt)
            with open(txt_file, 'r') as f:
                objects = f.readlines()
                for object in objects:
                    object = object.strip().split(' ')
                    print(object)
                    all_names.add(int(object[0]))
        return list(all_names)

    def yolo2voc(self):
        #  Create a folder to save the xml tag file
        if not os.path.exists(self.xmls_path):
            os.mkdir(self.xmls_path)

        imgs = os.listdir(self.imgs_path)
        txts = os.listdir(self.txts_path)
        txts = [txt for txt in txts if not txt.split('.')[0] == "classes"]
        print(txts)

        # Note: keep the number of pictures equal to the number of label txt files, and ensure that the names correspond one by one
        if len(imgs) == len(txts):
            map_imgs_txts = [(img, txt) for img, txt in zip(imgs, txts)]
            txts = [txt for txt in txts if txt.split('.')[-1] == 'txt']
            print(len(txts), txts)
            for img_name, txt_name in map_imgs_txts:
                # Read the scale information of the picture
                print("Read image：", img_name)
                img = cv2.imread(os.path.join(self.imgs_path, img_name))
                height_img, width_img, depth_img = img.shape
                print(height_img, width_img, depth_img)

                # Get the annotation information in the annotation file txt
                all_objects = []
                txt_file = os.path.join(self.txts_path, txt_name)
                with open(txt_file, 'r') as f:
                    objects = f.readlines()
                    for object in objects:
                        object = object.strip().split(' ')
                        all_objects.append(object)
                        print(object)

                # Create tags in xml tag files
                xmlBuilder = Document()
                # Create the annotation tag, which is also the root tag
                annotation = xmlBuilder.createElement("annotation")

                # Add a sub tag to the tag annotation
                xmlBuilder.appendChild(annotation)

                # Create sub tag folder
                folder = xmlBuilder.createElement("folder")
                folderContent = xmlBuilder.createTextNode(self.imgs_path.split('/')[-1])
                folder.appendChild(folderContent)
                annotation.appendChild(folder)

                # Create sub tag filename
                filename = xmlBuilder.createElement("filename")
                filenameContent = xmlBuilder.createTextNode(txt_name.split('.')[0] + '.jpg')
                filename.appendChild(filenameContent)
                annotation.appendChild(filename)

                # Store the shape of the picture into the xml tag
                size = xmlBuilder.createElement("size")
                width = xmlBuilder.createElement("width")
                widthContent = xmlBuilder.createTextNode(str(width_img))
                width.appendChild(widthContent)
                size.appendChild(width)

                height = xmlBuilder.createElement("height")
                heightContent = xmlBuilder.createTextNode(str(height_img))
                height.appendChild(heightContent)
                size.appendChild(height)

                depth = xmlBuilder.createElement("depth")
                depthContent = xmlBuilder.createTextNode(str(depth_img))
                depth.appendChild(depthContent)
                size.appendChild(depth)
                annotation.appendChild(size)


                for object_info in all_objects:
                    # Start creating labels that label the target's label information
                    object = xmlBuilder.createElement("object")

                    imgName = xmlBuilder.createElement("name")
                    imgNameContent = xmlBuilder.createTextNode(self.classes[int(object_info[0])])
                    imgName.appendChild(imgNameContent)
                    object.appendChild(imgName)

                    # Create pose tag
                    pose = xmlBuilder.createElement("pose")
                    poseContent = xmlBuilder.createTextNode("Unspecified")
                    pose.appendChild(poseContent)
                    object.appendChild(pose)

                    # Create truncated tag
                    truncated = xmlBuilder.createElement("truncated")
                    truncatedContent = xmlBuilder.createTextNode("0")
                    truncated.appendChild(truncatedContent)
                    object.appendChild(truncated)

                    # Create difficult tag
                    difficult = xmlBuilder.createElement("difficult")
                    difficultContent = xmlBuilder.createTextNode("0")
                    difficult.appendChild(difficultContent)
                    object.appendChild(difficult)

                    # Transform coordinates
                    # (objx_center, objy_center, obj_width, obj_height)->(xmin，ymin, xmax,ymax)
                    x_center = float(object_info[1]) * width_img + 1
                    y_center = float(object_info[2]) * height_img + 1
                    xminVal = int(x_center - 0.5 * float(object_info[3]) * width_img)
                    yminVal = int(y_center - 0.5 * float(object_info[4]) * height_img)
                    xmaxVal = int(x_center + 0.5 * float(object_info[3]) * width_img)
                    ymaxVal = int(y_center + 0.5 * float(object_info[4]) * height_img)

                    # Create bndbox tag
                    bndbox = xmlBuilder.createElement("bndbox")

                    # Create xmin tag
                    xmin = xmlBuilder.createElement("xmin")
                    xminContent = xmlBuilder.createTextNode(str(xminVal))
                    xmin.appendChild(xminContent)
                    bndbox.appendChild(xmin)
                    # Create ymin tag
                    ymin = xmlBuilder.createElement("ymin")
                    yminContent = xmlBuilder.createTextNode(str(yminVal))
                    ymin.appendChild(yminContent)
                    bndbox.appendChild(ymin)
                    # Create xmax tag
                    xmax = xmlBuilder.createElement("xmax")
                    xmaxContent = xmlBuilder.createTextNode(str(xmaxVal))
                    xmax.appendChild(xmaxContent)
                    bndbox.appendChild(xmax)
                    # Create ymax tag
                    ymax = xmlBuilder.createElement("ymax")
                    ymaxContent = xmlBuilder.createTextNode(str(ymaxVal))
                    ymax.appendChild(ymaxContent)
                    bndbox.appendChild(ymax)

                    object.appendChild(bndbox)
                    annotation.appendChild(object)
                f = open(os.path.join(self.xmls_path, txt_name.split('.')[0] + '.xml'), 'w')
                xmlBuilder.writexml(f, indent='\t', newl='\n', addindent='\t', encoding='utf-8')
                f.close()


if __name__ == '__main__':

    # Convert Yolo's txt label file into VOC format xml label file
    txts_path1 = './test_txt'
    xmls_path1 = './test_xml'
    imgs_path1 = './Images'

    yolo2voc_obj1 = YOLO2VOCConvert(txts_path1, xmls_path1, imgs_path1)
    labels = yolo2voc_obj1.search_all_classes()
    print('labels: ', labels)
=======
import os
import xml.etree.ElementTree as ET
from xml.dom.minidom import Document
import cv2

'''
import xml
xml.dom.minidom.Document().writexml()
def writexml(self,
             writer: Any,
             indent: str = "",
             addindent: str = "",
             newl: str = "",
             encoding: Any = None) -> None
'''


class YOLO2VOCConvert:
    def __init__(self, txts_path, xmls_path, imgs_path):
        self.txts_path = txts_path  # The Yolo format label file path of the label
        self.xmls_path = xmls_path  # Save path after converting to VOC format label
        self.imgs_path = imgs_path  # Read the path of film reading and the name of each picture, and store it in the XML tag file
        '''#Note: it can be changed according to your category name and category'''
        self.classes = ['0','1']  # Note: it can be changed according to your category name and category

    # Extract all categories from all txt files. The label format categories of Yolo format are numbers 0,1...
    def search_all_classes(self, writer=False):
        # Read each txt label file and take out the label information of each target
        all_names = set()
        txts = os.listdir(self.txts_path)
        txts = [txt for txt in txts if txt.split('.')[-1] == 'txt']
        print(len(txts), txts)
        for txt in txts:
            txt_file = os.path.join(self.txts_path, txt)
            with open(txt_file, 'r') as f:
                objects = f.readlines()
                for object in objects:
                    object = object.strip().split(' ')
                    print(object)
                    all_names.add(int(object[0]))
        return list(all_names)

    def yolo2voc(self):
        #  Create a folder to save the xml tag file
        if not os.path.exists(self.xmls_path):
            os.mkdir(self.xmls_path)

        imgs = os.listdir(self.imgs_path)
        txts = os.listdir(self.txts_path)
        txts = [txt for txt in txts if not txt.split('.')[0] == "classes"]
        print(txts)

        # Note: keep the number of pictures equal to the number of label txt files, and ensure that the names correspond one by one
        if len(imgs) == len(txts):
            map_imgs_txts = [(img, txt) for img, txt in zip(imgs, txts)]
            txts = [txt for txt in txts if txt.split('.')[-1] == 'txt']
            print(len(txts), txts)
            for img_name, txt_name in map_imgs_txts:
                # Read the scale information of the picture
                print("Read image：", img_name)
                img = cv2.imread(os.path.join(self.imgs_path, img_name))
                height_img, width_img, depth_img = img.shape
                print(height_img, width_img, depth_img)

                # Get the annotation information in the annotation file txt
                all_objects = []
                txt_file = os.path.join(self.txts_path, txt_name)
                with open(txt_file, 'r') as f:
                    objects = f.readlines()
                    for object in objects:
                        object = object.strip().split(' ')
                        all_objects.append(object)
                        print(object)

                # Create tags in xml tag files
                xmlBuilder = Document()
                # Create the annotation tag, which is also the root tag
                annotation = xmlBuilder.createElement("annotation")

                # Add a sub tag to the tag annotation
                xmlBuilder.appendChild(annotation)

                # Create sub tag folder
                folder = xmlBuilder.createElement("folder")
                folderContent = xmlBuilder.createTextNode(self.imgs_path.split('/')[-1])
                folder.appendChild(folderContent)
                annotation.appendChild(folder)

                # Create sub tag filename
                filename = xmlBuilder.createElement("filename")
                filenameContent = xmlBuilder.createTextNode(txt_name.split('.')[0] + '.jpg')
                filename.appendChild(filenameContent)
                annotation.appendChild(filename)

                # Store the shape of the picture into the xml tag
                size = xmlBuilder.createElement("size")
                width = xmlBuilder.createElement("width")
                widthContent = xmlBuilder.createTextNode(str(width_img))
                width.appendChild(widthContent)
                size.appendChild(width)

                height = xmlBuilder.createElement("height")
                heightContent = xmlBuilder.createTextNode(str(height_img))
                height.appendChild(heightContent)
                size.appendChild(height)

                depth = xmlBuilder.createElement("depth")
                depthContent = xmlBuilder.createTextNode(str(depth_img))
                depth.appendChild(depthContent)
                size.appendChild(depth)
                annotation.appendChild(size)


                for object_info in all_objects:
                    # Start creating labels that label the target's label information
                    object = xmlBuilder.createElement("object")

                    imgName = xmlBuilder.createElement("name")
                    imgNameContent = xmlBuilder.createTextNode(self.classes[int(object_info[0])])
                    imgName.appendChild(imgNameContent)
                    object.appendChild(imgName)

                    # Create pose tag
                    pose = xmlBuilder.createElement("pose")
                    poseContent = xmlBuilder.createTextNode("Unspecified")
                    pose.appendChild(poseContent)
                    object.appendChild(pose)

                    # Create truncated tag
                    truncated = xmlBuilder.createElement("truncated")
                    truncatedContent = xmlBuilder.createTextNode("0")
                    truncated.appendChild(truncatedContent)
                    object.appendChild(truncated)

                    # Create difficult tag
                    difficult = xmlBuilder.createElement("difficult")
                    difficultContent = xmlBuilder.createTextNode("0")
                    difficult.appendChild(difficultContent)
                    object.appendChild(difficult)

                    # Transform coordinates
                    # (objx_center, objy_center, obj_width, obj_height)->(xmin，ymin, xmax,ymax)
                    x_center = float(object_info[1]) * width_img + 1
                    y_center = float(object_info[2]) * height_img + 1
                    xminVal = int(x_center - 0.5 * float(object_info[3]) * width_img)
                    yminVal = int(y_center - 0.5 * float(object_info[4]) * height_img)
                    xmaxVal = int(x_center + 0.5 * float(object_info[3]) * width_img)
                    ymaxVal = int(y_center + 0.5 * float(object_info[4]) * height_img)

                    # Create bndbox tag
                    bndbox = xmlBuilder.createElement("bndbox")

                    # Create xmin tag
                    xmin = xmlBuilder.createElement("xmin")
                    xminContent = xmlBuilder.createTextNode(str(xminVal))
                    xmin.appendChild(xminContent)
                    bndbox.appendChild(xmin)
                    # Create ymin tag
                    ymin = xmlBuilder.createElement("ymin")
                    yminContent = xmlBuilder.createTextNode(str(yminVal))
                    ymin.appendChild(yminContent)
                    bndbox.appendChild(ymin)
                    # Create xmax tag
                    xmax = xmlBuilder.createElement("xmax")
                    xmaxContent = xmlBuilder.createTextNode(str(xmaxVal))
                    xmax.appendChild(xmaxContent)
                    bndbox.appendChild(xmax)
                    # Create ymax tag
                    ymax = xmlBuilder.createElement("ymax")
                    ymaxContent = xmlBuilder.createTextNode(str(ymaxVal))
                    ymax.appendChild(ymaxContent)
                    bndbox.appendChild(ymax)

                    object.appendChild(bndbox)
                    annotation.appendChild(object)
                f = open(os.path.join(self.xmls_path, txt_name.split('.')[0] + '.xml'), 'w')
                xmlBuilder.writexml(f, indent='\t', newl='\n', addindent='\t', encoding='utf-8')
                f.close()


if __name__ == '__main__':

    # Convert Yolo's txt label file into VOC format xml label file
    txts_path1 = './test_txt'
    xmls_path1 = './test_xml'
    imgs_path1 = './Images'

    yolo2voc_obj1 = YOLO2VOCConvert(txts_path1, xmls_path1, imgs_path1)
    labels = yolo2voc_obj1.search_all_classes()
    print('labels: ', labels)
>>>>>>> b201b31cc22f62ca79d4167c14e19308f24f3ead
    yolo2voc_obj1.yolo2voc()