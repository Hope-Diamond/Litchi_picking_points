import sys
import os
import time
import math
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torch.autograd import Variable

import itertools
import struct  # get_image_size
import imghdr  # get_image_size


def sigmoid(x):
    return 1.0 / (np.exp(-x) + 1.)


def softmax(x):
    x = np.exp(x - np.expand_dims(np.max(x, axis=1), axis=1))
    x = x / np.expand_dims(x.sum(axis=1), axis=1)
    return x


def bbox_iou(box1, box2, x1y1x2y2=True):
    if x1y1x2y2:
        mx = min(box1[0], box2[0])
        Mx = max(box1[2], box2[2])
        my = min(box1[1], box2[1])
        My = max(box1[3], box2[3])
        w1 = box1[2] - box1[0]
        h1 = box1[3] - box1[1]
        w2 = box2[2] - box2[0]
        h2 = box2[3] - box2[1]
    else:
        mx = min(box1[0] - box1[2] / 2.0, box2[0] - box2[2] / 2.0)
        Mx = max(box1[0] + box1[2] / 2.0, box2[0] + box2[2] / 2.0)
        my = min(box1[1] - box1[3] / 2.0, box2[1] - box2[3] / 2.0)
        My = max(box1[1] + box1[3] / 2.0, box2[1] + box2[3] / 2.0)
        w1 = box1[2]
        h1 = box1[3]
        w2 = box2[2]
        h2 = box2[3]
    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
    carea = 0
    if cw <= 0 or ch <= 0:
        return 0.0

    area1 = w1 * h1
    area2 = w2 * h2
    carea = cw * ch
    uarea = area1 + area2 - carea
    return carea / uarea


def bbox_ious(boxes1, boxes2, x1y1x2y2=True):
    if x1y1x2y2:
        mx = torch.min(boxes1[0], boxes2[0])
        Mx = torch.max(boxes1[2], boxes2[2])
        my = torch.min(boxes1[1], boxes2[1])
        My = torch.max(boxes1[3], boxes2[3])
        w1 = boxes1[2] - boxes1[0]
        h1 = boxes1[3] - boxes1[1]
        w2 = boxes2[2] - boxes2[0]
        h2 = boxes2[3] - boxes2[1]
    else:
        mx = torch.min(boxes1[0] - boxes1[2] / 2.0, boxes2[0] - boxes2[2] / 2.0)
        Mx = torch.max(boxes1[0] + boxes1[2] / 2.0, boxes2[0] + boxes2[2] / 2.0)
        my = torch.min(boxes1[1] - boxes1[3] / 2.0, boxes2[1] - boxes2[3] / 2.0)
        My = torch.max(boxes1[1] + boxes1[3] / 2.0, boxes2[1] + boxes2[3] / 2.0)
        w1 = boxes1[2]
        h1 = boxes1[3]
        w2 = boxes2[2]
        h2 = boxes2[3]
    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
    mask = ((cw <= 0) + (ch <= 0) > 0)
    area1 = w1 * h1
    area2 = w2 * h2
    carea = cw * ch
    carea[mask] = 0
    uarea = area1 + area2 - carea
    return carea / uarea


def nms(_boxes, _nms_thresh):
    if len(_boxes) == 0:
        return _boxes

    det_confs = torch.zeros(len(_boxes))
    for i in range(len(_boxes)):
        det_confs[i] = 1 - _boxes[i][4]

    _, sortIds = torch.sort(det_confs)
    out_boxes = []
    for i in range(len(_boxes)):
        box_i = _boxes[sortIds[i]]
        if box_i[4] > 0:
            out_boxes.append(box_i)
            for j in range(i + 1, len(_boxes)):
                box_j = _boxes[sortIds[j]]
                if bbox_iou(box_i, box_j, x1y1x2y2=False) > _nms_thresh:
                    # print(box_i, box_j, bbox_iou(box_i, box_j, x1y1x2y2=False))
                    box_j[4] = 0
    return out_boxes


def convert2cpu(gpu_matrix):
    return torch.FloatTensor(gpu_matrix.size()).copy_(gpu_matrix)


def convert2cpu_long(gpu_matrix):
    return torch.LongTensor(gpu_matrix.size()).copy_(gpu_matrix)


def get_region_boxes_in_model(output, conf_thresh, num_classes, anchors, num_anchors, only_objectness=1,
                              validation=False):
    anchor_step = len(anchors) // num_anchors
    if output.dim() == 3:
        output = output.unsqueeze(0)
    batch = output.size(0)
    assert (output.size(1) == (5 + num_classes) * num_anchors)
    h = output.size(2)
    w = output.size(3)

    t0 = time.time()
    all_boxes = []
    output = output.view(batch * num_anchors, 5 + num_classes, h * w).transpose(0, 1).contiguous().view(5 + num_classes,
                                                                                                        batch * num_anchors * h * w)

    grid_x = torch.linspace(0, w - 1, w).repeat(h, 1).repeat(batch * num_anchors, 1, 1).view(
        batch * num_anchors * h * w).type_as(output)  # cuda()
    grid_y = torch.linspace(0, h - 1, h).repeat(w, 1).t().repeat(batch * num_anchors, 1, 1).view(
        batch * num_anchors * h * w).type_as(output)  # cuda()
    xs = torch.sigmoid(output[0]) + grid_x
    ys = torch.sigmoid(output[1]) + grid_y

    anchor_w = torch.Tensor(anchors).view(num_anchors, anchor_step).index_select(1, torch.LongTensor([0]))
    anchor_h = torch.Tensor(anchors).view(num_anchors, anchor_step).index_select(1, torch.LongTensor([1]))
    anchor_w = anchor_w.repeat(batch, 1).repeat(1, 1, h * w).view(batch * num_anchors * h * w).type_as(output)  # cuda()
    anchor_h = anchor_h.repeat(batch, 1).repeat(1, 1, h * w).view(batch * num_anchors * h * w).type_as(output)  # cuda()
    ws = torch.exp(output[2]) * anchor_w
    hs = torch.exp(output[3]) * anchor_h

    det_confs = torch.sigmoid(output[4])

    cls_confs = torch.nn.Softmax()(Variable(output[5:5 + num_classes].transpose(0, 1))).data
    cls_max_confs, cls_max_ids = torch.max(cls_confs, 1)
    cls_max_confs = cls_max_confs.view(-1)
    cls_max_ids = cls_max_ids.view(-1)
    t1 = time.time()

    sz_hw = h * w
    sz_hwa = sz_hw * num_anchors
    det_confs = convert2cpu(det_confs)
    cls_max_confs = convert2cpu(cls_max_confs)
    cls_max_ids = convert2cpu_long(cls_max_ids)
    xs = convert2cpu(xs)
    ys = convert2cpu(ys)
    ws = convert2cpu(ws)
    hs = convert2cpu(hs)
    if validation:
        cls_confs = convert2cpu(cls_confs.view(-1, num_classes))
    t2 = time.time()
    for b in range(batch):
        boxes = []
        for cy in range(h):
            for cx in range(w):
                for i in range(num_anchors):
                    ind = b * sz_hwa + i * sz_hw + cy * w + cx
                    det_conf = det_confs[ind]
                    if only_objectness:
                        conf = det_confs[ind]
                    else:
                        conf = det_confs[ind] * cls_max_confs[ind]

                    if conf > conf_thresh:
                        bcx = xs[ind]
                        bcy = ys[ind]
                        bw = ws[ind]
                        bh = hs[ind]
                        cls_max_conf = cls_max_confs[ind]
                        cls_max_id = cls_max_ids[ind]
                        box = [bcx / w, bcy / h, bw / w, bh / h, det_conf, cls_max_conf, cls_max_id]
                        if (not only_objectness) and validation:
                            for c in range(num_classes):
                                tmp_conf = cls_confs[ind][c]
                                if c != cls_max_id and det_confs[ind] * tmp_conf > conf_thresh:
                                    box.append(tmp_conf)
                                    box.append(c)
                        boxes.append(box)
        all_boxes.append(boxes)
    t3 = time.time()
    if False:
        print('---------------------------------')
        print('matrix computation : %f' % (t1 - t0))
        print('        gpu to cpu : %f' % (t2 - t1))
        print('      tpz filter : %f' % (t3 - t2))
        print('---------------------------------')
    return all_boxes


def get_region_boxes_out_model(_output, _cfg, _anchors, _num_anchors, _only_objectness=1, _validation=False):
    anchor_step = len(_anchors) // _num_anchors
    if len(_output.shape) == 3:
        _output = np.expand_dims(_output, axis=0)
    batch = _output.shape[0]
    assert (_output.shape[1] == (5 + _cfg.classes) * _num_anchors)
    h = _output.shape[2]
    w = _output.shape[3]

    t0 = time.time()
    all_boxes = []
    _output = _output.reshape(batch * _num_anchors, 5 + _cfg.classes, h * w).transpose((1, 0, 2)).reshape(
        5 + _cfg.classes,
        batch * _num_anchors * h * w)

    grid_x = np.expand_dims(np.expand_dims(np.linspace(0, w - 1, w), axis=0).repeat(h, 0), axis=0).repeat(
        batch * _num_anchors, axis=0).reshape(
        batch * _num_anchors * h * w)
    grid_y = np.expand_dims(np.expand_dims(np.linspace(0, h - 1, h), axis=0).repeat(w, 0).T, axis=0).repeat(
        batch * _num_anchors, axis=0).reshape(
        batch * _num_anchors * h * w)

    xs = sigmoid(_output[0]) + grid_x
    ys = sigmoid(_output[1]) + grid_y

    anchor_w = np.array(_anchors).reshape((_num_anchors, anchor_step))[:, 0]
    anchor_h = np.array(_anchors).reshape((_num_anchors, anchor_step))[:, 1]
    anchor_w = np.expand_dims(np.expand_dims(anchor_w, axis=1).repeat(batch, 1), axis=2) \
        .repeat(h * w, axis=2).transpose(1, 0, 2).reshape(batch * _num_anchors * h * w)
    anchor_h = np.expand_dims(np.expand_dims(anchor_h, axis=1).repeat(batch, 1), axis=2) \
        .repeat(h * w, axis=2).transpose(1, 0, 2).reshape(batch * _num_anchors * h * w)
    ws = np.exp(_output[2]) * anchor_w
    hs = np.exp(_output[3]) * anchor_h

    det_confs = sigmoid(_output[4])

    cls_confs = softmax(_output[5:5 + _cfg.classes].transpose(1, 0))
    cls_max_confs = np.max(cls_confs, 1)
    cls_max_ids = np.argmax(cls_confs, 1)
    t1 = time.time()

    sz_hw = h * w
    sz_hwa = sz_hw * _num_anchors
    t2 = time.time()
    for b in range(batch):
        boxes = []
        for cy in range(h):
            for cx in range(w):
                for i in range(_num_anchors):
                    ind = b * sz_hwa + i * sz_hw + cy * w + cx
                    det_conf = det_confs[ind]
                    if _only_objectness:
                        conf = det_confs[ind]
                    else:
                        conf = det_confs[ind] * cls_max_confs[ind]

                    if conf > _cfg.conf_thresh:
                        bcx = xs[ind]
                        bcy = ys[ind]
                        bw = ws[ind]
                        bh = hs[ind]
                        cls_max_conf = cls_max_confs[ind]
                        cls_max_id = cls_max_ids[ind]
                        box = [bcx / w, bcy / h, bw / w, bh / h, det_conf, cls_max_conf, cls_max_id]
                        if (not _only_objectness) and _validation:
                            for c in range(_cfg.classes):
                                tmp_conf = cls_confs[ind][c]
                                if c != cls_max_id and det_confs[ind] * tmp_conf > _cfg.conf_thresh:
                                    box.append(tmp_conf)
                                    box.append(c)
                        boxes.append(box)
        all_boxes.append(boxes)
    t3 = time.time()
    if False:
        print('---------------------------------')
        print('matrix computation : %f' % (t1 - t0))
        print('        gpu to cpu : %f' % (t2 - t1))
        print('      tpz filter : %f' % (t3 - t2))
        print('---------------------------------')
    return all_boxes


def get_classtxt_out_model(_output, _cfg, _anchors, _num_anchors, _only_objectness=1, _validation=False):
    anchor_step = len(_anchors) // _num_anchors
    if len(_output.shape) == 3:
        _output = np.expand_dims(_output, axis=0)
    batch = _output.shape[0]
    assert (_output.shape[1] == (5 + _cfg.n_classes) * _num_anchors)
    h = _output.shape[2]
    w = _output.shape[3]

    t0 = time.time()
    all_boxes = []
    _output = _output.reshape(batch * _num_anchors, 5 + _cfg.n_classes, h * w).transpose((1, 0, 2)).reshape(
        5 + _cfg.n_classes,
        batch * _num_anchors * h * w)

    grid_x = np.expand_dims(np.expand_dims(np.linspace(0, w - 1, w), axis=0).repeat(h, 0), axis=0).repeat(
        batch * _num_anchors, axis=0).reshape(
        batch * _num_anchors * h * w)
    grid_y = np.expand_dims(np.expand_dims(np.linspace(0, h - 1, h), axis=0).repeat(w, 0).T, axis=0).repeat(
        batch * _num_anchors, axis=0).reshape(
        batch * _num_anchors * h * w)

    xs = sigmoid(_output[0]) + grid_x
    ys = sigmoid(_output[1]) + grid_y

    anchor_w = np.array(_anchors).reshape((_num_anchors, anchor_step))[:, 0]
    anchor_h = np.array(_anchors).reshape((_num_anchors, anchor_step))[:, 1]
    anchor_w = np.expand_dims(np.expand_dims(anchor_w, axis=1).repeat(batch, 1), axis=2) \
        .repeat(h * w, axis=2).transpose(1, 0, 2).reshape(batch * _num_anchors * h * w)
    anchor_h = np.expand_dims(np.expand_dims(anchor_h, axis=1).repeat(batch, 1), axis=2) \
        .repeat(h * w, axis=2).transpose(1, 0, 2).reshape(batch * _num_anchors * h * w)
    ws = np.exp(_output[2]) * anchor_w
    hs = np.exp(_output[3]) * anchor_h

    det_confs = sigmoid(_output[4])

    cls_confs = softmax(_output[5:5 + _cfg.n_classes].transpose(1, 0))
    cls_max_confs = np.max(cls_confs, 1)
    cls_max_ids = np.argmax(cls_confs, 1)
    t1 = time.time()

    sz_hw = h * w
    sz_hwa = sz_hw * _num_anchors
    t2 = time.time()
    for b in range(batch):
        boxes = []
        for cy in range(h):
            for cx in range(w):
                for i in range(_num_anchors):
                    ind = b * sz_hwa + i * sz_hw + cy * w + cx
                    det_conf = det_confs[ind]
                    if _only_objectness:
                        conf = det_confs[ind]
                    else:
                        conf = det_confs[ind] * cls_max_confs[ind]

                    if conf > _cfg.conf_thresh:
                        bcx = xs[ind]
                        bcy = ys[ind]
                        bw = ws[ind]
                        bh = hs[ind]
                        cls_max_conf = cls_max_confs[ind]
                        cls_max_id = cls_max_ids[ind]
                        box = [bcx / w, bcy / h, bw / w, bh / h, det_conf, cls_max_conf, cls_max_id]
                        if (not _only_objectness) and _validation:
                            for c in range(_cfg.classes):
                                tmp_conf = cls_confs[ind][c]
                                if c != cls_max_id and det_confs[ind] * tmp_conf > _cfg.conf_thresh:
                                    box.append(tmp_conf)
                                    box.append(c)
                        boxes.append(box)
        all_boxes.append(boxes)
    t3 = time.time()
    if False:
        print('---------------------------------')
        print('matrix computation : %f' % (t1 - t0))
        print('        gpu to cpu : %f' % (t2 - t1))
        print('      tpz filter : %f' % (t3 - t2))
        print('---------------------------------')
    return all_boxes


def plot_boxes_cv2(img, boxes, savename=None, class_names=None, color=None):
    import cv2
    colors = torch.FloatTensor([[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]]);

    def get_color(c, x, max_val):
        ratio = float(x) / max_val * 5
        i = int(math.floor(ratio))
        j = int(math.ceil(ratio))
        ratio = ratio - i
        r = (1 - ratio) * colors[i][c] + ratio * colors[j][c]
        return int(r * 255)

    width = img.shape[1]
    height = img.shape[0]
    for i in range(len(boxes)):
        box = boxes[i]
        x1 = int((box[0] - box[2] / 2.0) * width)
        y1 = int((box[1] - box[3] / 2.0) * height)
        x2 = int((box[0] + box[2] / 2.0) * width)
        y2 = int((box[1] + box[3] / 2.0) * height)

        if color:
            rgb = color
        else:
            rgb = (255, 0, 0)
        if len(box) >= 7 and class_names:
            cls_conf = box[5]
            cls_id = box[6]
            print('%s: %f' % (class_names[cls_id], cls_conf))
            classes = len(class_names)
            offset = cls_id * 123457 % classes
            red = get_color(2, offset, classes)
            green = get_color(1, offset, classes)
            blue = get_color(0, offset, classes)
            if color is None:
                rgb = (red, green, blue)
            img = cv2.putText(img, class_names[cls_id], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.2, rgb, 1)
        img = cv2.rectangle(img, (x1, y1), (x2, y2), rgb, 1)
    if savename:
        print("save plot results to %s" % savename)
        cv2.imwrite(savename, img)
    return img


def plot_boxes(_img,  _boxes, _savename=None, _class_names=None):
    font = ImageFont.truetype("consola.ttf", 40, encoding="unic")  # 设置字体

    colors = torch.FloatTensor([[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]]);

    def get_color(c, x, max_val):
        ratio = float(x) / max_val * 5
        i = int(math.floor(ratio))
        j = int(math.ceil(ratio))
        ratio = ratio - i
        r = (1 - ratio) * colors[i][c] + ratio * colors[j][c]
        return int(r * 255)

    # width = _img.shape[1]
    # height = _img.shape[0]
    draw = ImageDraw.Draw(_img)
    for i in range(len(_boxes)):
        box = _boxes[i]
        x1 = box[0]
        y1 = box[1]
        x2 = box[2]
        y2 = box[3]

        rgb = (255, 0, 0)
        if len(box) >= 7 and _class_names:
            cls_conf = box[5]
            cls_id = box[6]
            print('%s: %f' % (_class_names[cls_id], cls_conf))
            classes = len(_class_names)
            offset = cls_id * 123457 % classes
            red = get_color(2, offset, classes)
            green = get_color(1, offset, classes)
            blue = get_color(0, offset, classes)
            rgb = (red, green, blue)
            # draw.text((x1, y1), _class_names[cls_id], fill=rgb, font=font)
            draw.text((x1, y1), _class_names[cls_id], fill=rgb, font=font)
        draw.rectangle([x1, y1, x2, y2], outline=rgb, width=5)
    if _savename:
        print("save plot results to %s" % _savename)
        _img.save(_savename)
    return _img


def read_truths(lab_path):
    if not os.path.exists(lab_path):
        return np.array([])
    if os.path.getsize(lab_path):
        truths = np.loadtxt(lab_path)
        truths = truths.reshape(truths.size / 5, 5)  # to avoid single truth problem
        return truths
    else:
        return np.array([])


def load_class_names(_namesfile):
    class_names = []
    with open(_namesfile, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.rstrip()
        class_names.append(line)
    return class_names


def do_detect(_model, _img, _cfg, _use_cuda=1):
    _model.eval()
    t0 = time.time()

    if isinstance(_img, Image.Image):
        width = _img.width
        height = _img.height
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(_img.tobytes()))
        img = img.view(height, width, 3).transpose(0, 1).transpose(0, 2).contiguous()
        img = img.view(1, 3, height, width)
        img = img.float().div(255.0)
    elif type(_img) == np.ndarray and len(_img.shape) == 3:  # cv2 image
        img = torch.from_numpy(_img.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
    elif type(_img) == np.ndarray and len(_img.shape) == 4:
        img = torch.from_numpy(_img.transpose(0, 3, 1, 2)).float().div(255.0)
    else:
        print("unknow image type")
        exit(-1)

    t1 = time.time()

    if _use_cuda:
        img = img.cuda()
    img = torch.autograd.Variable(img)
    t2 = time.time()

    list_features = _model(img)

    list_features_numpy = []
    for feature in list_features:
        list_features_numpy.append(feature.data.cup().numpy())

    return post_processing(_img=img, _cfg=_cfg, _list_features_numpy=list_features_numpy, _t0=t0, _t1=t1, _t2=t2)


def post_processing(_img, _cfg, _list_features_numpy, _t0, _t1, _t2):
    anchor_step = len(_cfg.anchors) // _cfg.num_anchors
    boxes = []
    for i in range(3):
        masked_anchors = []
        for m in _cfg.anchor_masks[i]:
            masked_anchors += _cfg.anchors[m * anchor_step:(m + 1) * anchor_step]
        masked_anchors = [anchor / _cfg.strides[i] for anchor in masked_anchors]
        boxes.append(get_region_boxes_out_model(_output=_list_features_numpy[i], _cfg=_cfg, _anchors=masked_anchors,
                                                _num_anchors=len(_cfg.anchor_masks[i])))
    if _img.shape[0] > 1:
        bboxs_for_imgs = [
            boxes[0][index] + boxes[1][index] + boxes[2][index]
            for index in range(_img.shape[0])]
        # 分别对每一张图片的结果进行nms
        t3 = time.time()
        boxes = [nms(_boxes=bboxs, _nms_thresh=_cfg.nms_thresh) for bboxs in bboxs_for_imgs]
    else:
        boxes = boxes[0][0] + boxes[1][0] + boxes[2][0]
        t3 = time.time()
        boxes = nms(boxes, _cfg.nms_thresh)
    t4 = time.time()

    if True:
        print('-----------------------------------')
        print(' image to tensor : %f' % (_t1 - _t0))
        print('  tensor to cuda : %f' % (_t2 - _t1))
        print('         predict : %f' % (t3 - _t2))
        print('             nms : %f' % (t4 - t3))
        print('           total : %f' % (t4 - _t0))
        print('-----------------------------------')
    return boxes


def classtxt_processing(_img, _cfg, _list_features_numpy, _t0, _t1, _t2):
    anchor_step = len(_cfg.anchors) // _cfg.num_anchors
    boxes = []
    for i in range(3):
        masked_anchors = []
        for m in _cfg.anchor_masks[i]:
            masked_anchors += _cfg.anchors[m * anchor_step:(m + 1) * anchor_step]
        masked_anchors = [anchor / _cfg.strides[i] for anchor in masked_anchors]
        boxes.append(get_classtxt_out_model(_output=_list_features_numpy[i], _cfg=_cfg, _anchors=masked_anchors,
                                            _num_anchors=len(_cfg.anchor_masks[i])))
    if _img.shape[0] > 1:
        bboxs_for_imgs = [
            boxes[0][index] + boxes[1][index] + boxes[2][index]
            for index in range(_img.shape[0])]
        # 分别对每一张图片的结果进行nms
        t3 = time.time()
        boxes = [nms(_boxes=bboxs, _nms_thresh=_cfg.nms_thresh) for bboxs in bboxs_for_imgs]
    else:
        boxes = boxes[0][0] + boxes[1][0] + boxes[2][0]
        t3 = time.time()
        boxes = nms(boxes, _cfg.nms_thresh)
    t4 = time.time()

    if True:
        print('-----------------------------------')
        print(' image to tensor : %f' % (_t1 - _t0))
        print('  tensor to cuda : %f' % (_t2 - _t1))
        print('         predict : %f' % (t3 - _t2))
        print('             nms : %f' % (t4 - t3))
        print('           total : %f' % (t4 - _t0))
        print('-----------------------------------')
    return boxes


def gen_cls_txt(_model, _img, _cfg, _use_cuda):
    _model.eval()
    t0 = time.time()

    if isinstance(_img, Image.Image):
        width = _img.width
        height = _img.height
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(_img.tobytes()))
        img = img.view(height, width, 3).transpose(0, 1).transpose(0, 2).contiguous()
        img = img.view(1, 3, height, width)
        img = img.float().div(255.0)
    elif type(_img) == np.ndarray and len(_img.shape) == 3:  # cv2 image
        img = torch.from_numpy(_img.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
    elif type(_img) == np.ndarray and len(_img.shape) == 4:
        img = torch.from_numpy(_img.transpose(0, 3, 1, 2)).float().div(255.0)
    else:
        print("unknow image type")
        exit(-1)

    t1 = time.time()

    if _use_cuda:
        img = img.cuda()
    img = torch.autograd.Variable(img)
    t2 = time.time()

    list_features = _model(img)

    list_features_numpy = []
    for feature in list_features:
        list_features_numpy.append(feature.data.cpu().numpy())

    return classtxt_processing(_img=img, _cfg=_cfg, _list_features_numpy=list_features_numpy, _t0=t0, _t1=t1, _t2=t2)
