# -*- coding: utf-8 -*-
import os
import numpy as np
from yolov5_eval import yolov5_eval  # 注意将yolov4_eval.py和compute_mAP.py放在同一级目录下
from cfg_mAP import Cfg
import pickle
import shutil

cfg = Cfg
eval_classtxt_path = cfg.eval_classtxt_path  # 各类txt文件路径
eval_classtxt_files = os.listdir(eval_classtxt_path)

classes = cfg.names  # ['combustion_lining', 'fan', 'fan_stator_casing_and_support', 'hp_core_casing', 'hpc_spool', 'hpc_stage_5','mixer', 'nozzle', 'nozzle_cone', 'stand']

aps = []  # 保存各类ap
cls_rec = {}  # 保存recall
cls_prec = {}  # 保存精度
cls_ap = {}

annopath = cfg.eval_Annotations_path + '/{:s}.xml'  # annotations的路径，{:s}.xml方便后面根据图像名字读取对应的xml文件
imagesetfile = cfg.eval_imgs_name_txt  # 读取图像名字列表文件
cachedir = cfg.cachedir

if os.path.exists(cachedir):
    shutil.rmtree(cachedir)  # delete output folder
os.makedirs(cachedir)  # make new output folder

for cls in eval_classtxt_files:  # 读取cls类对应的txt文件
    filename = eval_classtxt_path + cls

    rec, prec, ap = yolov5_eval(  # yolov4_eval.py计算cls类的recall precision ap
        filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5,
        use_07_metric=False)

    aps += [ap]
    cls_ap[cls] = ap
    cls_rec[cls] = rec[-1]
    cls_prec[cls] = prec[-1]

    print('AP for {} = {:.4f}'.format(cls, ap))
    print('recall for {} = {:.4f}'.format(cls, rec[-1]))
    print('precision for {} = {:.4f}'.format(cls, prec[-1]))

with open(os.path.join(cfg.cachedir, 'cls_ap.pkl'), 'wb') as in_data:
    pickle.dump(cls_ap, in_data, pickle.HIGHEST_PROTOCOL)

with open(os.path.join(cfg.cachedir, 'cls_rec.pkl'), 'wb') as in_data:
    pickle.dump(cls_rec, in_data, pickle.HIGHEST_PROTOCOL)

with open(os.path.join(cfg.cachedir, 'cls_prec.pkl'), 'wb') as in_data:
    pickle.dump(cls_prec, in_data, pickle.HIGHEST_PROTOCOL)

print('Mean AP = {:.4f}'.format(np.mean(aps)))
print('~~~~~~~~')

print('Results:')
for ap in aps:
    print('{:.3f}'.format(ap))
print('~~~~~~~~')
print('{:.3f}'.format(np.mean(aps)))
print('~~~~~~~~')