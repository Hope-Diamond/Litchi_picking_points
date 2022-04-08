# -*- coding: utf-8 -*-

import os
from easydict import EasyDict

Cfg = EasyDict()

Cfg.names = ['combustion_lining', 'fan', 'fan_stator_casing_and_support', 'hp_core_casing', 'hpc_spool', 'hpc_stage_5',
             'mixer', 'nozzle', 'nozzle_cone', 'stand']
Cfg.textnames = ['combustion', 'fan', 'stator', 'core', 'spool', 'stage', 'mixer', 'nozzle', 'cone', 'stand']

Cfg.device = '0,1'

# manual
Cfg.origimgs_filepath = '../data_test/JPEGImages_manual'
Cfg.testimgs_filepath = '../data_test/JPEGImages_manual'
Cfg.eval_classtxt_path = '../data_test/class_txt_manual/'
Cfg.eval_Annotations_path = '../data_test/Annotations_manual'
Cfg.eval_imgs_name_txt = '../data_test/imgs_name_manual.txt'
Cfg.cachedir = '../data_test/cachedir_manual/'
Cfg.prediction_path = '../data_test/predictions_manual'

# mAP_line cachedir
Cfg.systhesis_valid_cachedir = '../data_test/cachedir_systhesis_valid/'
Cfg.manual_cachedir = '../data_test/cachedir_manual/'
